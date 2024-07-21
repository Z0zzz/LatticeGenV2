# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch OPT model."""
from typing import List, Optional, Tuple, Union
import copy
import inspect

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import AutoTokenizer
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.generation.configuration_utils import GenerationConfig
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_opt import OPTConfig
from transformers.generation.utils import GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput, GreedySearchOutput
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/opt-350m"
_CONFIG_FOR_DOC = "OPTConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ArthurZ/opt-350m-dummy-sc"
_SEQ_CLASS_EXPECTED_LOSS = 1.71
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_0'"

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://huggingface.co/models?filter=opt
]


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    # print("_make_causal_mask mask: ", mask.shape)
    # print("_make_causal_mask past_key_values_length", past_key_values_length)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class OPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=config.enable_bias,
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


OPT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OPTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTPreTrainedModel(PreTrainedModel):
    config_class = OPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPTDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (OPTDecoder)):
            module.gradient_checkpointing = value


OPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class OPTDecoder(OPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]

    Args:
        config: OPTConfig
    """

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        elif attention_mask.shape[1] != mask_seq_length:
            raise ValueError(
                f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                f"{mask_seq_length} (sum of the lengths of current and past inputs)"
            )
        causal_attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = OPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


class OPTForCausalLM(OPTPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)
        self.times = []
        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.sample`] directly. Use generate() instead.
        For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.SampleDecoderOnlyOutput`], [`~generation.SampleEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.SampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.SampleEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     TopKLogitsWarper,
        ...     TemperatureLogitsWarper,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.config.pad_token_id = model.config.eos_token_id
        >>> model.generation_config.pad_token_id = model.config.eos_token_id

        >>> input_prompt = "Today is a beautiful day, and"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> # instantiate logits processors
        >>> logits_warper = LogitsProcessorList(
        ...     [
        ...         TopKLogitsWarper(50),
        ...         TemperatureLogitsWarper(0.7),
        ...     ]
        ... )

        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
        >>> outputs = model.sample(
        ...     input_ids,
        ...     logits_processor=logits_processor,
        ...     logits_warper=logits_warper,
        ...     stopping_criteria=stopping_criteria,
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Today is a beautiful day, and we must do everything possible to make it a day of celebration.']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            import time
            start = time.time()
            # forward pass to get next token
            with torch.no_grad():
                
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            # if return_dict_in_generate:
            #     if output_scores:
            #         scores += (next_token_scores,)
            #     if output_attentions:
            #         decoder_attentions += (
            #             (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
            #         )
            #         if self.config.is_encoder_decoder:
            #             cross_attentions += (outputs.cross_attentions,)

            #     if output_hidden_states:
            #         decoder_hidden_states += (
            #             (outputs.decoder_hidden_states,)
            #             if self.config.is_encoder_decoder
            #             else (outputs.hidden_states,)
            #         )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            end = time.time()
            self.times.append(end - start)

            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


@add_start_docstrings(
    """
    The OPT Model transformer with a sequence classification head on top (linear layer).

    [`OPTForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    OPT_START_DOCSTRING,
)
class OPTForSequenceClassification(OPTPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = OPTModel(config)
        self.score = nn.Linear(config.word_embed_proj_dim, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        for_argmax_noise_generation: Optional[bool] = False,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value


@add_start_docstrings(
    """
    The OPT Model transformer with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    OPT_START_DOCSTRING,
)
class OPTForQuestionAnswering(OPTPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.model = OPTModel(config)
        self.qa_outputs = nn.Linear(config.word_embed_proj_dim, 2)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForQuestionAnswering
        >>> import torch

        >>> torch.manual_seed(4)  # doctest: +IGNORE_RESULT
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> # note: we are loading a OPTForQuestionAnswering from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random
        >>> model = OPTForQuestionAnswering.from_pretrained("facebook/opt-350m")

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

        >>> inputs = tokenizer(question, text, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> answer_start_index = outputs.start_logits.argmax()
        >>> answer_end_index = outputs.end_logits.argmax()

        >>> answer_offset = len(tokenizer(question)[0])

        >>> predict_answer_tokens = inputs.input_ids[
        ...     0, answer_offset + answer_start_index : answer_offset + answer_end_index + 1
        ... ]
        >>> predicted = tokenizer.decode(predict_answer_tokens)
        >>> predicted
        ' a nice puppet'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value



class OPTDiscreteTokenSecurityForCausalLM(OPTForCausalLM):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        self.seed = 1
        self.use_smaller_model = None
        self.encoder_type = None
        self.decoder_type = None
        self.embedding_model_name = None
        self.times = []

        self.large_prime = 914873376890177123195388945337
        self.loss_mask = []
        # Initialize weights and apply final processing
        self.post_init()

    '''
    ##########################BELOW ARE CUSTOM FUNCTIONS FOR NOISE GENERATION AND INFERENCE##########################
    '''
    import random

    '''
    Beam search functionalities
    '''
    def beam_search_sanity_check(self, logits, n=2, starting_index = 0):
        import torch.nn as nn
        m = nn.LogSoftmax(dim=0)
        bos_logits = m(logits[0,starting_index])
        # tok1 = self.input_ids[0,1] if self.loss_mask[0,1] == 0 else self.input_ids[0,2]
        for i in range(starting_index,starting_index+self.noise_magnitude):
            if self.loss_mask[0,i] == 0:
                tok1 = self.input_ids[0,i]
                offset = i-1
                break
        # offset = 0 if self.loss_mask[0,1] == 0 else 1
        original_text = [tok1]
        score = bos_logits[tok1]
        for idx in range(starting_index+self.noise_magnitude, logits.shape[1]-self.noise_magnitude, self.noise_magnitude):
            prob = m(logits[0,idx + offset])
            for i in range(self.noise_magnitude):
                if self.loss_mask[0,idx+self.noise_magnitude+i] == 0:
                    tok = self.input_ids[0,idx+self.noise_magnitude+i]
                    offset = i
            score += prob[tok]
            original_text.append(tok)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
        print("original text: ", "".join(tokenizer.batch_decode(torch.tensor(original_text))))
        print("original text score: ", score)
    
    def beam_search_sanity_check_bigram(self, logits, n = 2, starting_index = 0):
        import torch.nn as nn
        m = nn.LogSoftmax(dim=0)
        print("n = ", self.noise_magnitude)
        bos_logits = m(logits[0,starting_index])
        # tok1 = self.input_ids[0,1] if self.loss_mask[0,1] == 0 else self.input_ids[0,2]
        for i in range(starting_index+1,starting_index+1+self.noise_magnitude):
            if self.loss_mask[0,i] == 0:
                tok1 = self.input_ids[0,i]
                offset = i-1
                break
        # offset = 0 if self.loss_mask[0,1] == 0 else 1
        original_text = [tok1]
        score = bos_logits[tok1]
        for idx in range(starting_index+1, logits.shape[1]-self.noise_magnitude, self.noise_magnitude):
            prob = m(logits[0,idx + offset])
            for offset in range(self.noise_magnitude):
                if self.loss_mask[0,idx+self.noise_magnitude+offset] == 0:
                    tok = self.input_ids[0,idx+self.noise_magnitude+offset]
                    break
            score += prob[tok]
            original_text.append(tok)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
        print("original text: ", "".join(tokenizer.batch_decode(torch.tensor(original_text))))
        print("original text score: ", score)
    
    def beam_search_attack(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        starting_index: Optional[int] = 0,
        noise_order: Optional[int] = 2,
        noise_input_ids: Optional[torch.LongTensor] = None,
        noise_input_loss_mask: Optional[torch.LongTensor] = None,
        noise_type: Optional[str] = "unigram",
        run_baseline: Optional[bool] = False,
        noise_generate: Optional[bool] = True,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        import torch.nn as nn
        m = nn.LogSoftmax(dim=0)
        if noise_input_ids is not None:
            print("###beam search input size: ", noise_input_ids.shape)
            self.input_ids = noise_input_ids
            self.attention_mask = torch.ones(self.input_ids.shape)
            self.loss_mask = noise_input_loss_mask
            self.noise_magnitude = noise_order
        else:
            self.generate_noised_input_2way1order_argmax(input_ids)

        with torch.no_grad():
            output, _, _ = self.forward(
                noise_type = noise_type,
                run_baseline = run_baseline,
                noise_generate = noise_generate,
            )
        if self.do_beam_search_sanity_check:
            self.beam_search_sanity_check(output.logits, n=noise_order, starting_index =starting_index)
            self.do_beam_search_sanity_check = False

        logits = output.logits
        # every element in the cur_sequences is of a tuple [[sequence of tokens], probability, # of tokens to trace back to find the next distribution]
        # from the bos token, fetch the distribution of the first token of the sentence
        bos_logits = m(logits[0,starting_index])
        top_k_sequences = {i:list() for i in range(1,noise_order+1)}
        
        for i in range(starting_index,starting_index+noise_order):
            tok1 = self.input_ids[0,i]  # starting from index 1, b/c index 0 is the bos token
            top_k_sequences[1].append([[tok1], m(logits[0,i]), i])
        
        new_top_k_sequences = {}
        k = 1
        for i in range(starting_index+noise_order, starting_index+2*noise_order):
            for seq, cur_logits, prev_idx in top_k_sequences[1]:
                new_top_k_sequences[k] = new_top_k_sequences.get(k, list())
                tok1 = self.input_ids[0,prev_idx]
                tok2 = self.input_ids[0,i]
                new_top_k_sequences[k].append([[tok1, tok2], cur_logits[tok2], i])
        
        top_k_sequences = new_top_k_sequences.copy()
        for k,v in top_k_sequences.items():
            v.sort(key=lambda x:x[1], reverse=True)
            top_k_sequences[k] = [v[0]]
            
        for idx in range(starting_index+noise_order, logits.shape[1]-noise_order, noise_order):
            new_top_k_sequences = {}
            for i in range(noise_order):
                token1 = self.input_ids[0,idx+noise_order+i]
                cur_sequences = []
                for k, v in top_k_sequences.items():
                    for seq in v:
                        prob = m(logits[0,idx + seq[2]])
                        seq_copy1 = [seq[0].copy() + [token1], seq[1] + prob[token1], i]
                        cur_sequences.append(seq_copy1)
                cur_sequences.sort(reverse=True, key=lambda x: x[1])
                new_top_k_sequences[i+1] = [cur_sequences[0]]
            top_k_sequences = new_top_k_sequences.copy()
        
        all_sequences = [v[0] for k, v in top_k_sequences.items()]
        all_sequences.sort(reverse=True, key=lambda x: x[1])
        best_sequence = all_sequences[0]
        return best_sequence

    def beam_search_attack_bigram(
        self,
        starting_index: Optional[int] = 0,
        input_ids: Optional[torch.LongTensor] = None,
        n: Optional[int] = 2,
        noise_order: Optional[int] = 8,
        noise_input_ids: Optional[torch.LongTensor] = None,
        noise_input_loss_mask: Optional[torch.LongTensor] = None,
        noise_type: Optional[str] = "bigram",
        run_baseline: Optional[bool] = False,
        noise_generate: Optional[bool] = True,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        import torch.nn as nn
        m = nn.LogSoftmax(dim=0)
        if noise_input_ids is not None:
            print("###beam search input size: ", noise_input_ids.shape)
            
            self.input_ids = noise_input_ids
            self.attention_mask = torch.ones(self.input_ids.shape)
            self.loss_mask = noise_input_loss_mask
            if not self.noise_magnitude:
                self.noise_magnitude = n**2*2
        else:
            self.generate_noised_input_2way1order_argmax(input_ids)

        with torch.no_grad():
            output, _, _ = self.forward(
                noise_type = noise_type,
                run_baseline = run_baseline,
                noise_generate = noise_generate,
            )

        # self.beam_search_sanity_check_bigram(output.logits, n = n, starting_index = starting_index)
        
        print("###starting_index in beam attack: ", starting_index)
        print("###noise magnitude: ", self.noise_magnitude)

        logits = output.logits
        # every element in the cur_sequences is of a tuple [[sequence of tokens], probability, # of tokens to trace back to find the next distribution]
        # from the bos token, fetch the distribution of the first token of the sentence
        bos_logits = m(logits[0,starting_index])
        top_k_sequences = dict()
        # bos_token = self.input_ids[0,starting_index]
        # for i in range(starting_index+2,starting_index+self.noise_magnitude+2,2):
        #     tok1 = self.input_ids[0,i]  # starting from index 2, b/c index 0 and 1 are the bos token, every second index from there on is a next token
        #     t1 = bos_token.item()
        #     t2 = tok1.item()
        #     top_k_sequences[(t1, t2)] = top_k_sequences.get((t1, t2), list())
        #     top_k_sequences[(t1, t2)].append(([[bos_token, tok1], bos_logits[tok1], i-1]))
        
        # get logits of first tokens
        for i in range(starting_index+1,starting_index+self.noise_magnitude,2):
            tok1 = self.input_ids[0,i]
            t1 = tok1.item()
            top_k_sequences[(t1,)] = top_k_sequences.get((t1, ), list())
            top_k_sequences[(t1,)].append([[tok1], 0, i])
        new_top_k_sequences = dict()
        print("###input ids for beam attack: ", noise_input_ids.shape)
        # get the first tokens
        for i in range(starting_index+self.noise_magnitude+1, starting_index+2*self.noise_magnitude, 2):
            if (self.input_ids[0,i-1].item(),) not in list(top_k_sequences.keys()):
                continue
            else:
                for seq, score, prev_idx in top_k_sequences[(self.input_ids[0,i-1].item(),)]:
                    bos_logits = m(logits[0,prev_idx])
                    # tok1 is bos
                    tok1 = torch.tensor(seq[-1])
                    t1 = seq[-1].item()
                    tok2 = self.input_ids[0,i]
                    t2 = tok2.item()
                    new_top_k_sequences[(t1,t2)] = new_top_k_sequences.get((t1, t2), list())
                    new_top_k_sequences[(t1,t2)].append([[tok1, tok2], bos_logits[tok2], i])
        
        top_k_sequences = new_top_k_sequences.copy()
        for k,v in top_k_sequences.items():
            v.sort(key=lambda x:x[1], reverse=True)
            top_k_sequences[k] = [v[0]]

        for idx in range(starting_index+self.noise_magnitude, logits.shape[1]-self.noise_magnitude, self.noise_magnitude):
            new_top_k_sequences = {}
            # first find the next token and its noise 
            next_toks = set()
            for offset in range(1,self.noise_magnitude+1,2):
                next_toks.add(self.input_ids[0,idx+self.noise_magnitude+offset].item())
            next_toks = list(next_toks)
            next_toks = [torch.tensor(x) for x in next_toks]

            for k,v in top_k_sequences.items():
                for seq in v:
                    for offset in range(0,self.noise_magnitude,2):
                        if k == tuple(self.input_ids[0,idx+offset:idx+offset+2]):     # check matching bigram history
                            for next_tok in next_toks:  # fetch scores for both true next token and noise next token from the distribution of current last token 
                                prob = m(logits[0,idx+offset+1])
                                seq_copy1 = [seq[0].copy() + [next_tok], seq[1] + prob[next_tok], i]
                                key = (seq_copy1[0][-2].item(), seq_copy1[0][-1].item())
                                new_top_k_sequences[key] = new_top_k_sequences.get(key, list())
                                new_top_k_sequences[key].append(seq_copy1)
            # for each beam maintain the highest scoring
            for k,v in new_top_k_sequences.items():
                v.sort(key=lambda x:x[1], reverse=True)
                new_top_k_sequences[k] = [v[0]]
            top_k_sequences = new_top_k_sequences.copy()
            
        all_sequences = [v[0] for k, v in top_k_sequences.items()]
        all_sequences.sort(reverse=True, key=lambda x: x[1])
        best_sequence = all_sequences[0]
        return best_sequence

    def multiple_beam_searches(
        self, 
        starting_index: Optional[int] = 0,
        search_rounds: Optional[int] = 1,
        input_ids: Optional[torch.LongTensor] = None,
        noise_order: Optional[int] = 2,
        noise_input_ids: Optional[torch.LongTensor] = None,
        noise_input_loss_mask: Optional[torch.LongTensor] = None,
        noise_type: Optional[str] = "unigram",
        run_baseline: Optional[bool] = False,
        noise_generate: Optional[bool] = True,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        true_token_percentage = {round:list() for round in range(noise_order)}
        original_noise_input_ids = noise_input_ids.clone()[:,starting_index:]
        noise_input_ids2 = noise_input_ids.clone()[:,starting_index:]
        original_noise_input_loss_mask = noise_input_loss_mask.clone()[:,starting_index:]
        original_noise_order = noise_order
        prefix = noise_input_ids.clone()[:,:starting_index]
        prefix_mask = noise_input_loss_mask.clone()[:,:starting_index]
        recovered_texts = []

        # this list records whether a spot is using any noise token or not
        is_true = [True] * 300
        for round in range(search_rounds):
            print(f"round {round} beam search")
            result = self.beam_search_attack(
                input_ids=input_ids,
                starting_index = starting_index,
                noise_order=noise_order,
                noise_input_ids=noise_input_ids,
                noise_input_loss_mask=noise_input_loss_mask,
                noise_type=noise_type,
                run_baseline=run_baseline,
                noise_generate=noise_generate,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            print("score: ", result[1])
            result = result[0]
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
            
            # calculate percentage of true tokens beam attack has recovered
            # print recovered text
            true_tokens = 0
            total_tokens = 0
            filter_index = 1
            result_for_print = []
            for base in range(original_noise_order, original_noise_input_ids.shape[-1], original_noise_order):
                for offset in range(original_noise_order):
                    if original_noise_input_loss_mask[0,base+offset] == 0:
                        if original_noise_input_ids[0,base+offset] == result[filter_index]:
                            true_tokens += 1
                            is_true[filter_index] = False
                            
                result_for_print.append(result[filter_index])
                total_tokens += 1
                filter_index += 1
            print(f"number of true tokens after round {round}: ", true_tokens)
            print(f"total tokens {round}: ", total_tokens)
            print(f"recovered text after round {round}: ", tokenizer.decode(torch.tensor(result_for_print)))
            print(f"Percentage of true tokens recovered after beam search round {round}: ", true_tokens/total_tokens)
            recovered_texts.append(tokenizer.decode(torch.tensor(result_for_print)))

            # record percentage of true tokens beam attack has recovered
            true_token_percentage[round].append(true_tokens/total_tokens)

            # filter out the recovered tokens in the input
            curr_noise_order = noise_order - 1
            filtered_noised_input = torch.tensor(noise_input_ids.clone()[0,starting_index:starting_index+curr_noise_order]).to(torch.long)
            filtered_loss_mask = torch.tensor([0]*curr_noise_order).to(torch.long)
            filter_index = 1
            print("#####self.noise_magnitude: ", self.noise_magnitude)
            for base in range(self.noise_magnitude, noise_input_ids2.shape[-1], self.noise_magnitude):
                flag = False
                for offset in range(self.noise_magnitude):
                    if noise_input_ids2[0,base+offset] != result[filter_index] or flag:
                        filtered_noised_input = torch.cat((filtered_noised_input, torch.tensor([noise_input_ids2[0,base+offset]])))
                        filtered_loss_mask = torch.cat((filtered_loss_mask, torch.tensor([original_noise_input_loss_mask[0,base+offset]])))
                    else:
                        flag = True
                filter_index += 1
            
            noise_order -= 1
            # noise_input_ids = filtered_noised_input.clone().unsqueeze(dim=0)
            # noise_input_loss_mask = filtered_loss_mask.clone().unsqueeze(dim=0)
            noise_input_ids2 = filtered_noised_input.clone().unsqueeze(dim=0)
            noise_input_ids = torch.cat((prefix, noise_input_ids2), dim=1)
            noise_input_loss_mask = filtered_loss_mask.clone().unsqueeze(dim=0)
            noise_input_loss_mask = torch.cat((prefix_mask, noise_input_loss_mask), dim=1)

        # calculate percentage of true tokens beam attack has recovered for last recovery
        # print recovered text for last recovery
        print("filtered noise input ids: ", filtered_noised_input.shape)
        true_tokens = 0
        total_tokens = 0
        filter_index = 1
        result_for_print = []
        for base in range(original_noise_order, original_noise_input_ids.shape[-1], original_noise_order):
            for offset in range(original_noise_order):
                if original_noise_input_loss_mask[0,base+offset] == 0:
                    if original_noise_input_ids[0,base+offset] == filtered_noised_input[filter_index]:
                        # if is_true[filter_index]:
                        true_tokens += 1
            result_for_print.append(filtered_noised_input[filter_index])
            total_tokens += 1
            filter_index += 1
        print(f"number of true tokens after round {round+1}: ", true_tokens)
        print(f"total tokens {round}: ", total_tokens)
        print(f"recovered text after round {round+1}: ", tokenizer.decode(result_for_print))
        print(f"Percentage of true tokens recovered after beam search round {round+1}: ", true_tokens/total_tokens)
        true_token_percentage[round+1].append(true_tokens/total_tokens)
        recovered_texts.append(tokenizer.decode(result_for_print))
        return filtered_noised_input, true_token_percentage, recovered_texts

    def multiple_beam_searches_bigram(
        self, 
        starting_index: Optional[int] = 0,
        search_rounds: Optional[int] = 1,
        input_ids: Optional[torch.LongTensor] = None,
        noise_order: Optional[int] = 8,
        n: Optional[int] = 2,
        noise_input_ids: Optional[torch.LongTensor] = None,
        noise_input_loss_mask: Optional[torch.LongTensor] = None,
        noise_type: Optional[str] = "bigram",
        run_baseline: Optional[bool] = False,
        noise_generate: Optional[bool] = True,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        true_token_percentage = {round:list() for round in range(n)}
        original_noise_input_ids = noise_input_ids.clone()[:,starting_index:]
        noise_input_ids2 = noise_input_ids.clone()[:,starting_index:]
        prefix = noise_input_ids.clone()[:,:starting_index]
        original_noise_input_loss_mask = noise_input_loss_mask.clone()[:,starting_index:]
        prefix_mask = noise_input_loss_mask.clone()[:,:starting_index]
        original_noise_order = n**2*2
        pairs = n**2
        original_n = n
        recovered_texts = []

        # this list records whether a spot is using any noise token or not
        is_true = [True] * 300
        for round in range(n-1):
            self.noise_magnitude = original_noise_order - round * original_n * 2
            print(f"round {round} beam search")
            result = self.beam_search_attack_bigram(
                input_ids=input_ids,
                n = n,
                starting_index = starting_index,
                noise_order=noise_order,
                noise_input_ids=noise_input_ids,
                noise_input_loss_mask=noise_input_loss_mask,
                noise_type=noise_type,
                run_baseline=run_baseline,
                noise_generate=noise_generate,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            print("score: ", result[1])
            if starting_index == 0:
                result = result[0][1:]
            else:
                result = result[0][:]
            print("length of results: ", len(result))
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
            # calculate percentage of true tokens beam attack has recovered
            # print recovered text
            true_tokens = 0
            total_tokens = 0
            filter_index = 0
            result_for_print = []
            # print(original_noise_input_ids.shape[-1])
            if starting_index == 0:
                starting_point = starting_index + original_noise_order
            else:
                starting_point = 0
            for base in range(starting_point, original_noise_input_ids.shape[-1], original_noise_order):
                for offset in range(original_noise_order):
                    if original_noise_input_loss_mask[0,base+offset] == 0:
                        if original_noise_input_ids[0,base+offset] == result[filter_index]:
                            true_tokens += 1
                            is_true[filter_index] = False
                            
                result_for_print.append(result[filter_index])
                total_tokens += 1
                filter_index += 1
            print(f"number of true tokens after round {round}: ", true_tokens)
            print(f"total tokens {round}: ", total_tokens)
            print(f"recovered text after round {round}: ", torch.tensor(result_for_print))
            print(f"recovered text after round {round}: ", tokenizer.decode(torch.tensor(result_for_print)))
            recovered_texts.append(tokenizer.decode(torch.tensor(result_for_print)))

            # record percentage of true tokens beam attack has recovered
            true_token_percentage[round].append(true_tokens/total_tokens)

            # filter out the recovered tokens in the input
            next_round_magnitude = self.noise_magnitude - original_n * 2
            filtered_loss_mask = torch.tensor([0]*next_round_magnitude).to(torch.long)
            filter_index = 0
            if starting_index == 0:
                filtered_noised_input = torch.tensor(noise_input_ids.clone()[0,starting_index:starting_index+next_round_magnitude])
                starting_point = starting_index + self.noise_magnitude
            else:
                filtered_noised_input = torch.tensor([]).to(torch.long)
                starting_point = 0
            for base in range(starting_point, noise_input_ids2.shape[-1], self.noise_magnitude):
                flag = 0
                for offset in range(0,self.noise_magnitude,2):
                    if noise_input_ids2[0,base+offset+1] != result[filter_index] and flag < pairs - original_n:
                        filtered_noised_input = torch.cat((filtered_noised_input, noise_input_ids2[0,base+offset:base+offset+2]))
                        filtered_loss_mask = torch.cat((filtered_loss_mask, noise_input_loss_mask[0,base+offset:base+offset+2]))
                        flag += 1
                filter_index += 1
            print("### filtered noised input: ", filtered_noised_input.shape)
            noise_order -= 1
            pairs -= original_n
            n -= 1
            noise_input_ids2 = filtered_noised_input.clone().unsqueeze(dim=0)
            noise_input_ids = torch.cat((prefix, noise_input_ids2), dim=1)
            noise_input_loss_mask = filtered_loss_mask.clone().unsqueeze(dim=0)
            noise_input_loss_mask = torch.cat((prefix_mask, noise_input_loss_mask), dim=1)
        
        # calculate percentage of true tokens beam attack has recovered for last recovery
        # print recovered text for last recovery
        true_tokens = 0
        total_tokens = 0
        filter_index = 1
        result_for_print = []
        if original_n == 2:
            p_offset = 4
        elif original_n == 3:
            p_offset = 12

        if starting_index == 0:
            starting_point = starting_index + original_noise_order
        else:
            starting_point = 0
            filter_index = 0
        print(filtered_noised_input.shape)
        for base in range(starting_point, original_noise_input_ids.shape[-1], original_noise_order):
            for offset in range(0,original_noise_order,2):
                if original_noise_input_loss_mask[0,base+offset+1] == 0:
                    if original_noise_input_ids[0,base+offset+1] == filtered_noised_input[base+1-filter_index*p_offset]:
                        # if is_true[filter_index]:
                        is_true[filter_index] = False
                        true_tokens += 1
                
            result_for_print.append(filtered_noised_input[base+1-filter_index*p_offset])
                            
            total_tokens += 1
            filter_index += 1
        print(f"number of true tokens after round {round+1}: ", true_tokens)
        print(f"total tokens {round}: ", total_tokens)
        print(f"recovered text after round {round+1}: ", result_for_print)
        print(f"recovered text after round {round+1}: ", tokenizer.decode(result_for_print))
        print(f"Percentage of true tokens recovered after beam search round {round+1}: ", true_tokens/total_tokens)
        true_token_percentage[round+1].append(true_tokens/total_tokens)
        recovered_texts.append(tokenizer.decode(result_for_print))
        return filtered_noised_input, true_token_percentage, recovered_texts

    def get_embedding_database(self, path):
        import pickle
        dbfile = open(path, 'rb')
        self.vdb = pickle.load(dbfile)

    def generate_noised_input_unigram(
        self, 
        input_ids, 
        noise_order = 2, 
        is_label=False, 
        use_start_token = True
    ):
        import random
        
        self.noise_magnitude = noise_order

        b, s = input_ids.shape
        noised_input_ids = []
        batch_loss_mask = []

        for i in range(b):
            noised_inputs = []
            loss_mask = []
            for idx in range(len(input_ids[i])):
                if idx == 0:
                    token = input_ids[i,idx]
                    noise_token_indices = [token]*(noise_order - 1) + [-1]
                    random.shuffle(noise_token_indices)
                else:
                    token = input_ids[i,idx]
                    results = list(map(int, self.vdb[int(token)]))[10:20]
                    noise_token_indices = random.sample(results, noise_order-1)
                    noise_token_indices = noise_token_indices + [-1]
                    random.shuffle(noise_token_indices)

                for tok in noise_token_indices:
                    if tok == -1:
                        noised_inputs.append(token)
                        loss_mask.append(0)
                    else:
                        noised_inputs.append(tok)
                        loss_mask.append(-1)
            batch_loss_mask.append(loss_mask)
            noised_input_ids.append(noised_inputs)
        self.input_ids = torch.tensor(noised_input_ids)
        self.attention_mask = torch.ones(self.input_ids.shape)
        self.loss_mask = torch.tensor(batch_loss_mask)
        return torch.tensor(noised_input_ids), torch.tensor(batch_loss_mask)

    def generate_masked_shifted_labels_unigram(
        self, 
        original_input_ids, 
        label, 
        mask
    ):
        b, s = label.shape
        masked_labels = []
        for i in range(b):
            seq_labels = []
            for idx, token in enumerate(label[i]):
                if idx == s - self.noise_magnitude:
                    break
                if mask[i][idx] == 0:
                    seq_labels.append(original_input_ids[i,idx//self.noise_magnitude+1])
                else:
                    seq_labels.append(-100)
            masked_labels.append(seq_labels)
        return torch.tensor(masked_labels)

    def generate_baseline_input(self):
        new_inputs = []
        new_labels = []
        new_attention_masks = []
        for b in range(self.input_ids.shape[0]):
            new_label = []
            new_input = []
            new_attention_mask = []
            for tok in range(self.input_ids.shape[1]):
                if self.loss_mask[b,tok] == 0:
                    new_label.append(self.input_ids[b,tok])
                else:
                    new_input.append(self.input_ids[b,tok])
                    new_attention_mask.append(1)
            new_attention_masks.append(new_attention_mask)
            new_inputs.append(new_input)
            new_labels.append(new_label)
        self.input_ids = torch.tensor(new_inputs)
        self.attention_mask = torch.tensor(new_attention_masks)
        return torch.tensor(new_inputs), torch.tensor(new_labels)
    
    '''
    def generate_noised_input_bigram(self, input_ids, use_start_token = True):
        import random
        b, s = input_ids.shape
        noised_input_ids = []
        batch_loss_mask = []
        self.noise_magnitude = 8
        for i in range(b):
            noised_inputs = []
            loss_mask = []
            for idx in range(s-1):
                
                token1 = input_ids[i,idx]
                token2 = input_ids[i,idx+1]
                results1 = self.vdb[int(token1)]
                results2 = self.vdb[int(token2)]

                if idx == 0:
                    noise_token1 = random.choice(list( (10,20)))# int(results1[random.choice(list( (10,20)))])
                    noise_token2 = random.choice(list( (10,20)))
                else:
                    # noise_token1 = self.temp_token 
                    noise_token1 = random.choice(list( (10,20)))
                    noise_token2 = random.choice(list( (10,20)))

                bigram_construct = [[token1,token2],[token1,noise_token2],[noise_token1,token2],[noise_token1,noise_token2]]
                self.temp_token = noise_token2
                random.shuffle(bigram_construct)
                for perm in bigram_construct:
                    noised_inputs.extend(perm)
                    if perm == [token1, token2]:
                        loss_mask.extend([-1,0])
                    else:
                        loss_mask.extend([-1,-1])
            batch_loss_mask.append(loss_mask)
            noised_input_ids.append(noised_inputs)
        
        self.input_ids = torch.tensor(noised_input_ids)
        self.attention_mask = torch.ones(self.input_ids.shape)
        self.loss_mask = torch.tensor(batch_loss_mask)

        return torch.tensor(noised_input_ids), torch.tensor(batch_loss_mask)
    '''
    
    def generate_noised_input_bigram(self, input_ids, n = 3, use_start_token = True):
        import random
        b, s = input_ids.shape
        self.noise_magnitude = n**2*2
        noised_input_ids = []
        batch_loss_mask = []
        for i in range(b):
            noised_inputs = []
            loss_mask = []
            for idx in range(-1, s-1):
                if idx == -1:
                    if use_start_token:
                        token1 = 2
                        token2 = input_ids[i,0]
                        # results1 = self.vdb[int(token1)]
                        # results2 = self.vdb[int(token2)]

                        tokens1 = [2]*n
                        tokens2 = [token2]*n
                        # choices = list( (10,20))
                        # for k in range(1,n):   
                        #     choice = random.choice(choices)
                        #     noise_token1 = int(results1[choice])
                        #     choices.remove(choice)
                        #     tokens1.append(noise_token1)
                        # tokens1.append(token1)

                        # tokens2 = []
                        # choices = list( (10,20))
                        # for k in range(1,n):    
                        #     choice = random.choice(choices)
                        #     noise_token2 = int(results2[choice])
                        #     choices.remove(choice)
                        #     tokens2.append(noise_token2)
                        # tokens2.append(token2)

                        self.temp_tokens = tokens2.copy()

                        bigram_construct = []
                        for k in range(n):
                            for j in range(n):
                                bigram_construct.append([tokens1[k], tokens2[j]])
                    else:
                        continue
                else:
                    token1 = input_ids[i,idx]
                    token2 = input_ids[i,idx+1]
                    results2 = self.vdb[int(token2)]
                    tokens1 = self.temp_tokens

                    tokens2 = []
                    choices = list( (10,20))
                    for k in range(1,n):    
                        choice = random.choice(choices)
                        noise_token2 = int(results2[choice])
                        choices.remove(choice)
                        tokens2.append(noise_token2)
                    tokens2.append(token2)

                    self.temp_tokens = tokens2.copy()

                    bigram_construct = []
                    for k in range(n):
                        for j in range(n):
                            bigram_construct.append([tokens1[k], tokens2[j]])

                random.shuffle(bigram_construct)

                lis1 = list( (n**2))
                rand_for_first_tok = random.choice(lis1)
                cnt_for_first_tok = 0
                lis2 = list( (n))
                rand_for_second_tok = random.choice(lis2)
                cnt_for_second_tok = 0

                for ii, perm in enumerate(bigram_construct):
                    noised_inputs.extend(perm)
                    if idx == -1:
                        if ii == rand_for_first_tok:
                            loss_mask.extend([-1,0])
                        else:
                            loss_mask.extend([-1,-1])
                    elif idx == 0:
                        if cnt_for_second_tok == rand_for_second_tok and perm == [token1, token2]:
                            cnt_for_second_tok += 1
                            loss_mask.extend([-1,0])
                        elif perm == [token1, token2]:
                            cnt_for_second_tok += 1
                            loss_mask.extend([-1,-1])
                        else:
                            loss_mask.extend([-1,-1])
                    else:
                        if  perm == [token1, token2]:
                            loss_mask.extend([-1,0])
                        else:
                            loss_mask.extend([-1,-1])

            batch_loss_mask.append(loss_mask)
            noised_input_ids.append(noised_inputs)
        self.input_ids = torch.tensor(noised_input_ids)
        self.attention_mask = torch.ones(self.input_ids.shape)
        self.loss_mask = torch.tensor(batch_loss_mask)

        return torch.tensor(noised_input_ids), torch.tensor(batch_loss_mask)
    
    def generate_masked_shifted_labels_bigram(self, original_input_ids, label, mask):
        b, s = label.shape
        masked_labels = []
        for i in range(b):
            seq_labels = []
            for idx, token in enumerate(label[i]):
                if idx == s - self.noise_magnitude:
                    break
                if mask[i][idx] == 0:
                    seq_labels.append(original_input_ids[i,idx//self.noise_magnitude+1])
                else:
                    seq_labels.append(-100)
            masked_labels.append(seq_labels)
        
        return torch.tensor(masked_labels)
    '''
    def generate_noised_input_trigram(self, input_ids, n = 2, use_start_token = True):
        import random
        b, s = input_ids.shape
        self.noise_magnitude = n**3*3
        noised_input_ids = []
        batch_loss_mask = []
        for i in range(b):
            noised_inputs = []
            loss_mask = []
            for idx in range(-1, s-1):
                if idx == -1:
                    if use_start_token:
                        token1 = 2
                        token2 = 2
                        token3 = input_ids[i,0]
                        results1 = self.vdb[int(token1)]
                        results2 = self.vdb[int(token2)]

                        tokens1 = [2] * n
                        tokens2 = [2] * n
                        tokens3 = [token3]*n

                        self.temp_tokens1 = tokens2.copy()
                        self.temp_tokens2 = tokens3.copy()
                        trigram_construct = []
                        for k in range(n):
                            for j in range(n):
                                for q in range(n):
                                    trigram_construct.append([tokens1[k], tokens2[j], tokens3[q]])
                    else:
                        continue
                else:
                    if idx == 0:
                        token1 = input_ids[i,0]
                    else:
                        token1 = input_ids[i,idx-1]
                    token2 = input_ids[i,idx]
                    token3 = input_ids[i,idx+1]
                    results3 = self.vdb[int(token3)]
                    tokens1 = self.temp_tokens1
                    tokens2 = self.temp_tokens2

                    tokens3 = []
                    choices = list( (10,20))
                    for k in range(1,n):    
                        choice = random.choice(choices)
                        noise_token3 = int(results3[choice])
                        choices.remove(choice)
                        tokens3.append(noise_token3)
                    tokens3.append(token3)

                    self.temp_tokens1 = tokens2.copy()
                    self.temp_tokens2 = tokens3.copy()

                    trigram_construct = []
                    for k in range(n):
                        for j in range(n):
                            for l in range(n):
                                trigram_construct.append([tokens1[k], tokens2[j], tokens3[l]])

                random.shuffle(trigram_construct)
                lis1 = list( (n**3))
                rand_for_first_tok = random.choice(lis1)
                lis2 = list( (n**2))
                rand_for_second_tok = random.choice(lis2)
                cnt_for_second_tok = 0
                lis3 = list( (n))
                rand_for_third_tok = random.choice(lis3)
                cnt_for_third_tok = 0

                for ii, perm in enumerate(trigram_construct):
                    noised_inputs.extend(perm)
                    if idx == -1:
                        if ii == rand_for_first_tok:
                            loss_mask.extend([-1,-1,0])
                        else:
                            loss_mask.extend([-1,-1,-1])
                    elif idx == 0:
                        if cnt_for_second_tok == rand_for_second_tok and perm == [token1, token2, token3]:
                            cnt_for_second_tok += 1
                            loss_mask.extend([-1,-1,0])
                        elif perm == [token1, token2, token3]:
                            cnt_for_second_tok += 1
                            loss_mask.extend([-1,-1,-1])
                        else:
                            loss_mask.extend([-1,-1,-1])
                    elif idx == 1:
                        if cnt_for_third_tok == rand_for_third_tok and perm == [token1, token2, token3]:
                            cnt_for_third_tok += 1
                            loss_mask.extend([-1,-1,0])
                        elif perm == [token1, token2, token3]:
                            cnt_for_third_tok += 1
                            loss_mask.extend([-1,-1,-1])
                        else:
                            loss_mask.extend([-1,-1,-1])
                    else:
                        if  perm == [token1, token2, token3]:
                            loss_mask.extend([-1,-1,0])
                        else:
                            loss_mask.extend([-1,-1,-1])
            batch_loss_mask.append(loss_mask)
            noised_input_ids.append(noised_inputs)
        self.input_ids = torch.tensor(noised_input_ids)
        self.attention_mask = torch.ones(self.input_ids.shape)
        self.loss_mask = torch.tensor(batch_loss_mask)

        return torch.tensor(noised_input_ids), torch.tensor(batch_loss_mask)
    '''
    def generate_noised_input_trigram(self, input_ids, n = 2, use_start_token = True):
        import random
        b, s = input_ids.shape
        self.noise_magnitude = n**2*3
        noised_input_ids = []
        batch_loss_mask = []
        for i in range(b):
            noised_inputs = []
            loss_mask = []
            for idx in range(-1, s-1):
                if idx == -1:
                    token1 = 2
                    token2 = 2
                    token3 = input_ids[i,0]
                    results1 = self.vdb[int(token1)]
                    results2 = self.vdb[int(token2)]

                    tokens1 = [2] * n
                    tokens2 = [2] * n
                    tokens3 = [token3]*n

                    self.temp_tokens1 = tokens2.copy()
                    self.temp_tokens2 = tokens3.copy()
                    trigram_construct = []
                    for k in range(n):
                        for j in range(n):
                            for q in range(n):
                                trigram_construct.append([tokens1[k], tokens2[j], tokens3[q]])
                else:
                    if idx == 0:
                        token1 = input_ids[i,0]
                    else:
                        token1 = input_ids[i,idx-1]
                    token2 = input_ids[i,idx]
                    token3 = input_ids[i,idx+1]
                    results3 = self.vdb[int(token3)]
                    tokens1 = self.temp_tokens1
                    tokens2 = self.temp_tokens2

                    tokens3 = []
                    choices = list( (10,20))
                    for k in range(1,n):    
                        choice = random.choice(choices)
                        noise_token3 = int(results3[choice])
                        choices.remove(choice)
                        tokens3.append(noise_token3)
                    tokens3.append(token3)

                    self.temp_tokens1 = tokens2.copy()
                    self.temp_tokens2 = tokens3.copy()

                    trigram_construct = []
                    for k in range(n):
                        for j in range(n):
                            for l in range(n):
                                trigram_construct.append([tokens1[k], tokens2[j], tokens3[l]])

                lis1 = list( (n**2))
                rand_for_first_tok = random.choice(lis1)
                lis2 = list( (n**2))
                rand_for_second_tok = random.choice(lis2)
                cnt_for_second_tok = 0
                lis3 = list( (n))
                rand_for_third_tok = random.choice(lis3)
                cnt_for_third_tok = 0

                if idx == -1:
                    trigram_construct = trigram_construct[:4]
                elif idx == 0:
                    temp = []
                    for construct in trigram_construct:
                        if construct not in temp:
                            temp.append(construct)
                    for a in range(2):
                        temp.append(trigram_construct[a])
                    trigram_construct = temp[:]
                elif idx == 1:
                    temp = [[token1, token2, token3], [tokens1[0], tokens2[0], tokens3[0]]]
                    for a in range(2):
                        temp.append(trigram_construct[a])
                    trigram_construct = temp[:]
                random.shuffle(trigram_construct)

                fft_count = 0
                for ii, perm in enumerate(trigram_construct):
                    if idx == -1:
                        if ii == rand_for_first_tok:
                            loss_mask.extend([-1,-1,0])
                        else:
                            loss_mask.extend([-1,-1,-1])
                    elif idx == 0:
                        if cnt_for_second_tok == 0 and perm == [token1, token2, token3]:
                            cnt_for_second_tok += 1
                            loss_mask.extend([-1,-1,0])
                        else:
                            loss_mask.extend([-1,-1,-1])
                    elif idx == 1:
                        if cnt_for_third_tok == 0 and perm == [token1, token2, token3]:
                            cnt_for_third_tok += 1
                            loss_mask.extend([-1,-1,0])
                        else:
                            loss_mask.extend([-1,-1,-1])
                    else:
                        if perm == [token1, token2, token3]:
                            loss_mask.extend([-1,-1,0])
                        elif perm == [tokens1[0], tokens2[0], tokens3[0]]:
                            loss_mask.extend([-1,-1,1])
                        else:
                            if fft_count >= 2:
                                continue
                            loss_mask.extend([-1,-1,-1])
                            fft_count += 1
                    noised_inputs.extend(perm)

            batch_loss_mask.append(loss_mask)
            noised_input_ids.append(noised_inputs)
        self.input_ids = torch.tensor(noised_input_ids)
        self.attention_mask = torch.ones(self.input_ids.shape)
        self.loss_mask = torch.tensor(batch_loss_mask)

        return torch.tensor(noised_input_ids), torch.tensor(batch_loss_mask)
    
    def generate_masked_shifted_labels_trigram(self, original_input_ids, label, mask):
        b, s = label.shape
        masked_labels = []
        for i in range(b):
            seq_labels = []
            for idx, token in enumerate(label[i]):
                if idx == s - self.noise_magnitude:
                    break
                if mask[i][idx] == 0:
                    seq_labels.append(original_input_ids[i,idx//self.noise_magnitude+1])
                else:
                    seq_labels.append(-100)
            masked_labels.append(seq_labels)
        return torch.tensor(masked_labels)
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        noise_mode: Optional[str] = "unigram",
        n: Optional[int] = 2,
        noise_order: Optional[int] = 2,
        with_prompt: Optional[bool] = True,
        use_noise_threshold: Optional[float] = 0.0,
        threshold_get_from_true_token: Optional[float] = 0.7,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs
    ):
        self.noise_magnitude = noise_order
        if noise_mode == "unigram":
            self.noise_type = "argmax"
        else:
            self.noise_type = noise_mode
        self.true_input_ids = torch.tensor([])
        self.total_noised_input_ids = torch.tensor([])
        self.total_generated_input_ids = torch.tensor([])
        self.total_generated_loss_mask = torch.tensor([])
        self.noise_past_key_values = None
        self.do_beam_search_sanity_check = True
        self.total_noised_input_ids_for_display = torch.tensor([])
        self.noise_tokens_for_display = []

        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation -- update the generation config
            # model attribute accordingly, if it was created from the model config
            if self.generation_config._from_model_config:
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use a generation configuration file (see"
                        " https://huggingface.co/docs/transformers/main_classes/text_generation)"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config.pad_token_id is not None
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 7. determine generation mode
        is_constraint_gen_mode = (
            generation_config.constraints is not None or generation_config.force_words_ids is not None
        )

        is_contrastive_search_gen_mode = (
            (generation_config.num_beams == 1)
            and generation_config.top_k is not None
            and generation_config.top_k > 1
            and generation_config.do_sample is False
            and generation_config.penalty_alpha is not None
            and generation_config.penalty_alpha > 0
        )

        is_greedy_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_sample_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_beam_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_beam_sample_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_group_beam_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups > 1)
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_assisted_gen_mode = False
        if assistant_model is not None:
            if not (is_greedy_gen_mode or is_sample_gen_mode):
                raise ValueError(
                    "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                    "is only supported with Greedy Search and Sample."
                )
            is_assisted_gen_mode = True

        if generation_config.num_beam_groups > generation_config.num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and generation_config.do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        # 9. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        # 10. go into different generation modes
        if is_assisted_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")

            # 11. If the assistant model is an encoder-decoder, prepare its encoder outputs
            if assistant_model.config.is_encoder_decoder:
                assistant_model_kwargs = copy.deepcopy(model_kwargs)
                inputs_tensor, model_input_name, assistant_model_kwargs = assistant_model._prepare_model_inputs(
                    inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_model_kwargs
                )
                assistant_model_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor, assistant_model_kwargs, model_input_name
                )
                model_kwargs["assistant_encoder_outputs"] = assistant_model_kwargs["encoder_outputs"]

            # 12. run assisted generate
            return self.assisted_decoding(
                input_ids,
                assistant_model=assistant_model,
                do_sample=generation_config.do_sample,
                logits_processor=logits_processor,
                logits_warper=self._get_logits_warper(generation_config) if generation_config.do_sample else None,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        if is_greedy_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing greedy search, "
                    f"but is {generation_config.num_return_sequences}."
                )

            # 11. run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif is_contrastive_search_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing contrastive search, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")

            return self.contrastive_search(
                input_ids,
                top_k=generation_config.top_k,
                penalty_alpha=generation_config.penalty_alpha,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample
            return self.sample(
                input_ids,
                noise_order=noise_order,
                n = n,
                use_noise_threshold = use_noise_threshold,
                with_prompt = with_prompt,
                logits_processor=logits_processor,
                threshold_get_from_true_token=threshold_get_from_true_token,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")
            # 12. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size * generation_config.num_return_sequences,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                max_length=generation_config.max_length,
            )

            # 13. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams * generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 14. run beam sample
            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if generation_config.num_beams % generation_config.num_beam_groups != 0:
                raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            has_default_typical_p = kwargs.get("typical_p") is None and generation_config.typical_p == 1.0
            if not has_default_typical_p:
                raise ValueError("Decoder argument `typical_p` is not supported with beam groups.")

            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_constraint_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            if generation_config.num_beams <= 1:
                raise ValueError("`num_beams` needs to be greater than 1 for constrained generation.")

            if generation_config.do_sample:
                raise ValueError("`do_sample` needs to be false for constrained generation.")

            if generation_config.num_beam_groups is not None and generation_config.num_beam_groups > 1:
                raise ValueError("`num_beam_groups` not supported yet for constrained generation.")

            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]`"
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

    def sample(
        self,
        input_ids: torch.LongTensor,
        noise_order: Optional[int] = 2,
        n: Optional[int] = 2,
        threshold_get_from_true_token: Optional[float] = 0.7,
        use_noise_threshold: Optional[float] = 0.0,
        with_prompt: Optional[bool] = True,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs
    ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        self.time_step = 1
        self.loss_mask = []
        self.loss_mask_all_noise = []
        noise_generate = True
        # generate noise for prompt, give one token to the server at a time
        if with_prompt:
            prompt_length = input_ids.shape[1]
            for idx in range(1, prompt_length):
                self.time_step += 1
                token = input_ids[0,idx]
                if self.noise_type == "bigram":
                    self.generate_noised_input_bigram_argmax(
                        token,
                        noise_order = 8,
                        use_noise_threshold = use_noise_threshold,
                        n = n,
                        is_prompt = True,
                        is_start_of_sentence = True if idx == 1 else False,
                        threshold_get_from_true_token = 0.4
                    )
                elif self.noise_type == "trigram":
                    continue
                    # TODO
                else:
                    self.generate_noised_input_nway2order_argmax(
                        token,
                        noise_order = noise_order,
                        use_noise_threshold = use_noise_threshold,
                        is_prompt = True,
                        is_start_of_sentence = True if idx == 1 else False,
                        threshold_get_from_true_token = 0,
                    )
        else:
            prompt_length = input_ids.shape[1]
            for idx in range(1, prompt_length):
                self.time_step += 1
                token = input_ids[0,idx]
                # self.input_ids = input_ids
                # self.input_ids_all_noise = input_ids
                if self.noise_type == "bigram":
                    self.generate_noised_input_bigram_argmax(
                        token,
                        noise_order = 8,
                        use_noise_threshold = use_noise_threshold,
                        n = n,
                        is_prompt = True,
                        noise_scheme = "none",
                        is_start_of_sentence = True if idx == 1 else False,
                        # threshold_get_from_true_token = threshold_get_from_true_token
                    )
                # elif self.noise_type == "trigram":
                    # TODO
                else:
                    self.generate_noised_input_nway2order_argmax(
                        token,
                        noise_order = noise_order,
                        use_noise_threshold = use_noise_threshold,
                        is_prompt = True,
                        noise_scheme = "none",
                        is_start_of_sentence = True if idx == 1 else False,
                        # threshold_get_from_true_token = threshold_get_from_true_token
                    )
        print("#####", self.input_ids.shape)
        generation_index = 0
        # auto-regressive generation
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
            self.time_step += 1
            # prepare model inputs
            # model_inputs = self.prepare_inputs_for_generation(input_ids, noise_generate = noise_generate, noise_order = noise_order, **model_kwargs)
            if self.noise_type == "bigram":
                if with_prompt:
                    _, loss_mask, outputs = self.generate_noised_input_bigram_argmax(
                        None, 
                        noise_order = 8, 
                        is_prompt = False, 
                        n = n,
                        use_noise_threshold = use_noise_threshold,
                        is_start_of_sentence = True if generation_index == 0 else False,
                        threshold_get_from_true_token = threshold_get_from_true_token,
                        logits_processor=logits_processor,
                        logits_warper=logits_warper
                    )
                else:
                    _, loss_mask, outputs = self.generate_noised_input_bigram_argmax(
                        None, 
                        noise_order = 8, 
                        is_prompt = False, 
                        n = n,
                        use_noise_threshold = use_noise_threshold,
                        is_start_of_sentence = True if generation_index == 0 else False,
                        threshold_get_from_true_token = threshold_get_from_true_token,
                        logits_processor=logits_processor,
                        logits_warper=logits_warper
                    )
            elif self.noise_type == "trigram":
                if with_prompt:
                    # TODO
                   continue
                else:
                    _, loss_mask, outputs = self.generate_noised_input_trigram_no_noise_for_prompt(
                        None, 
                        noise_order = 8, 
                        is_prompt = False, 
                        n = n,
                        use_noise_threshold = use_noise_threshold,
                        is_start_of_sentence = True if generation_index == 0 else False,
                        threshold_get_from_true_token = threshold_get_from_true_token,
                        logits_processor=logits_processor,
                        logits_warper=logits_warper
                    )
            elif self.noise_type == "direct_noise":
                _, loss_mask, outputs = self.generate_noised_input_baseline(
                        None, 
                        noise_order = 1, 
                        is_prompt = False, 
                        use_noise_threshold = use_noise_threshold,
                        is_start_of_sentence = True if generation_index == 0 else False,
                        threshold_get_from_true_token = threshold_get_from_true_token,
                        logits_processor=logits_processor,
                        logits_warper=logits_warper
                    )
            else:
                if with_prompt:
                    _, loss_mask, outputs = self.generate_noised_input_nway2order_argmax(
                        None, 
                        noise_order = noise_order, 
                        is_prompt = False, 
                        use_noise_threshold = use_noise_threshold,
                        is_start_of_sentence = True if generation_index == 0 else False,
                        threshold_get_from_true_token = threshold_get_from_true_token,
                        logits_processor=logits_processor,
                        logits_warper=logits_warper
                    )
                else:
                    # _, loss_mask, outputs = self.generate_noised_input_nway2order_no_noise_for_prompt_all_noise(
                    #     None, 
                    #     noise_order = noise_order, 
                    #     is_prompt = False,
                    #     use_noise_threshold = use_noise_threshold,
                    #     is_start_of_sentence = True if generation_index == 0 else False,
                    #     threshold_get_from_true_token = threshold_get_from_true_token,
                    #     logits_processor=logits_processor,
                    #     logits_warper=logits_warper
                    # )
                    _, loss_mask, outputs = self.generate_noised_input_unigram_no_noise_for_prompt(
                        None, 
                        noise_order = noise_order, 
                        is_prompt = False,
                        use_noise_threshold = use_noise_threshold,
                        is_start_of_sentence = True if generation_index == 0 else False,
                        threshold_get_from_true_token = threshold_get_from_true_token,
                        logits_processor=logits_processor,
                        logits_warper=logits_warper
                    )
            # end generation early
            if outputs == False:
                break

            generation_index += 1
            if generation_index >= 60:
                break
            # forward pass to get next token
            # outputs contains the usual outputs
            # loss_mask contains the mask of which token is the true token
            # noise_magnitude contains a scalar that indicates how many noise options are added for each input token
            
            # outputs, loss_mask, noise_magnitude = self(
            #     **model_inputs,
            #     noise_generate=noise_generate,
            #     return_dict=True,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            # )
            # print("########noise magnitude: ", noise_magnitude)

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # if noise_generate:
                # for idx in range(1, self.noise_magnitude + 1):
                    # if loss_mask[-1 * idx] == 0:
                        # break
                # print("idx: ", idx)
                # next_token_logits = outputs.logits[:, -1 * idx, :]
            # else:
            # next_token_logits = outputs.logits[:, -1, :]
            

            # pre-process distribution
            # next_token_scores = logits_processor(input_ids, next_token_logits)
            # next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            # if return_dict_in_generate:
            #     if output_scores:
            #         scores += (next_token_scores,)
            #     if output_attentions:
            #         decoder_attentions += (
            #             (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
            #         )
            #         if self.config.is_encoder_decoder:
            #             cross_attentions += (outputs.cross_attentions,)

            #     if output_hidden_states:
            #         decoder_hidden_states += (
            #             (outputs.decoder_hidden_states,)
            #             if self.config.is_encoder_decoder
            #             else (outputs.hidden_states,)
            #         )

            # sample
            # probs = nn.functional.softmax(next_token_scores, dim=-1)
            # next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            # next_token_scores_argmax = logits_processor(input_ids, next_token_logits)
            # next_tokens_argmax = torch.argmax(next_token_scores_argmax, dim=-1)

            # finished sentences should have their next token be a padding token
            # if eos_token_id is not None:
            #     if pad_token_id is None:
            #         raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            #     next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # update generated ids, model inputs, and length for next step
            # input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            
            # if streamer is not None:
            #     streamer.put(next_tokens.cpu())
            # model_kwargs = self._update_model_kwargs_for_generation(
            #     outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            # )

            # if eos_token was found in one sentence, set sentence to finished
            # if eos_token_id_tensor is not None:
            #     unfinished_sequences = unfinished_sequences.mul(
            #         next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            #     )

            #     # stop when each sentence is finished
            #     if unfinished_sequences.max() == 0:
            #         this_peer_finished = True

            # # stop if we exceed the maximum length
            # if stopping_criteria(input_ids, scores):
            #     this_peer_finished = True

            # if this_peer_finished and not synced_gpus:
            #     break
        # print("#############argmax gen percentage: ", self.argmax_gen_cnt / self.total_gen_cnt)
        # if streamer is not None:
        #     streamer.end()

        # if return_dict_in_generate:
        #     if self.config.is_encoder_decoder:
        #         return SampleEncoderDecoderOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             encoder_attentions=encoder_attentions,
        #             encoder_hidden_states=encoder_hidden_states,
        #             decoder_attentions=decoder_attentions,
        #             cross_attentions=cross_attentions,
        #             decoder_hidden_states=decoder_hidden_states,
        #         )
        #     else:
        #         return SampleDecoderOnlyOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             attentions=decoder_attentions,
        #             hidden_states=decoder_hidden_states,
        #         )
        else:
            return input_ids

    def get_generated_token(
        self,
        logits_processor,
        logits_warper,
        outputs,
        loss_mask,
        modify_true_generation=True,
    ):
        for idx in range(1, self.noise_magnitude + 1):
            if loss_mask[-1 * idx] == 0:
                break
        next_token_logits = outputs.logits[:, -1*idx, :]
        self.true_input_ids =  self.true_input_ids.to(torch.int64)
        next_token_scores = logits_processor(self.true_input_ids.unsqueeze(dim=0), next_token_logits)
        # next_tokens = torch.argmax(next_token_scores, dim=-1)

        next_token_scores = logits_warper(self.true_input_ids, next_token_scores)

        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        if modify_true_generation:
            self.true_input_ids = torch.cat((self.true_input_ids, next_tokens), dim=-1)

        return next_tokens
    '''
    different modes of generating noise:
    1. is_prompt == True and is_start_of_sentence == True: generate noise for the first token based on <bos>
    2. is_prompt == False and is_start_of_sentence == True: generating the first token that is not in the prompt
    '''
    '''
    def generate_noised_input_nway2order_argmax(
        self, 
        input_ids, 
        noise_order = 3, 
        is_prompt=True, 
        is_start_of_sentence=False,
        threshold_get_from_true_token = 0.5,
        logits_processor=None,
        logits_warper=None,
    ):

        import random
        
        new_input = []
        # loss_masks = []
        # key: -1 for true token, -2, -3... for noises
        # value: -1, -2, -3... for position in logits
        toks_to_indices = {k:0 for k in range(-1*noise_order, 0)}
        # key: -1 for true token, -2, -3... for noises
        # value: token number
        tok_to_noise = {k:-1 for k in range(-1*noise_order, 0)}

        if not is_prompt:
            if is_start_of_sentence:
                new_input = self.input_ids
                toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
                self.total_noised_input_ids = new_input.clone()
            else:
                new_input = self.sequence_last_time_stamp
                toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
        else:
            if is_start_of_sentence:
                self.loss_mask.append(0)
                new_input.append(2)
                toks_to_indices[-1] = -1
            else:
                new_input = self.input_ids
                toks_to_indices = self.toks_to_indices_last_time_stamp.copy()

        new_input = torch.tensor(new_input).unsqueeze(dim=0).to(torch.int)
        self.input_ids = new_input
        if is_prompt or (not is_prompt and is_start_of_sentence):
            self.attention_mask = torch.ones(self.input_ids.shape)
        else:
            self.attention_mask = torch.cat((self.attention_mask, torch.ones(noise_order).unsqueeze(dim=0)),dim=1)
        
        loss_mask = self.loss_mask
        self.noise_magnitude = noise_order

        with torch.no_grad():
            # generation with past key values
            if not is_prompt and not is_start_of_sentence: 
                outputs, _, _ = self(
                    new_input, 
                    past_key_values=self.argmax_past_key_values, 
                    for_argmax_noise_generation = True
                )
            else:
                outputs, _, _ = self(
                    new_input, 
                    for_argmax_noise_generation = True
                )
            self.argmax_past_key_values = outputs.past_key_values

        # if is in generation mode, sample a token first
        if not is_prompt:
            input_ids = self.get_generated_token(
                logits_processor,
                logits_warper,
                outputs,
                loss_mask
            )
            
        # if currently on the start token, get all noises from the logits of the start token
        if is_start_of_sentence and is_prompt:
            next_token_logits = outputs.logits[:, -1, :]
            noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
            noise_tok_cnt = 2
            seen_toks = [input_ids, torch.tensor(2)]
            while noise_tok_cnt <= noise_order:
                while noise_tok in seen_toks:
                    noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                tok_to_noise[-1*noise_tok_cnt] = noise_tok
                noise_tok_cnt += 1
                seen_toks.append(noise_tok.squeeze(dim=0))
        else:
            seen_toks = [input_ids]
            for k,v in toks_to_indices.items():
                if k == -1:
                    continue
                next_token_logits = outputs.logits[:, v, :]
                noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                while noise_tok in seen_toks:
                    noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)                        
                tok_to_noise[k] = noise_tok
                seen_toks.append(noise_tok.squeeze(dim=0))

        if not is_prompt:
            new_input = torch.tensor([])

        shuffled_noises = list(tok_to_noise.items())
        random.shuffle(shuffled_noises)

        # let first noise token always come from bos
        seen_toks.append(input_ids)
        old_real_tok_idx = toks_to_indices[-1]
        token_index = noise_order
        for idx,token in shuffled_noises:
            # if is the true token
            if idx == -1:
                loss_mask.append(0)
                if is_prompt:
                    new_input = torch.cat((new_input.squeeze(dim=0),input_ids.unsqueeze(dim=0)))
                else:
                    new_input = torch.cat((new_input,input_ids))
            else:
                # if it is a noise token
                # determine if we are going to sample from true token distribution
                if is_prompt == True and is_start_of_sentence == True:
                    use_noise_from_true_token = float("inf")
                else:
                    use_noise_from_true_token = random.uniform(0,1)
                    
                next_token_logits = outputs.logits[:, old_real_tok_idx, :]
                noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)

                if use_noise_from_true_token < threshold_get_from_true_token:
                    while noise_tok in seen_toks:
                        noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                    seen_toks.append(noise_tok.squeeze(dim=0))
                    loss_mask.append(-1)
                    if is_prompt:
                        new_input = torch.cat((new_input.squeeze(dim=0),noise_tok))
                    else:
                        new_input = torch.cat((new_input,noise_tok))
                else:
                    loss_mask.append(-1)
                    if is_prompt:
                        new_input = torch.cat((new_input.squeeze(dim=0),token))
                    else:
                        new_input = torch.cat((new_input, token))
            toks_to_indices[idx] = -1 * token_index
            token_index -= 1

        # keep track of the lattice for the last token and their index mappings for generation
        self.sequence_last_time_stamp = new_input[-1*noise_order:]
        self.toks_to_indices_last_time_stamp = toks_to_indices.copy()
        
        self.input_ids = torch.tensor(new_input)
        self.loss_mask = loss_mask
        self.noise_magnitude = noise_order

        if is_prompt:
            self.true_input_ids = torch.cat((self.true_input_ids, input_ids.unsqueeze(dim=0)), dim=0)
        else:
            self.total_noised_input_ids = torch.cat((self.total_noised_input_ids, self.input_ids), dim=0)
        return self.input_ids, self.loss_mask, outputs
    '''

    def generate_noised_input_bigram_argmax(
        self, 
        input_ids, 
        noise_order = 8, 
        is_prompt=True, 
        prompt_noised = True,
        noise_scheme = "mix",
        n = 2,
        is_start_of_sentence=False,
        threshold_get_from_true_token = 0.6,
        use_noise_threshold = -1,
        logits_processor=None,
        logits_warper=None,
    ):
        import random
        new_input = []
        # loss_masks = []
        # key: -1 for previous time stamp true token, -2 for previous time stamp noise token
        # value: -1, -2, -3... for position in logits
        toks_to_indices = {k:0 for k in range(-1*n, 0)}
        # key: -1 for current time stamp true token, -2 for current time stamp noise token, -3 for previous time stamp true token, -4 for previous time stamp noise token
        # value: token number
        tok_to_noise = {k:-1 for k in range(-2*n, 0)}
        self.noise_magnitude = n**2*2
        if not is_prompt:
            if is_start_of_sentence:
                # new_input = self.input_ids
                new_input = self.sequence_last_time_stamp
                toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
                # self.attention_mask = torch.ones(self.input_ids.shape)
                # self.total_noised_input_ids = new_input.clone()
                tok_to_noise = self.toks_to_noise_last_time_stamp.copy()
            else:
                new_input = self.sequence_last_time_stamp
                # new_input = self.input_ids
                toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
                tok_to_noise = self.toks_to_noise_last_time_stamp.copy()
        else:
            if is_start_of_sentence:
                true_tok = random.choice(list( (-1*self.noise_magnitude+1,0,2)))
                self.loss_mask.extend([-1]*self.noise_magnitude)
                self.loss_mask[true_tok] = 0
                new_input.extend([2]*self.noise_magnitude)
                toks_to_indices[-1] = true_tok
                self.total_noised_input_ids = torch.tensor([2]*self.noise_magnitude)
            else:
                # new_input = self.input_ids
                new_input = self.sequence_last_time_stamp
                toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
                tok_to_noise = self.toks_to_noise_last_time_stamp.copy()


        new_input = torch.tensor(new_input).unsqueeze(dim=0).to(torch.int)
        self.input_ids = new_input
        if is_start_of_sentence and is_prompt:
            self.attention_mask = torch.ones(self.input_ids.shape)
        else:
            self.attention_mask = torch.cat((self.attention_mask, torch.ones(n**2*2).unsqueeze(dim=0)),dim=1)
        
        loss_mask = self.loss_mask

        with torch.no_grad():
            outputs, _, _ = self(
                new_input, 
                past_key_values=self.noise_past_key_values, 
                for_argmax_noise_generation = True
            )
            self.noise_past_key_values = outputs.past_key_values

        # if is in generation mode, sample a token first
        if not is_prompt:
            input_ids = self.get_generated_token(
                logits_processor,
                logits_warper,
                outputs,
                loss_mask
            )

            # if starts to generate pad tokens, end generation.
            if input_ids.squeeze(dim=0) == 1:
                return None, None, False
            
        # if currently on the start token, get all noises from the logits of the start token
        if is_start_of_sentence and is_prompt:
            next_token_logits = outputs.logits[:, toks_to_indices[-1], :]
            seen_toks = [input_ids]
            if noise_scheme == "mix":
                noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
            elif noise_scheme == "synonym":
                results = self.vdb[int(input_ids)]
                choices = list( (10,20))
                choice = random.choice(choices)
                noise_tok = int(results[choice])
            else:   # prompt not noised
                noise_tok = input_ids
            for i in range(-1*n,-1):
                while noise_tok in seen_toks:
                    if noise_scheme == "mix":
                        noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                    elif noise_scheme == "synonym":   
                        choice = random.choice(choices)
                        noise_tok = int(results[choice])       
                    else:
                        noise_tok = input_ids
                        break         
                tok_to_noise[i] = noise_tok
                seen_toks.append(noise_tok)
            tok_to_noise[-1] = input_ids
            for i in range(-2*n,-1*n):
                tok_to_noise[i] = 2
        else:
            seen_toks = [input_ids]
            for i in range(-1*n, -1):
                v = toks_to_indices[i]
                next_token_logits = outputs.logits[:, v, :]
                if noise_scheme == "mix":
                    noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                elif noise_scheme == "synonym":
                    results = self.vdb[int(input_ids)]
                    choices = list( (10,20))
                    choice = random.choice(choices)
                    noise_tok = int(results[choice])
                else:
                    noise_tok = input_ids
                while noise_tok in seen_toks:
                    if noise_scheme == "mix":
                        noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5) 
                    elif noise_scheme == "synonym":
                        choice = random.choice(choices)
                        noise_tok = int(results[choice])  
                    else:
                        noise_tok = input_ids
                        break
                tok_to_noise[i] = noise_tok
                seen_toks.append(noise_tok)
            tok_to_noise[-1] = input_ids                      
        
        use_mix_flag = False
        # key: -1 for current time stamp true token, -2 for current time stamp noise token, -3 for previous time stamp true token, -4 for previous time stamp noise token      
        if noise_scheme == "mix":
            for i in range(-1*n,-1):
                mix_ratio = random.uniform(0,1)
                if mix_ratio < threshold_get_from_true_token:
                    use_mix_flag = True
                    v = toks_to_indices[-1]
                    next_token_logits = outputs.logits[:, v, :]
                    while noise_tok in seen_toks:
                        noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                    seen_toks.append(noise_tok.squeeze(dim=0))
                    tok_to_noise[i] = noise_tok
                    self.noise_tokens_for_display.append([noise_tok])

        ft_keys = 0
        bigram_construct = {}
        for i in range(-1*n,0):
            for j in range(-1*2*n,-1*n):
                if i == -1 and j == -1*n-1:
                    bigram_construct["tt"] = [tok_to_noise[j],tok_to_noise[i]]
                elif abs(i - j) == n:
                    bigram_construct[f"ff{i}"] = [tok_to_noise[j],tok_to_noise[i]]
                else:
                    bigram_construct[f"ft{ft_keys}"] = [tok_to_noise[j],tok_to_noise[i]]
                    ft_keys += 1        

        shuffled_noise = list(bigram_construct.items())
        random.seed(self.time_step * self.large_prime)
        random.shuffle(shuffled_noise)

        seen_toks.append(input_ids)
        old_real_tok_idx = toks_to_indices[-1]
        token_index = n**2*2-1

        for code, construct in shuffled_noise:
            construct = torch.tensor(construct)
            new_input = torch.cat((new_input.squeeze(dim=0),construct))
            if code == "tt":
                loss_mask.extend([-1,0])
                toks_to_indices[-1] = -1 * token_index
            elif "ff" in code:
                noise_seq_id = int(code[-2:])
                toks_to_indices[noise_seq_id] = -1 * token_index
                loss_mask.extend([-1,-1])
            else:
                loss_mask.extend([-1,-1])
            token_index -= 2
        
        # change from this time stamp to last time stamp
        self.toks_to_noise_last_time_stamp = tok_to_noise.copy()
        for i in range(-1*n,0):
            self.toks_to_noise_last_time_stamp[i-n] = tok_to_noise[i]
        self.toks_to_indices_last_time_stamp = toks_to_indices.copy()
        self.sequence_last_time_stamp = new_input[-1*n**2*2:]
        self.input_ids = torch.tensor(new_input)
        self.loss_mask = loss_mask
        self.noise_magnitude = noise_order

        if is_prompt:
            self.true_input_ids = torch.cat((self.true_input_ids, input_ids.unsqueeze(dim=0)), dim=0)
            self.total_noised_input_ids = torch.cat((self.total_noised_input_ids, self.sequence_last_time_stamp), dim=0)
            self.total_noised_input_ids_for_display = torch.cat((self.total_noised_input_ids_for_display, self.sequence_last_time_stamp), dim=0)
        else:
            self.total_noised_input_ids = torch.cat((self.total_noised_input_ids, self.sequence_last_time_stamp), dim=0)
            self.total_noised_input_ids_for_display = torch.cat((self.total_noised_input_ids_for_display, self.sequence_last_time_stamp), dim=0)
        if use_mix_flag:
            self.total_noised_input_ids_for_display = torch.cat((self.total_noised_input_ids_for_display, torch.tensor([1])), dim=0)
        
        return self.input_ids, self.loss_mask, outputs

    '''
    def generate_noised_input_bigram_no_noise_for_prompt(
        self, 
        input_ids, 
        noise_order = 8, 
        is_prompt=True, 
        is_start_of_sentence=False,
        threshold_get_from_true_token = 0.7,
        use_noise_threshold = -1,
        logits_processor=None,
        logits_warper=None,
    ):
        import random
        new_input = []
        # loss_masks = []
        # key: -1 for previous time stamp true token, -2 for previous time stamp noise token
        # value: -1, -2, -3... for position in logits
        toks_to_indices = {k:0 for k in range(-2, 0)}
        # key: -1 for current time stamp true token, -2 for current time stamp noise token, -3 for previous time stamp true token, -4 for previous time stamp noise token
        # value: token number
        tok_to_noise = {k:-1 for k in range(-4, 0)}

        if not is_prompt:
            if is_start_of_sentence:
                self.true_input_ids = torch.cat((self.true_input_ids, self.input_ids.squeeze(dim=0)), dim=0)
                new_input = self.input_ids.squeeze(dim=0)
                self.loss_mask.extend([0]*self.input_ids.shape[1])
                # toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
                self.total_noised_input_ids = new_input.clone()
                last_prompt_tok = self.input_ids[0,-1]
            else:
                new_input = self.input_ids
                toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
                tok_to_noise = self.toks_to_noise_last_time_stamp.copy()

        new_input = torch.tensor(new_input).unsqueeze(dim=0).to(torch.int)
        self.input_ids = new_input
        if is_prompt or (not is_prompt and is_start_of_sentence):
            self.attention_mask = torch.ones(self.input_ids.shape)
        else:
            self.attention_mask = torch.cat((self.attention_mask, torch.ones(noise_order).unsqueeze(dim=0)),dim=1)
        
        loss_mask = self.loss_mask
        self.noise_magnitude = noise_order

        with torch.no_grad():
            # generation with past key values
            if not is_prompt and not is_start_of_sentence: 
                outputs, _, _ = self(
                    new_input, 
                    # past_key_values=self.argmax_past_key_values, 
                    for_argmax_noise_generation = True
                )
            else:
                outputs, _, _ = self(
                    new_input, 
                    for_argmax_noise_generation = True
                )
            self.argmax_past_key_values = outputs.past_key_values

        seen = []

        # if is in generation mode, sample a token first
        if not is_prompt:
            input_ids = self.get_generated_token(
                logits_processor,
                logits_warper,
                outputs,
                loss_mask
            )
            # while input_ids in seen:
            #     input_ids = self.get_generated_token(
            #         logits_processor,
            #         logits_warper,
            #         outputs,
            #         loss_mask
            #     )
            # if starts to generate pad tokens, end generation.
            if input_ids.squeeze(dim=0) == 1:
                return None, None, False

        # if currently on the start token, get all noises from the logits of the start token
        if is_start_of_sentence:
            next_token_logits = outputs.logits[:, -1, :]
            noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
            seen_toks = [input_ids, torch.tensor(2), last_prompt_tok]
            while noise_tok in seen_toks:
                noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
            tok_to_noise[-1] = input_ids
            tok_to_noise[-2] = noise_tok
            tok_to_noise[-3] = last_prompt_tok
            tok_to_noise[-4] = last_prompt_tok
            seen_toks.append(noise_tok.squeeze(dim=0))
        else:
            seen_toks = [input_ids, tok_to_noise[-3], tok_to_noise[-4]]
            v = toks_to_indices[-2]
            next_token_logits = outputs.logits[:, v, :]
            noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
            while noise_tok in seen_toks:
                noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)  
            tok_to_noise[-1] = input_ids                      
            tok_to_noise[-2] = noise_tok
            seen_toks.append(noise_tok.squeeze(dim=0))
        
        # key: -1 for current time stamp true token, -2 for current time stamp noise token, -3 for previous time stamp true token, -4 for previous time stamp noise token
        use_noise = random.uniform(0,1)
        if use_noise < use_noise_threshold:
            bigram_construct = {"tt":[tok_to_noise[-3],tok_to_noise[-1]], "tf":[tok_to_noise[-3],tok_to_noise[-1]],"ft":[tok_to_noise[-4],tok_to_noise[-1]],"ff":[tok_to_noise[-4],tok_to_noise[-1]]}
            tok_to_noise[-2] = tok_to_noise[-1]
        else:
            bigram_construct = {"tt":[tok_to_noise[-3],tok_to_noise[-1]], "tf":[tok_to_noise[-3],tok_to_noise[-2]],"ft":[tok_to_noise[-4],tok_to_noise[-1]],"ff":[tok_to_noise[-4],tok_to_noise[-2]]}
        shuffled_noise = list(bigram_construct.items())
        random.shuffle(shuffled_noise)

        seen_toks.append(input_ids)
        old_real_tok_idx = toks_to_indices[-1]
        token_index = noise_order-1

        for code, construct in shuffled_noise:
            
            construct = torch.tensor(construct)
            new_input = torch.cat((new_input.squeeze(dim=0),construct))
            if code == "tt":
                loss_mask.extend([-1,0])
                toks_to_indices[-1] = -1 * token_index
            elif code == "tf":
                loss_mask.extend([-1,-1])
            elif code == "ft":
                loss_mask.extend([-1,-1])
            elif code == "ff":
                loss_mask.extend([-1,-1])
                toks_to_indices[-2] = -1 * token_index
            token_index -= 2
        
        # change from this time stamp to last time stamp
        self.toks_to_noise_last_time_stamp = tok_to_noise.copy()
        self.toks_to_noise_last_time_stamp[-3] = tok_to_noise[-1]
        self.toks_to_noise_last_time_stamp[-4] = tok_to_noise[-2]
        self.toks_to_indices_last_time_stamp = toks_to_indices.copy()
        self.sequence_last_time_stamp = new_input[-1*noise_order:]
        self.input_ids = torch.tensor(new_input)
        self.loss_mask = loss_mask
        self.noise_magnitude = noise_order
        seen_toks = [self.toks_to_noise_last_time_stamp[-3], self.toks_to_noise_last_time_stamp[-4]]
        if is_prompt:
            self.true_input_ids = torch.cat((self.true_input_ids, input_ids.unsqueeze(dim=0)), dim=0)
            self.total_noised_input_ids = new_input.clone()
        else:
            self.total_noised_input_ids = torch.cat((self.total_noised_input_ids, self.sequence_last_time_stamp), dim=0)
        
        return self.input_ids, self.loss_mask, outputs
    '''
    
    def generate_noised_input_bigram_no_noise_for_prompt(
        self, 
        input_ids, 
        noise_scheme = "mix",
        n = 2,
        noise_order = 8, 
        is_prompt=True, 
        is_start_of_sentence=False,
        threshold_get_from_true_token = 0.7,
        use_noise_threshold = -1,
        logits_processor=None,
        logits_warper=None,
    ):
        import time
        import random
        start = time.time()
        new_input = []
        # loss_masks = []
        # key: -1 for previous time stamp true token, -2 for previous time stamp noise token
        # value: -1, -2, -3... for position in logits
        toks_to_indices = {k:0 for k in range(-1*n, 0)}
        # key: -1 for current time stamp true token, -2 for current time stamp noise token1,..-n for current time stamp noise token n
        # key: -(n+1) for previous time stamp true token, -(n+2) for previous time stamp noise token...-2n for previous time stamp noise token n 
        # value: token number
        tok_to_noise = {k:-1 for k in range(-2*n, 0)}

        if not is_prompt:
            if is_start_of_sentence:
                self.true_input_ids = torch.cat((self.true_input_ids, self.input_ids.squeeze(dim=0)), dim=0)
                new_input = self.input_ids.squeeze(dim=0)
                self.loss_mask.extend([0]*self.input_ids.shape[1])
                # toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
                self.total_noised_input_ids = new_input.clone()
                last_prompt_tok = self.input_ids[0,-1]
            else:
                new_input = self.sequence_last_time_stamp #self.input_ids
                toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
                tok_to_noise = self.toks_to_noise_last_time_stamp.copy()
        
        new_input = torch.tensor(new_input).unsqueeze(dim=0).to(torch.int)
        self.input_ids = new_input
        if is_prompt or (not is_prompt and is_start_of_sentence):
            self.attention_mask = torch.ones(self.input_ids.shape)
        else:
            self.attention_mask = torch.cat((self.attention_mask, torch.ones(n**2*2).unsqueeze(dim=0)),dim=1)
        
        loss_mask = self.loss_mask
        self.noise_magnitude = n**2*2

        with torch.no_grad():
            # generation with past key values
            outputs, _, _ = self(
                new_input, 
                past_key_values=self.noise_past_key_values, 
                for_argmax_noise_generation = True
            )
            self.noise_past_key_values = outputs.past_key_values
        seen = []

        # if is in generation mode, sample a token first
        if not is_prompt:
            input_ids = self.get_generated_token(
                logits_processor,
                logits_warper,
                outputs,
                loss_mask
            )
            # if starts to generate pad tokens, end generation.
            if input_ids.squeeze(dim=0) == 1:
                return None, None, False

        # if currently on the start token, get all noises from the logits of the start token
        if is_start_of_sentence:
            next_token_logits = outputs.logits[:, -1, :]
            seen_toks = [input_ids, torch.tensor(2)]#, last_prompt_tok]
            if noise_scheme == "mix":
                noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
            else:
                results = self.vdb[int(input_ids)]
                choices = list( (10,20))
                choice = random.choice(choices)
                noise_tok = int(results[choice])
            for i in range(-1*n,-1):
                while noise_tok in seen_toks:
                    if noise_scheme == "mix":
                        noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                    else:   # synonym scheme
                        choice = random.choice(choices)
                        noise_tok = int(results[choice])
                tok_to_noise[i] = noise_tok
                seen_toks.append(noise_tok)
            tok_to_noise[-1] = input_ids
            for i in range(-2*n,-1*n):
                tok_to_noise[i] = last_prompt_tok
        else:
            seen_toks = [input_ids] # + [tok_to_noise[i] for i in range(-1*2*n,-1*n)]
            for i in range(-1*n,-1):
                v = toks_to_indices[i]
                next_token_logits = outputs.logits[:, v, :]
                if noise_scheme == "mix":
                    noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                else:
                    results = self.vdb[int(input_ids)]
                    choices = list( (10,20))
                    choice = random.choice(choices)
                    noise_tok = int(results[choice])
                while noise_tok in seen_toks:
                    if noise_scheme == "mix":
                        noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5) 
                    else:
                        choice = random.choice(choices)
                        noise_tok = int(results[choice])
                tok_to_noise[i] = noise_tok
                seen_toks.append(noise_tok)
            tok_to_noise[-1] = input_ids                      
        
        # using mixed noise ratio if the scheme is mix scheme
        noise_scheme_branched = {x:False for x in range(-1*n,-1)}
        use_mix_flag = False
        if noise_scheme == "mix":
            for i in range(-1*n,-1):
                mix_ratio = random.uniform(0,1)
                if mix_ratio < threshold_get_from_true_token:
                    noise_scheme_branched[i] = True
                    use_mix_flag = True
                    v = toks_to_indices[-1]
                    next_token_logits = outputs.logits[:, v, :]
                    while noise_tok in seen_toks:
                        noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                    seen_toks.append(noise_tok.squeeze(dim=0))
                    tok_to_noise[i] = noise_tok
                    self.noise_tokens_for_display.append([noise_tok])

        # print(seen_toks)
        # key: -1 for current time stamp true token, -2 for current time stamp noise token, -3 for previous time stamp true token, -4 for previous time stamp noise token
        ft_keys = 0
        bigram_construct = {}
        for i in range(-1*n,0):
            for j in range(-2*n,-1*n):
                if i == -1 and j == -1*n-1:
                    bigram_construct["tt"] = [tok_to_noise[j],tok_to_noise[i]]
                elif abs(i - j) == n:
                    bigram_construct[f"ff{i}"] = [tok_to_noise[j],tok_to_noise[i]]
                else:
                    bigram_construct[f"ft{ft_keys}"] = [tok_to_noise[j],tok_to_noise[i]]
                    ft_keys += 1


        shuffled_noise = list(bigram_construct.items())
        random.seed(self.time_step * self.large_prime)  # setting random seed to be t*p, where p is a large prime number input by user
        random.shuffle(shuffled_noise)

        seen_toks.append(input_ids)
        old_real_tok_idx = toks_to_indices[-1]
        token_index = n**2*2-1

        for code, construct in shuffled_noise:
            
            construct = torch.tensor(construct)
            new_input = torch.cat((new_input.squeeze(dim=0),construct))
            
            for kkk in range(-1*n,-1):
                if construct[-1] == tok_to_noise[kkk] and noise_scheme_branched[kkk]:
                    self.total_noised_input_ids_for_display = torch.cat((self.total_noised_input_ids_for_display, construct, torch.tensor([1])), dim=0)
                    break
            else:
                self.total_noised_input_ids_for_display = torch.cat((self.total_noised_input_ids_for_display, construct), dim=0)

            if code == "tt":
                loss_mask.extend([-1,0])
                toks_to_indices[-1] = -1 * token_index
            elif "ff" in code:
                noise_seq_id = int(code[-2:])
                toks_to_indices[noise_seq_id] = -1 * token_index
                loss_mask.extend([-1,-1])
            else:
                loss_mask.extend([-1,-1])

            token_index -= 2
        
        # change from this time stamp to last time stamp
        self.toks_to_noise_last_time_stamp = tok_to_noise.copy()
        for i in range(-1*n,0):
            self.toks_to_noise_last_time_stamp[i-n] = tok_to_noise[i]
        self.toks_to_indices_last_time_stamp = toks_to_indices.copy()
        self.sequence_last_time_stamp = new_input[-1*n**2*2:]
        self.input_ids = torch.tensor(new_input)
        self.loss_mask = loss_mask
        self.noise_magnitude = noise_order
        # seen_toks = [self.toks_to_noise_last_time_stamp[i] for i in range(-1*2*n,-1*n)]
        if is_prompt:
            self.true_input_ids = torch.cat((self.true_input_ids, input_ids.unsqueeze(dim=0)), dim=0)
            self.total_noised_input_ids = new_input.clone()
        else:
            self.total_noised_input_ids = torch.cat((self.total_noised_input_ids, self.sequence_last_time_stamp), dim=0)
            
            # if use_mix_flag:
            #     if self.total_noised_input_ids_for_display is None:
            #         self.total_noised_input_ids_for_display = self.total_noised_input_ids.clone()
            #         self.total_noised_input_ids_for_display = torch.cat((self.total_noised_input_ids_for_display, self.sequence_last_time_stamp), dim=0)
            #         self.total_noised_input_ids_for_display = torch.cat((self.total_noised_input_ids_for_display, torch.tensor([1])), dim=0)
            #     else:
            #         self.total_noised_input_ids_for_display = torch.cat((self.total_noised_input_ids_for_display, self.sequence_last_time_stamp), dim=0)
            #         self.total_noised_input_ids_for_display = torch.cat((self.total_noised_input_ids_for_display, torch.tensor([1])), dim=0)
        end = time.time()
        self.times.append(end - start)

        return self.input_ids, self.loss_mask, outputs
    
    '''
    def generate_noised_input_trigram_no_noise_for_prompt(
        self, 
        input_ids, 
        noise_scheme = "mix",
        n = 2,
        noise_order = 8, 
        is_prompt=True, 
        is_start_of_sentence=False,
        threshold_get_from_true_token = 0.7,
        use_noise_threshold = -1,
        logits_processor=None,
        logits_warper=None,
    ):
        import random
        new_input = []
        # loss_masks = []
        # key: -1 for previous time stamp true token, -2 for previous time stamp noise token
        # value: -1, -2, -3... for position in logits
        toks_to_indices = {k:0 for k in range(-1*n, 0)}
        # key: -1 for current time stamp true token, -2 for current time stamp noise token1,..-n for current time stamp noise token n
        # key: -(n+1) for previous time stamp true token, -(n+2) for previous time stamp noise token...-2n for previous time stamp noise token n 
        # value: token number
        tok_to_noise = {k:-1 for k in range(-3*n, 0)}

        if not is_prompt:
            if is_start_of_sentence:
                self.true_input_ids = torch.cat((self.true_input_ids, self.input_ids.squeeze(dim=0)), dim=0)
                new_input = self.input_ids.squeeze(dim=0)
                self.loss_mask.extend([0]*self.input_ids.shape[1])
                # toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
                self.total_noised_input_ids = new_input.clone()
                last_prompt_tok2 = self.input_ids[0,-2]
                last_prompt_tok1 = self.input_ids[0,-1]
            else:
                new_input = self.sequence_last_time_stamp # self.input_ids
                toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
                tok_to_noise = self.toks_to_noise_last_time_stamp.copy()

        new_input = torch.tensor(new_input).unsqueeze(dim=0).to(torch.int)
        
        self.input_ids = new_input
        if is_prompt or (not is_prompt and is_start_of_sentence):
            self.attention_mask = torch.ones(self.input_ids.shape)
        else:
            self.attention_mask = torch.cat((self.attention_mask, torch.ones(n**3*3).unsqueeze(dim=0)),dim=1)
        
        loss_mask = self.loss_mask
        self.noise_magnitude = n**3*3

        with torch.no_grad():
            # generation with past key values
            outputs, _, _ = self(
                new_input, 
                past_key_values=self.noise_past_key_values if self.noise_past_key_values else None, 
                for_argmax_noise_generation = True
            )
            self.noise_past_key_values = outputs.past_key_values
        
        seen = []
        # if is in generation mode, sample a token first
        if not is_prompt:
            input_ids = self.get_generated_token(
                logits_processor,
                logits_warper,
                outputs,
                loss_mask
            )
            # if starts to generate pad tokens, end generation.
            if input_ids.squeeze(dim=0) == 1:
                return None, None, False

        # if currently on the start token, get all noises from the logits of the start token
        if is_start_of_sentence:
            next_token_logits = outputs.logits[:, -1, :]
            seen_toks = [input_ids, torch.tensor(2)]
            if noise_scheme == "mix":
                noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
            else:
                results = self.vdb[int(input_ids)]
                choices = list( (10,20))
                choice = random.choice(choices)
                noise_tok = int(results[choice])
            for i in range(-1*n,-1):
                while noise_tok in seen_toks:
                    if noise_scheme == "mix":
                        noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                    else:   # synonym scheme
                        choice = random.choice(choices)
                        noise_tok = int(results[choice])
                tok_to_noise[i] = noise_tok
                seen_toks.append(noise_tok)
            tok_to_noise[-1] = input_ids
            for i in range(-3*n,-2*n):
                tok_to_noise[i] = last_prompt_tok2
            for i in range(-2*n,-1*n):
                tok_to_noise[i] = last_prompt_tok1
        else:
            seen_toks = [input_ids] + [tok_to_noise[i] for i in range(-3*n,-1*n)]
            for i in range(-1*n,-1):
                v = toks_to_indices[i]
                next_token_logits = outputs.logits[:, v, :]
                if noise_scheme == "mix":
                    noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                else:
                    results = self.vdb[int(input_ids)]
                    choices = list( (10,20))
                    choice = random.choice(choices)
                    noise_tok = int(results[choice])
                while noise_tok in seen_toks:
                    if noise_scheme == "mix":
                        noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5) 
                    else:
                        choice = random.choice(choices)
                        noise_tok = int(results[choice])
                tok_to_noise[i] = noise_tok
                seen_toks.append(noise_tok)
            tok_to_noise[-1] = input_ids                      
        
        # using mixed noise ratio if the scheme is mix scheme
        # if noise_scheme == "mix":
        #     for i in range(-1*n,-1):
        #         mix_ratio = random.uniform(0,1)
        #         if mix_ratio < threshold_get_from_true_token:
        #             v = toks_to_indices[-1]
        #             next_token_logits = outputs.logits[:, v, :]
        #             while noise_tok in seen_toks:
        #                 noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
        #             seen_toks.append(noise_tok.squeeze(dim=0))
        #             tok_to_noise[i] = noise_tok

        # key: -1 for current time stamp true token, -2 for current time stamp noise token, -3 for previous time stamp true token, -4 for previous time stamp noise token
        ft_keys = 0
        trigram_construct = {}
        for i in range(-1*n,0):
            for j in range(-3*n,-2*n):
                for k in range(-2*n,-1*n):
                    if i == -1 and j == -2*n-1 and k == -1*n-1:
                        trigram_construct["ttt"] = [tok_to_noise[j],tok_to_noise[k],tok_to_noise[i]]
                    elif abs(i-k) == n and abs(j-k) == n:
                        trigram_construct[f"fff{i}"] = [tok_to_noise[j],tok_to_noise[k],tok_to_noise[i]]
                    else:
                        if ft_keys >= 2:
                            continue
                        trigram_construct[f"fft{ft_keys}"] = [tok_to_noise[j],tok_to_noise[k],tok_to_noise[i]]
                        ft_keys += 1

        shuffled_noise = list(trigram_construct.items())
        random.seed(self.time_step * self.large_prime)  # setting random seed to be t*p, where p is a large prime number input by user
        random.shuffle(shuffled_noise)

        seen_toks.append(input_ids)
        old_real_tok_idx = toks_to_indices[-1]
        # token_index = n**3*3-2
        token_index = n**3*3-2
        for code, construct in shuffled_noise:
            construct = torch.tensor(construct)
            new_input = torch.cat((new_input.squeeze(dim=0),construct))
            if code == "ttt":
                loss_mask.extend([-1,-1,0])
                toks_to_indices[-1] = -1 * token_index
            elif "fff" in code:
                noise_seq_id = int(code[-2:])
                toks_to_indices[noise_seq_id] = -1 * token_index
                loss_mask.extend([-1,-1,-1])
            else:
                loss_mask.extend([-1,-1,-1])
            token_index -= 3
        
        # change from this time stamp to last time stamp
        self.toks_to_noise_last_time_stamp = tok_to_noise.copy()
        for i in range(-2*n,0):
            self.toks_to_noise_last_time_stamp[i-n] = tok_to_noise[i]
        self.toks_to_indices_last_time_stamp = toks_to_indices.copy()
        self.sequence_last_time_stamp = new_input[-1*n**3*3:]
        # print(self.sequence_last_time_stamp)
        self.input_ids = torch.tensor(new_input)
        self.loss_mask = loss_mask
        self.noise_magnitude =  n**3*3
        # seen_toks = [self.toks_to_noise_last_time_stamp[i] for i in range(-1*2*n,-1*n)]
        
        if is_prompt:
            self.true_input_ids = torch.cat((self.true_input_ids, input_ids.unsqueeze(dim=0)), dim=0)
            self.total_noised_input_ids = new_input.clone()
        else:
            self.total_noised_input_ids = torch.cat((self.total_noised_input_ids, self.sequence_last_time_stamp), dim=0)
        
        return self.input_ids, self.loss_mask, outputs
    '''

    def generate_noised_input_trigram_no_noise_for_prompt(
        self, 
        input_ids, 
        noise_scheme = "mix",
        n = 2,
        noise_order = 8, 
        is_prompt=True, 
        is_start_of_sentence=False,
        threshold_get_from_true_token = 0.7,
        use_noise_threshold = -1,
        logits_processor=None,
        logits_warper=None,
    ):
        import random
        new_input = []
        # loss_masks = []
        # key: -1 for previous time stamp true token, -2 for previous time stamp noise token
        # value: -1, -2, -3... for position in logits
        toks_to_indices = {k:0 for k in range(-1*n, 0)}
        # key: -1 for current time stamp true token, -2 for current time stamp noise token1,..-n for current time stamp noise token n
        # key: -(n+1) for previous time stamp true token, -(n+2) for previous time stamp noise token...-2n for previous time stamp noise token n 
        # value: token number
        tok_to_noise = {k:-1 for k in range(-3*n, 0)}

        if not is_prompt:
            if is_start_of_sentence:
                self.true_input_ids = torch.cat((self.true_input_ids, self.input_ids.squeeze(dim=0)), dim=0)
                new_input = self.input_ids.squeeze(dim=0)
                self.loss_mask.extend([0]*self.input_ids.shape[1])
                # toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
                self.total_noised_input_ids = new_input.clone()
                last_prompt_tok2 = self.input_ids[0,-2]
                last_prompt_tok1 = self.input_ids[0,-1]
            else:
                new_input = self.sequence_last_time_stamp # self.input_ids
                toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
                tok_to_noise = self.toks_to_noise_last_time_stamp.copy()

        new_input = torch.tensor(new_input).unsqueeze(dim=0).to(torch.int)
        
        self.input_ids = new_input
        if is_prompt or (not is_prompt and is_start_of_sentence):
            self.attention_mask = torch.ones(self.input_ids.shape)
        else:
            self.attention_mask = torch.cat((self.attention_mask, torch.ones(n**2*3).unsqueeze(dim=0)),dim=1)
        
        loss_mask = self.loss_mask
        self.noise_magnitude = n**2*3

        with torch.no_grad():
            # generation with past key values
            outputs, _, _ = self(
                new_input, 
                past_key_values=self.noise_past_key_values if self.noise_past_key_values else None, 
                for_argmax_noise_generation = True
            )
            self.noise_past_key_values = outputs.past_key_values
        
        seen = []
        # if is in generation mode, sample a token first
        if not is_prompt:
            input_ids = self.get_generated_token(
                logits_processor,
                logits_warper,
                outputs,
                loss_mask
            )
            # if starts to generate pad tokens, end generation.
            if input_ids.squeeze(dim=0) == 1:
                return None, None, False

        # if currently on the start token, get all noises from the logits of the start token
        if is_start_of_sentence:
            next_token_logits = outputs.logits[:, -1, :]
            seen_toks = [input_ids, torch.tensor(2)]
            if noise_scheme == "mix":
                noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
            else:
                results = self.vdb[int(input_ids)]
                choices = list( (10,20))
                choice = random.choice(choices)
                noise_tok = int(results[choice])
            for i in range(-1*n,-1):
                while noise_tok in seen_toks:
                    if noise_scheme == "mix":
                        noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                    else:   # synonym scheme
                        choice = random.choice(choices)
                        noise_tok = int(results[choice])
                tok_to_noise[i] = noise_tok
                seen_toks.append(noise_tok)
            tok_to_noise[-1] = input_ids
            for i in range(-3*n,-2*n):
                tok_to_noise[i] = last_prompt_tok2
            for i in range(-2*n,-1*n):
                tok_to_noise[i] = last_prompt_tok1
        else:
            seen_toks = [input_ids] # + [tok_to_noise[i] for i in range(-3*n,-1*n)]
            for i in range(-1*n,-1):
                v = toks_to_indices[i]
                next_token_logits = outputs.logits[:, v, :]
                if noise_scheme == "mix":
                    noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                else:
                    results = self.vdb[int(input_ids)]
                    choices = list( (10,20))
                    choice = random.choice(choices)
                    noise_tok = int(results[choice])
                while noise_tok in seen_toks:
                    if noise_scheme == "mix":
                        noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5) 
                    else:
                        choice = random.choice(choices)
                        noise_tok = int(results[choice])
                tok_to_noise[i] = noise_tok
                seen_toks.append(noise_tok)
            tok_to_noise[-1] = input_ids                      
        
        # using mixed noise ratio if the scheme is mix scheme
        if noise_scheme == "mix":
            for i in range(-1*n,-1):
                mix_ratio = random.uniform(0,1)
                if mix_ratio < threshold_get_from_true_token:
                    v = toks_to_indices[-1]
                    next_token_logits = outputs.logits[:, v, :]
                    while noise_tok in seen_toks:
                        noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                    seen_toks.append(noise_tok.squeeze(dim=0))
                    tok_to_noise[i] = noise_tok

        # key: -1 for current time stamp true token, -2 for current time stamp noise token, -3 for previous time stamp true token, -4 for previous time stamp noise token
        ft_keys = 0
        trigram_construct = {}
        for i in range(-1*n,0):
            for j in range(-3*n,-2*n):
                for k in range(-2*n,-1*n):
                    if i == -1 and j == -2*n-1 and k == -1*n-1:
                        trigram_construct["ttt"] = [tok_to_noise[j],tok_to_noise[k],tok_to_noise[i]]
                    elif abs(i-k) == n and abs(j-k) == n:
                        trigram_construct[f"fff{i}"] = [tok_to_noise[j],tok_to_noise[k],tok_to_noise[i]]
                    else:
                        if ft_keys >= 2:
                            continue
                        trigram_construct[f"fft{ft_keys}"] = [tok_to_noise[j],tok_to_noise[k],tok_to_noise[i]]
                        ft_keys += 1

        shuffled_noise = list(trigram_construct.items())
        random.seed(self.time_step * self.large_prime)  # setting random seed to be t*p, where p is a large prime number input by user
        random.shuffle(shuffled_noise)

        seen_toks.append(input_ids)
        old_real_tok_idx = toks_to_indices[-1]
        # token_index = n**3*3-2
        token_index = n**2*3-2
        for code, construct in shuffled_noise:
            construct = torch.tensor(construct)
            new_input = torch.cat((new_input.squeeze(dim=0),construct))
            if code == "ttt":
                loss_mask.extend([-1,-1,0])
                toks_to_indices[-1] = -1 * token_index
            elif "fff" in code:
                noise_seq_id = int(code[-2:])
                toks_to_indices[noise_seq_id] = -1 * token_index
                loss_mask.extend([-1,-1,-1])
            else:
                loss_mask.extend([-1,-1,-1])
            token_index -= 3
        
        # change from this time stamp to last time stamp
        self.toks_to_noise_last_time_stamp = tok_to_noise.copy()
        for i in range(-2*n,0):
            self.toks_to_noise_last_time_stamp[i-n] = tok_to_noise[i]
        self.toks_to_indices_last_time_stamp = toks_to_indices.copy()
        self.sequence_last_time_stamp = new_input[-1*n**2*3:]
        # print(self.sequence_last_time_stamp)
        self.input_ids = torch.tensor(new_input)
        self.loss_mask = loss_mask
        self.noise_magnitude =  n**2*3
        # seen_toks = [self.toks_to_noise_last_time_stamp[i] for i in range(-1*2*n,-1*n)]
        
        if is_prompt:
            self.true_input_ids = torch.cat((self.true_input_ids, input_ids.unsqueeze(dim=0)), dim=0)
            self.total_noised_input_ids = new_input.clone()
        else:
            self.total_noised_input_ids = torch.cat((self.total_noised_input_ids, self.sequence_last_time_stamp), dim=0)
        
        return self.input_ids, self.loss_mask, outputs
    
    def generate_noised_input_nway2order_argmax(
        self, 
        input_ids, 
        noise_order = 3, 
        noise_scheme = "mix",
        is_prompt=True, 
        is_start_of_sentence=False,
        threshold_get_from_true_token = 0.7,
        use_noise_threshold = -1,
        logits_processor=None,
        logits_warper=None,
    ):

        import random
        new_input = []
        # loss_masks = []
        # key: -1 for true token, -2, -3... for noises
        # value: -1, -2, -3... for position in logits
        toks_to_indices = {k:0 for k in range(-1*noise_order, 0)}
        # key: -1 for true token, -2, -3... for noises
        # value: token number
        tok_to_noise = {k:-1 for k in range(-1*noise_order, 0)}

        if not is_prompt:
            if is_start_of_sentence:
                # new_input = self.input_ids
                new_input = self.sequence_last_time_stamp
                toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
                # tok_to_noise = self.toks_to_noise_last_time_stamp.copy()
                # self.total_noised_input_ids = new_input.clone()
            else:
                new_input = self.sequence_last_time_stamp
                # new_input = self.input_ids
                toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
                # tok_to_noise = self.toks_to_noise_last_time_stamp.copy()
        else:
            if is_start_of_sentence:

                true_tok = random.choice(list( (-1*self.noise_magnitude+1,0)))
                self.loss_mask.extend([-1]*self.noise_magnitude)
                self.loss_mask[true_tok] = 0
                new_input.extend([2]*self.noise_magnitude)
                toks_to_indices[-1] = true_tok
                self.total_noised_input_ids = torch.tensor([2]*self.noise_magnitude)
            else:
                # new_input = self.input_ids
                new_input = self.sequence_last_time_stamp
                toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
                # tok_to_noise = self.toks_to_noise_last_time_stamp.copy()

        new_input = torch.tensor(new_input).unsqueeze(dim=0).to(torch.int)
        self.input_ids = new_input
        if is_prompt and is_start_of_sentence:
            self.attention_mask = torch.ones(self.input_ids.shape)
        else:
            self.attention_mask = torch.cat((self.attention_mask, torch.ones(noise_order).unsqueeze(dim=0)),dim=1)
        
        loss_mask = self.loss_mask
        self.noise_magnitude = noise_order

        with torch.no_grad():
            # generation with past key values
            outputs, _, _ = self(
                new_input, 
                past_key_values=self.noise_past_key_values, 
                for_argmax_noise_generation = True
            )
            self.noise_past_key_values = outputs.past_key_values

        # if is in generation mode, sample a token first
        if not is_prompt:
            input_ids = self.get_generated_token(
                logits_processor,
                logits_warper,
                outputs,
                loss_mask
            )

            # if starts to generate pad tokens, end generation.
            if input_ids.squeeze(dim=0) == 1:
                return None, None, False
            
        # if currently on the start token, get all noises from the logits of the start token
        if is_start_of_sentence and is_prompt:
            next_token_logits = outputs.logits[:, toks_to_indices[-1], :]
            noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
            noise_tok_cnt = 2
            seen_toks = [input_ids]
            if noise_scheme == "mix":
                noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
            else:
                results = self.vdb[int(input_ids)]
                choices = list( (10,20))
                choice = random.choice(choices)
                noise_tok = torch.tensor(int(results[choice])).unsqueeze(dim=0)
            while noise_tok_cnt <= noise_order:
                while noise_tok in seen_toks:
                    if noise_scheme == "mix":
                        noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                    else:   # synonym scheme
                        choice = random.choice(choices)
                        noise_tok = torch.tensor(int(results[choice])).unsqueeze(dim=0)
                tok_to_noise[-1*noise_tok_cnt] = noise_tok
                noise_tok_cnt += 1
                seen_toks.append(noise_tok.squeeze(dim=0))
        else:
            seen_toks = [input_ids]
            for k,v in toks_to_indices.items():
                if k == -1:
                    continue
                next_token_logits = outputs.logits[:, v, :]
                if noise_scheme == "mix":
                    noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                else:
                    results = self.vdb[int(input_ids)]
                    choices = list( (10,20))
                    choice = random.choice(choices)
                    noise_tok = torch.tensor(int(results[choice])).unsqueeze(dim=0)
                while noise_tok in seen_toks:
                    if noise_scheme == "mix":
                        noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5) 
                    else:
                        choice = random.choice(choices)
                        noise_tok = torch.tensor(int(results[choice])).unsqueeze(dim=0)                 
                tok_to_noise[k] = noise_tok
                seen_toks.append(noise_tok.squeeze(dim=0))

        # if not is_prompt:
        #     new_input = torch.tensor([])

        shuffled_noises = list(tok_to_noise.items())
        random.seed(self.time_step * self.large_prime)
        random.shuffle(shuffled_noises)

        # let first noise token always come from bos
        seen_toks.append(input_ids)
        old_real_tok_idx = toks_to_indices[-1]
        token_index = noise_order
        use_noise = random.uniform(0,1)
 
        for idx,token in shuffled_noises:
            # if is the true token
            if idx == -1:
                loss_mask.append(0)
                if is_prompt:
                    new_input = torch.cat((new_input.squeeze(dim=0),input_ids.unsqueeze(dim=0)))
                else:
                    new_input = torch.cat((new_input.squeeze(dim=0),input_ids))
                    self.total_generated_input_ids = torch.cat((self.total_generated_input_ids, input_ids))
                    self.total_generated_loss_mask = torch.cat((self.total_generated_loss_mask, torch.tensor(0).unsqueeze(dim=0)))
            else:
                # if it is a noise token
                # determine if we are going to sample from true token distribution
                if is_prompt == True and is_start_of_sentence == True:
                    use_noise_from_true_token = float("inf")
                else:
                    use_noise_from_true_token = random.uniform(0,1)
                    
                next_token_logits = outputs.logits[:, old_real_tok_idx, :]
                noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)

                if use_noise < use_noise_threshold:
                    loss_mask.append(-1)
                    if is_prompt:
                        new_input = torch.cat((new_input.squeeze(dim=0),input_ids.unsqueeze(dim=0)))
                    else:
                        new_input = torch.cat((new_input.squeeze(dim=0),input_ids))
                        self.total_generated_input_ids = torch.cat((self.total_generated_input_ids, input_ids))
                        self.total_generated_loss_mask = torch.cat((self.total_generated_loss_mask, torch.tensor(-1).unsqueeze(dim=0)))
                else:
                    if use_noise_from_true_token < threshold_get_from_true_token:
                        while noise_tok in seen_toks:
                            noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                        seen_toks.append(noise_tok.squeeze(dim=0))
                        loss_mask.append(-1)
                        if is_prompt:
                            new_input = torch.cat((new_input.squeeze(dim=0),noise_tok))
                        else:
                            new_input = torch.cat((new_input.squeeze(dim=0),noise_tok))
                            self.total_generated_input_ids = torch.cat((self.total_generated_input_ids, noise_tok))
                            self.total_generated_loss_mask = torch.cat((self.total_generated_loss_mask, torch.tensor(-1).unsqueeze(dim=0)))
                    else:
                        loss_mask.append(-1)
                        if is_prompt:
                            new_input = torch.cat((new_input.squeeze(dim=0),token))
                        else:
                            new_input = torch.cat((new_input.squeeze(dim=0), token))
                            self.total_generated_input_ids = torch.cat((self.total_generated_input_ids, token))
                            self.total_generated_loss_mask = torch.cat((self.total_generated_loss_mask, torch.tensor(-1).unsqueeze(dim=0)))
            toks_to_indices[idx] = -1 * token_index
            token_index -= 1

        # keep track of the lattice for the last token and their index mappings for generation
        self.sequence_last_time_stamp = new_input[-1*noise_order:]
        self.toks_to_indices_last_time_stamp = toks_to_indices.copy()
        
        self.input_ids = torch.tensor(new_input)
        self.loss_mask = loss_mask
        self.noise_magnitude = noise_order

        if is_prompt:
            self.true_input_ids = torch.cat((self.true_input_ids, input_ids.unsqueeze(dim=0)), dim=0)
            self.total_noised_input_ids = torch.cat((self.total_noised_input_ids, self.sequence_last_time_stamp), dim=0)        
        else:
            self.total_noised_input_ids = torch.cat((self.total_noised_input_ids, self.sequence_last_time_stamp), dim=0)
        
        return self.input_ids, self.loss_mask, outputs

    def generate_noised_input_unigram_no_noise_for_prompt(
        self, 
        input_ids, 
        noise_order = 3, 
        noise_scheme = "mix",
        is_prompt=True, 
        is_start_of_sentence=False,
        threshold_get_from_true_token = 0,
        use_noise_threshold = -1,
        logits_processor=None,
        logits_warper=None,
    ):
        import time
        import random
        start = time.time()
        new_input = []
        # loss_masks = []
        # key: -1 for true token, -2, -3... for noises
        # value: -1, -2, -3... for position in logits
        toks_to_indices = {k:0 for k in range(-1*noise_order, 0)}
        # key: -1 for true token, -2, -3... for noises
        # value: token number
        tok_to_noise = {k:-1 for k in range(-1*noise_order, 0)}

        if not is_prompt:
            if is_start_of_sentence:
                self.true_input_ids = torch.cat((self.true_input_ids, self.input_ids.squeeze(dim=0)), dim=0)
                new_input = self.input_ids.squeeze(dim=0)
                self.loss_mask.extend([0]*self.input_ids.shape[1])
                # toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
                self.total_noised_input_ids = new_input.clone()
            else:
                new_input = self.sequence_last_time_stamp # self.input_ids
                toks_to_indices = self.toks_to_indices_last_time_stamp.copy()

        new_input = torch.tensor(new_input).unsqueeze(dim=0).to(torch.int)
        self.input_ids = new_input
        if is_prompt or (not is_prompt and is_start_of_sentence):
            self.attention_mask = torch.ones(self.input_ids.shape)
        else:
            self.attention_mask = torch.cat((self.attention_mask, torch.ones(noise_order).unsqueeze(dim=0)),dim=1)
        
        loss_mask = self.loss_mask
        self.noise_magnitude = noise_order
        
        with torch.no_grad():
            # generation with past key values
            outputs, _, _ = self(
                new_input, 
                past_key_values=self.noise_past_key_values, 
                for_argmax_noise_generation = True
            )
            self.noise_past_key_values = outputs.past_key_values

        # if is in generation mode, sample a token first
        if not is_prompt:
            input_ids = self.get_generated_token(
                logits_processor,
                logits_warper,
                outputs,
                loss_mask
            )

            # if starts to generate pad tokens, end generation.
            if input_ids.squeeze(dim=0) == 1:
                return None, None, False
        
        # if currently on the start token, get all noises from the logits of the start token
        if is_start_of_sentence:
            seen_toks = [input_ids]
            next_token_logits = outputs.logits[:, -1, :]
            noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
            noise_tok_cnt = 2
            seen_toks = [input_ids, torch.tensor(2)]
            if noise_scheme == "mix":
                noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
            else:
                results = self.vdb[int(input_ids)]
                choices = list( (10,20))
                choice = random.choice(choices)
                noise_tok = torch.tensor(int(results[choice])).unsqueeze(dim=0)
            while noise_tok_cnt <= noise_order:
                while noise_tok in seen_toks:
                    if noise_scheme == "mix":
                        noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                    else:   # synonym scheme
                        choice = random.choice(choices)
                        noise_tok = torch.tensor(int(results[choice])).unsqueeze(dim=0)
                tok_to_noise[-1*noise_tok_cnt] = noise_tok
                noise_tok_cnt += 1
                seen_toks.append(noise_tok)
        else:
            seen_toks = [input_ids]
            for k,v in toks_to_indices.items():
                if k == -1:
                    continue
                next_token_logits = outputs.logits[:, v, :]
                if noise_scheme == "mix":
                    noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                else:
                    results = self.vdb[int(input_ids)]
                    choices = list( (10,20))
                    choice = random.choice(choices)
                    noise_tok = torch.tensor(int(results[choice])).unsqueeze(dim=0)
                while noise_tok in seen_toks:
                    if noise_scheme == "mix":
                        noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5) 
                    else:
                        choice = random.choice(choices)
                        noise_tok = torch.tensor(int(results[choice])).unsqueeze(dim=0)
                tok_to_noise[k] = noise_tok
                seen_toks.append(noise_tok)
 
        shuffled_noises = list(tok_to_noise.items())
        random.seed(self.time_step * self.large_prime)
        random.shuffle(shuffled_noises)

        # let first noise token always come from bos
        seen_toks.append(input_ids)
        old_real_tok_idx = toks_to_indices[-1]
        token_index = noise_order
        use_noise = random.uniform(0,1)
 
        for idx,token in shuffled_noises:
            # if is the true token
            if idx == -1:
                loss_mask.append(0)
                if is_prompt:
                    new_input = torch.cat((new_input.squeeze(dim=0),input_ids.unsqueeze(dim=0)))
                else:
                    new_input = torch.cat((new_input.squeeze(dim=0),input_ids))
                    self.total_generated_input_ids = torch.cat((self.total_generated_input_ids, input_ids))
                    self.total_generated_loss_mask = torch.cat((self.total_generated_loss_mask, torch.tensor(0).unsqueeze(dim=0)))
            else:
                # if it is a noise token
                # determine if we are going to sample from true token distribution
                if is_prompt == True and is_start_of_sentence == True:
                    use_noise_from_true_token = float("inf")
                else:
                    use_noise_from_true_token = random.uniform(0,1)
                    
                next_token_logits = outputs.logits[:, old_real_tok_idx, :]
                noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)

                if use_noise < use_noise_threshold:
                    # print("here")
                    loss_mask.append(-1)
                    if is_prompt:
                        new_input = torch.cat((new_input.squeeze(dim=0),input_ids.unsqueeze(dim=0)))
                    else:
                        new_input = torch.cat((new_input.squeeze(dim=0),input_ids))
                        self.total_generated_input_ids = torch.cat((self.total_generated_input_ids, input_ids))
                        self.total_generated_loss_mask = torch.cat((self.total_generated_loss_mask, torch.tensor(-1).unsqueeze(dim=0)))
                else:
                    if use_noise_from_true_token < threshold_get_from_true_token:
                        while noise_tok in seen_toks:
                            noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                        seen_toks.append(noise_tok.squeeze(dim=0))
                        loss_mask.append(-1)
                        if is_prompt:
                            new_input = torch.cat((new_input.squeeze(dim=0),noise_tok))
                        else:
                            new_input = torch.cat((new_input.squeeze(dim=0),noise_tok))
                            self.total_generated_input_ids = torch.cat((self.total_generated_input_ids, noise_tok))
                            self.total_generated_loss_mask = torch.cat((self.total_generated_loss_mask, torch.tensor(-1).unsqueeze(dim=0)))
                    else:
                        loss_mask.append(-1)
                        if is_prompt:
                            new_input = torch.cat((new_input.squeeze(dim=0),token))
                            # new_input = torch.cat((new_input.squeeze(dim=0),self.toks_to_noise_tok_mapping[idx]))
                        else:
                            new_input = torch.cat((new_input.squeeze(dim=0), token))
                            self.total_generated_input_ids = torch.cat((self.total_generated_input_ids, token))
                            # new_input = torch.cat((new_input.squeeze(dim=0), self.toks_to_noise_tok_mapping[idx]))
                            # self.total_generated_input_ids = torch.cat((self.total_generated_input_ids, self.toks_to_noise_tok_mapping[idx]))
                            
                            self.total_generated_loss_mask = torch.cat((self.total_generated_loss_mask, torch.tensor(-1).unsqueeze(dim=0)))
            toks_to_indices[idx] = -1 * token_index
            token_index -= 1

        # keep track of the lattice for the last token and their index mappings for generation
        self.sequence_last_time_stamp = new_input[-1*noise_order:]
        self.toks_to_indices_last_time_stamp = toks_to_indices.copy()
        
        self.input_ids = torch.tensor(new_input)
        self.loss_mask = loss_mask
        self.noise_magnitude = noise_order

        if is_prompt:
            self.true_input_ids = torch.cat((self.true_input_ids, input_ids), dim=0)
            self.total_noised_input_ids = new_input.clone()
        else:
            self.total_noised_input_ids = torch.cat((self.total_noised_input_ids, self.sequence_last_time_stamp), dim=0)
        end = time.time()
        self.times.append(end - start)
        return self.input_ids, self.loss_mask, outputs

    def generate_noised_input_baseline(
        self, 
        input_ids, 
        noise_order = 1, 
        noise_scheme = "synonym",
        is_prompt=True, 
        is_start_of_sentence=False,
        threshold_get_from_true_token = 0,
        use_noise_threshold = -1,
        logits_processor=None,
        logits_warper=None,
    ):
        import random
        noise_order = 1
        new_input = []
        # loss_masks = []
        # key: -1 for true token, -2, -3... for noises
        # value: -1, -2, -3... for position in logits
        toks_to_indices = {k:0 for k in range(-1*noise_order, 0)}
        # key: -1 for true token, -2, -3... for noises
        # value: token number
        tok_to_noise = {k:-1 for k in range(-1*noise_order, 0)}

        if not is_prompt:
            if is_start_of_sentence:
                self.true_input_ids = torch.cat((self.true_input_ids, self.input_ids.squeeze(dim=0)), dim=0)
                new_input = self.input_ids.squeeze(dim=0)
                self.loss_mask.extend([0]*self.input_ids.shape[1])
                self.total_noised_input_ids = new_input.clone()
            else:
                new_input = self.sequence_last_time_stamp # self.input_ids
                toks_to_indices = self.toks_to_indices_last_time_stamp.copy()

        new_input = torch.tensor(new_input).unsqueeze(dim=0).to(torch.int)
        self.input_ids = new_input
        if is_prompt or (not is_prompt and is_start_of_sentence):
            self.attention_mask = torch.ones(self.input_ids.shape)
        else:
            self.attention_mask = torch.cat((self.attention_mask, torch.ones(noise_order).unsqueeze(dim=0)),dim=1)
        
        loss_mask = self.loss_mask
        self.noise_magnitude = noise_order
        
        with torch.no_grad():
            # generation with past key values
            outputs, _, _ = self(
                new_input, 
                past_key_values=self.noise_past_key_values, 
                for_argmax_noise_generation = True
            )
            self.noise_past_key_values = outputs.past_key_values

        # if is in generation mode, sample a token first
        if not is_prompt:
            input_ids = self.get_generated_token(
                logits_processor,
                logits_warper,
                outputs,
                loss_mask
            )

            # if starts to generate pad tokens, end generation.
            if input_ids.squeeze(dim=0) == 1:
                return None, None, False
                
        results = self.vdb[int(input_ids)]
        choices = list( (1,10))
        choice = random.choice(choices)
        noise_tok = torch.tensor(int(results[choice])).unsqueeze(dim=0)
        
        import random
        if random.random() > 0.5:
        # keep track of the lattice for the last token and their index mappings for generation
            self.sequence_last_time_stamp = noise_tok
        else:
            self.sequence_last_time_stamp = input_ids

        self.toks_to_indices_last_time_stamp = toks_to_indices.copy()
        
        self.input_ids = torch.tensor(new_input)
        self.loss_mask = loss_mask
        self.noise_magnitude = noise_order

        if is_prompt:
            self.true_input_ids = torch.cat((self.true_input_ids, input_ids), dim=0)
            self.total_noised_input_ids = new_input.clone()
        else:
            self.total_noised_input_ids = torch.cat((self.total_noised_input_ids, self.sequence_last_time_stamp), dim=0)
        
        return self.input_ids, self.loss_mask, outputs

    def generate_noised_input_nway2order_no_noise_for_prompt_modified(
        self, 
        input_ids, 
        noise_order = 3, 
        is_prompt=True, 
        is_start_of_sentence=False,
        threshold_get_from_true_token = 0,
        use_noise_threshold = -1,
        logits_processor=None,
        logits_warper=None,
    ):
        import random
        new_input = []
        # key: -1 for true token, -2, -3... for noises
        # value: -1, -2, -3... for position in logits
        toks_to_indices_all_noise = {k:0 for k in range(-1*noise_order, 0)}
        # key: -1 for true token, -2, -3... for noises
        # value: token number
        tok_to_noise_all_noise = {k:-1 for k in range(-1*noise_order, 0)}
        
        if not is_prompt:
            if is_start_of_sentence:
                self.true_input_ids = torch.cat((self.true_input_ids, self.input_ids.squeeze(dim=0)), dim=0)
                new_input = self.input_ids.squeeze(dim=0)
                new_input_all_noise = self.input_ids.squeeze(dim=0)
                self.loss_mask.extend([0]*self.input_ids.shape[1])
                self.total_noised_input_ids = new_input.clone()
            else:
                new_input = self.input_ids
                new_input_all_noise = self.input_ids_all_noise
                toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
                toks_to_indices_all_noise = self.toks_to_indices_last_time_stamp_all_noise.copy()

        new_input = torch.tensor(new_input).unsqueeze(dim=0).to(torch.int)
        new_input_all_noise = torch.tensor(new_input_all_noise).unsqueeze(dim=0).to(torch.int)
        self.input_ids = new_input
        if is_prompt or (not is_prompt and is_start_of_sentence):
            self.attention_mask = torch.ones(self.input_ids.shape)
        else:
            self.attention_mask = torch.cat((self.attention_mask, torch.ones(noise_order).unsqueeze(dim=0)),dim=1)
        
        loss_mask = self.loss_mask
        loss_mask_all_noise = self.loss_mask_all_noise
        self.noise_magnitude = noise_order
        
        with torch.no_grad():
            # generation with past key values
            if not is_prompt and not is_start_of_sentence: 
        
                self.input_ids = new_input_all_noise
                outputs_all_noise, _, _ = self(
                    new_input, 
                    for_argmax_noise_generation = True
                )
            else:
                outputs, _, _ = self(
                    new_input, 
                    for_argmax_noise_generation = True
                )

                self.input_ids = new_input_all_noise
                outputs_all_noise, _, _ = self(
                    new_input, 
                    for_argmax_noise_generation = True
                )

            self.argmax_past_key_values = outputs.past_key_values

        # if is in generation mode, sample a token first
        if not is_prompt:
            input_ids = self.get_generated_token(
                logits_processor,
                logits_warper,
                outputs,
                loss_mask
            )

            input_ids_all_noise = self.get_generated_token(
                logits_processor,
                logits_warper,
                outputs_all_noise,
                loss_mask_all_noise,
            )
            # if starts to generate pad tokens, end generation.
            if input_ids.squeeze(dim=0) == 1:
                return None, None, False
        
        # if currently on the start token, get all noises from the logits of the start token
        if is_start_of_sentence:
            seen_toks = [input_ids]
            next_token_logits = outputs.logits[:, -1, :]
            noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
            noise_tok_cnt = 2
            seen_toks = [input_ids, torch.tensor(2)]
            while noise_tok_cnt <= noise_order:
                while noise_tok in seen_toks:
                    noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                tok_to_noise[-1*noise_tok_cnt] = noise_tok
                noise_tok_cnt += 1
                seen_toks.append(noise_tok.squeeze(dim=0))
            tok_to_noise_all_noise = tok_to_noise.copy()
        else:
            seen_toks = [input_ids]
            for k,v in toks_to_indices.items():
                if k == -1:
                    continue
                next_token_logits = outputs.logits[:, v, :]
                noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                while noise_tok in seen_toks:
                    noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)                        
                tok_to_noise[k] = noise_tok
                seen_toks.append(noise_tok.squeeze(dim=0))

            seen_toks_all_noise = [input_ids_all_noise]
            for k,v in toks_to_indices_all_noise.items():
                if k == -1:
                    continue
                next_token_logits = outputs.logits[:, v, :]
                noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                while noise_tok in seen_toks:
                    noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)                        
                tok_to_noise_all_noise[k] = noise_tok
                seen_toks.append(noise_tok.squeeze(dim=0))
 
        shuffled_noises = list(tok_to_noise.items())
        shuffled_noises_all_noise = list(tok_to_noise_all_noise.items())
        random.shuffle(shuffled_noises)
        random.shuffle(shuffled_noises_all_noise)

        # let first noise token always come from bos
        seen_toks.append(input_ids)
        seen_toks_all_noise.append(input_ids_all_noise)
        old_real_tok_idx = toks_to_indices[-1]
        token_index = noise_order
        use_noise = random.uniform(0,1)
 
        for idx,token in shuffled_noises:
            # if is the true token
            if idx == -1:
                loss_mask.append(0)
                if is_prompt:
                    new_input = torch.cat((new_input.squeeze(dim=0),input_ids.unsqueeze(dim=0)))
                else:
                    new_input = torch.cat((new_input.squeeze(dim=0),input_ids))
                    self.total_generated_input_ids = torch.cat((self.total_generated_input_ids, input_ids))
                    self.total_generated_loss_mask = torch.cat((self.total_generated_loss_mask, torch.tensor(0).unsqueeze(dim=0)))
            else:
                # if it is a noise token
                # determine if we are going to sample from true token distribution
                if is_prompt == True and is_start_of_sentence == True:
                    use_noise_from_true_token = float("inf")
                else:
                    use_noise_from_true_token = random.uniform(0,1)
                    
                next_token_logits = outputs.logits[:, old_real_tok_idx, :]
                noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)

                if use_noise < use_noise_threshold:
                    # print("here")
                    loss_mask.append(-1)
                    if is_prompt:
                        new_input = torch.cat((new_input.squeeze(dim=0),input_ids.unsqueeze(dim=0)))
                    else:
                        new_input = torch.cat((new_input.squeeze(dim=0),input_ids))
                        self.total_generated_input_ids = torch.cat((self.total_generated_input_ids, input_ids))
                        self.total_generated_loss_mask = torch.cat((self.total_generated_loss_mask, torch.tensor(-1).unsqueeze(dim=0)))
                else:
                    if use_noise_from_true_token < threshold_get_from_true_token:
                        while noise_tok in seen_toks:
                            noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                        seen_toks.append(noise_tok.squeeze(dim=0))
                        loss_mask.append(-1)
                        if is_prompt:
                            new_input = torch.cat((new_input.squeeze(dim=0),noise_tok))
                        else:
                            new_input = torch.cat((new_input.squeeze(dim=0),noise_tok))
                            self.total_generated_input_ids = torch.cat((self.total_generated_input_ids, noise_tok))
                            self.total_generated_loss_mask = torch.cat((self.total_generated_loss_mask, torch.tensor(-1).unsqueeze(dim=0)))
                    else:
                        loss_mask.append(-1)
                        if is_prompt:
                            new_input = torch.cat((new_input.squeeze(dim=0),token))
                        else:
                            new_input = torch.cat((new_input.squeeze(dim=0), token))
                            self.total_generated_input_ids = torch.cat((self.total_generated_input_ids, token))
                            self.total_generated_loss_mask = torch.cat((self.total_generated_loss_mask, torch.tensor(-1).unsqueeze(dim=0)))
            toks_to_indices[idx] = -1 * token_index
            token_index -= 1

        token_index = noise_order
        for idx,token in shuffled_noises_all_noise:
            # if is the true token
            if idx == -1:
                loss_mask.append(0)
                if is_prompt:
                    new_input = torch.cat((new_input.squeeze(dim=0),input_ids_all_noise.unsqueeze(dim=0)))
                else:
                    new_input = torch.cat((new_input.squeeze(dim=0),input_ids_all_noise))
                    self.total_generated_loss_mask = torch.cat((self.total_generated_loss_mask, torch.tensor(0).unsqueeze(dim=0)))
            else:
                # if it is a noise token
                # determine if we are going to sample from true token distribution
                if is_prompt == True and is_start_of_sentence == True:
                    use_noise_from_true_token = float("inf")
                else:
                    use_noise_from_true_token = random.uniform(0,1)
                    
                next_token_logits = outputs.logits[:, old_real_tok_idx, :]
                noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)

                if use_noise_from_true_token < threshold_get_from_true_token:
                    while noise_tok in seen_toks_all_noise:
                        noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                    seen_toks_all_noise.append(noise_tok.squeeze(dim=0))
                    loss_mask.append(-1)
                    if is_prompt:
                        new_input_all_noise = torch.cat((new_input.squeeze(dim=0),noise_tok))
                    else:
                        new_input_all_noise = torch.cat((new_input.squeeze(dim=0),noise_tok))
                        self.total_generated_loss_mask = torch.cat((self.total_generated_loss_mask, torch.tensor(-1).unsqueeze(dim=0)))
                else:
                    loss_mask.append(-1)
                    if is_prompt:
                        new_input = torch.cat((new_input.squeeze(dim=0),token))
                    else:
                        new_input = torch.cat((new_input.squeeze(dim=0), token))
                        self.total_generated_loss_mask = torch.cat((self.total_generated_loss_mask, torch.tensor(-1).unsqueeze(dim=0)))
            toks_to_indices_all_noise[idx] = -1 * token_index
            token_index -= 1

        # keep track of the lattice for the last token and their index mappings for generation
        self.sequence_last_time_stamp = new_input[-1*noise_order:]
        self.sequence_last_time_stamp_all_noise = new_input_all_noise[-1*noise_order:]

        self.toks_to_indices_last_time_stamp = toks_to_indices.copy()
        self.toks_to_indices_last_time_stamp_all_noise = toks_to_indices_all_noise.copy()

        self.input_ids = torch.tensor(new_input)
        self.loss_mask = loss_mask
        self.noise_magnitude = noise_order

        if is_prompt:
            self.true_input_ids = torch.cat((self.true_input_ids, input_ids), dim=0)
            print("#####################", self.true_input_ids.shape)
            self.total_noised_input_ids = new_input.clone()
        else:
            self.total_noised_input_ids = torch.cat((self.total_noised_input_ids, self.sequence_last_time_stamp), dim=0)
        
        return self.input_ids, self.loss_mask, outputs

    def custom_sampling_topk(
            self, 
            input_ids, 
            next_token_logits,
            topk
        ):

        self.config.pad_token_id = self.config.eos_token_id
        self.generation_config.pad_token_id = self.config.eos_token_id

        logits_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(150, eos_token_id=self.generation_config.eos_token_id),
            ]
        )

        # instantiate logits processors
        logits_warper = LogitsProcessorList(
            [
                TopKLogitsWarper(topk),
                TemperatureLogitsWarper(0.7),
            ]
        )

        next_token_scores = logits_processor(input_ids, next_token_logits)
        # next_tokens = torch.argmax(next_token_scores, dim=-1)
        next_token_scores = logits_warper(input_ids, next_token_scores)
        
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        return next_tokens

    def generate_noised_input_2way1order_argmax(self, input_ids, past_key_values = None, use_start_token=True):
        import random
        new_inputs = []
        loss_masks = []
        attention_masks = []
        b,s = input_ids.shape
        for batch in range(b):
            new_input = []
            new_input_total = []
            loss_mask = []
            end_range = input_ids[batch].shape[-1]
            self.noise_past_key_values = None
            print(1, flush=True)
            for i in range(0, end_range-1):
                if i == 0:
                    new_input = [2]*2
                    new_input_total = new_input[:]
                    self.argmax_prev_idx = random.choice([-1,-2])
                    self.input_ids = torch.tensor(new_input).unsqueeze(dim=0)
                    self.attention_mask = torch.ones(self.input_ids.shape)
                    if self.argmax_prev_idx == -2:
                        loss_mask = [-1,0]
                    else:
                        loss_mask = [0,-1]
                else:
                    self.input_ids = torch.tensor(new_input).unsqueeze(dim=0)
                    self.attention_mask = torch.cat((self.attention_mask,torch.ones(self.input_ids.shape)),dim=1)
                self.loss_mask = loss_mask
                self.noise_magnitude = 2
                
                with torch.no_grad():
                    # if not use_start_token: # generation with past key values
                    output, _, _ = self(torch.tensor(new_input).unsqueeze(dim=0), past_key_values=self.noise_past_key_values, for_argmax_noise_generation = True)
                    # else:
                        # output, _, _ = self(torch.tensor(new_input).unsqueeze(dim=0), for_argmax_noise_generation = True)
                    self.noise_past_key_values = output.past_key_values
                    # make sure to use the first greedy token that is not the same as the true token
                    greedy_next_token = torch.topk(output.logits[:, self.argmax_prev_idx, :], k = 3, dim=1)
                    for noise_tok in greedy_next_token.indices.squeeze(dim=0):
                        if noise_tok != input_ids[batch, i+1]:
                            break
                    self.temp_token = noise_tok
  
                random_position = random.choice([0,1])
                if random_position == 0:
                    new_input_total.append(input_ids[batch, i+1])
                    new_input_total.append(noise_tok)
                    loss_mask.extend([0,-1])     
                    self.argmax_prev_idx = -1       
                else:
                    new_input_total.append(noise_tok)
                    new_input_total.append(input_ids[batch, i+1])
                    loss_mask.extend([-1,0])
                    self.argmax_prev_idx = -2
                new_input = new_input_total[-2:]
            new_inputs.append(new_input_total)
            loss_masks.append(loss_mask)
        self.input_ids = torch.tensor(new_inputs) 
        self.loss_mask = torch.tensor(loss_masks)
        self.attention_mask = torch.ones(self.input_ids.shape)
        self.noise_magnitude = 2
        print(self.input_ids.shape)
        print(self.loss_mask.shape)
        print(self.attention_mask.shape)
        # print("argmax noise input shape: ", self.input_ids.shape)
        return self.input_ids, self.loss_mask
    
    
    def generate_noised_input_bigram_argmax_train(self, input_ids, past_key_values = None, n=2, use_start_token=True):
        import random
        self.noise_magnitude = 2**n * 2
        new_inputs = []
        loss_masks = []
        attention_masks = []
        b,s = input_ids.shape
        for batch in range(b):
            print(1)
            new_input = []
            new_input_total = []
            loss_mask = []
            end_range = input_ids[batch].shape[-1]
            self.noise_past_key_values = None
            toks_to_indices = {k:0 for k in range(-1*n, 0)}
            tok_to_noise = {k:-1 for k in range(-2*n, 0)}
            for i in range(0, end_range-1):
                if i == 0:
                    new_input = [2]*self.noise_magnitude
                    new_input_total = new_input[:]
                    noise_idx, true_idx = random.sample(list( (2**n)),2)
                    print(noise_idx, " ", true_idx)
                    self.input_ids = torch.tensor(new_input).unsqueeze(dim=0)
                    self.attention_mask = torch.ones(self.input_ids.shape)
                    for lattice in range(2**n-1,-1,-1):
                        if lattice == noise_idx:
                            toks_to_indices[-2] = -2*lattice+1
                            loss_mask.extend([-1,-1])
                        elif lattice == true_idx:
                            toks_to_indices[-1] = -2*lattice+1
                            loss_mask.extend([-1,0])
                        else:
                            loss_mask.extend([-1,-1])
                    for index in range(-2**n,0):
                        tok_to_noise[index] = input_ids[batch, i]
                else:
                    self.input_ids = torch.tensor(new_input).unsqueeze(dim=0)
                    self.attention_mask = torch.cat((self.attention_mask,torch.ones(self.input_ids.shape)),dim=1)
                self.loss_mask = loss_mask
                seen_toks = [input_ids[batch, i+1]]
                # find noise token for current step
                with torch.no_grad():
                    # if not use_start_token: # generation with past key values
                    output, _, _ = self(torch.tensor(new_input).unsqueeze(dim=0), past_key_values=self.noise_past_key_values, for_argmax_noise_generation = True)
                    self.noise_past_key_values = output.past_key_values
                next_token_logits = output.logits[:, toks_to_indices[-2], :]
                noise_tok = self.custom_sampling_topk(self.input_ids, next_token_logits, topk=5)
                for index in range(-1*n,-1):
                    while noise_tok in seen_toks:
                        noise_tok = self.custom_sampling_topk(self.input_ids, next_token_logits, topk=5)
                    tok_to_noise[index] = noise_tok
                    seen_toks.append(noise_tok)
                tok_to_noise[-1] = input_ids[batch, i+1]

                # shuffle noise
                ft_keys = 0
                bigram_construct = {}
                for ii in range(-1*n,0):
                    for j in range(-1*2*n,-1*n):
                        if ii == -1 and j == -1*n-1:
                            bigram_construct["tt"] = [tok_to_noise[j],tok_to_noise[ii]]
                        elif abs(ii - j) == n:
                            bigram_construct[f"ff{ii}"] = [tok_to_noise[j],tok_to_noise[ii]]
                        else:
                            bigram_construct[f"ft{ft_keys}"] = [tok_to_noise[j],tok_to_noise[ii]]
                            ft_keys += 1

                shuffled_noise = list(bigram_construct.items())
                # random.seed(self.time_step * self.large_prime)
                random.shuffle(shuffled_noise)
                token_index = n**2*2-1
                for code, construct in shuffled_noise:
                    new_input_total.extend(construct)
                    if code == "tt":
                        loss_mask.extend([-1,0])
                        toks_to_indices[-1] = -1 * token_index
                    elif "ff" in code:
                        noise_seq_id = int(code[-2:])
                        toks_to_indices[noise_seq_id] = -1 * token_index
                        loss_mask.extend([-1,-1])
                    else:
                        loss_mask.extend([-1,-1])
                    token_index -= 2
                new_input = new_input_total[-1*self.noise_magnitude:]
                for jj in range(-1*n,0):
                    tok_to_noise[jj - n] = tok_to_noise[jj]
            print(torch.sum(torch.tensor(loss_mask) == 0))
            new_inputs.append(new_input_total)
            loss_masks.append(loss_mask)
        self.input_ids = torch.tensor(new_inputs) 
        self.loss_mask = torch.tensor(loss_masks)
        self.attention_mask = torch.ones(self.input_ids.shape)
        print(self.input_ids.shape)
        print(self.loss_mask.shape)
        print(self.attention_mask.shape)
        # print("argmax noise input shape: ", self.input_ids.shape)
        return self.input_ids, self.loss_mask
    
    '''
    def generate_noised_input_bigram_argmax(
        self, 
        input_ids, 
        noise_scheme = "synonym",
        n = 2,
        noise_order = 8, 
        is_prompt=True, 
        is_start_of_sentence=False,
        threshold_get_from_true_token = 0.7,
        use_noise_threshold = -1,
        logits_processor=None,
        logits_warper=None,
    ):
        self.noise_magnitude = 2**n*2
        import random
        b,s = input_ids.shape
        for batch in range(b):
            for seq in range(s):
                new_input = []
                # loss_masks = []
                # key: -1 for previous time stamp true token, -2 for previous time stamp noise token
                # value: -1, -2, -3... for position in logits
                toks_to_indices = {k:0 for k in range(-1*n, 0)}
                # key: -1 for current time stamp true token, -2 for current time stamp noise token1,..-n for current time stamp noise token n
                # key: -(n+1) for previous time stamp true token, -(n+2) for previous time stamp noise token...-2n for previous time stamp noise token n 
                # value: token number
                tok_to_noise = {k:-1 for k in range(-2*n, 0)}

                if seq == 0:
                    new_input = input_ids[batch, seq] * self.noise_magnitude
                    self.loss_mask.extend([0]*self.input_ids.shape[1])
                    # toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
                    
                    
                else:
                    new_input = self.sequence_last_time_stamp
                    toks_to_indices = self.toks_to_indices_last_time_stamp.copy()
                    tok_to_noise = self.toks_to_noise_last_time_stamp.copy()

                new_input = torch.tensor(new_input).unsqueeze(dim=0).to(torch.int)
                self.input_ids = new_input
                if is_prompt or (not is_prompt and is_start_of_sentence):
                    self.attention_mask = torch.ones(self.input_ids.shape)
                else:
                    self.attention_mask = torch.cat((self.attention_mask, torch.ones(n**2*2).unsqueeze(dim=0)),dim=1)
                
                loss_mask = self.loss_mask
                self.noise_magnitude = n**2*2

                with torch.no_grad():
                    # generation with past key values
                    outputs, _, _ = self(
                        new_input, 
                        past_key_values=self.noise_past_key_values, 
                        for_argmax_noise_generation = True
                    )
                    self.noise_past_key_values = outputs.past_key_values
                seen = []

                # if is in generation mode, sample a token first
                if not is_prompt:
                    input_ids = self.get_generated_token(
                        logits_processor,
                        logits_warper,
                        outputs,
                        loss_mask
                    )
                    # if starts to generate pad tokens, end generation.
                    if input_ids.squeeze(dim=0) == 1:
                        return None, None, False

                # if currently on the start token, get all noises from the logits of the start token
                if is_start_of_sentence:
                    next_token_logits = outputs.logits[:, -1, :]
                    seen_toks = [input_ids, torch.tensor(2), last_prompt_tok]
                    noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                    for i in range(-1*n,-1):
                        while noise_tok in seen_toks:
                            noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                        tok_to_noise[i] = noise_tok
                        seen_toks.append(noise_tok)
                    tok_to_noise[-1] = input_ids
                    for i in range(-2*n,-1*n):
                        tok_to_noise[i] = last_prompt_tok
                else:
                    seen_toks = [input_ids] # + [tok_to_noise[i] for i in range(-1*2*n,-1*n)]
                    for i in range(-1*n,-1):
                        v = toks_to_indices[i]
                        next_token_logits = outputs.logits[:, v, :]
                        noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5)
                        while noise_tok in seen_toks:
                            noise_tok = self.custom_sampling_topk(new_input, next_token_logits, topk=5) 
                        tok_to_noise[i] = noise_tok
                        seen_toks.append(noise_tok)
                    tok_to_noise[-1] = input_ids                      

                # key: -1 for current time stamp true token, -2 for current time stamp noise token, -3 for previous time stamp true token, -4 for previous time stamp noise token
                ft_keys = 0
                bigram_construct = {}
                for i in range(-1*n,0):
                    for j in range(-1*2*n,-1*n):
                        if i == -1 and j == -1*n-1:
                            bigram_construct["tt"] = [tok_to_noise[j],tok_to_noise[i]]
                        elif abs(i - j) == n:
                            bigram_construct[f"ff{i}"] = [tok_to_noise[j],tok_to_noise[i]]
                        else:
                            bigram_construct[f"ft{ft_keys}"] = [tok_to_noise[j],tok_to_noise[i]]
                            ft_keys += 1

                shuffled_noise = list(bigram_construct.items())
                random.seed(self.time_step * self.large_prime)  # setting random seed to be t*p, where p is a large prime number input by user
                random.shuffle(shuffled_noise)

                seen_toks.append(input_ids)
                old_real_tok_idx = toks_to_indices[-1]
                token_index = n**2*2-1

                for code, construct in shuffled_noise:
                    
                    construct = torch.tensor(construct)
                    new_input = torch.cat((new_input.squeeze(dim=0),construct))

                    if code == "tt":
                        loss_mask.extend([-1,0])
                        toks_to_indices[-1] = -1 * token_index
                    elif "ff" in code:
                        noise_seq_id = int(code[-2:])
                        toks_to_indices[noise_seq_id] = -1 * token_index
                        loss_mask.extend([-1,-1])
                    else:
                        loss_mask.extend([-1,-1])
                    token_index -= 2
                
                # change from this time stamp to last time stamp
                self.toks_to_noise_last_time_stamp = tok_to_noise.copy()
                for i in range(-1*n,0):
                    self.toks_to_noise_last_time_stamp[i-n] = tok_to_noise[i]
                self.toks_to_indices_last_time_stamp = toks_to_indices.copy()
                self.sequence_last_time_stamp = new_input[-1*n**2*2:]
                self.input_ids = torch.tensor(new_input)
                self.loss_mask = loss_mask
                self.noise_magnitude = noise_order
                seen_toks = [self.toks_to_noise_last_time_stamp[i] for i in range(-1*2*n,-1*n)]
                
                if is_prompt:
                    self.true_input_ids = torch.cat((self.true_input_ids, input_ids.unsqueeze(dim=0)), dim=0)
                    self.total_noised_input_ids = new_input.clone()
                else:
                    self.total_noised_input_ids = torch.cat((self.total_noised_input_ids, self.sequence_last_time_stamp), dim=0)
                
        return self.input_ids, self.loss_mask, outputs
    '''

    '''
    ##########################END OF CUSTOM FUNCTIONS##########################
    '''

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        noise_generate: Optional[bool] = True,
        run_baseline: Optional[bool] = False,
        noise_order: Optional[int] = 2,
        noise_type: Optional[str] = "unigram",
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        for_argmax_noise_generation: Optional[bool] = False,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        loss_mask = torch.tensor([])

        
        from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
        # if noise_generate:
        #     print(tokenizer.decode(self.input_ids[0,:]), flush=True)
        # else:
        #     print(tokenizer.decode(input_ids[0,:]), flush=True)
        
        # print(self.loss_mask)
        # print("input_ids: ", input_ids.shape)
        # print("self input_ids: ", self.input_ids.shape)
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if "mix" in noise_type:
            self.noise_magnitude = noise_order
        if noise_type == "direct_noise":
            outputs = self.model.decoder(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                head_mask=head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif not noise_generate or "mix" in noise_type:
            outputs = self.model.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            outputs = self.model.decoder(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                head_mask=head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            if noise_type == "direct_noise":
                print("calculating loss for baseline test...")
            elif "mix" in noise_type:
                shift_labels = labels[..., self.noise_magnitude:].to(torch.long).contiguous()
            elif noise_generate:
                if noise_type == "unigram" or "argmax" in noise_type:
                    shift_labels = self.generate_masked_shifted_labels_unigram(input_ids, self.input_ids, self.loss_mask)
                elif noise_type == "bigram":
                    shift_labels = self.generate_masked_shifted_labels_bigram(input_ids, self.input_ids, self.loss_mask)
                elif noise_type == "trigram":
                    shift_labels = self.generate_masked_shifted_labels_trigram(input_ids, self.input_ids, self.loss_mask)
            else:
                shift_labels = labels[..., 1:].to(torch.long).contiguous()
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            if noise_type == "direct_noise":
                shift_logits = logits[..., :-1, :].contiguous()
            else:
                shift_logits = logits[..., :-1*self.noise_magnitude, :].contiguous()
            if noise_type == "direct_noise":
                shift_labels = labels[..., 1:].to(torch.long).contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        # print("forward noise magnitude: ", self.noise_magnitude)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), self.loss_mask, self.noise_magnitude

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past





class OPTLatticeGenV2(OPTForCausalLM):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        import pickle
        self.model = OPTModel(config)
        self.times = []
        self.generation_time = list()
        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # additional attributes for latticegen
        self.tokenizer = AutoTokenizer.from_pretrained("opt-models/base-opt1.3b-tokenizer")
        self.prediction_token = self.tokenizer.encode("<predict>")[1]
        # dbfile = open("/home/gridsan/groups/txml/michael/LatticeGenV2/opt1.3b-vdb-top20", 'rb')
        # self.vdb = pickle.load(dbfile)

        # print("#"*100)
        # print(self.prediction_token)
        # print("#"*100)

        # Initialize weights and apply final processing
        self.post_init()
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.prediction_token = self.tokenizer.encode("<predict>")[1]

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()
        # print("decoded text: ", self.tokenizer.batch_decode(input_ids,skip_special_tokens=False)[-1])
        # print("noised tokens: ", input_ids[-1], " noised token length: ", len(input_ids[-1]))
        # print("labels: ", labels[-1], " label length: ", len(labels[-1]))
        # print("#"*10)
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            # loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


    # custom functions
    @torch.no_grad()
    def generate(
        self,
        ngram: Optional[int] = 2,
        extended_history: Optional[bool] = False,
        repetition_penalty: Optional[int] = 1,
        n_noise_tokens: Optional[int] = 2,
        noise_sample_topk: Optional[int] = 5,
        mix_ratio: Optional[float] = 0.0,
        noise_scheme: Optional[str] = "topk",
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ):
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
                `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
                generating before other GPUs. Otherwise it'll be set to `False`.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            kwargs:
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchDecoderOnlyOutput`],
                    - [`~generation.SampleDecoderOnlyOutput`],
                    - [`~generation.BeamSearchDecoderOnlyOutput`],
                    - [`~generation.BeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchEncoderDecoderOutput`],
                    - [`~generation.SampleEncoderDecoderOutput`],
                    - [`~generation.BeamSearchEncoderDecoderOutput`],
                    - [`~generation.BeamSampleEncoderDecoderOutput`]
        """

        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation -- update the generation config
            # model attribute accordingly, if it was created from the model config
            if self.generation_config._from_model_config:
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use a generation configuration file (see"
                        " https://huggingface.co/docs/transformers/main_classes/text_generation)"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config.pad_token_id is not None
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 7. determine generation mode
        is_constraint_gen_mode = (
            generation_config.constraints is not None or generation_config.force_words_ids is not None
        )

        is_contrastive_search_gen_mode = (
            (generation_config.num_beams == 1)
            and generation_config.top_k is not None
            and generation_config.top_k > 1
            and generation_config.do_sample is False
            and generation_config.penalty_alpha is not None
            and generation_config.penalty_alpha > 0
        )

        is_greedy_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_sample_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_beam_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_beam_sample_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_group_beam_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups > 1)
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_assisted_gen_mode = False
        if assistant_model is not None:
            if not (is_greedy_gen_mode or is_sample_gen_mode):
                raise ValueError(
                    "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                    "is only supported with Greedy Search and Sample."
                )
            is_assisted_gen_mode = True

        if generation_config.num_beam_groups > generation_config.num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and generation_config.do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        # 9. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        # 10. go into different generation modes
        if is_assisted_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")

            # 11. If the assistant model is an encoder-decoder, prepare its encoder outputs
            if assistant_model.config.is_encoder_decoder:
                assistant_model_kwargs = copy.deepcopy(model_kwargs)
                inputs_tensor, model_input_name, assistant_model_kwargs = assistant_model._prepare_model_inputs(
                    inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_model_kwargs
                )
                assistant_model_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor, assistant_model_kwargs, model_input_name
                )
                model_kwargs["assistant_encoder_outputs"] = assistant_model_kwargs["encoder_outputs"]

            # 12. run assisted generate
            return self.assisted_decoding(
                input_ids,
                assistant_model=assistant_model,
                do_sample=generation_config.do_sample,
                logits_processor=logits_processor,
                logits_warper=self._get_logits_warper(generation_config) if generation_config.do_sample else None,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        if is_greedy_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing greedy search, "
                    f"but is {generation_config.num_return_sequences}."
                )

            # 11. run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif is_contrastive_search_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing contrastive search, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")

            return self.contrastive_search(
                input_ids,
                top_k=generation_config.top_k,
                penalty_alpha=generation_config.penalty_alpha,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample
            return self.sample(
                input_ids,
                extended_history = extended_history,
                ngram = ngram,
                repetition_penalty = repetition_penalty,
                noise_sample_topk=noise_sample_topk,
                n_noise_tokens = n_noise_tokens,
                mix_ratio = mix_ratio,
                noise_scheme=noise_scheme,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")
            # 12. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size * generation_config.num_return_sequences,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                max_length=generation_config.max_length,
            )

            # 13. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams * generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 14. run beam sample
            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if generation_config.num_beams % generation_config.num_beam_groups != 0:
                raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            has_default_typical_p = kwargs.get("typical_p") is None and generation_config.typical_p == 1.0
            if not has_default_typical_p:
                raise ValueError("Decoder argument `typical_p` is not supported with beam groups.")

            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_constraint_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            if generation_config.num_beams <= 1:
                raise ValueError("`num_beams` needs to be greater than 1 for constrained generation.")

            if generation_config.do_sample:
                raise ValueError("`do_sample` needs to be false for constrained generation.")

            if generation_config.num_beam_groups is not None and generation_config.num_beam_groups > 1:
                raise ValueError("`num_beam_groups` not supported yet for constrained generation.")

            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]`"
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.LongTensor,
        extended_history: Optional[bool] = False,
        repetition_penalty: Optional[int] = 1,
        noise_sample_topk: Optional[int] = 5,
        ngram: Optional[int] = 2,
        n_noise_tokens: Optional[int] = 2,
        mix_ratio: Optional[int] = 0.0,
        noise_scheme: Optional[str] = "topk",
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.sample`] directly. Use generate() instead.
        For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.SampleDecoderOnlyOutput`], [`~generation.SampleEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.SampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.SampleEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     TopKLogitsWarper,
        ...     TemperatureLogitsWarper,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.config.pad_token_id = model.config.eos_token_id
        >>> model.generation_config.pad_token_id = model.config.eos_token_id

        >>> input_prompt = "Today is a beautiful day, and"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> # instantiate logits processors
        >>> logits_warper = LogitsProcessorList(
        ...     [
        ...         TopKLogitsWarper(50),
        ...         TemperatureLogitsWarper(0.7),
        ...     ]
        ... )

        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
        >>> outputs = model.sample(
        ...     input_ids,
        ...     logits_processor=logits_processor,
        ...     logits_warper=logits_warper,
        ...     stopping_criteria=stopping_criteria,
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Today is a beautiful day, and we must do everything possible to make it a day of celebration.']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only

        # define variables to keep track of history of generation
        import random

        self.past_key_values = None
        self.noised_history = [2]*(ngram*n_noise_tokens)
        self.sequence_idx_map = [list() for _ in range(n_noise_tokens)] # entry 0 is true sequence, the rest are noise sequences # random.choices(list( (ngram)), k=ngram)   # randonly setting bos tokens to be the true token
        self.true_sequence = []
        self.noise_sequences = [list() for _ in range(n_noise_tokens)]
        self.timestep_logits = dict()   # dictionary of dictionaries, outer key is the integer from 0 to total_length - 1, indicating the timestep of taking the logit, inner key are the ngrams
        if noise_scheme == "paralleldata":
          self.noise_datas = []
          for _ in range(1,n_noise_tokens):
            # import pdb
            # tttemp = self.dataset[random.randint(0, len(self.dataset)-1)]["input_ids"]
            # pdb.set_trace()
            self.noise_datas.append(self.dataset[random.randint(0, len(self.dataset)-1)]["input_ids"][1:])  # first token is bos
        time_step = 0

        bank = [list(range(n_noise_tokens)) for _ in range(ngram)]
        for k1 in range(n_noise_tokens):
            sequence = []
            for k2 in range(ngram):
                random_element = random.choice(bank[k2])
                bank[k2].remove(random_element)
                sequence.append(random_element)
            self.sequence_idx_map[k1] = sequence[:]

        # get noise for prompt
        for i in range(input_ids.shape[-1] - 1):
            if extended_history:
                self.sample_noise_tokens_extended_history(
                    input_ids[0,i + 1], 
                    time_step,
                    ngram=ngram, 
                    n_noise_tokens=n_noise_tokens,
                    repetition_penalty = repetition_penalty,
                )
            else:
                if noise_scheme == "topk":
                    self.sample_noise_tokens(
                        input_ids[0,i + 1],
                        time_step, 
                        ngram=ngram, 
                        n_noise_tokens=n_noise_tokens,
                        repetition_penalty = repetition_penalty,
                        noise_sample_topk=noise_sample_topk,
                        mix_ratio=mix_ratio,
                    )
                elif noise_scheme == "paralleldata":
                    self.sample_noise_tokens_synonym_and_paralleldata(
                        input_ids[0,i + 1],
                        time_step,
                        ngram=ngram,
                        n_noise_tokens=n_noise_tokens,
                        repetition_penalty = repetition_penalty,
                        noise_sample_topk=noise_sample_topk,
                        mix_ratio=mix_ratio,
                    )
            time_step += 1
            
        # auto-regressive generation with noise
        generation_count = 0
        while generation_count < 60:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
            if extended_history:
                self.sample_noise_tokens_extended_history(
                    input_ids[0,0],
                    time_step,
                    ngram=ngram, 
                    n_noise_tokens=n_noise_tokens,
                    is_prompt = False,
                    repetition_penalty = repetition_penalty,
                )
            else:
                generation_mix_ratio = 0.1
                if noise_scheme == "topk":
                    self.sample_noise_tokens(
                        input_ids[0,0],
                        time_step,
                        ngram=ngram, 
                        n_noise_tokens=n_noise_tokens,
                        is_prompt = False,
                        repetition_penalty = repetition_penalty,
                        noise_sample_topk=noise_sample_topk,
                        mix_ratio = generation_mix_ratio,
                    )
                elif noise_scheme == "paralleldata":
                  self.sample_noise_tokens_synonym_and_paralleldata(
                      input_ids[0,i + 1],
                      time_step,
                      ngram=ngram,
                      is_prompt = False,
                      n_noise_tokens=n_noise_tokens,
                      repetition_penalty = repetition_penalty,
                      noise_sample_topk=noise_sample_topk,
                      mix_ratio=mix_ratio,
                  )

            # print("generation content mix ratio: ", generation_mix_ratio)
            generation_count += 1
            time_step += 1
            # prepare model inputs
        #     model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        #     # forward pass to get next token
        #     outputs = self(
        #         **model_inputs,
        #         return_dict=True,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #     )

        #     if synced_gpus and this_peer_finished:
        #         continue  # don't waste resources running the code we don't need

        #     next_token_logits = outputs.logits[:, -1, :]

        #     # pre-process distribution
        #     next_token_scores = logits_processor(input_ids, next_token_logits)
        #     next_token_scores = logits_warper(input_ids, next_token_scores)

        #     # Store scores, attentions and hidden_states when required
        #     if return_dict_in_generate:
        #         if output_scores:
        #             scores += (next_token_scores,)
        #         if output_attentions:
        #             decoder_attentions += (
        #                 (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
        #             )
        #             if self.config.is_encoder_decoder:
        #                 cross_attentions += (outputs.cross_attentions,)

        #         if output_hidden_states:
        #             decoder_hidden_states += (
        #                 (outputs.decoder_hidden_states,)
        #                 if self.config.is_encoder_decoder
        #                 else (outputs.hidden_states,)
        #             )

        #     # sample
        #     probs = nn.functional.softmax(next_token_scores, dim=-1)
        #     next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        #     # finished sentences should have their next token be a padding token
        #     if eos_token_id is not None:
        #         if pad_token_id is None:
        #             raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
        #         next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        #     # update generated ids, model inputs, and length for next step
        #     input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        #     if streamer is not None:
        #         streamer.put(next_tokens.cpu())
        #     model_kwargs = self._update_model_kwargs_for_generation(
        #         outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        #     )

        #     # if eos_token was found in one sentence, set sentence to finished
        #     if eos_token_id_tensor is not None:
        #         unfinished_sequences = unfinished_sequences.mul(
        #             next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
        #         )

        #         # stop when each sentence is finished
        #         if unfinished_sequences.max() == 0:
        #             this_peer_finished = True

        #     # stop if we exceed the maximum length
        #     if stopping_criteria(input_ids, scores):
        #         this_peer_finished = True

        #     if this_peer_finished and not synced_gpus:
        #         break

        # if streamer is not None:
        #     streamer.end()

        # if return_dict_in_generate:
        #     if self.config.is_encoder_decoder:
        #         return SampleEncoderDecoderOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             encoder_attentions=encoder_attentions,
        #             encoder_hidden_states=encoder_hidden_states,
        #             decoder_attentions=decoder_attentions,
        #             cross_attentions=cross_attentions,
        #             decoder_hidden_states=decoder_hidden_states,
        #         )
        #     else:
        #         return SampleDecoderOnlyOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             attentions=decoder_attentions,
        #             hidden_states=decoder_hidden_states,
        #         )
        # else:
        #     return input_ids
    
    @torch.no_grad()
    def sample_noise_tokens(
        self,
        input_id: torch.LongTensor,
        time_step,
        repetition_penalty: Optional[int] = 1,
        ngram: Optional[int] = 2,
        n_noise_tokens: Optional[int] = 2,
        is_prompt: Optional[bool] = True,
        noise_sample_topk: Optional[int] = 5,
        mix_ratio: Optional[float] = 0.0,
    ):
        import itertools, random
        import time
        # import pdb
        # manual batch process when not enough space
        if n_noise_tokens > 3:
            batchify = True
        else:
            batchify = False

        start = time.time()
        # print("sample_noise_tokens, n_noise_tokens: ", n_noise_tokens, flush=True)
        # get all combinations of the ngram history
        lattice_histories = []
        for i in range(-1, -1 * n_noise_tokens * ngram, -1 * n_noise_tokens):
            if i == -1: 
                lattice_histories.append(self.noised_history[i-n_noise_tokens+1:])
            else:
                lattice_histories.append(self.noised_history[i-n_noise_tokens+1:i+1])  
        # reversing list of lists
        lattice_histories = lattice_histories[::-1]
        # pdb.set_trace()
        combinations = [list(range(n_noise_tokens)) for _ in range(ngram)]
        cur_sequence_indices = dict()
        noised_histories = []

        for seq_idx, combination in enumerate(itertools.product(*combinations)):
            noised_lattice = []
            # keep track of which combination to sample noise/true token from
            for comb_idx, seq_comb in enumerate(self.sequence_idx_map):
                seq = seq_comb[-1*ngram:]
                if list(combination) == seq:
                    cur_sequence_indices[comb_idx] = seq_idx
            for history, token_idx in enumerate(combination):
                noised_lattice.append(lattice_histories[history][token_idx])
            noised_histories.append(noised_lattice)
        
        seqs_to_process = []
        for i in range(n_noise_tokens**ngram):
            seqs_to_process.append(self.noised_history + [self.prediction_token] + noised_histories[i])
        seqs_to_process = torch.tensor(seqs_to_process).to(self.device)
        
        # save past key values to speed up generation
        # if is the very first sequence
        # total_num_sequences = n_noise_tokens ** ngram
        # total_batches = total_num_sequences // 64
        # remainder = total_num_sequences % 64
        # output_logits = None
        # output_past_key_values = None
        # for batch in range(total_batches):
        if batchify:
            # hardcoded batch number
            batches = 4
            batch_start = 0
            for batch in range(batches):
                
                batch_end = batch_start + n_noise_tokens**ngram // batches
                tmp_seq_to_process = seqs_to_process[batch_start:batch_end,:]
                # tmp_past_key_values = tuple(tuple(tensor[:, :, :-1*ngram-1, :].to(self.device) for tensor in inner_tuple) for inner_tuple in self.past_key_values)
                if seqs_to_process.shape[1] == (ngram*n_noise_tokens + 1 + ngram):
                    outputs = self(
                        seqs_to_process,
                        past_key_values = self.past_key_values,
                    )
                else:
                    outputs = self(
                        seqs_to_process[:,(-1*ngram-1 + -1*n_noise_tokens):],
                        # seqs_to_process,
                        past_key_values = self.past_key_values,
                    )
                batch_start = batch_end

        else:
            if seqs_to_process.shape[1] == (ngram*n_noise_tokens + 1 + ngram):
                outputs = self(
                    seqs_to_process,
                    past_key_values = self.past_key_values,
                )
            else:
                outputs = self(
                    seqs_to_process[:,(-1*ngram-1 + -1*n_noise_tokens):],
                    # seqs_to_process,
                    past_key_values = self.past_key_values,
                )

        #     if output_logits == None:
        #         output_logits = outputs.logits
        #     else:
        #         output_logits = torch.cat((output_logits, outputs.logits), dim = 0)
        #     if output_past_key_values == None:
        #         output_past_key_values = outputs.past_key_values
        #     else:
        #         output_past_key_values = tuple(
        #             tuple(torch.cat((output_past_key_values[inner_tuple][tensor],outputs.past_key_values[inner_tuple][tensor].clone())) 
        #                   for tensor in range(len(outputs.past_key_values[inner_tuple]))) 
        #                     for inner_tuple in range(len(outputs.past_key_values))
        #             )
        
        # print("past key value size: ", len(output_past_key_values))
        # print("past key value size1: ", len(output_past_key_values[0]))
        # print("past key value size2: ", output_past_key_values[0][0].shape)
        if batchify:
            self.past_key_values = tuple(tuple(tensor[:, :, :-1*ngram-1, :].cpu() for tensor in inner_tuple) for inner_tuple in outputs.past_key_values)
        else:
            self.past_key_values = tuple(tuple(tensor[:, :, :-1*ngram-1, :] for tensor in inner_tuple) for inner_tuple in outputs.past_key_values)
        # self.past_key_values = tuple(tuple(tensor[:, :, :-1*ngram-1, :] for tensor in inner_tuple) for inner_tuple in output_past_key_values)
        
        # saving the ngram history for beam search
        self.timestep_logits[time_step] = self.timestep_logits.get(time_step, dict())
        for history_idx in range(n_noise_tokens**ngram):
            key = []
            for k in noised_histories[history_idx][-1*ngram:]:
                if type(k) == int:
                    key.append(k)
                else:
                    key.append(k.item())
            self.timestep_logits[time_step][tuple(key)] = outputs.logits[history_idx][-1].cpu()
        
        seen = []
        # sample next true token
        logits = outputs.logits[cur_sequence_indices[0]][-1].unsqueeze(dim=0)
        current_lattice = dict()
        input_id = input_id.unsqueeze(dim=0)
        next_true_token = self.custom_sampling_topk(
            torch.tensor(self.true_sequence).unsqueeze(dim=0), 
            logits, 
            topk = 50, 
            repetition_penalty = repetition_penalty,
        )

        if is_prompt:
            self.true_sequence.append(input_id)
            current_lattice[0] = input_id 
            seen.append(input_id)
        else:
            current_lattice[0] = next_true_token
            self.true_sequence.append(next_true_token)
            seen.append(next_true_token)

        # sample noise tokens, with mix ratio
        for i in range(1, n_noise_tokens):
            sample_mix_ratio = random.random()
            # print("sample mix ratio: ", sample_mix_ratio)
            # print("mix ratio: ", mix_ratio)
            if sample_mix_ratio < mix_ratio:
                # sampling from true sequence
                logits = outputs.logits[cur_sequence_indices[0]][-1].unsqueeze(dim=0)
                next_noise_token = self.custom_sampling_topk(
                    torch.tensor(self.noise_sequences[i]).unsqueeze(dim=0), 
                    logits, 
                    topk = 5, 
                    repetition_penalty = repetition_penalty,
                    seen = seen,
                )
                while next_noise_token in seen:
                    # print("here2")
                    next_noise_token = self.custom_sampling_topk(
                        torch.tensor(self.true_sequence).unsqueeze(dim=0), 
                        logits, 
                        topk = noise_sample_topk,
                        repetition_penalty = repetition_penalty,
                        seen = seen
                    )
            else:
                # sampling from corresponding noise sequence
                logits = outputs.logits[cur_sequence_indices[i]][-1].unsqueeze(dim=0)
                next_noise_token = self.custom_sampling_topk(
                    torch.tensor(self.noise_sequences[i]).unsqueeze(dim=0), 
                    logits, 
                    topk = 5, 
                    repetition_penalty = repetition_penalty,
                    seen = seen,
                )
                while next_noise_token in seen:
                    # pdb.set_trace()
                    # print("here")
                    next_noise_token = self.custom_sampling_topk(
                        torch.tensor(self.noise_sequences[i]).unsqueeze(dim=0), 
                        logits, 
                        topk = noise_sample_topk,
                        repetition_penalty = repetition_penalty,
                        seen = seen,
                    )
                    # logits[0, next_noise_token.item()] = -100000
            # print("####################")
            current_lattice[i] = next_noise_token
            seen.append(next_noise_token)
            self.noise_sequences[i].append(next_noise_token)

        current_lattice_items = list(current_lattice.items())
        random.shuffle(current_lattice_items)
        next_lattice = []
        for idx, pair in enumerate(current_lattice_items):
            self.sequence_idx_map[pair[0]].append(idx)
            next_lattice.append(pair[1])
        self.noised_history.extend(next_lattice)

        end = time.time()
        if not is_prompt:
            self.generation_time.append(end - start)

    
    @torch.no_grad()
    def sample_noise_tokens_synonym_and_paralleldata(
        self,
        input_id: torch.LongTensor,
        time_step,
        repetition_penalty: Optional[int] = 1,
        ngram: Optional[int] = 2,
        n_noise_tokens: Optional[int] = 2,
        is_prompt: Optional[bool] = True,
        noise_sample_topk: Optional[int] = 5,
        mix_ratio: Optional[float] = 0.0,
    ):
        import itertools, random
        import time
        start = time.time()
        # print("sample_noise_tokens, n_noise_tokens: ", n_noise_tokens, flush=True)
        lattice_histories = []
        for i in range(-1, -1 * n_noise_tokens * ngram, -1 * n_noise_tokens):
            if i == -1:
                lattice_histories.append(self.noised_history[i-n_noise_tokens+1:])
            else:
                lattice_histories.append(self.noised_history[i-n_noise_tokens+1:i+1])
        # reversing list of lists
        lattice_histories = lattice_histories[::-1]
        cur_sequence_indices = dict()
        noised_histories = []
        combinations = [list(range(n_noise_tokens)) for _ in range(ngram)]

        for seq_idx, combination in enumerate(itertools.product(*combinations)):
            noised_lattice = []
            # keep track of which combination to sample noise/true token from
            for comb_idx, seq_comb in enumerate(self.sequence_idx_map):
                seq = seq_comb[-1*ngram:]
                if list(combination) == seq:
                    cur_sequence_indices[comb_idx] = seq_idx
            for history, token_idx in enumerate(combination):
                noised_lattice.append(lattice_histories[history][token_idx])
            noised_histories.append(noised_lattice)

        seqs_to_process = []
        for i in range(n_noise_tokens**ngram):
            seqs_to_process.append(self.noised_history + [self.prediction_token] + noised_histories[i])
        seqs_to_process = torch.tensor(seqs_to_process)

        if seqs_to_process.shape[1] == (ngram*n_noise_tokens + 1 + ngram):
            outputs = self(
                seqs_to_process.to(self.device),
                past_key_values = self.past_key_values,
            )
        else:
            outputs = self(
                seqs_to_process[:,(-1*ngram-1 + -1*n_noise_tokens):].to(self.device),
                # seqs_to_process,
                past_key_values = self.past_key_values,
            )

        self.past_key_values = tuple(tuple(tensor[:, :, :-1*ngram-1, :] for tensor in inner_tuple) for inner_tuple in outputs.past_key_values)

        # saving the ngram history for beam search
        self.timestep_logits[time_step] = self.timestep_logits.get(time_step, dict())
        for history_idx in range(n_noise_tokens**ngram):
            key = []
            for k in noised_histories[history_idx][-1*ngram:]:
                if type(k) == int:
                    key.append(k)
                else:
                    key.append(k.item())
            self.timestep_logits[time_step][tuple(key)] = outputs.logits[history_idx][-1].cpu()

        seen = []
        # sample next true token
        logits = outputs.logits[cur_sequence_indices[0]][-1].unsqueeze(dim=0)
        # import pdb
        # pdb.set_trace()
        current_lattice = dict()
        input_id = input_id.unsqueeze(dim=0)
        
        next_true_token = self.custom_sampling_topk(
            torch.tensor(self.true_sequence).unsqueeze(dim=0),
            logits,
            topk = 50,
            repetition_penalty = repetition_penalty,
            seen = seen,
        )

        if is_prompt:
            self.true_sequence.append(input_id)
            current_lattice[0] = input_id
            seen.append(input_id)
        else:
            current_lattice[0] = next_true_token
            self.true_sequence.append(next_true_token)
            seen.append(next_true_token)

        token = current_lattice[0]
        results = self.vdb[int(token)]
        choices = list(range(10,20))
        for k in range(1,n_noise_tokens):
            choice = random.choice(choices)
            next_noise_token = int(results[choice])
            while next_noise_token == self.prediction_token:
                choice = random.choice(choices)
                next_noise_token = int(results[choice])
            current_lattice[k] = next_noise_token
            choices.remove(choice)
            self.noise_sequences[k].append(next_noise_token)

        current_lattice_items = list(current_lattice.items())
        random.shuffle(current_lattice_items)
        next_lattice = []
        for idx, pair in enumerate(current_lattice_items):
            self.sequence_idx_map[pair[0]].append(idx)
            next_lattice.append(pair[1])
        self.noised_history.extend(next_lattice)
        # print("self.noised_history: ", self.noised_history)
        end = time.time()
        if not is_prompt:
            self.generation_time.append(end - start)
    
    @torch.no_grad()
    def custom_sampling_topk(
            self, 
            input_ids, 
            next_token_logits,
            topk,
            repetition_penalty = 1,
            seen = [],
        ):
        import pdb
        # pdb.set_trace()
        # print(input_ids)
        # print(next_token_logits)
        input_ids = input_ids.to(self.device)
        self.config.pad_token_id = self.config.eos_token_id
        self.generation_config.pad_token_id = self.config.eos_token_id

        # TODO: check this doesn't affect
        logits_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(150, eos_token_id=self.generation_config.eos_token_id),
            ]
        )
        if repetition_penalty != 1:
            logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))

        # instantiate logits processors
        # TODO: check if this is consistent when added temperature
        logits_warper = LogitsProcessorList(
            [
                TopKLogitsWarper(topk),
                TemperatureLogitsWarper(0.7),
            ]
        )
        
        for idd in seen:
            next_token_logits[0, idd.item()] = float("-inf")
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)
        # print(next_token_logits)
        # pdb.set_trace()
        # TODO set seen token scores to -100 
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        return next_tokens

    
    def generate_training_batch(self, inputs, ngram = 5, n_noise_toks = 3, nrepeats = 8):

        # TODO: every sequence has 4 different seq_lens
        # TODO: debug bigram, then use all ngrams
        # TODO: add more bos at the start accordingly based on what ngram

        import random
        input_ids = inputs["input_ids"]
        b, orig_seq_len = input_ids.shape
        # self.noise_magnitude = n**2*2

        noised_input_ids = []
        batch_labels = []
        batch_attention_masks = []
        for i in range(b):

            for j in range(nrepeats):

                seq_len = min(int(random.random() * orig_seq_len) + 1, orig_seq_len - 1)   # first token will always be bos, so there will always be at least 1 token
                noised_inputs = []
                labels = []
                attention_mask = []

                for idx in range(seq_len):
                    token = input_ids[i,idx]
                    # if is start token, then the noise token(s) are all start tokens, device a set of ngram * n_noise_toks bos tokens
                    if idx == 0:
                        noised_inputs.extend([token] * n_noise_toks * ngram)
                        labels.extend([-100] * n_noise_toks * ngram)
                        attention_mask.extend([1] * n_noise_toks * ngram)
                    else:
                        all_toks = []
                        results = self.vdb[int(token)]
                        choices = list(range(10,20))
                        for k in range(1,n_noise_toks):    
                            choice = random.choice(choices)
                            noise_token = int(results[choice])
                            while noise_token == self.prediction_token:
                                choice = random.choice(choices)
                                noise_token = int(results[choice])
                            choices.remove(choice)
                            all_toks.append(noise_token)
                        all_toks.append(token)
                        random.shuffle(all_toks)
                        
                        noised_inputs.extend(all_toks)
                        labels.extend([-100]*n_noise_toks)
                        attention_mask.extend([1] * n_noise_toks)

                noised_inputs.append(self.prediction_token)
                # TODO: small bug
                if seq_len <= ngram:
                    seq_to_append = [input_ids[i,0].detach().tolist()] * (ngram - seq_len) + input_ids[i, 0 : seq_len].detach().tolist()
                    noised_inputs.extend(seq_to_append)
                    # print("seq_to_append1: ", idx, len(seq_to_append))
                    # noised_inputs.extend([input_ids[i,0].detach().tolist()] * (ngram + 1 - seq_len))
                    # noised_inputs.extend(input_ids[i, 1 : seq_len - 1])
                else:
                    # TODO: confirm is correct
                    cur_idx = orig_seq_len - idx
                    seq_to_append = input_ids[i, -1*cur_idx - ngram + 1: -1*cur_idx + 1]
                    # seq_to_append = input_ids[i, idx - ngram + 1: idx + 1]
                    noised_inputs.extend(seq_to_append)
                    # print("seq_to_append2: ", idx, len(seq_to_append))
                noised_inputs.append(1)
                labels.extend([-100]*(ngram+1))
                labels.append(input_ids[i, idx+1])
                attention_mask.extend([1]*(ngram+1))
                attention_mask.append(0)

                # print("label len: ", len(labels))
                # print("input ids len: ", len(noised_inputs))
                
                batch_labels.append(labels)
                noised_input_ids.append(noised_inputs)
                batch_attention_masks.append(attention_mask)

                # if seq_len <= ngram:
                # print("decoded text: ", self.tokenizer.batch_decode(noised_input_ids,skip_special_tokens=False)[-1])
                # print("noised tokens: ", noised_input_ids[-1], " noised token length: ", len(noised_input_ids[-1]))
                # print("labels: ", batch_labels[-1], " label length: ", len(batch_labels[-1]))
                # print("inputs: ", input_ids[i])
                # print("#"*10)
        # noised_inputs = self.tokenizer(self.tokenizer.batch_decode(noised_input_ids), padding=True, return_tensors="pt", add_special_tokens = False)
        # print(noised_inputs)

        # padd to max seq length
        noised_inputs_to_return = {}
        max_seq_len = max(list(map(len, noised_input_ids)))
        for i in range(b*nrepeats):
            batch_labels[i].extend([-100]*(max_seq_len - len(batch_labels[i])))
            noised_input_ids[i].extend([1]*(max_seq_len - len(noised_input_ids[i])))
            batch_attention_masks[i].extend([0]*(max_seq_len - len(batch_attention_masks[i])))
        
        noised_inputs_to_return["input_ids"] = torch.tensor(noised_input_ids)
        noised_inputs_to_return["attention_mask"] = torch.tensor(batch_attention_masks)
        noised_inputs_to_return["labels"] = torch.tensor(batch_labels)
        
        # print("#######################################")
        # print(noised_inputs_to_return["input_ids"].shape)
        # print(torch.max(noised_inputs_to_return["input_ids"]))
        # print(torch.min(noised_inputs_to_return["attention_mask"]))
        # print(torch.min(noised_inputs_to_return["labels"]))

        # noised_inputs_to_return["labels"] = -100 * torch.ones(noised_inputs["input_ids"].shape, dtype = torch.long)
        # for k in range(b):
        #     seq_last_idx = len(batch_labels[k]) - 1
        #     noised_inputs["labels"][k][seq_last_idx] = batch_labels[k][-1]
        
        return noised_inputs_to_return
    

    def generate_training_batch_extended_history(
        self, 
        inputs, 
        ngram = 4, 
        n_noise_toks = 2, 
        nrepeats = 8, 
        history_pattern = [-1,-2,-4,-6],
    ):

        import random
        input_ids = inputs["input_ids"]
        b, orig_seq_len = input_ids.shape

        ngram = len(history_pattern)
        noised_input_ids = []
        batch_labels = []
        batch_attention_masks = []
        for i in range(b):

            for j in range(nrepeats):

                seq_len = min(int(random.random() * orig_seq_len) + 1, orig_seq_len - 1)   # first token will always be bos, so there will always be at least 1 token
                noised_inputs = []
                labels = []
                attention_mask = []

                for idx in range(seq_len):
                    token = input_ids[i,idx]
                    # if is start token, then the noise token(s) are all start tokens, device a set of ngram * n_noise_toks bos tokens
                    if idx == 0:
                        noised_inputs.extend([token] * n_noise_toks * ngram)
                        labels.extend([-100] * n_noise_toks * ngram)
                        attention_mask.extend([1] * n_noise_toks * ngram)
                    else:
                        all_toks = []
                        results = self.vdb[int(token)]
                        choices = list(range(10,20))
                        for k in range(1,n_noise_toks):    
                            choice = random.choice(choices)
                            noise_token = int(results[choice])
                            while noise_token == self.prediction_token:
                                choice = random.choice(choices)
                                noise_token = int(results[choice])
                            choices.remove(choice)
                            all_toks.append(noise_token)
                        all_toks.append(token)
                        random.shuffle(all_toks)
                        
                        noised_inputs.extend(all_toks)
                        labels.extend([-100]*n_noise_toks)
                        attention_mask.extend([1] * n_noise_toks)

                noised_inputs.append(self.prediction_token)
                # # TODO: small bug
                # if seq_len <= ngram:
                #     seq_to_append = [input_ids[i,0].detach().tolist()] * (ngram - seq_len) + input_ids[i, 1 : seq_len+1].detach().tolist()
                #     noised_inputs.extend(seq_to_append)

                # else:
                #     # TODO: confirm is correct
                #     cur_idx = orig_seq_len - idx
                #     seq_to_append = input_ids[i, -1*cur_idx - ngram + 1: -1*cur_idx + 1]
                #     noised_inputs.extend(seq_to_append)

                seq_to_append = list()
                for history_index in history_pattern:
                    # edge case: if sample sequence shorter than required history
                    if abs(history_index) > seq_len:
                        seq_to_append.append(input_ids[i,0].detach().tolist())
                    else:
                        cur_idx = idx
                        seq_to_append.append((input_ids[i, cur_idx + history_index + 1]))
                noised_inputs.extend(seq_to_append[::-1])

                noised_inputs.append(1)
                labels.extend([-100]*(ngram+1))
                labels.append(input_ids[i, idx+1])
                attention_mask.extend([1]*(ngram+1))
                attention_mask.append(0)
                
                batch_labels.append(labels)
                noised_input_ids.append(noised_inputs)
                batch_attention_masks.append(attention_mask)

        # padd to max seq length
        noised_inputs_to_return = {}
        max_seq_len = max(list(map(len, noised_input_ids)))
        for i in range(b*nrepeats):
            batch_labels[i].extend([-100]*(max_seq_len - len(batch_labels[i])))
            noised_input_ids[i].extend([1]*(max_seq_len - len(noised_input_ids[i])))
            batch_attention_masks[i].extend([0]*(max_seq_len - len(batch_attention_masks[i])))
        
        noised_inputs_to_return["input_ids"] = torch.tensor(noised_input_ids)
        noised_inputs_to_return["attention_mask"] = torch.tensor(batch_attention_masks)
        noised_inputs_to_return["labels"] = torch.tensor(batch_labels)
        

        # print(self.tokenizer.batch_decode(noised_input_ids,skip_special_tokens=False)[0])
        # print(self.tokenizer.batch_decode(inputs["input_ids"],skip_special_tokens=False)[0])
        
        return noised_inputs_to_return

    def generate_training_batch_parallel_datas(
        self, 
        inputs, 
        ngram = 4, 
        n_noise_toks = 2, 
        nrepeats = 8, 
        batch_size = 1,
    ):
        
        import random
        noised_input_ids = []
        batch_labels = []
        batch_attention_masks = []
        input_ids = inputs["input_ids"]
        b, orig_seq_len = input_ids.shape
        
        for i in range(b):
            pairs = random.sample(range(b), n_noise_toks)
            for j in range(nrepeats):
                seq_len = min(int(random.random() * orig_seq_len) + 1, orig_seq_len - 1)   # first token will always be bos, so there will always be at least 1 token
                noised_inputs = []
                labels = []
                attention_mask = []

                for idx in range(seq_len):
                    token = input_ids[pairs[0], idx]
                    # if is start token, then the noise token(s) are all start tokens, device a set of ngram * n_noise_toks bos tokens
                    if idx == 0:
                        noised_inputs.extend([token] * n_noise_toks * ngram)
                        labels.extend([-100] * n_noise_toks * ngram)
                        attention_mask.extend([1] * n_noise_toks * ngram)
                    else:
                        all_toks = []
                        for k in range(1,n_noise_toks):
                            noise_token = input_ids[pairs[k], idx]
                            all_toks.append(noise_token)
                        all_toks.append(token)
                        random.shuffle(all_toks)

                        noised_inputs.extend(all_toks)
                        labels.extend([-100]*n_noise_toks)
                        attention_mask.extend([1] * n_noise_toks)

                noised_inputs.append(self.prediction_token)

                if seq_len <= ngram:
                    seq_to_append = [input_ids[pairs[0], 0].detach().tolist()] * (ngram - seq_len) + input_ids[pairs[0], 0 : seq_len].detach().tolist()
                    noised_inputs.extend(seq_to_append)
                else:
                    cur_idx = orig_seq_len - idx
                    seq_to_append = input_ids[pairs[0], -1*cur_idx - ngram + 1: -1*cur_idx + 1]
                    noised_inputs.extend(seq_to_append)
                    
                noised_inputs.append(1)
                labels.extend([-100]*(ngram+1))
                labels.append(input_ids[pairs[0], idx+1])
                attention_mask.extend([1]*(ngram+1))
                attention_mask.append(0)

                batch_labels.append(labels)
                noised_input_ids.append(noised_inputs)
                batch_attention_masks.append(attention_mask)


        noised_inputs_to_return = {}
        max_seq_len = max(list(map(len, noised_input_ids)))
        for i in range(b*nrepeats):
            batch_labels[i].extend([-100]*(max_seq_len - len(batch_labels[i])))
            noised_input_ids[i].extend([1]*(max_seq_len - len(noised_input_ids[i])))
            batch_attention_masks[i].extend([0]*(max_seq_len - len(batch_attention_masks[i])))
        
        noised_inputs_to_return["input_ids"] = torch.tensor(noised_input_ids)[:batch_size*nrepeats].cuda()
        noised_inputs_to_return["attention_mask"] = torch.tensor(batch_attention_masks)[:batch_size*nrepeats].cuda()
        noised_inputs_to_return["labels"] = torch.tensor(batch_labels)[:batch_size*nrepeats].cuda()

        # print(self.tokenizer.batch_decode(noised_inputs_to_return["input_ids"])[0])
        return noised_inputs_to_return
    
    @torch.no_grad()
    def sample_noise_tokens_extended_history(
        self,
        input_id: torch.LongTensor,
        time_step,
        history_pattern = [-1,-2,-4,-6],
        repetition_penalty: Optional[int] = 1,
        ngram: Optional[int] = 2,
        n_noise_tokens: Optional[int] = 2,
        is_prompt: Optional[bool] = True,
    ):
        import itertools, random
        max_history_len = max([abs(h) for h in history_pattern])

        # get all combinations of the ngram history
        lattice_histories = []
        if (ngram + len(self.true_sequence)) < max_history_len:
            # take last ngram from real history, and pad the rest with bos
            for i in range(-1, -1 * n_noise_tokens * (ngram + len(self.true_sequence)), -1 * n_noise_tokens):
                if i == -1: 
                    lattice_histories.append(self.noised_history[i-n_noise_tokens+1:])
                else:
                    lattice_histories.append(self.noised_history[i-n_noise_tokens+1:i+1]) 
            
            # print("Padding length: ", max_history_len - (ngram + len(self.true_sequence)))
            for j in range(max_history_len - (ngram + len(self.true_sequence))):
                lattice_histories.append([2]*n_noise_tokens)
        else:
             for i in range(-1, -1 * n_noise_tokens * max_history_len, -1 * n_noise_tokens):
                if i == -1: 
                    lattice_histories.append(self.noised_history[i-n_noise_tokens+1:])
                else:
                    lattice_histories.append(self.noised_history[i-n_noise_tokens+1:i+1]) 
        # reversing list of lists
        lattice_histories = lattice_histories[::-1]
        # print(lattice_histories)
        # print(self.noised_history)
        history_pattern_reverse = history_pattern[::-1]
        combinations = [list(range(n_noise_tokens)) for _ in range(ngram)]
        cur_sequence_indices = dict()
        noised_histories = []
        for seq_idx, combination in enumerate(itertools.product(*combinations)):
            noised_lattice = []
            # keep track of which combination to sample noise/true token from
            for comb_idx, seq_comb in enumerate(self.sequence_idx_map):
                seq = []
                for history_idx in history_pattern_reverse:
                    if len(seq_comb) < abs(history_idx):
                        seq.append(0)
                    else:
                        seq.append(seq_comb[history_idx])
                # seq = seq_comb[-1*ngram:]
                if list(combination) == seq:
                    cur_sequence_indices[comb_idx] = seq_idx
            for history, token_idx in enumerate(combination[::-1]):
                # use history_pattern as indices
                noised_lattice.append(lattice_histories[history_pattern[history]][token_idx])
            noised_lattice = noised_lattice[::-1]
            noised_histories.append(noised_lattice)
        seqs_to_process = []
        for i in range(n_noise_tokens ** ngram):
            seqs_to_process.append(self.noised_history + [self.prediction_token] + noised_histories[i])
        seqs_to_process = torch.tensor(seqs_to_process)
        
        # save past key values to speed up generation
        # if is the very first sequence
        if seqs_to_process.shape[1] == (ngram*n_noise_tokens + 1 + ngram):
            outputs = self(
                seqs_to_process,
                past_key_values = self.past_key_values,
            )
        else:
            outputs = self(
                seqs_to_process[:,(-1*ngram-1 + -1*n_noise_tokens):],
                past_key_values = self.past_key_values,
            )
        self.past_key_values = tuple(tuple(tensor[:, :, :-1*ngram-1, :] for tensor in inner_tuple) for inner_tuple in outputs.past_key_values)

        # for kkk in range(len(self.past_key_values)):
        #     for kkkk in range(len(self.past_key_values[kkk])):
        #         print(self.past_key_values[0][0] == self.past_key_values[kkk][kkkk])
        
        seen = []
        # sample next true token
        logits = outputs.logits[cur_sequence_indices[0]][-1].unsqueeze(dim=0)
        current_lattice = dict()
        input_id = input_id.unsqueeze(dim=0)
        next_true_token = self.custom_sampling_topk(
            torch.tensor(self.true_sequence).unsqueeze(dim=0), 
            logits, 
            topk = 50, 
            repetition_penalty = repetition_penalty,
        )

        if is_prompt:
            self.true_sequence.append(input_id)
            current_lattice[0] = input_id 
            seen.append(input_id)
        else:
            current_lattice[0] = next_true_token
            self.true_sequence.append(next_true_token)
            seen.append(next_true_token)

        # sample noise tokens 
        for i in range(1, n_noise_tokens):
            logits = outputs.logits[cur_sequence_indices[i]][-1].unsqueeze(dim=0)
            next_noise_token = self.custom_sampling_topk(torch.tensor(self.noise_sequences[i]).unsqueeze(dim=0), logits, topk = 5)
            while next_noise_token in seen:
                next_noise_token = self.custom_sampling_topk(torch.tensor(self.noise_sequences[i]).unsqueeze(dim=0), logits, topk = 5)
            current_lattice[i] = next_noise_token
            seen.append(next_noise_token)
            self.noise_sequences[i].append(next_noise_token)

        current_lattice_items = list(current_lattice.items())
        random.shuffle(current_lattice_items)
        next_lattice = []
        for idx, pair in enumerate(current_lattice_items):
            self.sequence_idx_map[pair[0]].append(idx)
            next_lattice.append(pair[1])
        self.noised_history.extend(next_lattice)

    def multiple_beam_searches(
        self,
        input_ids: torch.LongTensor,
        num_beams: Optional[int] = 100,
        ngram: Optional[int] = 2,
        n_noise_tokens: Optional[int] = 2,
    ):
        seq_to_process = input_ids[:]
        generated_seq_len = len(self.true_sequence)
        true_seq_idx_map = self.sequence_idx_map[0][(ngram-1):]
        seq_len = len(input_ids)
        recovered_ratios = {k:0 for k in range(n_noise_tokens)}
        recovered_sequences = []
        self.beam_search_attack_sanity_check(
                input_ids = seq_to_process,
                ngram = ngram,
                n_noise_tokens = n_noise_tokens,
            )
        
        for round in range(n_noise_tokens - 1):
            top_k_sequences, recovered_sequence = self.beam_search_attack(
                input_ids = seq_to_process,
                num_of_beams = num_beams,
                ngram = ngram,
                n_noise_tokens = n_noise_tokens - round,
            )
            recovered_sequences.append(recovered_sequence)
            seq_recovered = top_k_sequences[0][0][(ngram-1):]
            
            # filter the sequence, assuming no repeating tokens within a single lattice  
            starting_idx = ngram * n_noise_tokens
            recovered_tokens = 0
            # pad the beginning of the sequence with bos, this doesn't affect the outcome, just a convenient way to code
            new_seq_to_process = [2] * ngram * (n_noise_tokens - round - 1)
            k = 1
            j = ngram * (n_noise_tokens - round)
            for i in range(starting_idx, seq_len, n_noise_tokens):
                recovered_tok = seq_recovered[k]
                true_tok = input_ids[i + true_seq_idx_map[k]]
                for _ in range(n_noise_tokens - round):
                    cur_lattice_tok = seq_to_process[j]
                    if recovered_tok != cur_lattice_tok:
                        new_seq_to_process.append(cur_lattice_tok)
                    else:
                        if recovered_tok == true_tok:
                            recovered_tokens += 1
                    j += 1
                k += 1
            print(f"RBS round {round + 1} recovery ratio: ", recovered_tokens/generated_seq_len)
            print("filtered sequence length: ", len(new_seq_to_process))
            seq_to_process = new_seq_to_process[:]
            recovered_ratios[round] = recovered_tokens/generated_seq_len

        seq_to_process = new_seq_to_process[(ngram-1):]
        print("last RBS seq length: ", len(seq_to_process))
        recovered_tokens = 0
        k = 1
        for i in range(starting_idx, seq_len, n_noise_tokens):
            recovered_tok = seq_to_process[k]
            true_tok = input_ids[i + true_seq_idx_map[k]]
            if recovered_tok == true_tok:
                recovered_tokens += 1
            k += 1
        print(f"RBS round {round + 1} recovery ratio: ", recovered_tokens/generated_seq_len)
        recovered_ratios[round+1] = recovered_tokens/generated_seq_len
        recovered_sequence = self.tokenizer.batch_decode(torch.tensor(seq_to_process).unsqueeze(dim=0))[0]
        print("recovered sequence: ", recovered_sequence)
        recovered_sequences.append(recovered_sequence)
        return recovered_ratios, recovered_sequences
    
    @torch.no_grad()
    def beam_search_attack_sanity_check(
            self,
            input_ids: torch.LongTensor,
            ngram: Optional[int] = 2,
            n_noise_tokens: Optional[int] = 2,
        ):

        import torch.nn as nn
        m = nn.LogSoftmax(dim=0)
        input_ids = torch.tensor(input_ids).unsqueeze(dim=0)
        true_sequence = [2]*ngram + self.true_sequence
        seq_len = len(true_sequence)
        time_step = 0

        score = 0
        for i in range(seq_len-ngram-1):
            key = tuple(true_sequence[i:i+ngram])
            history = []
            for k in key:
                if type(k) == int:
                    history.append(k)
                else:
                    history.append(k.item())
            logits = self.timestep_logits[time_step][tuple(history)]
            score += m(logits)[true_sequence[i+ngram].cpu()]
            time_step += 1
        print("RBS sanity check score: ", score)


    @torch.no_grad()
    def beam_search_attack(
        self,
        input_ids: torch.LongTensor,
        num_of_beams: Optional[int] = 100,
        ngram: Optional[int] = 4,
        n_noise_tokens: Optional[int] = 2,
    ):
        # TODO: debug, use both next_token_score and next_token logits, apply logsoftmax
        import torch.nn as nn
        seq_len = len(input_ids)
        m = nn.LogSoftmax(dim=0)
        input_ids = torch.tensor(input_ids).unsqueeze(dim=0)

        # top_k_sequences = [[[2]*(ngram),0] for _ in range(n_noise_tokens**ngram)]
        top_k_sequences = [[[2]*(ngram),0] for _ in range(1)]

        # deal with bos tokens
        # for i in range(0, ngram * n_noise_tokens, n_noise_tokens):
        #     for j in range(n_noise_tokens):
        #         tok1 = input_ids[0,i+j]
        #         t1 = tok1.item()
        #         for k in range(len(top_k_sequences)):
        #             new_entry = [top_k_sequences[k][0] + [t1], 0]
        #             temp_top_k_sequences.append(new_entry)
        # top_k_sequences = sorted(temp_top_k_sequences, key=lambda x: x[1], reverse = True)[:topk]
        
        time_step = 0
        for i in range(ngram * n_noise_tokens, seq_len, n_noise_tokens):
            temp_top_k_sequences = []
            # print(f"timestep {time_step} keys: ", self.timestep_logits[time_step].keys())
            for j in range(n_noise_tokens):
                tok1 = input_ids[0,i+j]
                t1 = tok1.item()
                for k in range(len(top_k_sequences)):
                    history = tuple(top_k_sequences[k][0][-1*ngram:])
                    # print("current ngram history: ", history)
                    logits = self.timestep_logits[time_step][history]
                    score = m(logits)[t1]
                    new_entry = [top_k_sequences[k][0] + [t1], top_k_sequences[k][1] + score]
                    # print("new entry: ", new_entry)
                    temp_top_k_sequences.append(new_entry)
            top_k_sequences = sorted(temp_top_k_sequences, key=lambda x: x[1], reverse = True)[:num_of_beams]
            # for jj in range(min(10,len(top_k_sequences))):
            #     print(top_k_sequences[jj][0])
            # print("$"*10)
            time_step += 1

        top_k_sequences = sorted(temp_top_k_sequences, key=lambda x: x[1], reverse = True)[:num_of_beams]
        # output = self(input_ids)

        
        # logits = output.logits
        # top_k_sequences = list()
 
        # starting_idx = ngram * n_noise_tokens - n_noise_tokens     # the starting index is the last set of bos tokens
        # for bos_idx in range(starting_idx, starting_idx + n_noise_tokens):
        #     tok1 = input_ids[0,bos_idx]
        #     t1 = tok1.item()
        #     top_k_sequences.append([[t1], logits[0,bos_idx,t1]])

        # top_k_sequences = sorted(top_k_sequences, key=lambda x: x[1], reverse = True)[:topk]
        
        # temp_top_k_sequences = []
        # for i in range(bos_idx+1, seq_len, n_noise_tokens):
        #     for j in range(i, i+n_noise_tokens):
        #         tok1 = input_ids[0,j]
        #         t1 = tok1.item()
        #         for k in range(min(topk, len(top_k_sequences))):
        #             new_entry = [top_k_sequences[k][0] + [t1], top_k_sequences[k][1] + logits[0,j,t1]]
        #             temp_top_k_sequences.append(new_entry)
        #     top_k_sequences = sorted(temp_top_k_sequences, key=lambda x: x[1], reverse = True)[:topk]
        print("recovered sequence length: ", len(top_k_sequences[0][0]))
        recovered_sequence = self.tokenizer.batch_decode(torch.tensor(top_k_sequences[0][0]).unsqueeze(dim=0))[0]
        print("recovered sequence: ", recovered_sequence)
        print("score: ", top_k_sequences[0][1])
        return top_k_sequences, recovered_sequence
    
    def get_true_ratio(
        self,
        input_ids: torch.LongTensor,
        ngram: Optional[int] = 2,
        n_noise_tokens: Optional[int] = 2,
    ):
        seq_to_process = input_ids[:]
        generated_seq_len = len(self.true_sequence)
        true_seq_idx_map = self.sequence_idx_map[0][(ngram-1):]
        seq_len = len(input_ids)
        
        starting_idx = ngram * n_noise_tokens
        recovered_tokens = 0
        k = ngram
        k1 = 1
        for i in range(starting_idx, seq_len, n_noise_tokens):
            recovered_tok = seq_to_process[k]
            true_tok = self.noised_history[i + true_seq_idx_map[k1]]
            if recovered_tok == true_tok:
                recovered_tokens += 1
            k += 1
            k1 += 1
        # print(f"baseline assessment recovery ratio: ", recovered_tokens/generated_seq_len)
        return recovered_tokens/generated_seq_len

    def multiple_beam_searches_100beams(
        self,
        input_ids: torch.LongTensor,
        num_beams: Optional[int] = 100,
        ngram: Optional[int] = 2,
        n_noise_tokens: Optional[int] = 2,
    ):
        seq_to_process = input_ids[:]
        generated_seq_len = len(self.true_sequence)
        true_seq_idx_map = self.sequence_idx_map[0][(ngram-1):]
        seq_len = len(input_ids)
        recovered_ratios = []
        recovered_sequences = []
        self.beam_search_attack_sanity_check(
                input_ids = seq_to_process,
                ngram = ngram,
                n_noise_tokens = n_noise_tokens,
            )
        
        for round in range(1):
            top_k_sequences, recovered_sequence = self.beam_search_attack(
                input_ids = seq_to_process,
                num_of_beams = num_beams,
                ngram = ngram,
                n_noise_tokens = n_noise_tokens - round,
            )
            recovered_sequences.append(recovered_sequence)
            
            # filter the sequence, assuming no repeating tokens within a single lattice  
            starting_idx = ngram * n_noise_tokens
            
            # pad the beginning of the sequence with bos, this doesn't affect the outcome, just a convenient way to code
            for z in range(100):
                seq_recovered = top_k_sequences[z][0][(ngram-1):]
                # print("score: ", top_k_sequences[z][1])
                new_seq_to_process = [2] * ngram * (n_noise_tokens - round - 1)
                k = 1
                j = ngram * (n_noise_tokens - round)
                recovered_tokens = 0
                for i in range(starting_idx, seq_len, n_noise_tokens):
                    recovered_tok = seq_recovered[k]
                    true_tok = input_ids[i + true_seq_idx_map[k]]
                    for _ in range(n_noise_tokens - round):
                        cur_lattice_tok = seq_to_process[j]
                        if recovered_tok != cur_lattice_tok:
                            new_seq_to_process.append(cur_lattice_tok)
                        else:
                            if recovered_tok == true_tok:
                                recovered_tokens += 1
                        j += 1
                    k += 1
                recovered_ratios.append(recovered_tokens/generated_seq_len)

            for kk in range(len(top_k_sequences)):
                recovered_sequences.append(self.tokenizer.batch_decode(torch.tensor(top_k_sequences[kk][0]).unsqueeze(dim=0))[0])

        return recovered_ratios, recovered_sequences

'''
 TODO:
1. mechanism to allow repeat, and modify beam search accordingly when there is repeat.
'''
