from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, LlamaTokenizer, AutoModel
from transformers.generation.configuration_utils import GenerationConfig

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.deepspeed import is_deepspeed_zero3_enabled

from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)

from transformers.generation.logits_process import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
)
from typing import List, Optional, Tuple, Union
import copy
import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from custom_datasets.custom_datasets import DailyDialogueDataset, WritingPromptsDataset, WritingPromptsDatasetExampleGeneration
import random

# torch.manual_seed(10)
# random.seed(10)

logger = logging.get_logger(__name__)
print("importing LatticeGen...")
    
class LatticeGenLlamaModel(LlamaModel):
    config_class = LlamaConfig

    def __init__(self, config: AutoConfig):
        super().__init__(config)

class LatticeGenLlamaForCausalLM(LlamaForCausalLM):
    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        import pickle
        self.model = LatticeGenLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        dbfile = open("./vdbs/llama_vdb_extended.pickle", "rb")
        self.vdb = pickle.load(dbfile)

        print("initialzing...")
        # self.tokenizer = AutoTokenizer.from_pretrained("./base/llama/base_vanilla_tokenizer_hf")
        # self.prediction_token = self.tokenizer.encode("<predict>")[1]
        # self.bos_token = self.tokenizer.encode("<predict>")[0]  # hard code in the bos token, supposed to be the first token of every encoding operation   
        self.times = []
        self.generation_time = list()
        # torch.manual_seed(10)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def set_seed(self, seed):
        self.seed = seed

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

        self.prediction_token = self.tokenizer.encode("<predict>")[1]
        self.bos_token = self.tokenizer.encode("<predict>")[0]
        self.pad_token = self.tokenizer.encode("[PAD]")[1]
        
        self.dataset = WritingPromptsDataset(tokenizer, 300, "train", size=1000)
      
        print("prediction token: ", self.prediction_token)
        print("bos token: ", self.bos_token)
        print("pad token: ", self.pad_token)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
        allow_repeat_positions: Optional[list] = [],
        generation_temperature: Optional[float] = 0.7,
        prompt_mix_ratio: Optional[float] = 0.2,
        generation_mix_ratio: Optional[float] = 0.05,
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

        '''
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
        '''
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
                noise_scheme = noise_scheme,
                allow_repeat_positions = allow_repeat_positions,
                generation_temperature = generation_temperature,
                prompt_mix_ratio = prompt_mix_ratio,
                generation_mix_ratio = generation_mix_ratio,
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
        allow_repeat_positions: Optional[list] = [],
        generation_temperature: Optional[float] = 0.7,
        prompt_mix_ratio: Optional[float] = 0.2,
        generation_mix_ratio: Optional[float] = 0.05,
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
        # import random

        self.past_key_values = None
        self.noised_history = [self.bos_token]*(ngram*n_noise_tokens)
        self.sequence_idx_map = [list() for _ in range(n_noise_tokens)] # entry 0 is true sequence, the rest are noise sequences # random.choices(list( (ngram)), k=ngram)   # randonly setting bos tokens to be the true token
        self.true_sequence = []
        self.noise_sequences = [list() for _ in range(n_noise_tokens)]
        self.timestep_logits = dict()   # dictionary of dictionaries, outer key is the integer from 0 to total_length - 1, indicating the timestep of taking the logit, inner key are the ngrams
        self.allow_repeat_positions = allow_repeat_positions
        self.allow_repeat_tokens = list()
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
                      mix_ratio=prompt_mix_ratio,
                      generation_temperature = generation_temperature
                  )
                elif noise_scheme == "paralleldata":
                  self.sample_noise_tokens_synonym_and_paralleldata(
                      input_ids[0,i + 1],
                      time_step, 
                      ngram=ngram, 
                      n_noise_tokens=n_noise_tokens,
                      repetition_penalty = repetition_penalty,
                      noise_sample_topk=noise_sample_topk,
                      mix_ratio=prompt_mix_ratio,
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
                generation_mix_ratio = generation_mix_ratio
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
                      generation_temperature = generation_temperature,
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
        mix_ratio: Optional[float] = 0.2,
        generation_temperature: Optional[float] = 0.7,
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
        
        ######### SERVER STEP ##############
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
        ######### END OF SERVER STEP ##############
        
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
            seen = seen, 
            generation_temperature = generation_temperature,
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
            if sample_mix_ratio < mix_ratio and time_step not in self.allow_repeat_positions:
                # sampling from true sequence
                print("sample from true sequence")
                print("time step: ", time_step)
                
                logits = outputs.logits[cur_sequence_indices[0]][-1].unsqueeze(dim=0)
                next_noise_token = self.custom_sampling_topk(
                    # torch.tensor(self.noise_sequences[i]).unsqueeze(dim=0), 
                    torch.tensor(self.true_sequence).unsqueeze(dim=0),
                    logits, 
                    topk = noise_sample_topk, 
                    repetition_penalty = repetition_penalty,
                    seen = seen,
                    generation_temperature = generation_temperature,
                )
                print("noise token: ", self.tokenizer.decode(next_noise_token.item()))
            elif sample_mix_ratio >= mix_ratio and time_step not in self.allow_repeat_positions:
                # sampling from corresponding noise sequence
                logits = outputs.logits[cur_sequence_indices[i]][-1].unsqueeze(dim=0)
                next_noise_token = self.custom_sampling_topk(
                    torch.tensor(self.noise_sequences[i]).unsqueeze(dim=0), 
                    logits, 
                    topk = noise_sample_topk, 
                    repetition_penalty = repetition_penalty,
                    seen = seen,
                    generation_temperature = generation_temperature,
                    )
                
            else:   # if time_step in allow_repeat_positions
                next_noise_token = input_id.squeeze(dim=0)
                self.allow_repeat_tokens.append(next_noise_token)

            current_lattice[i] = next_noise_token
            seen.append(next_noise_token)
            self.noise_sequences[i].append(next_noise_token)
        
        random.seed(self.seed*time_step)
        torch.manual_seed(self.seed*time_step)

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
            generation_temperature = 0.7,
        ):

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
        
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        return next_tokens


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

        '''
        outputs = self(
                seqs_to_process.cuda(),
                # past_key_values = self.past_key_values,
            )
        '''
      
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
        # print(self.tokenizer.batch_decode(torch.tensor(self.true_sequence).unsqueeze(dim=0)))
        # print(self.tokenizer.batch_decode(torch.tensor(self.noise_sequences[1]).unsqueeze(dim=0)))
        next_true_token = self.custom_sampling_topk(
            torch.tensor(self.true_sequence).unsqueeze(dim=0), 
            logits, 
            topk = 50, 
            repetition_penalty = repetition_penalty,
            seen = seen, 
        )
        # pdb.set_trace() 
        if is_prompt:
            self.true_sequence.append(input_id)
            current_lattice[0] = input_id 
            seen.append(input_id)
        else:
            current_lattice[0] = next_true_token
            self.true_sequence.append(next_true_token)
            seen.append(next_true_token)
        
        '''
        # sample noise tokens from parallel data
        for i in range(1, n_noise_tokens):  
          # next_noise_token = self.noise_datas[i-1][time_step]
          next_noise_token = self.true_sequence[-1]
          current_lattice[i] = next_noise_token
          seen.append(next_noise_token)
          self.noise_sequences[i].append(next_noise_token)
        '''
        
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

  
    def generate_training_batch(self, inputs, ngram = 4, n_noise_toks = 2, nrepeats = 8):

        # TODO: every sequence has 4 different seq_lens
        # TODO: debug bigram, then use all ngrams
        # TODO: add more bos at the start accordingly based on what ngram

        #  import random
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
        
        return noised_inputs_to_return

    def generate_training_batch_parallel_datas(
        self, 
        inputs, 
        ngram = 2, 
        n_noise_toks = 2, 
        nrepeats = 4,
        force_batch = 1,
    ):
        print(f"generating parallel data training batch, ngram {ngram}, n={n_noise_toks}")
        # import random
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
                # print("generate_training_batch_parallel_datas: ", "sampling...")
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
                    
                noised_inputs.append(self.pad_token)
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
            noised_input_ids[i].extend([self.pad_token]*(max_seq_len - len(noised_input_ids[i])))
            batch_attention_masks[i].extend([0]*(max_seq_len - len(batch_attention_masks[i])))
        
        noised_inputs_to_return["input_ids"] = torch.tensor(noised_input_ids)[:force_batch*nrepeats].cuda()
        noised_inputs_to_return["attention_mask"] = torch.tensor(batch_attention_masks)[:force_batch*nrepeats].cuda()
        noised_inputs_to_return["labels"] = torch.tensor(batch_labels)[:force_batch*nrepeats].cuda()

        print(self.tokenizer.batch_decode(noised_inputs_to_return["input_ids"])[0])
        print(len(noised_inputs_to_return["input_ids"]))
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
            new_seq_to_process = [self.bos_token] * ngram * (n_noise_tokens - round - 1)
            k = 1
            j = ngram * (n_noise_tokens - round)
            print("sequence length: ", seq_len)
            time_step = 0
            # import pdb
            # pdb.set_trace()
            for i in range(starting_idx, seq_len, n_noise_tokens):
                recovered_tok = seq_recovered[k]
                true_tok = input_ids[i + true_seq_idx_map[k]]
                # print("###")
                for _ in range(n_noise_tokens - round):
                    cur_lattice_tok = seq_to_process[j]
                    # print(self.allow_repeat_tokens)
                    # print(cur_lattice_tok)
                    if time_step in self.allow_repeat_positions:
                        # print(1)
                        recovered_tokens += 1
                        j += (n_noise_tokens - round)
                        new_seq_to_process.extend([cur_lattice_tok]*(n_noise_tokens - round - 1))
                        break
                    elif recovered_tok != cur_lattice_tok:
                        # print(2)
                        new_seq_to_process.append(cur_lattice_tok)
                    else:
                        # print(3)
                        if recovered_tok == true_tok:
                            recovered_tokens += 1
                    j += 1
                # print(j)
                # print("###")
                k += 1
                time_step += 1
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
        # print("recovered sequence: ", recovered_sequence)
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
        true_sequence = [self.bos_token]*ngram + self.true_sequence
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
            # print(m(logits).device)
            # print(true_sequence)
            score += m(logits)[true_sequence[i+ngram].cpu()]
            time_step += 1
        print("RBS sanity check score: ", score)

    @torch.no_grad()
    def beam_search_attack(
        self,
        input_ids: torch.LongTensor,
        num_of_beams: Optional[int] = 50,
        ngram: Optional[int] = 4,
        n_noise_tokens: Optional[int] = 2,
    ):
        # TODO: debug, use both next_token_score and next_token logits, apply logsoftmax
        import torch.nn as nn
        seq_len = len(input_ids)
        m = nn.LogSoftmax(dim=0)
        input_ids = torch.tensor(input_ids).unsqueeze(dim=0)

        top_k_sequences = [[[self.bos_token]*(ngram),0] for _ in range(1)]

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

        # print("recovered sequence length: ", len(top_k_sequences[0][0]))
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
                new_seq_to_process = [self.bos_token] * ngram * (n_noise_tokens - round - 1)
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

    
    
