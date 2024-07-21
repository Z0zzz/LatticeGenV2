from transformers import OPTForCausalLM, AutoTokenizer
import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from custom_datasets.custom_datasets import WritingPromptsDataset, WritingPromptsParellelNoiseDataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from accelerate import Accelerator

from models import LatticeGenLlamaForCausalLM 
import torch
# from transformers import OPTForCausalLM, AutoTokenizer
from custom_datasets.custom_datasets import DailyDialogueDataset, WritingPromptsDataset, WritingPromptsParellelNoiseDataset
import pdb

device = "cuda" # the device to load the model onto
block_size = 30
trust_remote_code = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
weight_decay = 0.0
num_train_epochs = 1
lr_scheduler_type = "linear"
learning_rate = 5e-5
num_warmup_steps = 0
gradient_accumulation_steps = 1

model_path = "base/llama"
model = LatticeGenLlamaForCausalLM.from_pretrained(model_path+"/base_model", device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path+"/base_tokenizer", trust_remote_code = True)
config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)
config.vocab_size = 32001
# accelerator = Accelerator()


embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
train_dataset = WritingPromptsDataset(tokenizer, block_size, "train",size=10000)
eval_dataset = WritingPromptsDataset(tokenizer, block_size, "valid", size=500)
print(train_dataset[0])
# pdb.set_trace()
train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=per_device_train_batch_size
    )
eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=per_device_eval_batch_size
    )


num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
max_train_steps = num_train_epochs * num_update_steps_per_epoch


no_decay = ["bias", "layer_norm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

# train_dataloader, model, optimizer = accelerator.prepare(train_dataloader, model, optimizer)

progress_bar = tqdm(range(max_train_steps))
starting_epoch = 0
import os
print(os.system("nvidia-smi"))
# pdb.set_trace()
for epoch in range(starting_epoch, num_train_epochs):
    model.train()

    active_dataloader = train_dataloader
    print(model)
    
    # pdb.set_trace()
    train_losses = []
    for step, batch in enumerate(active_dataloader):
        new_batch = model.generate_training_batch_parallel_datas(batch)
        print(os.system("nvidia-smi"))
        pdb.set_trace()
        outputs = model(**new_batch)
        loss = outputs.loss
        train_losses.append(loss.detach().cpu())
        if step % 10 == 0:
            print("checkpoint step loss: ", sum(train_losses)/len(train_losses), flush=True)
            train_losses = []
        # We keep track of the loss at each epoch
        # pdb.set_trace()
        print(os.system("nvidia-smi"))
        loss.backward()
        # accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
