# from transformers import OPTForCausalLM, AutoTokenizer

# model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b")
# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

# new_tokens = ["<predict>"]
# tokenizer.add_tokens(list(new_tokens))
# model.resize_token_embeddings(len(tokenizer))

# print(tokenizer)

# model.save_pretrained("tmp/test_model")
# tokenizer.save_pretrained("tmp/test_tokenizer")

import torch
from transformers import OPTForCausalLM, AutoTokenizer
from custom_datasets.custom_datasets import DailyDialogueDataset, WritingPromptsDataset, WritingPromptsParellelNoiseDataset
device = "cuda" # the device to load the model onto

model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("tmp/base-opt1.3b-tokenizer")
test_dataset = WritingPromptsParellelNoiseDataset(tokenizer, 32, "train", size=10)
text = test_dataset[1]
# prompt = ("A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I am the "
#               "Statue of Liberty.\nHuman: Where do you live?\nStatue: New York City.\nHuman: How long have you lived "
#               "there?")

# model_inputs = tokenizer([prompt], return_tensors="pt")
# model.to(device)
# print("model use cache: ", model.config.use_cache)
# generated_ids = model.generate(**text, max_new_tokens=30, do_sample=True)
# print(tokenizer.batch_decode(generated_ids)[0])