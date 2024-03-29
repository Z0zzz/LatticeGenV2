from transformers import OPTForCausalLM, AutoTokenizer 
from models import LatticeGenLlamaForCausalLM
from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# new_tokens = ["<predict>"]
# tokenizer.add_tokens(list(new_tokens))
# model.resize_token_embeddings(len(tokenizer))
# print("new token space len: ", len(tokenizer))

# model.save_pretrained("base/llama/base_vanilla_model")
# tokenizer.save_pretrained("base/llama/base_vanilla_tokenizer")
# print("saved")
model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

new_tokens = ["<predict>"]
tokenizer.add_tokens(list(new_tokens))
model.resize_token_embeddings(len(tokenizer))

# print(tokenizer)

model.save_pretrained("opt-models/base-opt1.3b-model")
tokenizer.save_pretrained("opt-models/base-opt1.3b-tokenizer")

# import torch
# from transformers import OPTForCausalLM, AutoTokenizer
# from custom_datasets.custom_datasets import DailyDialogueDataset, WritingPromptsDataset, WritingPromptsParellelNoiseDataset
# device = "cuda" # the device to load the model onto

# model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
# tokenizer = AutoTokenizer.from_pretrained("tmp/base-opt1.3b-tokenizer")
# test_dataset = WritingPromptsParellelNoiseDataset(tokenizer, 32, "train", size=10)
# text = test_dataset[1]
# from models import LatticeGenLlamaForCausalLM
# from transformers import AutoModelForCausalLM, AutoTokenizer
# model_path = 'meta-llama/Llama-2-7b-chat-hf'
# model = LatticeGenLlamaForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# model = AutoModelForCausalLM.from_pretrained("/gscratch/argon/tianxing/projects/michael/llama/llama-2-7b-chat", device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("/gscratch/argon/tianxing/projects/michael/llama", trust_remote_code=True)


# test_dataset = WritingPromptsDataset(tokenizer, 32, "test", size=3)

# prompt = ("A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I am the "
#              "Statue of Liberty.\nHuman: Where do you live?\nStatue: New York City.\nHuman: How long have you lived "
#               "there?")
# text = test_dataset[1]
# print(model.beam_search_attack)
# model_inputs = tokenizer([prompt], return_tensors="pt")
# model.to(device)
# print("model use cache: ", model.config.use_cache)
# generated_ids = model.generate(**text, max_new_tokens=30, do_sample=True)
# print(generated_ids)
# print(tokenizer.batch_decode(generated_ids))
