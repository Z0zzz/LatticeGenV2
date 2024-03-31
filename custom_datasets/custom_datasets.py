import torch
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
class DailyDialogueDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, text, max_length):
        self.tokenizer = tokenizer
        self.text = text
        self.max_length = max_length

    def __getitem__(self, idx):
        item = self.text[idx]
        text = []
        for i, dialog in enumerate(item):
            person = "A:" if i % 2 == 0 else "B:"
            text.append(person)
            text.append(dialog)
        text = " ".join(text)
        sample = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length)
        sample["label"] = sample["input_ids"][:]
        # print(sample["label"])
        return sample

    def __len__(self):
        return len(self.text)


# ds = load_dataset("daily_dialog")
# train = ds["train"]
# dataset = DailyDialogueDataset(tokenizer, train["dialog"], max_length = 128)
# # print(tokenizer.decode(dataset[0]["input_ids"]))
# train_dataloader = DataLoader(
#         dataset, shuffle=True, collate_fn=default_data_collator, batch_size=8
#     )
# print(next(iter(train_dataloader)))


# data = ["train", "test", "valid"]
# for name in data:
#     with open("./writingPrompts/" + name + ".wp_source") as f:
#         stories = f.readlines()
#     stories = [" ".join(i.split()[0:1000]) for i in stories]

# for name in data:
#     with open("./writingPrompts/" + name + ".wp_target") as f:
#         stories1 = f.readlines()
#     stories1 = [" ".join(i.split()[0:1000]) for i in stories1]
datapath = '/gscratch/argon/tianxing/projects/michael/LatticeGenV2/custom_datasets/writingPrompts/'

class WritingPromptsDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, max_length, split, size = 2000, idx_start = 0):
        self.tokenizer = tokenizer
        self.split = split
        with open(datapath + split + ".wp_source") as f:
            stories = f.readlines()
        self.prompt = [" ".join(i.split()[:]) for i in stories][idx_start:min(size,len(stories))]

        with open(datapath + split + ".wp_target") as f:
            stories = f.readlines()
        self.story = [" ".join(i.split()[:]) for i in stories][idx_start:min(size,len(stories))]

        self.max_length = max_length

    def __getitem__(self, idx):

        prompt = self.prompt[idx].replace("<newline>", "")
        story = self.story[idx].replace("<newline>", "")
        if self.split == "test":
            text = f"Prompt: {prompt[7:]}. Story:"
            sample = self.tokenizer(text, return_tensors="pt")
        else:
            text = f"Prompt: {prompt[7:]}. Story: {story}"
            sample = self.tokenizer(text, return_token_type_ids=False, padding='max_length', truncation=True, max_length=self.max_length)
            # delete bos token which is added back when generating noised input for training
            sample["input_ids"] = sample["input_ids"]
            sample["attention_mask"] = sample["attention_mask"]
            sample["label"] = sample["input_ids"]

        return sample

    def __len__(self):
        return len(self.prompt)


class WritingPromptsParellelNoiseDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, max_length, split, size = 2000):
        self.tokenizer = tokenizer
        self.split = split
        self.prediction_token = self.tokenizer.encode("<predict>")[1]

        with open("/home/gridsan/groups/txml/michael/EncDecLlama/custom_datasets/writingPrompts/" + split + ".wp_source") as f:
            stories = f.readlines()
        self.prompt = [" ".join(i.split()[0:300]) for i in stories][:size]
        self.prompt_all = [" ".join(i.split()[0:300]) for i in stories][:]

        with open("/home/gridsan/groups/txml/michael/EncDecLlama/custom_datasets/writingPrompts/" + split + ".wp_target") as f:
            stories = f.readlines()
        self.story = [" ".join(i.split()[0:300]) for i in stories][:size]
        self.story_all = [" ".join(i.split()[0:300]) for i in stories][:]
        
        self.nrepeat = 8
        self.train_idx = 0
        self.size = len(self.story_all)
        self.max_length = max_length
        self.seen = []

    def __getitem__(self, idx):
        idx = self.train_
        import random
        self.seen.append(idx)
        prompt1 = self.prompt[idx].replace("<newline>", "")
        story1 = self.story[idx].replace("<newline>", "")

        idx2 = random.choice(list(range(self.size)))

        while idx2 in self.seen:
            idx2 = random.choice(list(range(self.size)))
        self.seen.append(idx2)

        prompt2 = self.prompt_all[idx2].replace("<newline>", "")
        story2 = self.story_all[idx2].replace("<newline>", "")


        if self.split == "test":
            text1 = f"Prompt: {prompt1[7:]}. Story:"
            text2 = f"Prompt: {prompt2[7:]}. Story:"
            sample1 = self.tokenizer(text1, return_tensors="pt")
            sample2 = self.tokenizer(text2, return_tensors="pt")
            sample = self.generate_parallel_noise_n2(sample1, sample2)
        else:
            prompt_len = min(len(prompt1), len(prompt2))
            text1 = f"Prompt: {prompt1[7:prompt_len]}. Story: {story1}"
            text2 = f"Prompt: {prompt2[7:prompt_len]}. Story: {story2}"
            sample1 = self.tokenizer(text1, padding='max_length', truncation=True, max_length=self.max_length)
            sample2 = self.tokenizer(text2, padding='max_length', truncation=True, max_length=self.max_length)

            sample = self.generate_parallel_noise_n2(sample1, sample2, prompt_len)

        return sample

    def generate_parallel_noise_n2(self, sample1, sample2, prompt_len):
        import random
        noised_input_ids = []
        batch_labels = []
        batch_attention_masks = []
        nrepeats = 8
        n_noise_toks = 2
        ngram = 4
        orig_seq_len = min(prompt_len, self.max_length)
        for j in range(nrepeats):
            seq_len = min(int(random.random() * orig_seq_len) + 1, orig_seq_len - 1)   # first token will always be bos, so there will always be at least 1 token
            noised_inputs = []
            labels = []
            attention_mask = []

            for idx in range(seq_len):
                token = sample1["input_ids"][idx]
                # if is start token, then the noise token(s) are all start tokens, device a set of ngram * n_noise_toks bos tokens
                if idx == 0:
                    noised_inputs.extend([token] * n_noise_toks * ngram)
                    labels.extend([-100] * n_noise_toks * ngram)
                    attention_mask.extend([1] * n_noise_toks * ngram)
                else:
                    all_toks = []
                    noise_token = sample2["input_ids"][idx]
                    all_toks.append(noise_token)
                    all_toks.append(token)
                    random.shuffle(all_toks)

                    noised_inputs.extend(all_toks)
                    labels.extend([-100]*n_noise_toks)
                    attention_mask.extend([1] * n_noise_toks)

            noised_inputs.append(self.prediction_token)

            if seq_len <= ngram:
                seq_to_append = [sample1["input_ids"][0]] * (ngram - seq_len) + sample1["input_ids"][0 : seq_len]
                noised_inputs.extend(seq_to_append)
            else:
                cur_idx = orig_seq_len - idx
                seq_to_append = sample1["input_ids"][ -1*cur_idx - ngram + 1: -1*cur_idx + 1]
                noised_inputs.extend(seq_to_append)
                
            noised_inputs.append(1)
            labels.extend([-100]*(ngram+1))
            labels.append(sample1["input_ids"][idx+1])
            attention_mask.extend([1]*(ngram+1))
            attention_mask.append(0)

            batch_labels.append(labels)
            noised_input_ids.append(noised_inputs)
            batch_attention_masks.append(attention_mask)
        
        noised_inputs_to_return = {}
        max_seq_len = max(list(map(len, noised_input_ids)))
        for i in range(nrepeats):
            batch_labels[i].extend([-100]*(max_seq_len - len(batch_labels[i])))
            noised_input_ids[i].extend([1]*(max_seq_len - len(noised_input_ids[i])))
            batch_attention_masks[i].extend([0]*(max_seq_len - len(batch_attention_masks[i])))
        
        noised_inputs_to_return["input_ids"] = torch.tensor(noised_input_ids)
        noised_inputs_to_return["attention_mask"] = torch.tensor(batch_attention_masks)
        noised_inputs_to_return["labels"] = torch.tensor(batch_labels)

        print(self.tokenizer.batch_decode(noised_inputs_to_return["input_ids"])[0])
        return noised_inputs_to_return

    def __len__(self):
        return len(self.prompt)


class WritingPromptsDatasetExampleGeneration(torch.utils.data.Dataset):

    def __init__(self, tokenizer, max_length, split, size = 2000, idx_start = 0):
        self.tokenizer = tokenizer
        # self.split = split
        # with open("/home/gridsan/groups/txml/michael/EncDecLlama/custom_datasets/writingPrompts/" + split + ".wp_source") as f:
        #     stories = f.readlines()
        # self.prompt = [" ".join(i.split()[0:300]) for i in stories][idx_start:size]

        # with open("/home/gridsan/groups/txml/michael/EncDecLlama/custom_datasets/writingPrompts/" + split + ".wp_target") as f:
        #     stories = f.readlines()
        # self.story = [" ".join(i.split()[0:300]) for i in stories][idx_start:size]
        # self.prompt = ["Aliens start abducting humans", "The scientists have discovered something terrible.", "The silence before the storm comes"]
        self.prompt = [
                "Aliens have arrived, and ask for a single human to plead humanity's case and save them from extinction. The human is selected through a lottery of the entire human race, and on the day of the drawing, your name is picked..", "Every planet in our solar system has a `` champion '' being that takes on the attributes of the planet itself. The `` champion '' from the sun has created an army to destroy the planets and the 8 ( or 9 ) champions must save the solar system..", "As a Space marine you have an allowance of one call home a day. Today's battle was especially bad and your best friend died I'm the heat of it all. Time to call home.."]
        self.max_length = max_length

    def __getitem__(self, idx):

        prompt = self.prompt[idx].replace("<newline>", "")
        # story = self.story[idx].replace("<newline>", "")
        text = f"Prompt: {prompt}. Story:"
        sample = self.tokenizer(text, return_tensors="pt")
    
        return sample

    def __len__(self):
        return len(self.prompt)

if __name__ == "__main__":

    ds = WritingPromptsDataset(tokenizer,100000000,"train")
    avg_prompt = []
    for p in ds.prompt:
        print(p, flush=True)
        avg_prompt.append(len(p.split(" ")))
    ds = WritingPromptsDataset(tokenizer,100000000,"valid")
    for p in ds.prompt:
        print(p, flush=True)
        avg_prompt.append(len(p.split(" ")))
    ds = WritingPromptsDataset(tokenizer,100000000,"test")
    for p in ds.prompt:
        print(p, flush=True)
        avg_prompt.append(len(p.split(" ")))
    print("avg of story: ", sum(avg_prompt)/len(avg_prompt))
