from transformers import LlamaModel, LlamaTokenizer, LlamaForCausalLM
import torch
from transformers import GPT2Tokenizer, GPT2Model, AutoTokenizer, OPTForCausalLM
# from EncDecLlamaModel import EncDecLlamaForCausal
from transformers import GPT2ForNoiseCausalLM, OPTDiscreteTokenSecurityForCausalLM, OPTLatticeGenV2
import statistics
import math
from custom_datasets.custom_datasets import DailyDialogueDataset, WritingPromptsDataset, WritingPromptsDatasetExampleGeneration
import pickle
from torch.nn import CrossEntropyLoss, LogSoftmax
from bert_score import score
from nltk.util import ngrams
import collections
size = 500


# model_path = "./tmp/4gram_n4_model_8sample"
# model_path = "./corrected_models/5gram_n2_model_8sample/step_2000"
# model_path = "./tmp/tmp/"
# model_path = "./tmp/5gram_n3_model_8sample"
# model_path = "./tmp/4gram_n2_model"
# model_path = "./tmp/4gram_n3_model_8sample"

# model_path = "./opt-models/4gram-n3-paralleldata"
# model_path = "./opt-models/4gram-n2-paralleldata"
# model_path = "./opt-models/2gram-n2-model/step_1400"
# model_path = "./opt-models/1gram-n2-model/step_1600"
# model_path = "./opt-models/1gram-n2-model"
model_path = "./opt-models/2gram-n2-model"
ngram = 4
n_noise_tokens = 2
noise_sample_topk = 5
num_beams = 50
noise_scheme = "topk"
# mix_ratios = [0,0.1,0.2,0.3,0.4]
mix_ratios = [0.0]
repetition_ratios = [1.05]
full_test_suite = False

if model_path == "./tmp/2gram_n2_model/step_2300":
    model = OPTLatticeGenV2.from_pretrained("./tmp/2gram_n2_model/step_2300").cuda()
    ngram = 2
elif "2gram-n2" in model_path:
    model = OPTLatticeGenV2.from_pretrained(model_path).cuda()
    ngram = 2
    n_noise_tokens = 3
    mix_ratios = [0.0]
elif "1gram-n2" in model_path:
    model = OPTLatticeGenV2.from_pretrained(model_path).cuda()
    ngram = 1
    # n_noise_tokens = 3
    mix_ratios = [0.3]
elif "4gram-n2" in model_path:
    model = OPTLatticeGenV2.from_pretrained(model_path).cuda()
elif model_path == "./tmp/4gram_n3_model_8sample":
    model = OPTLatticeGenV2.from_pretrained("./tmp/4gram_n3_model_8sample").cuda()
    n_noise_tokens = 3
    noise_sample_topk = 10
elif "4gram-n3" in model_path:
    model = OPTLatticeGenV2.from_pretrained(model_path).cuda()
    n_noise_tokens = 3
    noise_sample_topk = 10
elif "4gram-n4" in model_path:
    model = OPTLatticeGenV2.from_pretrained(model_path).cuda()
    n_noise_tokens = 4
    noise_sample_topk = 10
elif "5gram_n2_model" in model_path:
    model = OPTLatticeGenV2.from_pretrained(model_path)
    ngram = 5
elif "5gram_n3_model" in model_path:
    model = OPTLatticeGenV2.from_pretrained(model_path)
    ngram = 5
    n_noise_tokens = 3
    noise_sample_topk = 10
elif model_path == "./tmp/6gram_n2_model_8sample":
    model = OPTLatticeGenV2.from_pretrained("./tmp/6gram_n2_model_8sample")
    ngram = 6
else:
    # manually configure here
    ngram = 4
    model = OPTLatticeGenV2.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained("./opt-models/base-opt1.3b-tokenizer")
# model.get_embedding_database("/home/gridsan/groups/txml/michael/EncDecLlama/opt-vdb_top20")
model.init_tokenizer(tokenizer)

model_gt = OPTForCausalLM.from_pretrained("facebook/opt-2.7b")
tokenizer_gt = AutoTokenizer.from_pretrained("facebook/opt-2.7b")

test_dataset = WritingPromptsDataset(tokenizer, 32, "test", size=size)
# test_dataset = WritingPromptsDatasetExampleGeneration(tokenizer, 32, "test", size=size)

repetition_list = {}
true_token_percentages = {}
alignments = {mix_ratios[k]:list() for k in range(0,len(mix_ratios))} 

# printing run information
print("Model: ", model_path)
print(f"ngram {ngram}, n={n_noise_tokens}, noise_sample_topk={noise_sample_topk}")
print(f"mix ratios: {mix_ratios}, repetition ratios: {repetition_ratios}")


'''
Testing perplexity, prompt-generation alignment, beam search attack
'''

# configs
# debugging pkv used 4-2-5
# ngram = 4
# n_noise_tokens = 3
# noise_sample_topk = 10
# mix_ratios = [0,0.1,0.15,0.2,0.25]
'''
Currently testing repetition ngram
'''
# repetition_ratios = [1, 1.05, 1.1, 1.15]
print("model device: ", model.device)
for mix_ratio in mix_ratios:
    true_token_percentages = {round:list() for round in range(n_noise_tokens)}
    max_recovered_ratios = []
    all_max_bertscore = []
    all_max_bertscore_100beams = []
    hundred_beam_max_true_ratios = []
    bert_scores = {round:list() for round in range(n_noise_tokens)}
    for repetition_ratio in repetition_ratios:
        perplexities = []
        repetition_list = {}
        for i in range(size):
            text = test_dataset[i]
            if i > 250: break
            for jj in range(1):
                Max = 0
                Max_bert = 0
                inputs = text
                # print("input devices: ", inputs["input_ids"].device)
                inputs = {k:v.to(model.device) for k,v in inputs.items()}
                import pdb
                # pdb.set_trace()
                generate_ids = model.generate(
                    **inputs,
                    repetition_penalty = repetition_ratio,
                    ngram = ngram,
                    noise_scheme = noise_scheme,
                    n_noise_tokens = n_noise_tokens,
                    noise_sample_topk = noise_sample_topk,
                    mix_ratio = mix_ratio,
                    max_new_tokens=60,
                    do_sample=True,
                    top_k=50
                )

                decoded_text = tokenizer.batch_decode(torch.tensor(model.true_sequence).unsqueeze(dim=0), skip_special_tokens=True)[0]
                print(decoded_text)
                generate_ids = tokenizer_gt(decoded_text, return_tensors="pt")["input_ids"]
                outputs = model_gt(
                    generate_ids,
                    labels = generate_ids,
                )
                loss = outputs.loss
                avg_loss = torch.mean(loss)
                perplexity = math.exp(avg_loss)
                print("perplexity: ", perplexity, flush=True)
                perplexities.append(perplexity)
                print("avg ppl: ", sum(perplexities) / len(perplexities))

                # test repetition ratio
                prompt_len = inputs["input_ids"].shape[1]
                generated_text_ids = model.true_sequence[prompt_len:]
                generated_text = tokenizer.batch_decode(torch.tensor(generated_text_ids).unsqueeze(dim=0), skip_special_tokens=True)[0]
                
                for n in range(1,4):
                    ngram_words = ngrams(generated_text.split(), n)
                    
                    result = collections.Counter(ngram_words)
                    unique_ngrams = len(list(result.keys()))
                    total_ngrams = sum(list(result.values()))
                    try:
                        repeat_ratio = 1 - unique_ngrams / total_ngrams
                    except:
                        print("skipping because occasional bad generation results, setting repetition ratio = 0.1 to accomondate for now")
                        repeat_ratio = 0.1
                    print("repetition ratio: ", repeat_ratio)
                    repetition_list[n] = repetition_list.get(n,list())
                    repetition_list[n].append(repeat_ratio)
                
                # repeated beam search attack
                print("total noised history: ", model.tokenizer.batch_decode(torch.tensor(model.noised_history).unsqueeze(dim=0)))
                recovered_ratios, recovered_texts = model.multiple_beam_searches(
                    input_ids = model.noised_history,
                    ngram = ngram,
                    num_beams = num_beams,
                    n_noise_tokens = n_noise_tokens, 
                )
                for k,v in recovered_ratios.items():
                    true_token_percentages[k].append(v)
                    max_recovered_ratio = max(list(recovered_ratios.values()))
                    max_recovered_ratios.append(max_recovered_ratio)
                
                # one beam search attack, 100 beams for baseline assessment
                if full_test_suite == True:
                    recovered_ratios, recovered_sequences = model.multiple_beam_searches_100beams(
                        input_ids = model.noised_history,
                        ngram = ngram,
                        n_noise_tokens = n_noise_tokens, 
                    )
                    hundred_beam_max_true_ratios.append(max(recovered_ratios))

                    Max_bert = 0
                    prompt_len = inputs["input_ids"].shape[1]
                    reference_text = tokenizer.batch_decode(generate_ids.to(torch.int), skip_special_tokens = True)[0]
                    for bert_idx, recovered_text in enumerate(recovered_sequences):
                        print("recovered text: ", recovered_text)
                        btScore = score([recovered_text], [reference_text], lang="en", rescale_with_baseline=True)[-1]  # get F1 score
                        Max_bert = max(Max_bert, btScore.item())
                    all_max_bertscore_100beams.append(Max_bert)

                    print("100 beam assessment max true ratio: ", max(recovered_ratios))
                    print("100 beam assessment max bertscore: ", Max_bert)

                # bert score
                Max_bert = 0
                prompt_len = inputs["input_ids"].shape[1]
                reference_text = tokenizer.batch_decode(generate_ids.to(torch.int), skip_special_tokens = True)[0]
                for bert_idx, recovered_text in enumerate(recovered_texts):
                    print("recovered text: ", recovered_text)
                    print("reference text: ", reference_text)
                    btScore = score([recovered_text], [reference_text], lang="en", rescale_with_baseline=True)[-1]  # get F1 score
                    bert_scores[bert_idx].append(btScore.item())
                    Max_bert = max(Max_bert, btScore.item())
                    print("bert score: ", btScore, flush=True)
                all_max_bertscore.append(Max_bert)


                # alignment
                m = LogSoftmax(dim=0)
                logits = outputs.logits[0, prompt_len-1:-1, :]
                logits = m(logits)
                labels = generate_ids[0, prompt_len:]
                logProb1 = []
                for idx, tok in enumerate(labels):
                    logProb1.append(logits[idx,tok])  #logProb(gen|prompt)

                generate_ids = torch.cat((torch.tensor([2]).unsqueeze(dim=0),generate_ids[:,prompt_len:]),dim=1)
                outputs = model_gt(
                    generate_ids,
                    labels = generate_ids,
                )
                logits = outputs.logits[0, :-1, :]
                logits = m(logits)
                logProb2 = []
                for idx, tok in enumerate(labels):
                    logProb2.append(logits[idx,tok])  #logProb(gen)

                avgLogProb1 = sum(logProb1) / len(logProb1)
                avgLogProb2 = sum(logProb2) / len(logProb2)
                alignment = avgLogProb1 - avgLogProb2
                alignments[mix_ratio].append(alignment.item())
                print("alignment: ", alignment.item(), flush=True)

                # speed slowdown
                print(f"{ngram} gram n = {n_noise_tokens} avg generation slow down: ", sum(model.generation_time)/len(model.generation_time))

                print("#"*20)
        
        print(f"repetition penalty {repetition_ratio}, avg ppl: ", sum(perplexities) / len(perplexities))
        for k,v in repetition_list.items():
            print(f"ngram={k} avg repetition ratio", sum(v) / len(v))
        for k,v in true_token_percentages.items():
            print(f"mix ratio {mix_ratio} {k}th round recovered: {sum(v)/len(v)}, standard deviation: {statistics.stdev(v)}")
        print(f"mix ratio {mix_ratio} max recovered: {sum(max_recovered_ratios)/len(max_recovered_ratios)}, standard deviation: {statistics.stdev(max_recovered_ratios)}")
        print(f"mix ratio {mix_ratio} avg alignment: {sum(alignments[mix_ratio])/len(alignments[mix_ratio])}, standard deviation: {statistics.stdev(alignments[mix_ratio])}")
        print(f"mix ratio {mix_ratio} avg bertscore first round: {sum(bert_scores[0])/len(bert_scores[0])}, standard deviation: {statistics.stdev(bert_scores[0])}")
        print(f"mix ratio {mix_ratio} avg of max bertscore: {sum(all_max_bertscore)/len(all_max_bertscore)}, standard deviation: {statistics.stdev(all_max_bertscore)}")
        if full_test_suite == True:
            print(f"mix ratio {mix_ratio} 100 beams max true ratio: {max(hundred_beam_max_true_ratios)}")
            print(f"mix ratio {mix_ratio} 100 beams avg max true ratio: {sum(hundred_beam_max_true_ratios)/len(hundred_beam_max_true_ratios)}, standard deviation: {statistics.stdev(hundred_beam_max_true_ratios)}")
            print(f"mix ratio {mix_ratio} 100 beams max bertscore: {max(all_max_bertscore_100beams)}")
            print(f"mix ratio {mix_ratio} 100 beams avg max bertscore: {sum(all_max_bertscore_100beams)/len(all_max_bertscore_100beams)}, standard deviation: {statistics.stdev(all_max_bertscore_100beams)}")
