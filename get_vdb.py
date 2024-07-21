import vdblite
from transformers import OPTModel, AutoTokenizer
import numpy as np
import pickle as pk
from transformers import AutoModelForCausalLM

# model = OPTModel.from_pretrained("facebook/opt-1.3b")
# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("base/llama/base_vanilla_model_hf")
tokenizer = AutoTokenizer.from_pretrained("base/llama/base_vanilla_tokenizer_hf")
vdb = vdblite.Vdb()

embeddings = model.get_input_embeddings().weight
embeddings.require_grad = False
for idx in range(embeddings.shape[0]):
    info = {"vector":embeddings[idx].detach().numpy(), "uuid": idx}
    vdb.add(info)
    # print("add: ", embeddings[idx])


test_uuid = 1234

print(embeddings[test_uuid].detach().numpy())
print(tokenizer.decode([test_uuid]))
top = vdb.search(embeddings[test_uuid].detach().numpy(), count = 5)
for i in top:
    uuid = int(i["uuid"])
    print(tokenizer.decode([uuid]))

db = {}
for idx in range(embeddings.shape[0]):
    top = vdb.search(embeddings[idx].detach().numpy(), count = 20)
    db[idx] = list()
    for i in top:
        uuid = int(i['uuid'])
        db[idx].append(uuid)
    if idx % 1000 == 0: print(idx)


with open('vdbs/llama_vdb_extended.pickle', 'wb') as handle:
    pk.dump(db, handle)
    

'''
import vdblite
from time import time
from uuid import uuid4
import sys
from pprint import pprint as pp


if __name__ == '__main__':
    vdb = vdblite.Vdb()
    dimension = 12    # dimensions of each vector                         
    n = 200    # number of vectors                   
    np.random.seed(1)             
    db_vectors = np.random.random((n, dimension)).astype('float32')
    print(db_vectors[0])
    for vector in db_vectors:
        info = {'vector': vector, 'time': time(), 'uuid': str(uuid4())}
        vdb.add(info)
    vdb.details()
    results = vdb.search(db_vectors[10])
    pp(results)
'''
