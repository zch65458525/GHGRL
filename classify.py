import transformers
import torch
import pandas as pd
import numpy as np
import accelerate
import bitsandbytes
import os
import re
import random
import csv
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import (LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel)
pattern = r'\[(.*?)\]'
feature='plot_keywords'
def get_text(path,attr):

    # load raw data, delete movies with no actor or director
    movies = pd.read_csv(path+'/movie_metadata.csv', encoding='utf-8').dropna(
        axis=0, subset=['actor_1_name', 'director_name']).reset_index(drop=True)
    labels = np.zeros((len(movies)), dtype=int)
    for movie_idx, genres in movies['genres'].items():
        labels[movie_idx] = -1
        for genre in genres.split('|'):
            if genre == 'Action':
                labels[movie_idx] = 0
                break
            elif genre == 'Comedy':
                labels[movie_idx] = 1
                break
            elif genre == 'Drama':
                labels[movie_idx] = 2
                break
    unwanted_idx = np.where(labels == -1)[0]
    movies = movies.drop(unwanted_idx).reset_index(drop=True)
    labels = np.delete(labels, unwanted_idx, 0)
    directors = list(set(movies['director_name'].dropna()))
    directors.sort()
    actors = list(set(movies['actor_1_name'].dropna().to_list() +
                    movies['actor_2_name'].dropna().to_list() +
                    movies['actor_3_name'].dropna().to_list()))
    actors.sort()
    dim = len(movies) + len(directors) + len(actors)
    type_mask = np.zeros((dim), dtype=int)
    type_mask[len(movies):len(movies)+len(directors)] = 1
    type_mask[len(movies)+len(directors):] = 2
    print(len(movies) ,len(directors) , len(actors))

    node_text_lst = []
    label_text_lst = []
    for node in range(0,len(movies)):#['movie_title','plot_keywords']
        if isinstance(attr,list):
            nodet=movies[attr].loc[node].to_dict()
        else:
            nodet=movies[[attr]].loc[node].to_dict()
        node_text_lst.append(nodet)
    #node_text_lst=actors
    random_sample = random.sample(node_text_lst, 5)
    for i in (0,2,4):
        tmp=""
        for k,v in random_sample[i].items():
                tmp+=str(k)+":"+str(v)
        random_sample[i]=tmp
    random_sample=[str(x) for x in random_sample]
    random_sample += random.sample(actors, 5)
    random_sample += random.sample(directors, 5)
    return random_sample
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
node_text_lst=get_text('imdb',feature)

pipeline = transformers.pipeline(#meta-llama/Meta-Llama-3-8B-Instruct
  "text-generation",
  model=model_id,
  model_kwargs={"torch_dtype": torch.bfloat16,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
        },
token="",
)
answer1=[]
answer2=[]
answer3=[]
answer4=[]
answer5=[]
answer6=[]
answer=[]
error=[]
feature='movie'

f1=open('./imdb/imdbclass.txt', 'w')

times=0
tt=""
for i in range(0,len(node_text_lst)):
    tt+="node"+str(i)+":"+node_text_lst[i]+"\n"
print(tt)
while True:
        #print(node_text_lst[i])
        messages = [
            {"role": "system", "content": """The following content is a description of some nodes in a heterogeneous graph with three types of nodes. 
            the description is IMDB data about movies.answer 2 questions of the content and format classification.
            1.format classify:distinguish the format of data into 2 types,for example,text or json or html or matrix.
            2.content classify:in movies,director and actors are different kinds of feature.if you think it is movie,tell me the content is what kind of feature of the movie,the feature can be "movie plot keyword","movie language"...Please distinguish which 3 types of nodes can be classified.
            The answer should be like this:[format_type1,format_type2],[content_type1,content_type2, content_type3]
        """},
            {"role": "user", "content": tt},
        ]

        prompt = pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
        )
        res=outputs[0]["generated_text"][len(prompt):]
        print(res)
        


        answer1.append(res)
        if times>=10:
            break
        times+=1

for i in answer1:
    f1.write(str(i)+"\n\n")
f1.close()

