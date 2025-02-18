import os
os.environ['CUDA_VISIBLE_DEVICES']="1,3,5,6,7"
import transformers
import torch
import pandas as pd
import numpy as np
import accelerate
import bitsandbytes

import re
import csv
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import (LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel)
pattern = r'\[(.*?)\]'
feature='plot_keywords'
def is_decimal(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
def get_text(path,attr):

    # load raw data, delete movies with no actor or director
    movies = pd.read_csv(path+'/movie_metadata.csv', encoding='utf-8').dropna(
        axis=0, subset=['actor_1_name', 'director_name']).reset_index(drop=True)

    # extract labels, and delete movies with unwanted genres
    # 0 for action, 1 for comedy, 2 for drama, -1 for others
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

    # build the adjacency matrix for the graph consisting of movies, directors and actors
    # 0 for movies, 1 for directors, 2 for actors
    dim = len(movies) + len(directors) + len(actors)
    type_mask = np.zeros((dim), dtype=int)
    type_mask[len(movies):len(movies)+len(directors)] = 1
    type_mask[len(movies)+len(directors):] = 2
    print(len(movies) ,len(directors) , len(actors))

    node_text_lst = []
    label_text_lst = []
    for node in range(0,len(movies)):#['movie_title','plot_keywords']
        if isinstance(attr,list):
            nodet=str(movies[attr].loc[node].to_dict())
        else:
            nodet=str(movies[[attr]].loc[node].to_dict())
        node_text_lst.append(nodet)
    node_text_lst=directors
    return node_text_lst
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
node_text_lst=get_text('imdb',feature)

pipeline = transformers.pipeline(#meta-llama/Meta-Llama-3-8B-Instruct
  "text-generation",
  model=model_id,
  model_kwargs={"torch_dtype": torch.bfloat16,
                "device_map": 'auto',
                "load_in_8bit" : False,
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
feature='director'
#f=open('llm_answer_'+feature+'.csv', 'w',newline='')
f1=open('./imdb/new_answer1_'+feature+'.txt', 'w')
f2=open('./imdb/new_answer2_'+feature+'.txt', 'w')
f3=open('./imdb/new_answer3_'+feature+'.txt', 'w')
f4=open('./imdb/new_answer4_'+feature+'.txt', 'w')
f5=open('./imdb/new_answer5_'+feature+'.txt', 'w')
f6=open('./imdb/new_answer6_'+feature+'.txt', 'w')
e=open('./imdb/error_'+feature+'.txt', 'w')
for i in range(0,len(node_text_lst)):
    times=0
    while True:
        print(node_text_lst[i])
        messages = [
            {"role": "system", "content": """The following is something of movies. I have 4 questions for you. Please provide your answers in the following format: [Answer 1], [Answer 2], [Answer 3], [Answer 4]. Each answer should be enclosed in its respective brackets.
        1.Describe the given content in coherent language.
        2.What kind of feature it is?there are 3 types in total," movie directors" or " movie actors" or "movie plot keywords".
        3.explain why you give your result in question 2?
        4.How confident are you in your evaluation?use a decimal number between 0 and 1 to answer.such as "0.9".just show the number.
        """},
            {"role": "user", "content": "this is content:\""+node_text_lst[i]+"\""},
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
        matches = re.findall(pattern, res)
        #print(matches)
        
        if len(matches)!=4:
            error.append(res)
            print("smddx "+str(res))
        elif is_decimal(matches[3])==False:
            error.append(res)
            print("smddx "+str(res))
        elif len(matches[0])<10 and len(matches[2])<10:
            error.append(res)
            print("smddx "+str(res))
        else:
            answer1.append(matches[0])
            answer2.append(matches[1])
            answer3.append(matches[2])
            answer4.append(matches[3])
            break
        if times>=15:
            answer1.append("unkown")
            answer2.append("unkown")
            answer3.append("unkown")
            answer4.append("1.0")
            #answer6.append("1.0")
            break
        times+=1

for i in answer1:
    f1.write(str(i)+"\n")
for i in answer2:
    f2.write(str(i)+"\n")
for i in answer3:
    f3.write(str(i)+"\n")
for i in answer4:
    f4.write(str(i)+"\n")

for i in error:
    e.write(str(i)+"\n")
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()

e.close()
