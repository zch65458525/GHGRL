import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid,IMDB
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv,SAGEConv,GATConv,Linear
import torch.nn as nn
import torch.optim as optim
import transformers
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from sklearn.utils import shuffle
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer
from transformers import (LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel) 
ENCODER_DIM_DICT = {"ST": 768}
device_ids = [0, 1,2,3,4,5,6,7]
class SentenceEncoder:
    def __init__(self, name, root="cache_data/model", batch_size=1, multi_gpu=False):
        self.name = name
        self.root = root
        self.device = torch.device('cuda:7')
        self.batch_size = batch_size
        self.multi_gpu = multi_gpu
        self.model = SentenceTransformer("multi-qa-distilbert-cos-v1", device=self.device, cache_folder=self.root, )
        self.encode = self.ST_encode
    def ST_encode(self, texts, to_tensor=True):
        if self.multi_gpu:
            # Start the multi-process pool on all available CUDA devices
            pool = self.model.start_multi_process_pool()
            embeddings = self.model.encode_multi_process(texts, pool=pool, batch_size=self.batch_size, )
            embeddings = torch.from_numpy(embeddings)
        else:
            embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=True,
                convert_to_tensor=to_tensor, convert_to_numpy=not to_tensor, )
        return embeddings
    def llama_encode(self, texts, to_tensor=True):

        # Add EOS token for padding
        self.tokenizer.pad_token = self.tokenizer.eos_token
        all_embeddings = []
        with torch.no_grad(): 
            for start_index in trange(0, len(texts), self.batch_size, desc="Batches", disable=False, ):
                sentences_batch = texts[start_index: start_index + self.batch_size]
                input_ids = self.tokenizer(sentences_batch, return_tensors="pt", padding="longest", truncation=True,
                                           max_length=500).input_ids.to(self.device)
                transformer_output = self.model(input_ids, return_dict=True, output_hidden_states=True)["hidden_states"]
                # No gradients on word_embeddings
                word_embeddings = transformer_output[-1].detach()
                sentence_embeddings = word_embeddings.mean(dim=1)
                all_embeddings.append(sentence_embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        if not to_tensor:
            all_embeddings = all_embeddings.numpy()

        return all_embeddings
tau=0.7
tau1=0

device=torch.device('cpu')
# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class GCN1(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        print(hidden_channels)
        self.conv1 = GCNConv(hidden_channels, out_channels,add_self_loops=False) # -1表示延迟初始化
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        return x

class CustomModel(nn.Module):
    def __init__(self, input_size1, hidden_size, output_size1, output_size2, gcn_hidden,gcn_hidden2, gcn_output):
        super(CustomModel, self).__init__()
        self.mlp = MLP(input_size1, hidden_size, output_size1)
        self.weight_matrix1 = nn.Parameter(torch.randn(3, 768,256))
        self.weight_matrix12 = nn.Parameter(torch.ones(3, 256))
        self.weight_matrix13 = nn.Parameter(torch.randn(768, 256))
        self.b1=nn.Parameter(torch.zeros(3,768))
        self.gcn1 = GCN1(768,256)
        self.weight_matrix2 = nn.Parameter(torch.randn(3,256,256))
        self.b2=nn.Parameter(torch.zeros(3,256))
        self.gcn2 = GCN1(256,32)
        self.fc = nn.Linear(32, gcn_output)
    def forward(self, data):
        x1 = data.x1  # 节点特征1
        x2 = data.x2  # 节点特征2\
        x3=data.x3
        edge_index=data.edge_index

        
        mlp_output = self.mlp(x1)
        one_hot_positions = torch.argmax(mlp_output, dim=1)
        one_hot_matrix = torch.zeros_like(mlp_output).scatter_(1, one_hot_positions.unsqueeze(1), 1)
        result_matrix = one_hot_matrix.clone()
        
        change_amount = one_hot_matrix[torch.arange(mlp_output.size(0)), one_hot_positions] * (1-tau)
        result_matrix[torch.arange(mlp_output.size(0)), one_hot_positions] *= tau
        
        change_amount_per_other_position = change_amount / 2
        mask = one_hot_matrix == 0
        result_matrix[mask] += change_amount_per_other_position.unsqueeze(1).repeat(1, 2).flatten()
        mlp_output=result_matrix
        mlp_output=mlp_output.to(device)
        weighted_output1 = torch.einsum('ij,jkl->ikl', mlp_output, self.weight_matrix1)
        weighted_output2 = torch.einsum('ij,jkl->ikl', mlp_output, self.weight_matrix2)
        weight_matrix12 = torch.einsum('ij,jk->ik', mlp_output, self.weight_matrix12)
        b1=torch.einsum('ij,jk->ik', mlp_output, self.b1)
        b2=torch.einsum('ij,jk->ik', mlp_output, self.b2)
        result1 = weighted_output1*x2.unsqueeze(-1)
        result1 = result1.sum(dim=-1)+b1
        gcn_output = self.gcn1(result1, edge_index)
        gcn_output=(weight_matrix12*gcn_output)+0.0002*result1@self.weight_matrix13
        result2 = weighted_output2*gcn_output.unsqueeze(-1)
        result2 = result2.sum(dim=-1)+b2
        gcn_output = self.gcn2(result2, edge_index)
        final_output = self.fc(gcn_output)
        return F.log_softmax(final_output, dim=1)




# 训练和评估模型
def train_custom_model(data,input_size1,input_size2):
    hidden_size = 256
    output_size1 = 3
    output_size2 = input_size2
    gcn_hidden = 32
    gcn_hidden2 = 16
    gcn_output = 3
    num_epochs = 300
    learning_rate = 0.001

    #device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    model = CustomModel(input_size1, hidden_size, output_size1, output_size2, gcn_hidden,gcn_hidden2, gcn_output).to(device)
    data.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2e-4)


    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        pred = out.max(dim=1)[1]
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
        accuracy = correct / data.test_mask.sum().item()
        precision, recall, f1, _ = precision_recall_fscore_support(data.y[data.test_mask].to('cpu'), pred[data.test_mask].to('cpu'), average='micro')
        precision, recall, f2, _ = precision_recall_fscore_support(data.y[data.test_mask].to('cpu'), pred[data.test_mask].to('cpu'), average='macro')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f},Accuracy: {accuracy:.4f},micro-f1: {f1:.4f},macro-f1: {f2:.4f}')

    model.eval()
    with torch.no_grad():
        pred = model(data).max(dim=1)[1]
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
        accuracy = correct / data.test_mask.sum().item()
        print(f'Accuracy: {accuracy:.4f}')

def sample_mask(idx, l):
    """Create mask."""
    mask = torch.zeros(l)
    mask[idx] = 1
    return torch.as_tensor(mask, dtype=torch.bool)

if __name__ == "__main__":
    encoder=SentenceEncoder('imdb')
    dataset = IMDB(root='./imdb')
    seed=torch.initial_seed()
    print(seed)
    data = dataset[0]
    y=data['movie'].y
    sample_number = len(y)
    seed = 5
    shuffled_idx = shuffle(np.array(range(len(y))), random_state=seed) # 已经被随机打乱
    train_idx = shuffled_idx[:int(0.8* y.shape[0])].tolist()
    test_idx = shuffled_idx[int(0.8*y.shape[0]):].tolist()
    train_mask = sample_mask(train_idx, sample_number)
    test_mask = sample_mask(test_idx, sample_number)
    data['movie'].train_mask=train_mask
    data['movie'].test_mask=test_mask
    data=data.to_homogeneous()
    print(encoder.name)
    feature1=[]
    with open('./imdb/new_answer2_movie.txt','r') as f:
        feature1=list(f.readlines())
    feature2=[]
    with open('./imdb/new_answer3_movie.txt','r') as f:
        feature2=list(f.readlines())
    feature1=[feature1[i]+feature2[i] for i in range(0,len(feature1))]
    embeddings1 = encoder.encode(feature1).cpu()
    feature2=[]
    with open('./imdb/new_answer1_movie.txt','r') as f:
        feature2=list(f.readlines())
    feature3=[]
    with open('./imdb/new_answer4_movie.txt','r') as f:
        feature3=list(f.readlines())
    embeddings3 = [float(i) for i in feature3]
    print(data)
    # 生成自定义的节点特征1和节点特征2
    num_nodes = data.num_nodes
    input_size1 = embeddings1.shape[1]  # 节点特征1的维度
    input_size2 = embeddings1.shape[1]  # 节点特征2的维度

    feature1=[]
    with open('./imdb/new_answer2_director.txt','r') as f:
        feature1=list(f.readlines())
    feature2=[]
    with open('./imdb/new_answer3_director.txt','r') as f:
        feature2=list(f.readlines())
    feature1=[feature1[i]+feature2[i] for i in range(0,len(feature1))]
    embeddings11 = encoder.encode(feature1).cpu()
    feature2=[]
    with open('./imdb/new_answer1_director.txt','r') as f:
        feature2=list(f.readlines())
    feature3=[]
    with open('./imdb/new_answer4_director.txt','r') as f:
        feature3=list(f.readlines())
    embeddings31 = [float(i) for i in feature3]
    num_nodes = data.num_nodes
    input_size1 = embeddings1.shape[1]  # 节点特征1的维度
    input_size2 = embeddings1.shape[1]  # 节点特征2的维度
    feature1=[]
    with open('./imdb/new_answer2_actor.txt','r') as f:
        feature1=list(f.readlines())
    feature2=[]
    with open('./imdb/new_answer3_actor.txt','r') as f:
        feature2=list(f.readlines())
    feature1=[feature1[i]+feature2[i] for i in range(0,len(feature1))]
    embeddings12 = encoder.encode(feature1).cpu()
    feature2=[]
    with open('./imdb/new_answer1_actor.txt','r') as f:
        feature2=list(f.readlines())
    feature3=[]
    with open('./imdb/new_answer4_actor.txt','r') as f:
        feature3=list(f.readlines())
    embeddings32 = [float(i) for i in feature3]
    feature1=[]
    with open('./imdb/enhanced_answer1_imdb.txt','r') as f:
        feature1=list(f.readlines())
    embeddings121 = encoder.encode(feature1).cpu()
    print(data)
    # 生成自定义的节点特征1和节点特征2
    num_nodes = data.num_nodes
    input_size1 = embeddings1.shape[1]  # 节点特征1的维度
    input_size2 = embeddings1.shape[1]  # 节点特征2的维度'
    data.x1=torch.cat((embeddings1,embeddings11,embeddings12),dim=0)
    data.x2=embeddings121
    data.x3=torch.tensor(embeddings3+embeddings31+embeddings32)
    train_custom_model(data,input_size1,input_size2)