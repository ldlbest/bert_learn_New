import sys
import os
import torch 
sys.path.append(os.getcwd())#必须要加工作目录
from torch.utils.data import Dataset
from project_utils.data_helper import get_data_dic
class MyDataSet(Dataset):
    def __init__(self,tokenize,Vocab,config,max_positon_embedding=512) -> None:
        super().__init__()
        self.data_dir=config.data_dir
        self.train_data_name=config.train_data_name
        self.tokenize=tokenize
        self.vocab=Vocab
        self.CLS_IDX=self.vocab[config.CLS_IDX]
        self.SEP_IDX=self.vocab[config.SEP_IDX]
        self.data=[]
        self.label=[]
        ######{id:,label:,title:,keywords:}
        self.max_position_embeddings=max_positon_embedding

        self.feature=[]
        self.data_process()

    def tokenize_and_pre(self, sent, tokenizer, CLS_IDX, SEP_IDX, max_position_embeddings):
        token = tokenizer(sent)  # ['今','天','几','号'],需要处理成[CLS]+[今天几号]+[SEP]
        token = [CLS_IDX] + [self.vocab[token] for token in token]  # [100,101,232,5,102]
        if len(token) > max_position_embeddings - 1:  # 如果长度大于最大长度，就截断
            token = token[:max_position_embeddings - 1]
        token += [SEP_IDX]  # 补上结尾符
        return token
    
    def data_process(self):
        data = get_data_dic(data_dir=self.data_dir, dataname=self.train_data_name)

        for item in data:
            self.label.append(int(item["classid"])-100)
            item = self.tokenize_and_pre(tokenizer=self.tokenize, sent=item["title"], CLS_IDX=self.CLS_IDX,
                                         SEP_IDX=self.SEP_IDX, max_position_embeddings=self.max_position_embeddings)
            self.data.append(item)  # 插入已经tokenize好的数据


    def __getitem__(self, index):##一定这样写，这样写是最快的，把其他数据处理方式一定不要加载这里

        return self.data[index],self.label[index] ##数据格式是列表，没进行padding
    
    def __len__(self):
        return len(self.data)

def pad_sequence(sequence,batch_first=False,max_len=None,padd_value=0):
    if max_len==None:
        max_len=max([len(seq) for seq in sequence]) #一个batch中的最大长度
    out_tensors=[]
    for tensor in sequence:
        if tensor.size(0)< max_len:
            tensor=torch.cat([tensor,torch.tensor([padd_value]*(max_len-tensor.size(0)))],dim=0)
        else:
            tensor=tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors=torch.stack(out_tensors,dim=1)
    if batch_first:
        return out_tensors.transpose(0,1)
    return out_tensors

def generate_batach(data_bacth,PAD_IDX=0):
    #print(data_bacth)data_batch里的元素是tuple,tuple的个数是batch_size
    batch_sentence,batch_label=[],[]
    for (sen,label) in data_bacth:
        batch_sentence.append(torch.tensor(sen,dtype=torch.long))
        batch_label.append(label)
    batch_sentence=pad_sequence(batch_sentence,batch_first=False,padd_value=PAD_IDX)
    batch_label=torch.tensor(batch_label,dtype=torch.long)
    padding_mask=(batch_sentence == 0).transpose(0, 1)
    return {"input_ids":batch_sentence,"label_ids":batch_label}
