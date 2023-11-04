from task.Bert_For_Sentence_Classificiation import BertForSentenceClassification
import torch
import os
from project_utils.log_helper import logger_init
import logging
from config.BertConfig import BertConfig
from transformers import BertTokenizer
from Myvocab.myvocab import Vocab
from dataset.mydata import MyDataSet,generate_batach
from torch.utils.data import DataLoader
import time
"""
自己写的训练代码，和trainer的效果一比差太多了，今后不要自己写训练代码
"""

def train(config):
    model=BertForSentenceClassification(config=config)
    model.train()
    model_save_path="./cache/bert_model.bin"
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")

    tokenizer=BertTokenizer.from_pretrained("/home/lidailin/pretrained_models/bert_case_chinese").tokenize#只分词
    vocab_path="./Myvocab/vocab.txt"
    vocab=Vocab(vocab_path)
    dataset=MyDataSet(tokenize=tokenizer,Vocab=vocab,data_dir="./data",data_name="toutiao_train.txt",max_positon_embedding=512)
    train_dataloader=DataLoader(dataset,batch_size=32,shuffle=True,num_workers=8,collate_fn=generate_batach)
    max_acc=0
    for epoch in range(50):
        losses=0
        start_time=0
        for idx,(sample,label) in enumerate(train_dataloader):
            sample=sample.to("cuda")
            label=label.to("cuda")
            optimizer=torch.optim.Adam(model.parameters(),lr=5e-5)
            padding_mask=(sample == 0).transpose(0, 1)
            mode=model.to("cuda")
            loss, logits = model(
                input_ids=sample,
                attention_mask=padding_mask,
                token_type_ids=None,
                position_ids=None,
                labels=label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            acc = (logits.argmax(1) == label).float().mean()
            if idx % 10 == 0:
                print(f"Epoch: {epoch}, Batch[{idx}/{len(train_dataloader)}], "
                             f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
                logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_dataloader)}], "
                             f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
        end_time = time.time()

        train_loss = losses / len(train_dataloader)
        logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
    torch.save(model.state_dict(), model_save_path)

def evaluate(data_iter, model, device, PAD_IDX):
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            padding_mask = (x == PAD_IDX).transpose(0, 1)
            logits = model(x, attention_mask=padding_mask)
            acc_sum += (logits.argmax(1) == y).float().sum().item()
            n += len(y)
        model.train()
        return acc_sum / n
    


if __name__=="__main__":
    config = BertConfig.from_json_file("./config/config.json")
    train(config=config)
    #bert_model=BertForSentenceClassification(config=config)
    #bert_model.load_state_dict("./cache")
    #print(bert_model)