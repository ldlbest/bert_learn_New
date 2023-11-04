import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dataclasses import dataclass, field
from transformers import HfArgumentParser,Trainer,TrainingArguments,BertTokenizer
from Myvocab import myvocab
from dataset.mydata import MyDataSet,generate_batach
from task.Bert_For_Sentence_Classificiation import BertForSentenceClassification


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="./bert_base_chinese",metadata={"help":"预训练模型的路径"})

@dataclass
class BertArguments:
    vocab_size: int = field(default=21128,metadata={"help":"词表大小"})
    hidden_size: int = field(default=768,metadata={"help":"模型的隐藏层大小"})
    num_hidden_layers: int = field(default=12,metadata={"help":"模型的隐藏层数"})
    num_attention_heads: int = field(default=12,metadata={"help":"模型的attention head的数量"})
    intermediate_size: int =field(default=3072,metadata={"help":"模型的中间层大小"})
    pad_token_id: int = field(default=0,metadata={"help":"pad token的id"}) 
    hidden_act: str = field(default="gelu",metadata={"help":"激活函数"})
    hidden_dropout_prob: float = field(default=0.1,metadata={"help":"隐藏层的dropout"})
    attention_probs_dropout_prob: float = field(default=0.1,metadata={"help":"attention层的dropout"})
    max_position_embeddings: int = field(default=512,metadata={"help":"最大的位置嵌入长度"})
    type_vocab_size: int = field(default=2,metadata={"help":"句子类型的数量"})
    initializer_range: float = field(default=0.02,metadata={"help":"初始化的范围"})
    pooler_type : str = field(default="first_token_transform",metadata={"help":"pooler的类型"})

@dataclass
class DataArguments:
    data_dir: str = field(default="./data",metadata={"help":"数据集的路径"})
    train_data_name: str = field(default="toutiao_train.txt",metadata={"help":"数据集的名称"})
    val_data_name: str = field(default="toutiao_val.txt",metadata={"help":"数据集的名称"})
    test_data_name: str = field(default="toutiao_test.txt",metadata={"help":"数据集的名称"})
    CLS_IDX : str=field(default="[CLS]",metadata={"help":"[CLS]的id"})
    SEP_IDX : str=field(default="[SEP]",metadata={"help":"[SEP]的id"})

@dataclass
class TrainArguments(TrainingArguments):
    output_dir: str = field(default="",metadata={"help":"输出的路径"})
    overwrite_output_dir: bool = field(default=True,metadata={"help":"是否覆盖输出路径"}) 
    num_train_epochs: int = field(default=3,metadata={"help":"训练的轮数"})
    use_mps_device: int = field(default=0,metadata={"help":"是否使用mps"}) 
    per_device_train_batch_size: int = field(default=32,metadata={"help":"每个设备的batch_size"})
    warmup_steps: int = field(default=500,metadata={"help":"warmup的步数"})
    weight_decay: float = field(default=0.01,metadata={"help":"权重衰减"})
    logging_dir: str = field(default="./logs",metadata={"help":"日志的路径"})
    logging_steps: int = field(default=10,metadata={"help":"日志的步数"})
    save_steps: int = field(default=10,metadata={"help":"保存的步数"})
    metric_for_best_model: str = field(default="accuracy",metadata={"help":"最好模型的指标"})
    greater_is_better: bool = field(default=True,metadata={"help":"是否越大越好"})
    save_total_limit: int = field(default=3,metadata={"help":"保存的模型数量"})
    learning_rate: float = field(default=5e-5,metadata={"help":"学习率"})
@dataclass
class VocabArguments:
    vocab_path: str = field(default="./Myvocab/vocab.txt",metadata={"help":"词表的路径"})


if __name__=="__main__":
    parser=HfArgumentParser(
        (ModelArguments,BertArguments,DataArguments,TrainArguments,VocabArguments)
    )
    (model_args,bert_args,data_args,train_args,vocab_args)=parser.parse_args_into_dataclasses()
    model=BertForSentenceClassification(config=bert_args)
    tokenizer=BertTokenizer.from_pretrained("/home/lidailin/pretrained_models/bert_case_chinese").tokenize
    vocab=myvocab.Vocab(vocab_args)
    dataset=MyDataSet(tokenize=tokenizer,Vocab=vocab,config=data_args)
    trainer=Trainer(model=model,args=train_args,train_dataset=dataset,data_collator=generate_batach)
    trainer.train()
    


