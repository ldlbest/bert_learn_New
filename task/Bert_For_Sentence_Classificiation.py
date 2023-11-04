import torch
import torch.nn as nn
from model.Bert import BertModel
from transformers import BertTokenizer
class BertForSentenceClassification(nn.Module):
    def __init__(self,config,num_labels=17):
        super(BertForSentenceClassification,self).__init__()
        self.bert=BertModel(config=config)
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.fc=nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size),
                              nn.ReLU(),
                              nn.Linear(config.hidden_size,num_labels)
                              )
        self.num_labels=num_labels
    def forward(self, input_ids,label_ids,attention_mask=None):
        """

        :param input_ids:  [src_len, batch_size]
        :param attention_mask: [batch_size, src_len]
        :param token_type_ids: 句子分类时为None
        :param position_ids: [1,src_len]
        :param labels: [batch_size,]
        :return:
        """
        pooled_output, _ = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=None,
                                     position_ids=None)  # [batch_size,hidden_size]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)  # [batch_size, num_label]
        if label_ids is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
            return (loss, logits)
        else:
            return logits
    