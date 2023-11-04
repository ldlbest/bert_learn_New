from model.Bert import BertModel
from config.BertConfig import BertConfig
config_path="./config/config.json"
config=BertConfig.from_json_file(config_path)
model=BertModel(config=config)
print("\n ====BertModel 参数：====")
print(len(model.state_dict()))#层数是199
for para_tensor in model.state_dict():
    print(para_tensor, "\t", model.state_dict()[para_tensor].size())#最后一层的输入出是768*768
