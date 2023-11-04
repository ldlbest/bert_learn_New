import copy,os,torch

def para_state_dict(model,model_save_dir):
    state_dict=copy.deepcopy(model.state_dict())#拷贝网络中原有的参数
    model_save_dir=os.path.join(model_save_dir,"model.pt")
    if os.path.exists(model_save_dir):
        loaded_paras=torch.load(model_save_dir)
        for key in state_dict:
            if key in loaded_paras and state_dict[key].size()==loaded_paras[key].size():
                print("成功初始化参数.key")
                state_dict[key]=loaded_paras[key]
    return state_dict

###test
###state_dict=para_state_dict(self.model,self.model_save_dir)
###self.model.load_state_dict(state_dict)