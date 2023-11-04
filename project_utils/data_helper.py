import os
import math
data_dir="./data"
dataname="toutiao_cat_data.txt"
###这个文件用一次就行了，主要工作是进行数据集划分
#将数据进行划分，以字典形式进行返回{id:,label:,title:,keywords:}
def get_data_dic(data_dir,dataname):
    dataset_dir=os.path.join(data_dir,dataname)
    with open(dataset_dir,"r",encoding="utf-8") as f:
        lines=f.readlines()

    listed_data=[]
    for item in lines:
        splited_item=item.split("_!_")
        dict_data={}
        dict_data["classid"]=splited_item[1]
        dict_data["classname"]=splited_item[2]
        dict_data["title"]=splited_item[3]
        dict_data["keywords"]=splited_item[4]
        listed_data.append(dict_data)
    return listed_data

#将数据集进行划分，并存入对应的文件中
def split_data(data_dir,dataname):
    dataset_dir=os.path.join(data_dir,dataname)
    train_data_name="toutiao_train.txt"
    test_data_name="toutiao_test.txt"
    val_data_name="toutiao_val.txt"
    with open(dataset_dir,"r",encoding="utf-8") as f:
        lines=f.readlines()
    nums_all=len(lines)
    print(nums_all)
    nums_train_dataset=math.floor(nums_all*0.7)
    print(nums_train_dataset)
    nums_val_dataset=math.floor(nums_all*0.2)
    print(nums_val_dataset)
    nums_test_dataset=nums_all-nums_train_dataset-nums_val_dataset
    
    with open(os.path.join(data_dir,train_data_name),"w",encoding="utf-8") as f:
        f.writelines(lines[0:nums_train_dataset-1])
    with open(os.path.join(data_dir,val_data_name),"w",encoding="utf-8") as f:
        f.writelines(lines[nums_train_dataset:nums_train_dataset+nums_val_dataset-1])
    with open(os.path.join(data_dir,test_data_name),"w",encoding="utf-8") as f:
        f.writelines(lines[nums_train_dataset+nums_val_dataset:])


    
###test
#listed_datas=get_data_dir(data_dir,dataname)

#if __name__=="__main__":
    #split_data(data_dir,dataname)