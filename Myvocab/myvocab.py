class Vocab(object):
    UNK="[UNK]"
    def __init__(self,config):
        super(Vocab,self).__init__()
        self.word2id={}
        self.id2word=[]
        self.vocab_path=config.vocab_path
        self.process_vocab()
    
    def process_vocab(self):
        with open(self.vocab_path,"r",encoding="utf-8") as f:
            lines=f.readlines()
        for id,line in enumerate(lines):##可以优化for i,word in enumerate(f):
            word=line.rstrip("\n")#去除结尾的"\n"
            self.word2id[word]=id
            self.id2word.append(word)

    def __len__(self):
        return len(self.id2word)
    def __getitem__(self,word):
        #self.word2id.get(Vocab.UNK)返回100
        return self.word2id.get(word,self.word2id.get(Vocab.UNK))

"""
if __name__=="__main__":
    vocab_path="./vocab/vocab.txt"
    vocab=Vocab(vocab_path)
    print(vocab["UNK"])
"""
