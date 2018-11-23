import spacy
import re
import ast
nlp = spacy.load('en_core_web_md')

import numpy as np

en_corpus = np.load('./dataset/translate/enCorpus.npy')
en_vocab = np.load('./dataset/translate/enVocab.npy').tolist() # use tolist() to transform back to dict()
en_rev = np.load('./dataset/translate/enRev.npy').tolist()


# load lines dictionary 

# load conversations
convs = open('dataset/chatbot/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
convs = [ str(i.replace("+++$+++ ",'')) for i in convs]
convlist = [ re.sub('.+m[0-9]+ ','',i)  for i in convs]
convlist = [ ast.literal_eval(i) for i in convlist[:-1]]

talk=[]
for j in convlist:
    for i  in range(len(j)-1):
        talk.append( j[i:i+2] )
        
for i in talk:
    if len(i)!=2:
        print("fuck")
# =============================split talk =================
        
        
lines = open('dataset/chatbot/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n') 
lines = [ str(i.replace("+++$+++ ",'')) for i in lines]
lines = [ re.sub(' u[0-9]+ m[0-9]+ [A-Z]+','',i).split(" ",1)  for i in lines]   
idict={}
for i in lines[:-1]:
    idict[i[0]]=i[1]
    
#===========================id sentence dict ===============

realtalk=[]
for i in talk:
    try:
        realtalk.append( [idict[i[0]],idict[i[1]]])
    except:
        pass



# ==========================================================
      

    
    
    
class BatchGenerator1:
    def __init__(self, dat, batch_size):

        self.batch_xs, self.batch_ys, self.reviews = [], [], []

        self.batch_x ,self.batch_y=[],[]
        
        for i in realtalk:
            if batch_size != 1:
                self.batch_x.append( i[0].lower())
                self.batch_y.append( i[1].lower())
            elif batch_size==1:
                self.batch_x=i[0].lower()
                self.batch_y=i[1].lower()
            
        for j in range(0,len(realtalk),batch_size):
            if batch_size != 1:
                self.batch_xs.append(self.batch_x[j:j+batch_size])
                self.batch_ys.append(self.batch_y[j:j+batch_size])
            elif batch_size==1:
                self.batch_xs.append( realtalk[j][0])
                self.batch_ys.append( realtalk[j][1])
            
    def get(self):
        return self.batch_xs, self.batch_ys
    

bg= BatchGenerator1(realtalk,1)
x,y = bg.get()

x=[i.lower() for i in x]
y=[i.lower() for i in y]
x = [re.sub(r'[?|$|.|!|,|-|"]|-+','',i) for i in x]
y = [re.sub(r'[?|$|.|!|,|-|"]|-+','',i) for i in y]
import copy
xx=copy.copy(x)




from collections import Counter
z=x+y
tid=[]
for i in (z):
    for j in re.findall(r'\S+', i):
        tid.append(j)
A=Counter(tid)
mydict = {x : A[x] for x in A if A[x] >= 30 }


A=1
for k,v in mydict.items():
    mydict[k]=A
    A+=1    


mydict["<PAD>"]=0
revdict = {v: k for k, v in mydict.items()}


for i in range(len(xx)-1,-1,-1):
    #print(i)
    state=0
    for j in re.findall(r'\S+', xx[i]):
        try:
            tid.append(mydict[j.lower()])
        except:
            state=1
    if state==1:
        del x[i];del y[i]     

    
yy=copy.copy(y)
for i in range(len(yy)-1,-1,-1):
    #print(i)
    state=0
    for j in re.findall(r'\S+', yy[i]):
        try:
            tid.append(mydict[j.lower()])
        except:
            state=1
    if state==1:
        del x[i];del y[i]     

    
xid , yid= [],[]
for i in range(len(x)):
    tid=[]
    for j in re.findall(r'\S+', x[i]):
        tid.append(mydict[j.lower()])
    xid.append(tid)
for i in range(len(y)):
    tid=[]
    for j in re.findall(r'\S+', y[i]):
        tid.append(mydict[j.lower()])
    yid.append(tid)
    


q_max_len = 12
a_max_len = 12
#for i in range(len(xid)): # caculate max length
#    q_max_len = max(q_max_len, len(xid[i]))
#    a_max_len = max(a_max_len, len(yid[i]))
#print(q_max_len, a_max_len)    
for i in range(len(xid)-1,-1,-1):
    if xid[i]==[] or yid[i]==[]:
        del xid[i];del yid[i]
    if len(xid[i])>q_max_len or len(yid[i])>q_max_len:
        del xid[i];del yid[i]


dbgx = []
for i in xid:
    tid=[]
    for j in i:
        tid.append(revdict[j])
    dbgx.append(tid)
    
dbgy = []
for i in yid:
    tid=[]
    for j in i:
        tid.append(revdict[j])
    dbgy.append(tid)



class BatchGenerator:
    def __init__(self, en_corpus, ch_corpus, en_pad, ch_pad, en_max_len, ch_max_len, batch_size):
        assert len(en_corpus) == len(ch_corpus)
        
        batch_num = len(en_corpus)//batch_size
        n = batch_num*batch_size
        
        self.xs = [np.zeros(n, dtype=np.int32) for _ in range(en_max_len)] # encoder inputs
        self.ys = [np.zeros(n, dtype=np.int32) for _ in range(ch_max_len)] # decoder inputs
        self.gs = [np.zeros(n, dtype=np.int32) for _ in range(ch_max_len)] # decoder outputs
        self.ws = [np.zeros(n, dtype=np.float32) for _ in range(ch_max_len)] # decoder weight for loss caculation
        
        self.en_max_len = en_max_len
        self.ch_max_len = ch_max_len
        self.batch_size = batch_size
        
        for b in range(batch_num):
            for i in range(b*batch_size, (b+1)*batch_size):
                for j in range(len(en_corpus[i])-2):
                    self.xs[j][i] = en_corpus[i][j+1]
                for j in range(j+1, en_max_len):
                    self.xs[j][i] = en_pad
                
                for j in range(len(ch_corpus[i])-1):
                    self.ys[j][i] = ch_corpus[i][j]
                    self.gs[j][i] = ch_corpus[i][j+1]
                    self.ws[j][i] = 1.0
                for j in range(j+1, ch_max_len): # don't forget padding and let loss weight zero
                    self.ys[j][i] = ch_pad
                    self.gs[j][i] = ch_pad
                    self.ws[j][i] = 0.0
    
    def get(self, batch_id):
        x = [self.xs[i][batch_id*self.batch_size:(batch_id+1)*self.batch_size] for i in range(self.en_max_len)]
        y = [self.ys[i][batch_id*self.batch_size:(batch_id+1)*self.batch_size] for i in range(self.ch_max_len)]
        g = [self.gs[i][batch_id*self.batch_size:(batch_id+1)*self.batch_size] for i in range(self.ch_max_len)]
        w = [self.ws[i][batch_id*self.batch_size:(batch_id+1)*self.batch_size] for i in range(self.ch_max_len)]
        
        return x, y, g, w



batch = BatchGenerator(xid, yid, 
                       mydict['<PAD>'], mydict['<PAD>'], q_max_len, a_max_len, 4)



x, y, g, w = batch.get(0)
for i in range(4):
    print(' '.join([revdict[x[j][i]] for j in range(q_max_len)]))
    print(' '.join([revdict[y[j][i]] for j in range(a_max_len)]))
    print(' '.join([revdict[g[j][i]] for j in range(a_max_len)]))
    print('')










