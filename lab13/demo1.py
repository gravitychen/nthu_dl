import spacy

nlp = spacy.load('en_core_web_md')




# load lines dictionary 

# load conversations
convs = open('dataset/chatbot/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')


convs = [ str(i.replace("+++$+++ ",'')) for i in convs]

import re
import ast
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
        
class BatchGenerator:
    def __init__(self, dat, batch_size):

        self.batch_xs, self.batch_ys, self.reviews = [], [], []

        self.batch_x ,self.batch_y=[],[]
        for i in realtalk:
            
            self.batch_x.append( i[0])
            self.batch_y.append( i[1])
        for j in range(0,len(realtalk),batch_size):
            self.batch_xs.append(self.batch_x[j:j+batch_size])
            self.batch_ys.append(self.batch_y[j:j+batch_size])
            
    def get(self):
        return self.batch_xs, self.batch_ys
    

bg= BatchGenerator(realtalk,16)
x,y = bg.get()












