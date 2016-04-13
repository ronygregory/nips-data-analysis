# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:41:59 2016

@author: Murali
"""
import gensim as gs
import pandas as pd
import scipy
import re

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
import re
from gensim import corpora, models, similarities
import nltk

def topics_creation(text_data_loc,num_of_topics,stp):
    
    #english_stemmer = nltk.stem.SnowballStemmer('english')
    final_text=text_data_loc
    #no_hashtags_text=[bsl.multiple_hashtag_deletion(i,"#")[0] for i in full_data]
    #final_text=[[j.lower() for j in i if j.lower() not in stp and rexp.sub("",j.lower())!=''] for i in full_data]  
    #final_text = [map(english_stemmer.stem,t) for t in final_text]
    dictionary = corpora.Dictionary(final_text)
    #print dictionary
    dictionary.save('tweets_lda.dict') # store the dictionary, for future reference
    corpus = [dictionary.doc2bow(text) for text in final_text]
    #print corpus[0]
    corpora.MmCorpus.serialize('tweets_lda_corpus.mm', corpus)
    model = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=num_of_topics,alpha="auto",eval_every=1,iterations=100000)
    topics=[model[c] for c in corpus]
    print model.print_topic(0,topn=20)
    print model.print_topic(2,topn=20)
    #print topics[0]
    #print topics[2]
    return model,topics
    

if __name__=="__main__":
    eng_stopwords=stopwords.words("english")
    domain_spec_stopwords=["press","foundations","trends","vol","editor","workshop","international","journal","research","paper","proceedings","conference","wokshop","acm","icml","sigkdd","ieee","pages","springer"]
    eng_stopwords=eng_stopwords+domain_spec_stopwords
    #normal_stopwords=[a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at,be,because,been,but,by,can,cannot,could,dear,did,do,does,either,else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,only,or,other,our,own,rather,said,say,says,she,should,since,so,some,than,that,the,their,them,then,there,these,they,this,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,who,whom,why,will,with,would,yet,you,your]
    with open(r"C:\Users\Murali\Desktop\full_stopwords.txt","r") as f:
        comp_st=[]
        for i in f.readlines():
           comp_st.append(i[:-1])
        
    compt_st=[i for i in comp_st if i!='']
    eng_stopwords=eng_stopwords+comp_st  
    papers=pd.read_csv(r"E:\Data mining project\output\papers.csv",delimiter=",")
    inp_column=raw_input("Do you want to compare similarity using the abstract or the full paper text, type a for abstract and anything else for full text")
    if inp_column=="a":
       abstract_data=list(papers["Abstract"])
    else:
        abstract_data=list(papers["PaperText"])
    for i in range(len(abstract_data)):
        empty_data=[]
        if len(abstract_data[i])==0:
           empty_data.append(i)
    for i in empty_data:
        print "Deleting row "+ str(i) +" from our dataset because it's empty"
        del abstract_data[i]
    abstract_data_ls_of_ls=[i.split() for i in abstract_data]
    rexp=re.compile('[^a-zA-Z]')
    abstract_data_ls_of_ls=[[j.lower() for j in i if j.lower() not in eng_stopwords and rexp.sub("",j.lower())!=''] for i in abstract_data_ls_of_ls]  
    phrases_model=gs.models.Phrases(abstract_data_ls_of_ls)    
    bip_data=phrases_model[abstract_data_ls_of_ls]    
    lda_model,topics_returned=topics_creation(bip_data,200,eng_stopwords)        