# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 16:01:12 2016

@author: Murali
"""

import pandas as pd
import nltk
import gensim as gs
from nltk.corpus import stopwords
import re
from sklearn.cluster import KMeans 
from operator import add
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.graph_objs import Scatter
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
#eng_stopwords=stopwords.words("english")
from sklearn.metrics.pairwise import euclidean_distances

"""
eng_stopwords=stopwords.words("english")

domain_spec_stopwords=["press","foundations","trends","vol","editor","workshop","international","journal","research","paper","proceedings","conference","wokshop","acm","icml","sigkdd","ieee","pages","springer"]
eng_stopwords=eng_stopwords+domain_spec_stopwords

with open(r"C:\Users\Murali\Desktop\full_stopwords.txt","r") as f:
    comp_st=[]
    for i in f.readlines():
        comp_st.append(i[:-1])
        
eng_stopwords=eng_stopwords+comp_st  
"""

def data_cleaning(datafile):
    regex = re.compile('[^a-zA-Z]')
    ls=[]
    for i in abstract_data:
        assert len(i.split())>0,"Make sure that the abstract is not empty"
        temp_ls=[]
        for j in regex.sub(" ",i).lower().split():
            if j not in eng_stopwords:
              temp_ls.append(j)
        ls.append(temp_ls)
    return ls
        
#cleaned_abstract=[" ".join(j).split() for i in abstract_data if len(i.split())>0 for j in re.sub("\d"," ",i).lower().split() if j not in eng_stopwords]
#cleaned_abstract=data_cleaning(abstract_data)
#del cleaned_abstract[125]
#model=gs.models.Word2Vec(cleaned_abstract,min_count=1)

def word2vec_model_creation(data_file,kgrams_or_phrases,length_of_kgrams_or_phrases,type_of_data):
    cleaned_abstract=data_file
    #print len(cleaned_abstract_data)
    if kgrams_or_phrases=="kgrams":
        n_grams=lambda datalist,n: zip(*[datalist[i:] for i in range(n)])
        if length_of_kgrams_or_phrases==2:
            combining_bigrams=lambda n_grams_res,n: [i[j]+"_"+i[j+1] for i in n_grams_res for j in range(n-1)]
            bi_grams=[combining_bigrams(n_grams(i,2),2) for i in cleaned_abstract] 
            model_bigrams=gs.models.Word2Vec(bi_grams,min_count=5)
            final_model=model_bigrams
            #return model_bigrams
            final_data=bi_grams
        elif length_of_kgrams_or_phrases==3:    
            combining_trigrams=lambda n_grams_res,n: [i[j]+"_"+i[j+1]+"_"+i[j+2] for i in n_grams_res for j in range(n-2)]
            tri_grams=[combining_trigrams(n_grams(i,3),3) for i in cleaned_abstract]
            model_trigrams=gs.models.Word2Vec(tri_grams,min_count=2)
            final_model=model_trigrams
            final_data=tri_grams
            #return model_trigrams
    elif kgrams_or_phrases=="phrases":
         bigram_phrases=gs.models.Phrases(cleaned_abstract)
         res_bi_phrases=bigram_phrases[cleaned_abstract]
         res_bi_phrases_ls=list(res_bi_phrases)
         print "here"
         if length_of_kgrams_or_phrases==2:   
             model_biphrases=gs.models.Word2Vec(res_bi_phrases_ls,min_count=3)
             final_model=model_biphrases
             #return model_biphrases
             final_data=res_bi_phrases_ls
             print "here"
         elif length_of_kgrams_or_phrases==3:
              trigram_phrases=gs.models.Phrases(res_bi_phrases)
              res_tri_phrases=list(trigram_phrases[res_bi_phrases])
              model_triphrases=gs.models.Word2Vec(res_tri_phrases,min_count=2)
              final_model=model_triphrases
              #return model_triphrases
              final_data=res_tri_phrases
              
    if type_of_data=="a":         
        final_model.save(kgrams_or_phrases+"_"+str(length_of_kgrams_or_phrases)+"_"+"abstract word2vec model.txt") 
    else:
         final_model.save(kgrams_or_phrases+"_"+str(length_of_kgrams_or_phrases)+"_"+"paper word2vec model.txt") 
    return final_model,final_data         



def sim_func(representation,num_of_similar_papers):
    assert (num_of_similar_papers<=20),"else please choose a value <=20 for the second argument"
    x=np.array(representation)
    nn_model=NearestNeighbors(n_neighbors=num_of_similar_papers, algorithm='ball_tree').fit(x)
    distances, indices = nn_model.kneighbors(x)
    return indices
    
    
def sim_comp(model,model_data_by_papers,method):
    assert (method=="average" or method=="clustering"),"Looks like the argument method was passes a parameter other than average or clustering"
    if method=="average":
        print "average"
        print model_data_by_papers[0]
        print model_data_by_papers[1]
        avg_representation=[]
        dim_of_model=len(model[model.vocab.keys()[0]])
        print dim_of_model
        print model.vocab.keys()
        each_paper=[0]*dim_of_model
        for i in model_data_by_papers:
            c=0
            for j in i:
                #print j
                if j in model.vocab.keys():
                   print "here" 
                   c=c+1 
                   each_paper=map(add,each_paper,list(model[j]))
            print c       
            each_paper=[float(k)/c for k in each_paper]    
            avg_representation.append(each_paper)    
        return sim_func(avg_representation,5)
        
    elif method=="clustering":
        print "clustering"
        word_vectors = model.syn0
        num_clusters = word_vectors.shape[0] / 20
        kmeans_clustering = KMeans(n_clusters=num_clusters,init="k-means++")
        idx = kmeans_clustering.fit_predict( word_vectors )
        word_centroid_map = dict(zip( model.index2word, idx ))
        for cluster in xrange(0,num_clusters):
            #print "\nCluster %d" % cluster
            words = []
            for i in xrange(0,len(word_centroid_map.values())):
                if word_centroid_map.values()[i] == cluster:
                   words.append(word_centroid_map.keys()[i])
            #print words
        cluster_representation=[]           
        for i in model_data_by_papers:
            cluster_representation.append(create_bag_of_centroids(i,word_centroid_map))
        return sim_func(cluster_representation,5)   


def create_bag_of_centroids(wordlist, word_centroid_map ):
    num_centroids = max( word_centroid_map.values() ) + 1
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids           

def vis_word_vector(word_2_vec_model):
    tsne_model=TSNE(n_components=2,random_state=5)
    data=tsne_model.fit_transform(word_2_vec_model.syn0)
   # names=word_2_vec_model.index2word
    plt.scatter(x=data[:,0],y=data[:,1])

def plotly_js_viz(word_2_vec_model):
    tsne_model=TSNE(n_components=2,random_state=5)
    data=tsne_model.fit_transform(word_2_vec_model.syn0)
    xd=list(data[:,0])
    yd=list(data[:,1])
    names_our=word_2_vec_model.index2word
    plot([Scatter(x=xd,y=yd,mode="markers",text=names_our)])

    
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
   papers=pd.read_csv("E:/Data mining project/output/Papers.csv",delimiter=",")
   authors=pd.read_csv("E:/Data mining project/output/Paperauthors.csv",delimiter=",")
   del authors["Id"]
   authors.columns=["Id","AuthorId"]
   joined_df=pd.merge(papers,authors,on="Id")
   authors_id_to_names=pd.read_csv("E:/Data mining project/output/Authors.csv",delimiter=",")
   #del authors_id_to_names["Id"]
   authors_id_to_names.columns=["AuthorId","AuthorName"]
   joined_df=pd.merge(joined_df,authors_id_to_names,on="AuthorId")
   #def similarity_simple(vectors):
   grped=joined_df.groupby(["AuthorName"])
   #grped_df=pd.DataFrame(grped)
   list_of_authors=grped.groups.keys()
   authors_to_papers=[]
   for i in list_of_authors:
      place_holder=[]
      for j in grped.groups[i]:
          #print int(j)
          place_holder.append(j)
      authors_to_papers.append(place_holder)
   choice=str(raw_input("What do you want to train the word2vec model on based on the papertext of all papers of the authors or the abstract or the title,input Abstract for abstract, PaperText for full paper text and Title for title"))
   final_data=[]
   for i in authors_to_papers:
       temp_str=" "
       for j in i:
           temp_str=temp_str+ joined_df.iloc[j][choice]+" "
       final_data.append(temp_str)
   final_data=[i.split() for i in final_data]
   abstract_data=final_data
   inp_kgrams_phrases=raw_input("Type kgrams for k-grams and phrases to create phrases from your data to load the word2vec model with: ") 
   inp_len=raw_input("Length of kgrams or phrases and make sure it's either 2 or 3: ")
   print inp_len
   print inp_kgrams_phrases
   print choice
   w2v_model,data_used=word2vec_model_creation(abstract_data,inp_kgrams_phrases,int(inp_len),choice)
   #init_notebook_mode()
   #vis_word_vector(w2v_model)
   #plotly_js_viz(w2v_model)
   inp_avg_or_clustering=raw_input("Type average for averaging the word vectors for each sentence or clustering inorder to create a bag of centroids model: ")    
   sim_indices=sim_comp(w2v_model,data_used,inp_avg_or_clustering) 
   
"""        
df=pd.DataFrame(authors_to_papers)
df=df.fillna(0)
dis_matrix=euclidean_distances(df,df)  
#for i in dis_matrix.shape[0]:
    
index_matrix=[np.argpartition(dis_matrix[i,:],-6)[-6:] for i in range(dis_matrix.shape[0])]
"""