#import logging
import os
import csv
import nltk
import gensim

def iter_docs(stoplist):
    with open('Papers.csv', 'rb') as csvfile:    
        f = csv.reader(csvfile, delimiter=',')
        readers = None
        document = []
        for row in f:
            text = row[5].replace(",","") #for author column
            yield (x for x in gensim.utils.tokenize(text, lowercase=True, deacc=True, errors="ignore") if x not in stoplist)

class MyCorpus(object):
    def __init__(self, stoplist):
        self.stoplist = stoplist
        self.dictionary = gensim.corpora.Dictionary(iter_docs(stoplist))
        
    def __iter__(self):
        for tokens in iter_docs(self.stoplist):
            yield self.dictionary.doc2bow(tokens)

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TEXTS_DIR = "."
MODELS_DIR = "."

stoplist = set(nltk.corpus.stopwords.words("english"))
corpus = MyCorpus(stoplist)

corpus.dictionary.save(os.path.join(MODELS_DIR, "mtsamples.dict"))
gensim.corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "mtsamples.mm"), corpus)


dictionary = gensim.corpora.Dictionary.load(os.path.join(MODELS_DIR, "mtsamples.dict"))
corpus = gensim.corpora.MmCorpus(os.path.join(MODELS_DIR, "mtsamples.mm"))

tfidf = gensim.models.TfidfModel(corpus, normalize=True)
corpus_tfidf = tfidf[corpus]

# project to 2 dimensions for visualization
lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)

# write out coordinates to file
fcoords = open(os.path.join(MODELS_DIR, "coords.csv"), 'wb')
for vector in lsi[corpus]:
    if len(vector) != 2:
        continue
    fcoords.write("%6.4f\t%6.4f\n" % (vector[0][1], vector[1][1]))
fcoords.close()