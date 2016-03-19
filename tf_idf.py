import csv
from nltk.corpus import stopwords
import numpy
from sklearn.feature_extraction.text import CountVectorizer
import heapq

# http://scikit-learn.org/stable/modules/feature_extraction.html
title_list = []
papertext_list = []
count = 5
stop_words = stopwords.words('english')
with open("data/Papers.csv") as csvfile:
    # Id,Title,EventType,PdfName,Abstract,PaperText
    next(csvfile)
    papers = csv.reader(csvfile, delimiter=',', quotechar='"')
    for paper in papers:
        title_list.append(paper[1])
        papertext = paper[5]
        papertext_list.append(papertext)
        
vectorizer = CountVectorizer(min_df=1, stop_words=stop_words)
X = vectorizer.fit_transform(papertext_list)

listOfLists = []
fullSet = set()

for i, rowOuter in enumerate(X.toarray()):
    row = []
    listOfLists.append(row)
    for j, rowInner in enumerate(X.toarray()):
        # print rowInner
        dist = None
        if i == j:
            row.append(0)
        else:
            dist = numpy.linalg.norm(rowOuter - rowInner)  # Calculate Euclidean distance
            row.append(dist)
            left = i if i < j else j
            right = j if i < j else i
            tupleVal = (dist, left, right)
            fullSet.add(tupleVal)
for row in listOfLists:
    print row

print 
topSimilarDocs = heapq.nsmallest(count, fullSet)
print topSimilarDocs
for doctuple in topSimilarDocs:
    print "{}, {} : {}".format(title_list[doctuple[1]], title_list[doctuple[2]], doctuple[0])
    
#Result obtained:
# Newton-Stein Method: A Second Order Method for GLMs via Stein's Lemma, Convergence rates of sub-sampled Newton methods : 118.410303606
# Learning structured densities via infinite dimensional exponential families, Robust Gaussian Graphical Modeling with the Trimmed Graphical Lasso : 134.733069437
# Learning with Incremental Iterative Regularization, Less is More: Nyström Computational Regularization : 137.393595193
# Robust Gaussian Graphical Modeling with the Trimmed Graphical Lasso, Distributionally Robust Logistic Regression : 139.789126902
# Bayesian Manifold Learning: The Locally Linear Latent Variable Model (LL-LVM), Probabilistic Curve Learning: Coulomb Repulsion and the Electrostatic Gaussian Process : 140.545366341