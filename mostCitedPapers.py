import csv
import nltk
import operator
import re


def kgram(a):
    return set(nltk.trigrams(a.split(" ")))

def similarity(a_kgram, b_kgram):
    intersectLen = len(a_kgram.intersection(b_kgram))
    unionLen = len(a_kgram.union(b_kgram))
    if unionLen == 0:
        return 0
    jaccardSim = intersectLen / (float)(unionLen)
    return jaccardSim

title_list = []
reference_list = []
ref_re = re.compile("\[\d+?\]\s.+?[0-9]{2}\.(?#=\s[[$])", re.MULTILINE)

with open("data/Papers.csv") as csvfile:
    next(csvfile)
    papers = csv.reader(csvfile, delimiter=',', quotechar='"')
    i = 0
    for paper in papers:
        title_list.append(paper[1])
        # references = paper[5].split("References")
        references = re.split(r'\bReferences\b|\bRefrences\b|\bBibliography\b', paper[5])
        if len(references) < 2:
            if i == 125 or i == 343:
                i += 1
                continue
            print paper[5]
            print paper[1]
            print i
            break
        references = references[1]
        # references = references.split('\n')
        # print references
        references = re.split('\[.+?\]', references)
        if len(references) == 1:
            references = re.split('\n', references[0])
        references = [" ".join(val.strip().decode('unicode_escape').\
                               encode('ascii', 'ignore').split("\n")) for val in references if len((val.strip())) > 10]
        print references
        # reference_list.append(references)
        reference_list += references
        i += 1
#         if i == 2:
#             break
matching_set = set()
count = 0
ref_count_map = {}

print len(reference_list)

with open("references.txt",'w') as f:
    for item in reference_list:
        f.write("%s\n" % item)
    
print "done"

done_list = []
kgram_map = {}
for i, reference in enumerate(reference_list):
    kgram_map[i] = kgram(reference)
    
for i, reference in enumerate(reference_list):
    if i % 100 == 0:
        print i, " ", count
    if i in done_list:
        continue
    for j, inner_reference in enumerate(reference_list[i + 1:], start = i + 1):
        if similarity(kgram_map[i], kgram_map[j]) > 0.5:
            # dprint i, j, reference
            count += 1
            ref_count_map[i] = 2 if i not in ref_count_map else ref_count_map[i] + 1
            done_list.append(j)
                 
print count
sorted_x = sorted(ref_count_map.items(), key=operator.itemgetter(1), reverse=True)

print sorted_x
for val in sorted_x:
    print reference_list[val[0]],", ", val[1]

# Top 10 cited papers:
# A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012. ,  14
# K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015. ,  12
# Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. 1998. ,  11
# Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S. Guadarrama, and T. Darrell. Caffe: Convolutional architecture for fast feature embedding. arXiv:1408.5093, 2014. ,  10
# S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural computation, 9(8):17351780, 1997. ,  10
# M. Lichman. UCI machine learning repository, 2013. ,  10
# A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems 27, pages 10971105, 2012. ,  9
# D. J. Rezende, S. Mohamed, and D. Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In ICML, 2014. ,  8
# John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. The Journal of Machine Learning Research, 12:21212159, 2011. ,  8
# R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In CVPR, 2014. ,  7
