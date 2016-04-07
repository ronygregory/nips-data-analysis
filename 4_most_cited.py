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

reference_list = reference_list[:10000] #TODO
print len(reference_list)
done_list = []
kgram_map = {}
for i, reference in enumerate(reference_list):
    kgram_map[i] = kgram(reference)
    
print i 
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