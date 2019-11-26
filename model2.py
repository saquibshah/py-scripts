from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from itertools import chain
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from scipy.spatial import distance
from nltk.corpus.reader.wordnet import WordNetError
from scipy import sparse
from scipy.sparse import csr_matrix
import sys
from sklearn.preprocessing import normalize
import numpy as np
import gensim
from collections import Counter
import itertools


language=str(sys.argv[1])
lang=str(sys.argv[2])
t1corpus = open(language+'corpus1.txt', 'r').readlines()
t2corpus = open(language+'corpus2.txt', 'r').readlines()
minfreq=int(sys.argv[3])

############read the raw corpus into a format that can be processed by the lesk algorithm
def corpusdata(corpus):
 corpuslist=[]
 wordfreq={}
 rarewordlist=[]
 for line in corpus:
   x=line.rstrip('\n').split()
   corpuslist.append(x)
 textdata = sorted(list(itertools.chain.from_iterable(corpuslist)))
 for sentence in corpuslist:                ################count word frequencies
  for word in sentence:
   if word not in wordfreq:
     wordfreq[word] = 0 
   wordfreq[word] += 1
 for word in wordfreq:
  if wordfreq[word]<minfreq:
   rarewordlist.append(word)
 return corpuslist, textdata, rarewordlist

corpust1,words1,rarewords1=corpusdata(t1corpus)
corpust2,words2,rarewords2=corpusdata(t2corpus)
ps = SnowballStemmer(lang)
rarewords=rarewords1+rarewords2

##########################perform WSD on a given word in the corpus
def lesk(context_sentence, word, pos=None, hyperhypo=True, stem=True):
    max_overlaps = 0; lesk_sense = None

    if wn.synsets(word):     ################only words with a synset will be accepted - if there is no synset, it will not be considered a proper word and disregarded
     for ss in wn.synsets(word):         ################## collect every possible meaning of the word and example sentences of how it is used in human language
        a=str(ss).replace('Synset','').replace('(','').replace(')','').replace("'","")
        try:
         examples=wn.synset(a).examples()
        except WordNetError:
         examples=None
        if pos and ss.pos is not pos:
            continue
 
        lesk_dictionary = []
        if hyperhypo == True:                                   ################collect all hypernyms and hypnonyms of the word for the lesk_dictionary
            lesk_dictionary+= list(chain(*[i.lemma_names() for i in ss.hypernyms()+ss.hyponyms()]))       

        if examples is not None:              #############if example sentences are available, add them to the lesk_dictionary
         for example in examples:
          example=example.split()
          lesk_dictionary+=example
        definition=str(ss.definition()).split()       
        lesk_dictionary+=definition    #############add the definitions of the meaning to the lesk_dictionary
        lesk_dictionary+=ss.lemma_names() #############add the lemma names of the meaning to the lesk_dictionary
        formatted=[]
        lesk_dictionary=[x.lower() for x in lesk_dictionary]

        for word in context_sentence:
            #############remove stopwords
          formatted.append(word.lower())
          formatted=[ps.stem(i) for i in formatted] 

        lesk_dictionary = [ps.stem(i) for i in lesk_dictionary]
        overlaps = set(lesk_dictionary).intersection(set(formatted))#########create the intersection of the lesk_dictionary for a given meaning with the context sentence
        if len(overlaps) > max_overlaps:     ############the meaning whose lesk_dictionary has the highest overlap with the context sentence wins
            lesk_sense = ss
            max_overlaps = len(overlaps)

    return lesk_sense

#####################counts how many times a given word occurs with its different meanings
def sensefreq(corpus,vocab, rw):
 print("Assembling Sense Frequency Dictionary")
 sensefreq={}
 sentences=[]
 for sentence in corpus:   ###############remove stopwords
   for word in sentence:
    if word.lower() not in stopwords.words(str(lang)) and word not in rw:
     if word in vocab:
      sentences.append(word.lower())

 voc=set(sentences)
 voc=list(voc)
 for word in voc:       #############assemble word-sense frequency dictionary - dictionary of dictionaries with the form [word:[sense1:n, sense2:n,...}}
   sensefreq[word]={}
 for sentence in corpus:
  for word in sentence:
   if word in vocab:
    word=word.lower()
    s=wn.synsets(word)
    for entry in s:
     if word not in stopwords.words(lang) and word not in rw:
      sense=str(entry).replace('Synset','').replace('(','').replace(')','').replace("'","")
      sensefreq[word][sense]=0
 print('Performing Lesk')
 for sentence in corpus:    ########for every sentence in the corpus, we perform WSD on every word
  for word in sentence:
   word=word.lower()
   try:
    if word not in stopwords.words(lang) and word not in rw:
     sense=lesk(sentence,word)
     a=str(sense).replace('Synset','').replace('(','').replace(')','').replace("'","")
     a=a.replace('Synset','').replace('(','').replace(')','').replace("'","")
     sensefreq[word][a]+=1       ###############the determined sense of the word will be stored in the sense frequency dictionary, increasing the corresponding sub-entry
   except KeyError:                ##############by 1
    if word in sensefreq: 
     del sensefreq[word]
 print("Sense Frequency Dictionary Assembled")
 return sensefreq

mutualwords=set(words1).intersection(set(words2))
mutualwords=mutualwords-set(rarewords)
senses1=sensefreq(corpust1, mutualwords, rarewords)
senses2=sensefreq(corpust2, mutualwords, rarewords)

#########################create vectors for every word from the sense frequency dictionary and normalize the vectors to get a relative frequency distribution
def vectorize(data):
 vectors={}
 print("Creating Vectors")
 for word in data:
  #print(word)
  vector=[]
  t=data[word]

  for entry in t:
   vector.append(t[entry])
  #print(vector)
  freqdist=normalize(np.array(vector).reshape(1,-1), norm='l1', axis=1)  
  print(freqdist)
  vectors[word]= freqdist
 print("Vectors Created")
 return vectors

a=vectorize(senses1)
b=vectorize(senses2)
ranking={}


######################compare the vectors for each word between t1 and t2, collect the word and its semantic difference between t1 and t2 in a dictionary

     

for x in a:
 try:
  if x in b:
   vector1=a[x]
   vector2=b[x]
   print(vector1)
   freqdist=sparse.csr_matrix(vector1)            
   freqdist2=sparse.csr_matrix(vector2)
   z= gensim.matutils.jensen_shannon(freqdist, freqdist2)  
   ranking[x]=z
 except IndexError:
  continue



print(dict(Counter(ranking).most_common(30)))

