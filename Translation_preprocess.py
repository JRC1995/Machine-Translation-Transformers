
# coding: utf-8

# # Preprocessing Translation Data

# ### Function for expanding english contractions
# 
# source: https://gist.github.com/nealrs/96342d8231b75cf4bb82

# In[1]:


import numpy as np
from __future__ import division
import io
import unicodedata
import nltk
from nltk import word_tokenize
import string
import re
import random


#source: https://gist.github.com/nealrs/96342d8231b75cf4bb82
cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)


# ### Loading Translation Data
# 
# Splitting the data into eng and beng.
# eng will contain the list of English lines, and beng will contain the corresponding list of Bengali lines.
# 
# 
# Source of data: http://www.manythings.org/anki/ (downloaded ben-eng)

# In[2]:


filename = 'ben.txt'
#Datasource: http://www.manythings.org/anki/
    
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    
def normalizeString(s):
    s = unicodeToAscii(expandContractions(s.lower().strip()))
    s = re.sub(r"([.!?,\"])", r" ", s)
    return s
    
def loaddata(filename):
    file = io.open(filename,'r')
    eng=[]
    beng = []
    for line in file.readlines():
        lang_pair = line.split('\t')
        lang_pair[0] = normalizeString(lang_pair[0])
        lang_pair[1] = normalizeString(lang_pair[1])
        eng.append(word_tokenize(lang_pair[0]))
        beng.append(word_tokenize(lang_pair[1]))
    file.close()
    return eng,beng

eng,beng = loaddata(filename)

#Example:
sample = random.randint(0,len(eng))
print "Example Sample #"+str(sample)+":\n"
string = "ENGLISH:"
for i in xrange(0,len(eng[sample])):
    string+=" "+eng[sample][i]
print string
    
string = "\nBENGALI:"
for i in xrange(0,len(beng[sample])):
    string+=" "+beng[sample][i]
print string


# ### Creating separate vocabulary lists for English words and Bengali words
# 
# The index of vocabulary will represent the numerical representation of the word which is the value of vocabulary at that index. 
# 

# In[3]:


import numpy as np

vocab_eng=[]
vocab_eng.append('<PAD>')
vocab_eng.append('<EOS>')

vocab_beng=[]
vocab_beng.append('<PAD>')
vocab_beng.append('<EOS>')

#The index of vocab will serve as an integer representation of the word

vectorized_eng = []
vectorized_beng = []

for i in xrange(len(eng)):
    
    vectorized_eng_line = []
    for word in eng[i]:
        if word not in vocab_eng:
            vocab_eng.append(word)
            vectorized_eng_line.append(vocab_eng.index(word))
        else:
            vectorized_eng_line.append(vocab_eng.index(word))
    vectorized_eng.append(vectorized_eng_line)
    
    vectorized_beng_line = []
    for word in beng[i]:
        if word not in vocab_beng:
            vocab_beng.append(word)
            vectorized_beng_line.append(vocab_beng.index(word))
        else:
            vectorized_beng_line.append(vocab_beng.index(word))
    vectorized_beng.append(vectorized_beng_line)
    
    


# ### Creating training dataset for word2vec embedding
# 
# if the sentence is "I am alright"
# 
# then for the word 'am', the context words with window size 1 will be "I" and "alright"
# i.e ["I","alright"]
# 
# For 'I' the context words will be "PAD" and "am"
# 
# For 'alright' the context words will be "am" and "PAD"
# 
# PAD represents empty and EOS represents end of sentence.
# 
# Later lots of pads may be applied after the end of sentence to fit sequence length.
# 
# So I also added the word PAD with context words being PADs, and PAD and EOS for embedding.
# 
# In this way, first, from each sentence, I am creating a list of words, and corresponding list of context words.
# Doing the same thing for

# In[4]:


words_eng = []
contexts_eng = []

words_beng = []
contexts_beng = []

words_eng.append(vocab_eng.index('<PAD>'))
contexts_eng.append([vocab_eng.index('<EOS>'),vocab_eng.index('<PAD>')])
words_eng.append(vocab_eng.index('<PAD>'))
contexts_eng.append([vocab_eng.index('<PAD>'),vocab_eng.index('<PAD>')])

words_beng.append(vocab_beng.index('<PAD>'))
contexts_beng.append([vocab_beng.index('<EOS>'),vocab_beng.index('<PAD>')])
words_beng.append(vocab_beng.index('<PAD>'))
contexts_beng.append([vocab_beng.index('<PAD>'),vocab_beng.index('<PAD>')])


for i in xrange(len(vectorized_eng)):
    
    for j in xrange(0,len(vectorized_eng[i])):
        
        context1=0
        context2=0
        
        if j==0:
            context1 = vocab_eng.index('<PAD>')
            if j!=len(vectorized_eng[i])-1:
                context2 = vectorized_eng[i][j+1]
        if j==len(vectorized_eng[i])-1:
            context2=vocab_eng.index('<EOS>')
            if j!=0:
                context1 = vectorized_eng[i][j-1]
        if j>0 and j<len(vectorized_eng[i])-1:
            context1 = vectorized_eng[i][j-1]
            context2 = vectorized_eng[i][j+1]
        
        words_eng.append(vectorized_eng[i][j])
        contexts_eng.append([context1,context2])
    
    rand = random.randint(0,3)
    if rand == 1: #reduce the freuency of <EOS> for training data
        words_eng.append(vocab_eng.index('<EOS>'))
        context1 = vectorized_eng[i][len(vectorized_eng[i])-1]
        context2 = vocab_eng.index('<PAD>')
        contexts_eng.append([context1,context2])
    
    for j in xrange(0,len(vectorized_beng[i])):
        
        context1=0
        context2=0
        
        if j==0:
            context1 = vocab_beng.index('<PAD>')
            if j!=len(vectorized_beng[i])-1:
                context2 = vectorized_beng[i][j+1]
        if j==len(vectorized_beng[i])-1:
            context2=vocab_beng.index('<EOS>')
            if j!=0:
                context1 = vectorized_beng[i][j-1]
        if j>0 and j<len(vectorized_beng[i])-1:
            context1 = vectorized_beng[i][j-1]
            context2 = vectorized_beng[i][j+1]
        
        words_beng.append(vectorized_beng[i][j])
        contexts_beng.append([context1,context2])
    
    rand = random.randint(0,3)
    if rand == 1: #reduce the freuency of <EOS> for training data
        words_beng.append(vocab_beng.index('<EOS>'))
        context1 = vectorized_beng[i][len(vectorized_beng[i])-1]
        context2 = vocab_beng.index('<PAD>')
        contexts_beng.append([context1,context2])
    
    
            


# If word = "am" and context = ["I","alright"],
# then, here I will reconstrcut the data as:
# 
# input = "am"
# output = "I"
# and 
# input = "am"
# label = "alright"
# 
# Like this I will construct a list of all training inputs (words) and training outputs\labels (context words)
# 
# embd_inputs_eng will contain all the English training inputs.
# embd_labels_eng will contain all the English training labels.
# 
# embd_inputs_beng will contain all the Bengali training inputs.
# embd_labels_beng will contain all the Bengali training labels.

# In[5]:


embd_inputs_eng = []
embd_labels_eng = []
for i in xrange(len(contexts_eng)):
    for context in contexts_eng[i]:
        embd_inputs_eng.append(words_eng[i])
        embd_labels_eng.append(context)
embd_inputs_eng = np.asarray(embd_inputs_eng,np.int32)
embd_labels_eng = np.asarray(embd_labels_eng,np.int32)

embd_inputs_beng = []
embd_labels_beng = []
for i in xrange(len(contexts_beng)):
    for context in contexts_beng[i]:
        embd_inputs_beng.append(words_beng[i])
        embd_labels_beng.append(context)
embd_inputs_beng = np.asarray(embd_inputs_beng,np.int32)
embd_labels_beng = np.asarray(embd_labels_beng,np.int32)
    


# ### Function for generating mini-batches from the total training set

# In[6]:


batch_size = 128

def generate_batch(inputs,labels,batch_size):
    rand = random.sample((np.arange(len(inputs))),batch_size)
    batch_inputs=[]
    batch_labels=[]
    for i in xrange(batch_size):
        batch_inputs.append(inputs[int(rand[i])])
        batch_labels.append(labels[int(rand[i])])
    batch_inputs = np.asarray(batch_inputs,np.int32)
    batch_labels = np.asarray(batch_labels,np.int32)
    return batch_inputs,batch_labels
    


# ### Preparing for word2vec embedding

# In[7]:


import tensorflow as tf
import math

#https://www.tensorflow.org/tutorials/word2vec
embedding_size = 256
vocabulary_size_eng = len(vocab_eng)
vocabulary_size_beng = len(vocab_beng)

# Placeholders for inputs
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])


# ### Training for word2vec embedding (For English words)
# 
# See: https://www.tensorflow.org/tutorials/word2vec
# 
# for details of word2vec and code description

# In[8]:


embeddings_eng = tf.Variable(
    tf.random_uniform([vocabulary_size_eng, embedding_size], -1.0, 1.0))

nce_weights_eng = tf.Variable(
  tf.truncated_normal([vocabulary_size_eng, embedding_size],
                      stddev=1.0 / math.sqrt(embedding_size)))
nce_biases_eng = tf.Variable(tf.zeros([vocabulary_size_eng]))

# Initializing the variables
init = tf.global_variables_initializer()


# In[9]:


embed_eng = tf.nn.embedding_lookup(embeddings_eng, train_inputs)

# Compute the NCE loss, using a sample of the negative labels each time.
loss = tf.reduce_mean(
  tf.nn.nce_loss(weights=nce_weights_eng,
                 biases=nce_biases_eng,
                 labels=train_labels,
                 inputs=embed_eng,
                 num_sampled=10, 
                 num_classes=vocabulary_size_eng)) #num_sampled = no. of negative samples

# We use the SGD optimizer.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

            


# In[10]:



with tf.Session() as sess:
    sess.run(init)
    convergence_threshold = 0.5
    training_iters = 500*(int((len(embd_inputs_eng))/batch_size))
    step=0
    n=5
    last_n_losses = np.zeros((n),np.float32)
    
    while step<training_iters:
        
        batch_inputs,batch_labels = generate_batch(embd_inputs_eng,embd_labels_eng,batch_size)
        
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels.reshape((-1,1))}
        
        _, np_embedding_eng, cur_loss = sess.run([optimizer, embeddings_eng, loss], feed_dict=feed_dict)
        
        print "Iter "+str(step)+", loss = "+str(cur_loss)
        
        last_n_losses[step%n]=cur_loss
        
        if step>=n:
            if np.mean(last_n_losses)<=convergence_threshold:
                break
        step+=1
                
print "\nOptimization Finished\n"


# ### Training for word2vec embedding (For Bengali words)
# 
# See: https://www.tensorflow.org/tutorials/word2vec
# 
# for details of word2vec and code description

# In[11]:


embeddings_beng = tf.Variable(
    tf.random_uniform([vocabulary_size_beng, embedding_size], -1.0, 1.0))

nce_weights_beng = tf.Variable(
  tf.truncated_normal([vocabulary_size_beng, embedding_size],
                      stddev=1.0 / math.sqrt(embedding_size)))
nce_biases_beng = tf.Variable(tf.zeros([vocabulary_size_beng]))

# Initializing the variables
init = tf.global_variables_initializer()


# In[12]:


embed_beng = tf.nn.embedding_lookup(embeddings_beng, train_inputs)

# Compute the NCE loss, using a sample of the negative labels each time.
loss = tf.reduce_mean(
  tf.nn.nce_loss(weights=nce_weights_beng,
                 biases=nce_biases_beng,
                 labels=train_labels,
                 inputs=embed_beng,
                 num_sampled=10, 
                 num_classes=vocabulary_size_beng)) #num_sampled = no. of negative samples

# We use the SGD optimizer.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

            


# In[13]:



with tf.Session() as sess:
    sess.run(init)
    convergence_threshold = 0.5
    training_iters = 500*(int((len(embd_inputs_beng))/batch_size))
    step=0
    n=5
    last_n_losses = np.zeros((n),np.float32)
    while step<training_iters:
        
        batch_inputs,batch_labels = generate_batch(embd_inputs_beng,embd_labels_beng,batch_size)
        
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels.reshape((-1,1))}
        _, np_embedding_beng, cur_loss = sess.run([optimizer, embeddings_beng, loss], feed_dict=feed_dict)
        
        print "Iter "+str(step)+", loss = "+str(cur_loss)
        last_n_losses[step%n]=cur_loss
        if step>=n:
            if np.mean(last_n_losses)<=convergence_threshold:
                break
        step+=1
                
print "\nOptimization Finished\n"


# ### Creating Train, Validation, and Test set
# 
# Randomly shuffling the complete dataset, and then splitting it into train, validation and test set

# In[14]:


shuffled_indices = np.arange(len(eng))
np.random.shuffle(shuffled_indices)

shuffled_vectorized_eng = []
shuffled_vectorized_beng = []

for i in xrange(len(eng)):
    shuffled_vectorized_eng.append(vectorized_eng[shuffled_indices[i]])
    shuffled_vectorized_beng.append(vectorized_beng[shuffled_indices[i]])

train_len = int(.75*len(eng))
val_len = int(.15*len(eng))

train_eng = shuffled_vectorized_eng[0:train_len]
train_beng = shuffled_vectorized_beng[0:train_len]

val_eng = shuffled_vectorized_eng[train_len:val_len]
val_beng = shuffled_vectorized_beng[train_len:val_len]

test_eng = shuffled_vectorized_eng[train_len+val_len:]
test_beng = shuffled_vectorized_beng[train_len+val_len:]


# ### Function for bucketing and generating batches
# 
# Mini-batch training requires all lines in a batch to be of equal length.
# We have different lines of different lengths. 
# 
# Solution is to fill shorter sentences with PADs so that length of all sentences become equal.
# But, if one sentence in a batch has 20 words, and the same batch has another sentence with one word, then the latter sentence will have to be filled in by at least 19 pads. If most of the sentences start to have more PADs than actual content, training will become troublesome.
# 
# The solution to that is bucketing. First the sentences in the total list are sorted. After that sentences of similar lengths are closer to each other. Batches are then formed with sentences of similar lengths. Much less padding will be required to turning sentences of similar lengths into senetences of equal lengths. 

# In[15]:


def bucket_and_batch(x,y,batch_size):
    
    len_x= np.zeros((len(x)),np.int32)
    
    for i in xrange(len(x)):
        len_x[i] = len(x[i])
        
    sorted_by_len_indices = np.flip(np.argsort(len_x),0)

    sorted_x = []
    sorted_y = []
    
    for i in xrange(len(x)):
        sorted_x.append(x[sorted_by_len_indices[i]])
        sorted_y.append(y[sorted_by_len_indices[i]])
        
    i=0
    batches_x = []
    batches_y = []
    
    while i<len(x):
        
        if i+batch_size>=len(x):
            break
        
        batch_x = []
        batch_y = []
    
        max_len_x = len(sorted_x[i])
    
        len_y= np.zeros((len(y)),np.int32)
    
        for j in xrange(i,i+batch_size):
            len_y[j] = len(sorted_y[j])
            
        max_len_y = np.amax(len_y)
        
        for j in xrange(i,i+batch_size):
            line=[]
            for k1 in xrange(max_len_x+1): #+1 to include <EOS>
                if k1==len(sorted_x[j]):
                    line.append(np_embedding_eng[vocab_eng.index('<EOS>')])
                elif k1>len(sorted_x[j]):
                    line.append(np_embedding_eng[vocab_eng.index('<PAD>')])
                else:
                    line.append(np_embedding_eng[sorted_x[j][k1]])
            batch_x.append(line)
        
            line=[]
            for k2 in xrange(max_len_y+1): #+1 to include <EOS>
                if k2>len(sorted_y[j]):
                    line.append(vocab_beng.index('<PAD>'))
                elif k2==len(sorted_y[j]):
                    line.append(vocab_beng.index('<EOS>'))
                else:
                    line.append(sorted_y[j][k2])
            batch_y.append(line)
    
        batch_x = np.asarray(batch_x,np.float32)
        batch_y = np.asarray(batch_y,np.int32)

        batches_x.append(batch_x)
        batches_y.append(batch_y)
    
        i+=batch_size
        
    return batches_x,batches_y



# ### Creating train, validation and test batches

# In[16]:


batch_size = 64

train_batch_eng,train_batch_beng = bucket_and_batch(train_eng,train_beng,batch_size)

val_batch_eng,val_batch_beng = bucket_and_batch(val_eng,val_beng,batch_size)

test_batch_eng,test_batch_beng = bucket_and_batch(test_eng,test_beng,batch_size)


# ### Saving processed data in another file.

# In[17]:


#Saving processed data in another file.

import pickle

PICK = [vocab_eng,vocab_beng,np_embedding_eng,np_embedding_beng,train_batch_eng,train_batch_beng,val_batch_eng,val_batch_beng,test_batch_eng,test_batch_beng]

with open('translationPICKLE', 'wb') as fp:
    pickle.dump(PICK, fp)

