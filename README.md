

# Machine Translation (English to Bengali) using Transformers    

The model is based on:

["Attention Is All You Need" by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. arXiv:1706.03762](https://arxiv.org/abs/1706.03762) 

# Preprocessing Translation Data
(from Translation_preprocess.py)

### Function for expanding English contractions

source: https://gist.github.com/nealrs/96342d8231b75cf4bb82


```python
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
```

### Loading Translation Data

Splitting the data into eng and beng.
eng will contain the list of English lines, and beng will contain the corresponding list of Bengali lines.


Source of data: http://www.manythings.org/anki/ (downloaded ben-eng)


```python
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

```

    Example Sample #646:
    
    ENGLISH: tom 's right
    
    BENGALI: টমই ঠিক।


### Creating separate vocabulary lists for English words and Bengali words

The index of vocabulary will represent the numerical representation of the word which is stored at that index. 



```python
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
    
    
```

### Creating training dataset for word2vec embedding

if the sentence is "I am alright"

then for the word 'am', the context words with window size 1 will be "I" and "alright"
i.e ["I","alright"]

For 'I' the context words will be "PAD" and "am"

For 'alright' the context words will be "am" and "PAD"

PAD represents empty and EOS represents end of sentence.

Later lots of pads may be applied after the end of sentence to fit sequence length.

So I also added the word PAD with context words being PADs, and PAD and EOS for embedding.

(Doing what I wrote directly above, was actually unnecessary but I already did it. We don't need to consider these cases. With masking I will ignore the effect of PADs on the cost, anyway, and the model doesn't need to predict pads correctly. Predicting the EOS properly will be enough. So PAD embedding doesn't need to be taken so seriously.)

In this way, first, from each sentence, I am creating a list of words, and a corresponding list of context words.
I am doing the same thing for both English and Bengali lines. 


```python
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
    
    
            
```

If word = "am" and context = ["I","alright"],
then, from this data I will create the following samples:

input = "am"
output = "I"
and 
input = "am"
label = "alright"

Like this I will construct a list of all training inputs (words) and training outputs\labels (context words)

embd_inputs_eng will contain all the English training inputs.
embd_labels_eng will contain all the English training labels.

embd_inputs_beng will contain all the Bengali training inputs.
embd_labels_beng will contain all the Bengali training labels.


```python
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
    
```

### Function for generating mini-batches from the total training set


```python
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
    
```

### Preparing for word2vec embedding


```python
import tensorflow as tf
import math

#https://www.tensorflow.org/tutorials/word2vec
embedding_size = 256
vocabulary_size_eng = len(vocab_eng)
vocabulary_size_beng = len(vocab_beng)

# Placeholders for inputs
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])

```

### Training for word2vec embedding (For English words)

See: https://www.tensorflow.org/tutorials/word2vec

for details of word2vec and code description. 

Most of the word2vec code used here are from the Tensorflow tutorial. 


```python
embeddings_eng = tf.Variable(
    tf.random_uniform([vocabulary_size_eng, embedding_size], -1.0, 1.0))

nce_weights_eng = tf.Variable(
  tf.truncated_normal([vocabulary_size_eng, embedding_size],
                      stddev=1.0 / math.sqrt(embedding_size)))
nce_biases_eng = tf.Variable(tf.zeros([vocabulary_size_eng]))

# Initializing the variables
init = tf.global_variables_initializer()
```


```python
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
```


```python

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
```

    Iter 172463, loss = 1.07693
    Iter 172464, loss = 1.23457
    Iter 172465, loss = 0.929267
    Iter 172466, loss = 0.951752
    Iter 172467, loss = 1.03454
    Iter 172468, loss = 1.04068
    Iter 172469, loss = 1.00835
    Iter 172470, loss = 0.724509
    Iter 172471, loss = 1.34491
    Iter 172472, loss = 1.40166
    Iter 172473, loss = 0.903883
    Iter 172474, loss = 0.820817
    Iter 172475, loss = 0.971768
    Iter 172476, loss = 1.13689
    Iter 172477, loss = 1.14364
    Iter 172478, loss = 0.898286
    Iter 172479, loss = 1.13082
    Iter 172480, loss = 0.942493
    Iter 172481, loss = 1.24602
    Iter 172482, loss = 1.41656
    Iter 172483, loss = 2.0268
    Iter 172484, loss = 1.85192
    Iter 172485, loss = 0.975864
    Iter 172486, loss = 1.64831
    Iter 172487, loss = 1.02136
    Iter 172488, loss = 1.39286
    Iter 172489, loss = 1.18648
    Iter 172490, loss = 0.867372
    Iter 172491, loss = 1.01302
    Iter 172492, loss = 1.37601
    Iter 172493, loss = 1.04559
    Iter 172494, loss = 1.44399
    Iter 172495, loss = 1.29064
    Iter 172496, loss = 1.1171
    Iter 172497, loss = 1.38955
    Iter 172498, loss = 1.26175
    Iter 172499, loss = 1.36192
    
    Optimization Finished
    


### Training for word2vec embedding (For Bengali words)


```python
embeddings_beng = tf.Variable(
    tf.random_uniform([vocabulary_size_beng, embedding_size], -1.0, 1.0))

nce_weights_beng = tf.Variable(
  tf.truncated_normal([vocabulary_size_beng, embedding_size],
                      stddev=1.0 / math.sqrt(embedding_size)))
nce_biases_beng = tf.Variable(tf.zeros([vocabulary_size_beng]))

# Initializing the variables
init = tf.global_variables_initializer()
```


```python
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

            

```


```python

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
```

    Iter 35806, loss = 0.918012
    Iter 35807, loss = 1.19433
    Iter 35808, loss = 0.576221
    Iter 35809, loss = 1.23828
    Iter 35810, loss = 1.04737
    Iter 35811, loss = 0.971104
    Iter 35812, loss = 0.607476
    Iter 35813, loss = 0.733661
    Iter 35814, loss = 0.612409
    Iter 35815, loss = 1.11281
    Iter 35816, loss = 1.00669
    Iter 35817, loss = 0.973409
    Iter 35818, loss = 0.56991
    Iter 35819, loss = 0.937719
    Iter 35820, loss = 0.389082
    Iter 35821, loss = 0.393635
    Iter 35822, loss = 0.385571
    Iter 35823, loss = 0.374355
    
    Optimization Finished
 
    


### Creating Train, Validation, and Test set

Randomly shuffling the complete dataset (not yet embedded with word2vec embeddings which was learned just now), 
and then splitting it into train, validation and test set


```python
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
```

### Function for bucketing and generating batches

Mini-batch training requires all lines in a batch to be of equal length.
We have different lines of different lengths. 

A solution is to fill shorter sentences with PADs so that length of all sentences become equal.
But, if one sentence in a batch has 20 words, and the same batch has another sentence with one word, then the latter sentence will have to be filled in by at least 19 pads. If most of the sentences start to have more PADs than actual content, training can be problematic.

The solution to that is bucketing. First the sentences in the total list are sorted. After that sentences of similar lengths will be closer to each other. Batches are then formed with sentences of similar lengths. Much less padding will be required to turn sentences of similar lengths into sentences of equal lengths. 

Also while creating the batch, the input samples (the Engish lines) will have their words embedded using the recently trained embedding matrix for English. The output samples (the labels) will simply contain the index of the target Bengali word in the Bengali vocabulary list. The labels being in this format will be easier to train with sparse_softmax_cross_entropy cost function of Tensorflow. 


```python
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


```

### Creating train, validation, and test batches


```python
batch_size = 64

train_batch_eng,train_batch_beng = bucket_and_batch(train_eng,train_beng,batch_size)

val_batch_eng,val_batch_beng = bucket_and_batch(val_eng,val_beng,batch_size)

test_batch_eng,test_batch_beng = bucket_and_batch(test_eng,test_beng,batch_size)

```

### Saving processed data in another file.


```python
#Saving processed data in another file.

import pickle

PICK = [vocab_eng,vocab_beng,np_embedding_eng,np_embedding_beng,train_batch_eng,train_batch_beng,val_batch_eng,val_batch_beng,test_batch_eng,test_batch_beng]

with open('translationPICKLE', 'wb') as fp:
    pickle.dump(PICK, fp)
    
```

### Loading Pre-processed Data
(start of Machine Translation.ipynb)


```python
import pickle
import math
import numpy as np


with open ('translationPICKLE', 'rb') as fp:
    PICK = pickle.load(fp)

vocab_eng = PICK[0] 
vocab_beng = PICK[1] 
vocab_len = len(vocab_beng)

np_embedding_eng = PICK[2]
np_embedding_beng = PICK[3]
np_embedding_eng = np.asarray(np_embedding_eng,np.float32)
np_embedding_beng = np.asarray(np_embedding_beng,np.float32)

word_vec_dim = np_embedding_eng.shape[1] 

train_batch_x = PICK[4]
train_batch_y = PICK[5]

val_batch_x = PICK[6]
val_batch_y = PICK[7]

test_batch_x = PICK[8]
test_batch_y = PICK[9]
    
```

### Function for converting vector of size word_vec_dim into the closest representative english word. 


```python
def most_similar_eucli_eng(x):
    xminusy = np.subtract(np_embedding_eng,x)
    sq_xminusy = np.square(xminusy)
    sum_sq_xminusy = np.sum(sq_xminusy,1)
    eucli_dists = np.sqrt(sum_sq_xminusy)
    return np.argsort(eucli_dists)
    
def vec2word_eng(vec):   # converts a given vector representation into the represented word 
    most_similars = most_similar_eucli_eng(np.asarray(vec,np.float32))
    return vocab_eng[most_similars[0]]
    
```

### Hyperparameters and Placeholders.


```python
import tensorflow as tf

#Hyperparamters

h=8 #no. of heads
N=1 #no. of decoder and encoder layers
learning_rate=0.001
epochs = 200
keep_prob = tf.placeholder(tf.float32)

#Placeholders

x = tf.placeholder(tf.float32, [None,None,word_vec_dim])
y = tf.placeholder(tf.int32, [None,None])

output_len = tf.placeholder(tf.int32)

teacher_forcing = tf.placeholder(tf.bool)

tf_pad_mask = tf.placeholder(tf.float32,[None,None])
tf_illegal_position_masks = tf.placeholder(tf.float32,[None,None,None])

tf_pe_out = tf.placeholder(tf.float32,[None,None,None]) #positional codes for output
```

### Model Parameters.


```python
   
# Dimensions for Q (Query),K (Keys) and V (Values) for attention layers.

dqkv = 32 
    
#Parameters for attention sub-layers for all n encoders

Wq_enc = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
Wk_enc = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
Wv_enc = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
Wo_enc = tf.Variable(tf.truncated_normal(shape=[N,h*dqkv,word_vec_dim],stddev=0.01))

#Parameters for position-wise fully connected layers for n encoders

d = 1024
W1_enc = tf.Variable(tf.truncated_normal(shape=[N,1,1,word_vec_dim,d],stddev=0.01))
b1_enc = tf.Variable(tf.constant(0,tf.float32,shape=[N,d]))
W2_enc = tf.Variable(tf.truncated_normal(shape=[N,1,1,d,word_vec_dim],stddev=0.01))
b2_enc = tf.Variable(tf.constant(0,tf.float32,shape=[N,word_vec_dim]))
    
#Parameters for 2 attention sub-layers for all n decoders

Wq_dec_1 = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
Wk_dec_1 = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
Wv_dec_1 = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
Wo_dec_1 = tf.Variable(tf.truncated_normal(shape=[N,h*dqkv,word_vec_dim],stddev=0.01))
Wq_dec_2 = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
Wk_dec_2 = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
Wv_dec_2 = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
Wo_dec_2 = tf.Variable(tf.truncated_normal(shape=[N,h*dqkv,word_vec_dim],stddev=0.01))
    
#Parameters for position-wise fully connected layers for n decoders

d = 1024
W1_dec = tf.Variable(tf.truncated_normal(shape=[N,1,1,word_vec_dim,d],stddev=0.01))
b1_dec = tf.Variable(tf.constant(0,tf.float32,shape=[N,d]))
W2_dec = tf.Variable(tf.truncated_normal(shape=[N,1,1,d,word_vec_dim],stddev=0.01))
b2_dec = tf.Variable(tf.constant(0,tf.float32,shape=[N,word_vec_dim]))
    
#Layer Normalization parameters for encoder

scale_enc_1 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
shift_enc_1 = tf.Variable(tf.zeros([N,1,1,word_vec_dim]),dtype=tf.float32)

scale_enc_2 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
shift_enc_2 = tf.Variable(tf.zeros([N,1,1,word_vec_dim]),dtype=tf.float32)

#Layer Normalization parameters for decoder   

scale_dec_1 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
shift_dec_1 = tf.Variable(tf.zeros([N,1,1,word_vec_dim]),dtype=tf.float32)

scale_dec_2 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
shift_dec_2 = tf.Variable(tf.zeros([N,1,1,word_vec_dim]),dtype=tf.float32)

scale_dec_3 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
shift_dec_3 = tf.Variable(tf.zeros([N,1,1,word_vec_dim]),dtype=tf.float32)
```

### Function for generating a sequence of positional codes for positional encoding.


```python
def positional_encoding(seq_len,model_dimensions):
    pe = np.zeros((seq_len,model_dimensions,),np.float32)
    for pos in xrange(0,seq_len):
        for i in xrange(0,model_dimensions):
            pe[pos][i] = math.sin(pos/(10000**(2*i/model_dimensions)))
    return pe.reshape((seq_len,model_dimensions))
```

### Function for Layer Normalization 

https://arxiv.org/abs/1607.06450


```python
#modified version of def LN used here: 
#https://theneuralperspective.com/2016/10/27/gradient-topics/

def layer_norm(inputs,scale,shift,epsilon = 1e-5):

    mean, var = tf.nn.moments(inputs, [1,2], keep_dims=True)

    LN = tf.multiply((scale / tf.sqrt(var + epsilon)),(inputs - mean)) + shift
 
    return LN
```

### Function to pre-generate masks for illegal positions. 

These masks are to be used to fill illegal positions with -infinity (or a very low value eg. -2^30).

Illegal positions are positions of the decoder input tokens that aren't predicted at a given timestep.

{ In a transformer, the decoder input is of the same shape as the WHOLE decoder output sequence. One word for the sequence is predicted at each timestep (from left to right). So in most timesteps, the left side of the decoder input sequence will contain valid previously predicted output words, but the right side -the yet to be predicted side should contain some values that should be ignored and never attended. We make sure that they're ignored by masking it }

So, the illegal positions depends on the total output length and the no. of predicted output tokens.

The appropriate mask when i output tokens are predicted can be retrieved from mask[i-1] where mask is the return value from this function. The argument out_len that function takes, signifies the total length of the output. 

The masks are used to assign the value -2^30 to all positions in the tensor influenced by the illegal ones.
After going through the softmax layer, these positions become close to 0, as it should be.

Dynamically creating masks depending on the current position\timestep (depending on which the program can know which positions are legal and which aren't) is, however,
a bit troublesome with tensorflow tf_while_loop. 

I will be pre-generating all the masks with Python native code and feed the list of all required masks to the network at each training step (output length can be different at different training steps). 
                                                                 


```python
def generate_masks_for_illegal_positions(out_len):
    
    masks=np.zeros((out_len-1,out_len,out_len),dtype=np.float32)
    
    for i in xrange(1,out_len):
        mask = np.zeros((out_len,out_len),dtype=np.float32)
        mask[i:out_len,:] = -2**30
        mask[:,i:out_len] = -2**30
        masks[i-1] = mask
        
    return masks
```

### Function for Multi-Headed Attention.

Details: https://arxiv.org/pdf/1706.03762.pdf

Q = Query

K = Key

V = Value

d is the dimension for Q, K and V. 


```python

def attention(Q,K,V,d,filled=0,mask=False):

    K = tf.transpose(K,[0,2,1])
    d = tf.cast(d,tf.float32)
    
    softmax_component = tf.div(tf.matmul(Q,K),tf.sqrt(d))
    
    if mask == True:
        softmax_component = softmax_component + tf_illegal_position_masks[filled-1]
        
    result = tf.matmul(tf.nn.dropout(tf.nn.softmax(softmax_component),keep_prob),V)
 
    return result
       

def multihead_attention(Q,K,V,d,weights,filled=0,mask=False):
    
    Q_ = tf.reshape(Q,[-1,tf.shape(Q)[2]])
    K_ = tf.reshape(K,[-1,tf.shape(Q)[2]])
    V_ = tf.reshape(V,[-1,tf.shape(Q)[2]])

    heads = tf.TensorArray(size=h,dtype=tf.float32)
    
    Wq = weights['Wq']
    Wk = weights['Wk']
    Wv = weights['Wv']
    Wo = weights['Wo']
    
    for i in xrange(0,h):
        
       Q_w = tf.matmul(Q_,Wq[i])
       Q_w = tf.reshape(Q_w,[tf.shape(Q)[0],tf.shape(Q)[1],d])
        
       K_w = tf.matmul(K_,Wk[i])
       K_w = tf.reshape(K_w,[tf.shape(K)[0],tf.shape(K)[1],d])
        
       V_w = tf.matmul(V_,Wv[i])
       V_w = tf.reshape(V_w,[tf.shape(V)[0],tf.shape(V)[1],d])

       head = attention(Q_w,K_w,V_w,d,filled,mask)
            
       heads = heads.write(i,head)
        
    heads = heads.stack()
    
    concated = heads[0]
    
    for i in xrange(1,h):
        concated = tf.concat([concated,heads[i]],2)

    concated = tf.reshape(concated,[-1,h*d])
    out = tf.matmul(concated,Wo)
    out = tf.reshape(out,[tf.shape(heads)[1],tf.shape(heads)[2],word_vec_dim])
    
    return out
    
```

### Function for encoder

More details: https://arxiv.org/pdf/1706.03762.pdf


```python
def encoder(x,weights,attention_weights,dqkv):

    W1 = weights['W1']
    W2 = weights['W2']
    b1 = weights['b1']
    b2 = weights['b2']
    
    scale1 = weights['scale1']
    shift1 = weights['shift1']
    scale2 = weights['scale2']
    shift2 = weights['shift2']
    
    # SUBLAYER 1 (MASKED MULTI HEADED SELF ATTENTION)
    
    sublayer1 = multihead_attention(x,x,x,dqkv,attention_weights)
    sublayer1 = tf.nn.dropout(sublayer1,keep_prob)
    sublayer1 = layer_norm(sublayer1 + x,scale1,shift1)
    
    sublayer1_ = tf.reshape(sublayer1,[tf.shape(sublayer1)[0],1,tf.shape(sublayer1)[1],word_vec_dim])
    
    # SUBLAYER 2 (TWO 1x1 CONVOLUTIONAL LAYERS AKA POSITION WISE FULLY CONNECTED NETWORKS)
    
    sublayer2 = tf.nn.conv2d(sublayer1_, W1, strides=[1,1,1,1], padding='SAME')
    sublayer2 = tf.nn.bias_add(sublayer2,b1)
    sublayer2 = tf.nn.relu(sublayer2)
    
    sublayer2 = tf.nn.conv2d(sublayer2, W2, strides=[1,1,1,1], padding='SAME')
    sublayer2 = tf.nn.bias_add(sublayer2,b2)
    
    sublayer2 = tf.reshape(sublayer2,[tf.shape(sublayer2)[0],tf.shape(sublayer2)[2],word_vec_dim])
    
    sublayer2 = tf.nn.dropout(sublayer2,keep_prob)
    sublayer2 = layer_norm(sublayer2 + sublayer1,scale2,shift2)
    
    return sublayer2

```

### Function for decoder

More details: https://arxiv.org/pdf/1706.03762.pdf


```python
def decoder(y,enc_out,weights,masked_attention_weights,attention_weights,dqkv,mask=False,filled=0):

    W1 = weights['W1']
    W2 = weights['W2']
    b1 = weights['b1']
    b2 = weights['b2']
    
    scale1 = weights['scale1']
    shift1 = weights['shift1']
    scale2 = weights['scale2']
    shift2 = weights['shift2']
    scale3 = weights['scale3']
    shift3 = weights['shift3']
    
    # SUBLAYER 1 (MASKED MULTI HEADED SELF ATTENTION)

    sublayer1 = multihead_attention(y,y,y,dqkv,masked_attention_weights,filled,mask)
    sublayer1 = tf.nn.dropout(sublayer1,keep_prob)
    sublayer1 = layer_norm(sublayer1 + y,scale1,shift1)
    
    # SUBLAYER 2 (MULTIHEADED ENCODER-DECODER INTERLAYER ATTENTION)
    
    sublayer2 = multihead_attention(sublayer1,enc_out,enc_out,dqkv,attention_weights)
    sublayer2 = tf.nn.dropout(sublayer2,keep_prob)
    sublayer2 = layer_norm(sublayer2 + sublayer1,scale2,shift2)
    
    # SUBLAYER 3 (TWO 1x1 CONVOLUTIONAL LAYERS AKA POSITION WISE FULLY CONNECTED NETWORKS)
    
    sublayer2_ = tf.reshape(sublayer2,[tf.shape(sublayer2)[0],1,tf.shape(sublayer2)[1],word_vec_dim])
    
    sublayer3 = tf.nn.conv2d(sublayer2_, W1, strides=[1,1,1,1], padding='SAME')
    sublayer3 = tf.nn.bias_add(sublayer3,b1)
    sublayer3 = tf.nn.relu(sublayer3)
    
    sublayer3 = tf.nn.conv2d(sublayer3, W2, strides=[1,1,1,1], padding='SAME')
    sublayer3 = tf.nn.bias_add(sublayer3,b2)
    
    sublayer3 = tf.reshape(sublayer3,[tf.shape(sublayer3)[0],tf.shape(sublayer3)[2],word_vec_dim])
    
    sublayer3 = tf.nn.dropout(sublayer3,keep_prob)
    sublayer3 = layer_norm(sublayer3 + sublayer2,scale3,shift3)
    
    return sublayer3
```

### Function for Stacking Encoders.


```python
def stacked_encoders(layer_num,encoderin):
    
    for i in xrange(0,layer_num):
        
        encoder_weights = {
            
            'W1': W1_enc[i],
            'W2': W2_enc[i],
            'b1': b1_enc[i],
            'b2': b2_enc[i],
            'scale1': scale_enc_1[i],
            'shift1': shift_enc_1[i],
            'scale2': scale_enc_2[i],
            'shift2': shift_enc_2[i],
        }
        
        attention_weights = {
            
            'Wq': Wq_enc[i],
            'Wk': Wk_enc[i],
            'Wv': Wv_enc[i],
            'Wo': Wo_enc[i],                       
        }
        
        encoderin = encoder(encoderin,encoder_weights,attention_weights,dqkv)
    
    return encoderin
    
```

### Function for Stacking Decoders.


```python
def stacked_decoders(layer_num,decoderin,encoderout,filled):
    
    for j in xrange(0,layer_num):
        
        decoder_weights = {
            
            'W1': W1_dec[j],
            'W2': W2_dec[j],
            'b1': b1_dec[j],
            'b2': b2_dec[j],
            'scale1': scale_dec_1[j],
            'shift1': shift_dec_1[j],
            'scale2': scale_dec_2[j],
            'shift2': shift_dec_2[j],
            'scale3': scale_dec_3[j],
            'shift3': shift_dec_3[j],
        }
            
        masked_attention_weights = {
            
            'Wq': Wq_dec_1[j],
            'Wk': Wk_dec_1[j],
            'Wv': Wv_dec_1[j],
            'Wo': Wo_dec_1[j],                       
        }
        
        attention_weights = {
            
            'Wq': Wq_dec_2[j],
            'Wk': Wk_dec_2[j],
            'Wv': Wv_dec_2[j],
            'Wo': Wo_dec_2[j],                       
        }
            
        decoderin = decoder(decoderin,encoderout,
                            decoder_weights,
                            masked_attention_weights,
                            attention_weights,
                            dqkv,
                            mask=True,filled=filled)
    return decoderin
    
```

### predicted_embedding():

Given a probability distribution and an embedding matrix, this function returns the embedding of the word with the maximum probability in the given distribution.

### replaceSOS():

SOS signifies the start of sentence for the decoder. Also often represented as 'GO'. I am using an all ones vector as the first decoder input token. 
In the next time step, the SOS will be forgotten, and only the context of the previously predicted output (or the target output at the previous timestep, if teacher forcing is on) will be used.

### add_pred_to_output_lists():

This function will concatenate the last predicted output into a tensor of concatenated sequence of output tokens. 


```python
def predicted_embedding(out_prob_dist,tf_embd):
    out_index = tf.cast(tf.argmax(out_prob_dist,1),tf.int32)
    return tf.gather(tf_embd,out_index)

def replaceSOS(output,out_prob_dist):
    return output,tf.constant(1),tf.reshape(out_prob_dist,[tf.shape(x)[0],1,vocab_len])

def add_pred_to_output_list(decoderin_part_1,output,filled,out_probs,out_prob_dist):
    decoderin_part_1 = tf.concat([decoderin_part_1,output],1)
    filled += 1
    out_probs = tf.concat([out_probs,tf.reshape(out_prob_dist,[tf.shape(x)[0],1,vocab_len])],1)
    return decoderin_part_1,filled,out_probs
```

### Model Definition

It follows the encoder-decoder paradigm. The main exception from standard encoder-decoder paradigm, is that it uses 'transformers' instead of Reccurrent networks. The decoder undergoes a sequential processing, though. 

If teacher forcing is True, the decoder is made to guess the next output from the previous words in the actual target output, else the decoder predicts the next output from the previously predicted output of the decoder.

Details about the model: https://arxiv.org/pdf/1706.03762.pdf


```python
def model(x,y,teacher_forcing=True):
    
        
    # NOTE: tf.shape(x)[0] == batch_size
    
    encoderin = x # (should be already positionally encoded) 
    encoderin = tf.nn.dropout(encoderin,keep_prob)

    
    # ENCODER LAYERS

    encoderout = stacked_encoders(N,encoderin)
    

    # DECODER LAYERS

    decoderin_part_1 = tf.ones([tf.shape(x)[0],1,word_vec_dim],dtype=tf.float32) #represents SOS
    
    filled = tf.constant(1) 
    # no. of output words that are filled i.e already predicted - are stored in 'filled'
    # filled value is used to retrieve appropriate mask for illegal positions. 
    
    
    tf_embd = tf.convert_to_tensor(np_embedding_beng)
    Wpd = tf.transpose(tf_embd)
    # Wpd the transpose of the output embedding matrix will be used to convert the decoder output
    # into a probability distribution over the output language vocabulary. 
    
    out_probs = tf.zeros([tf.shape(x)[0],output_len,vocab_len],tf.float32)
    # out_probs will contain the list of probability distributions.

    #tf_while_loop since output_len will be dynamically defined during session run
    
    i=tf.constant(0)
    
    def cond(i,filled,decoderin_part_1,out_probs):
        return i<output_len
    
    def body(i,filled,decoderin_part_1,out_probs):
        
        decoderin_part_2 = tf.zeros([tf.shape(x)[0],(output_len-filled),word_vec_dim],dtype=tf.float32)
        
        decoderin = tf.concat([decoderin_part_1,decoderin_part_2],1)
        
        decoderin = tf.nn.dropout(decoderin,keep_prob)
        
        decoderout = stacked_decoders(N,decoderin,encoderout,filled)
        
        # decoderout shape (now) = batch_size x seq_len x word_vec_dim

        decoderout = tf.reduce_sum(decoderout,1) 
        # A weighted summation of the attended decoder input
        # decoderout shape (now) = batch_size x word_vec_dim
        
        # converting decoderout to probability distributions
        
        out_prob_dist = tf.matmul(decoderout,Wpd)
   
        # If teacher forcing is false, initiate predicted_embedding(). It guesses the output embeddings
        # to be that whose vocabulary index has maximum probability in out_prob_dist
        # (the current output probability distribution). The embedding is used in the next
        # iteration. 
        
        # If teacher forcing is true, use the embedding of target index from y (laebls) 
        # for the next iteration.
        
        output = tf.cond(tf.equal(teacher_forcing,tf.convert_to_tensor(False)),
                         lambda: predicted_embedding(out_prob_dist,tf_embd),
                         lambda: tf.gather(tf_embd,y[:,i]))
        
        # Position Encoding the output
        
        output = output + tf_pe_out[i]
        output = tf.reshape(output,[tf.shape(x)[0],1,word_vec_dim])
                                
        
        #concatenate with list of previous predicted output tokens
        
        decoderin_part_1,filled,out_probs = tf.cond(tf.equal(i,0),
                                        lambda:replaceSOS(output,out_prob_dist),
                                        lambda:add_pred_to_output_list(decoderin_part_1,output,filled,out_probs,out_prob_dist))
        
        return i+1,filled,decoderin_part_1,out_probs
            
    _,_,_,out_probs = tf.while_loop(cond,body,[i,filled,decoderin_part_1,out_probs],
                      shape_invariants=[i.get_shape(),
                                        filled.get_shape(),
                                        tf.TensorShape([None,None,word_vec_dim]),
                                        tf.TensorShape([None,None,vocab_len])])

    return out_probs
```

### Setting up cost function and optimizer


```python
# Construct Model
output = model(x,y,teacher_forcing)

#OPTIMIZER

cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=y)
cost = tf.multiply(cost,tf_pad_mask) #mask used to remove loss effect due to PADS
cost = tf.reduce_mean(cost)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.98,epsilon=1e-9).minimize(cost)

#wanna add some temperature?

"""temperature = 0.7
scaled_output = tf.log(output)/temperature
softmax_output = tf.nn.softmax(scaled_output)"""

#(^Use it with "#prediction_int = np.random.choice(range(vocab_len), p=array.ravel())")

softmax_output = tf.nn.softmax(output)

```

### Function to create a Mask for pads' effect on cost. 

The mask will have the same shape as the batch of labels but with the value 0 wherever there is a PAD.
The mask will be element-wise multipled to the cost (before its averaged), so that any position in the cost tensor that is effected by the PAD will be multiplied by 0. This way, the effect of PADs (which we don't need to care about) on the cost (and therefore on the gradients) can be nullified. 


```python
def create_pad_Mask(output_batch):
    pad_index = vocab_beng.index('<PAD>')
    mask = np.ones_like((output_batch),np.float32)
    for i in xrange(len(mask)):
        for j in xrange(len(mask[i])):
            if output_batch[i,j]==pad_index:
                mask[i,j]=0
    return mask
```

### Training .....

The input batch is positionally encoded before its fed to the network.


```python
import string
import random
from __future__ import print_function

init = tf.global_variables_initializer()

with tf.Session() as sess: # Start Tensorflow Session
    
    saver = tf.train.Saver() 
    # Prepares variable for saving the model
    sess.run(init) #initialize all variables
    step = 0   
    best_loss = 999
    display_step = 1
    warm_up_steps = 7000
    
    while step < epochs:
        
        batch_len = len(train_batch_x)
        shuffled_indices = np.arange(batch_len)
        np.random.shuffle(shuffled_indices)
        
        for i in xrange(0,batch_len):
            
            # Adaptive learning rate formula
            #learning_rate = ((word_vec_dim)**(-0.5))*min((step*batch_len+i+1)**(-0.5),(step*batch_len+i+1)*warm_up_steps**(-1.5))

            sample_no = np.random.randint(0,len(train_batch_x[0]))
            print("\nCHOSEN SAMPLE NO.: "+str(sample_no))
            
            if i%display_step==0:
                
                print("\nEpoch: "+str(step+1)+" Iteration: "+str(i+1))
                print("\nSAMPLE TEXT:")
                for vec in train_batch_x[shuffled_indices[i]][sample_no]:
                    print(vec2word_eng(vec),end=" ")
                print("\n")
                
            input_seq_len = len(train_batch_x[shuffled_indices[i]][0])
            
            pe_in = positional_encoding(input_seq_len,word_vec_dim)
            pe_in = pe_in.reshape((1,input_seq_len,word_vec_dim))
            
            output_seq_len = len(train_batch_y[shuffled_indices[i]][0])
            
            
            
            illegal_position_masks = generate_masks_for_illegal_positions(output_seq_len)
            
            pe_out = positional_encoding(output_seq_len,word_vec_dim)
            pe_out = pe_out.reshape((output_seq_len,1,word_vec_dim))
    
            
            rand = random.randint(0,2) #determines chance of using Teacher Forcing
            if rand==1:
                random_bool = True
            else:
                random_bool = False
            
            pad_mask = create_pad_Mask(train_batch_y[shuffled_indices[i]])
            
            # Run optimization operation (backpropagation)
            _,loss,out = sess.run([optimizer,cost,softmax_output],
                                  feed_dict={x: (train_batch_x[shuffled_indices[i]]+pe_in), 
                                             y: train_batch_y[shuffled_indices[i]],
                                             keep_prob: 0.9,
                                             output_len: len(train_batch_y[shuffled_indices[i]][0]),
                                             tf_pad_mask: pad_mask,
                                             tf_illegal_position_masks: illegal_position_masks,
                                             tf_pe_out: pe_out,
                                             teacher_forcing: False #random_bool
                                             # feed random bool for randomized teacher forcing. 
                                             })
            
            if i%display_step==0:
                
                print("\nPREDICTED TRANSLATION OF THE SAMPLE:\n")
                flag = 0
                for array in out[sample_no]:
                    
                    #prediction_int = np.random.choice(range(vocab_len), p=array.ravel()) 
                    #(^use this if you want some variety)
                    #(or use this what's below:)
                    
                    prediction_int = np.argmax(array)
                    
                    if vocab_beng[prediction_int] in string.punctuation or flag==0: 
                        print(vocab_beng[prediction_int],end='')
                    else:
                        print(" "+vocab_beng[prediction_int],end='')
                    flag=1
                print("\n")
                
                print("ACTUAL TRANSLATION OF THE SAMPLE:\n")
                for index in train_batch_y[shuffled_indices[i]][sample_no]:
                    print(vocab_beng[index],end=" ")
                print("\n")
            
            print("loss="+str(loss))
                  
            if(loss<best_loss):
                best_loss = loss
                saver.save(sess, 'Model_Backup/translation_model.ckpt')

        step=step+1
    
```

    
    CHOSEN SAMPLE NO.: 29
    
    Epoch: 1 Iteration: 1
    
    SAMPLE TEXT:
    he is having lunch now <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    দেখছে দেখছে দেখছে দেখছে দেখছে দেখছে দেখছে দেখছে দেখছে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তিনি এখন লাঞচ করছেন। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=297.772
    
    CHOSEN SAMPLE NO.: 0
    
    Epoch: 1 Iteration: 2
    
    SAMPLE TEXT:
    they have got guns <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ঠিকানাটা ঠিকানাটা ঠিকানাটা ঠিকানাটা ঠিকানাটা ঠিকানাটা ঠিকানাটা ঠিকানাটা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ওদের কাছে বনদক রযেছে। <EOS> <PAD> <PAD> <PAD> 
    
    loss=242.409
    
    CHOSEN SAMPLE NO.: 55
    
    Epoch: 1 Iteration: 3
    
    SAMPLE TEXT:
    it seems that i have lost my keys <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তারাতারি তারাতারি তারাতারি তারাতারি তারাতারি তারাতারি তারাতারি তারাতারি তারাতারি তারাতারি তারাতারি তারাতারি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    মনে হচছে আমি আমার চাবি হারিযে ফেলেছি। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=358.392
    .
    .
    .
    .
        CHOSEN SAMPLE NO.: 20
    
    Epoch: 128 Iteration: 44
    
    SAMPLE TEXT:
    she knows where we live <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    উনি জানে আমরা কোথায থাকি। <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তিনি জানেন আমরা কোথায থাকি। <EOS> <PAD> <PAD> <PAD> 
    
    loss=1.27356
    
    CHOSEN SAMPLE NO.: 28
    
    Epoch: 128 Iteration: 45
    
    SAMPLE TEXT:
    tom 's strong <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টম শকতিশালী। <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম শকতিশালী। <EOS> <PAD> <PAD> <PAD> 
    
    loss=0.466606
    
    CHOSEN SAMPLE NO.: 8
    
    Epoch: 128 Iteration: 46
    
    SAMPLE TEXT:
    stop that woman <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ওই ফিরে আটকাও। <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ওই মহিলাটিকে থামান। <EOS> <PAD> <PAD> 
    
    loss=0.628224
    
    CHOSEN SAMPLE NO.: 30
    
    Epoch: 128 Iteration: 47
    
    SAMPLE TEXT:
    do you have my book <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমার কাছে কি ভালো বইটা আছে <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনার কাছে কি আমার বইটা আছে <EOS> <PAD> <PAD> 
    
    loss=1.2308
    
    CHOSEN SAMPLE NO.: 43
    
    Epoch: 128 Iteration: 48
    
    SAMPLE TEXT:
    would you like to come inside <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তমি কি ভেতরে আসবেন <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনারা কি ভেতরে আসবেন <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=1.67444
    
    CHOSEN SAMPLE NO.: 0
    
    Epoch: 128 Iteration: 49
    
    SAMPLE TEXT:
    are you busy tomorrow night <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কোনো কি কাল আছে <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তই কি কাল রাতে বযসত থাকবি <EOS> <PAD> 
    
    loss=0.907989
    
    CHOSEN SAMPLE NO.: 57
    
    Epoch: 128 Iteration: 50
    
    SAMPLE TEXT:
    stand up <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    দাডান। <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    দাডা <EOS> <PAD> <PAD> <PAD> 
    
    loss=0.790484
    
    CHOSEN SAMPLE NO.: 21
    
    Epoch: 128 Iteration: 51
    
    SAMPLE TEXT:
    tom did that himself <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টম ওটা বযবসথা <EOS> <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম ওটা নিজেই করলো। <EOS> <PAD> <PAD> <PAD> 
    
    loss=0.93948
    
    CHOSEN SAMPLE NO.: 56
    
    Epoch: 129 Iteration: 1
    
    SAMPLE TEXT:
    tom is still at school <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টম এখনো নামবার <EOS> <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম এখনো ইসকলে। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=0.921481
    
    CHOSEN SAMPLE NO.: 15
    
    Epoch: 129 Iteration: 2
    
    SAMPLE TEXT:
    were you there <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তমি কি ওখানে ছিলে <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি কি ওখানে ছিলে <EOS> <PAD> 
    
    loss=0.486593
    
    CHOSEN SAMPLE NO.: 58
    
    Epoch: 129 Iteration: 3
    
    SAMPLE TEXT:
    is there a public toilet in this building <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এই কি কি এই আছে আছে আছে <EOS> <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এই ইমারতটিতে কি কোনো সরবজনীন শৌচাগার আছে <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=1.76835
    
    CHOSEN SAMPLE NO.: 28
    
    Epoch: 129 Iteration: 4
    
    SAMPLE TEXT:
    i 'm not tom <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি এখনই নই। <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি টম নই। <EOS> <PAD> <PAD> <PAD> 
    
    loss=0.733902
    
    CHOSEN SAMPLE NO.: 59
    
    Epoch: 129 Iteration: 5
    
    SAMPLE TEXT:
    do you understand french <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তমি কি ফরাসি ভাষা বলতে <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি কি ফরাসি ভাষা বঝতে পারো <EOS> 
    
    loss=0.842568
    
    CHOSEN SAMPLE NO.: 27
    
    Epoch: 129 Iteration: 6
    
    SAMPLE TEXT:
    i 'm happy to see you again <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আপনাকে আবার দেখে খশি হযেছি। <EOS> <EOS> <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনাদেরকে আবার দেখে খশি হযেছি। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=1.91991
    
    CHOSEN SAMPLE NO.: 53
    
    Epoch: 129 Iteration: 7
    
    SAMPLE TEXT:
    i could not walk <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি এবার পারব <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি হাটতে পারিনি। <EOS> <PAD> <PAD> <PAD> 
    
    loss=0.91238
    
    CHOSEN SAMPLE NO.: 0
    
    Epoch: 129 Iteration: 8
    
    SAMPLE TEXT:
    i want to be as rich as tom <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি ওই টমের হতে হতে <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি টমের মত ধনী হতে চাই। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=1.78097
    
    CHOSEN SAMPLE NO.: 4
    
    Epoch: 129 Iteration: 9
    
    SAMPLE TEXT:
    you should eat vegetables <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তোমার শাকসবজি খাওযা উচিত। উচিত। <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তোমার শাকসবজি খাওযা উচিত। <EOS> <PAD> <PAD> <PAD> 
    
    loss=0.584272
    
    CHOSEN SAMPLE NO.: 52
    
    Epoch: 129 Iteration: 10
    
    SAMPLE TEXT:
    do come again <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আবার আসবে কিনত। <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আবার আসবে কিনত। <EOS> <PAD> <PAD> 
    
    loss=0.749034
    
    CHOSEN SAMPLE NO.: 58
    
    Epoch: 129 Iteration: 11
    
    SAMPLE TEXT:
    we will scream <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমরা চেচাবো। <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমরা চিৎকার করবো। <EOS> <PAD> <PAD> 
    
    loss=0.519659
    
    CHOSEN SAMPLE NO.: 18
    
    Epoch: 129 Iteration: 12
    
    SAMPLE TEXT:
    do you have time <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আপনার হাতে সময আছে আছে <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তোমার হাতে সময আছে <EOS> <PAD> <PAD> <PAD> 
    
    loss=0.776177
    
    CHOSEN SAMPLE NO.: 26
    
    Epoch: 129 Iteration: 13
    
    SAMPLE TEXT:
    i eat everything <EOS> 
    .
    .
    .



```python
def word2vec(word):
    if word in vocab_eng:
        return np_embedding_eng[vocab_eng.index(word)]
    else:
        return np_embedding_eng[vocab_eng.index('<PAD>')]
```

### Prediction.


```python
with tf.Session() as sess: # Begin session
    
    print('Loading pre-trained weights for the model...')
    saver = tf.train.Saver()
    saver.restore(sess, 'Model_Backup/translation_model.ckpt')
    sess.run(tf.global_variables())
    print('\nRESTORATION COMPLETE\n')
    
    
    test = ['who','are','you'] # Enter tokenized text here
    test = map(word2vec,test)
    test = np.asarray(test,np.float32)
    test = test.reshape((1,test.shape[0],test.shape[1]))
    
    input_seq_len = test.shape[0]
    pe_in = positional_encoding(input_seq_len,word_vec_dim)
    pe_in = pe_in.reshape((1,input_seq_len,word_vec_dim))
    test_pe = test+pe_in
    
    output_seq_len = int(input_seq_len+20)
    illegal_position_masks = generate_masks_for_illegal_positions(output_seq_len)
    pe_out = positional_encoding(output_seq_len,word_vec_dim) 
    pe_out = pe_out.reshape((output_seq_len,1,word_vec_dim))
        
    out = sess.run(softmax_output,
                          feed_dict={x: test_pe,
                                     y: np.zeros((1,1),np.int32), 
                                     # y value doesn't matter here.
                                     # feeding y, because the network graph requires y.
                                     # but its value won't actually be used in this case. 
                                     keep_prob: 1,
                                     output_len: output_seq_len,
                                     tf_pe_out: pe_out,
                                     tf_illegal_position_masks: illegal_position_masks,
                                     teacher_forcing: False
                                    })

    for array in out[0]:
        if vocab_beng[np.argmax(array)] != '<EOS>':
            print(vocab_beng[np.argmax(array)],end=' ')
    


```

    Loading pre-trained weights for the model...
    INFO:tensorflow:Restoring parameters from Model_Backup/translation_model.ckpt
    
    RESTORATION COMPLETE
    
    আপলোড করছিল। করছিস করছিল। করছিল। করছিল। করছিস করছিস করছিস করছিস করছিল। করছিস করছিস করছিস করছিল। 

### Some comments:

The model seems to fit well on the training data even using only 1 layer of encoder and decoder.
In fact, it seems to be fitting better when I am training with 1 layer of encoder and decoder.
However, I suppose the model is most likely 'memorizing' and overfitting. I tried some predictions below,
the results aren't good. Things will become clearer with validation, evaluation metrics and testing. 


At each timestep the decoder output is in the format batch_size x sequence_length x word_vector_dimensions.
I am first adding the decoder output along the second axis, to transform the shape into batch_size x word_vector_dimensions.
This should effectively be a weighted summation, and the 'weights' can be though to be assigned by all these encoder decoder attention layers.

I am not sure what the original implementation does with the decoder output before converting it linearly into a probability distribution.

I am using output language word embedding matrix (say E) to convert the transformed decoder output into a probability distribution.

Probability distribution of next word = transpose(E) x (decoder_output after summation along 2nd axis)

The paper recommended something along that line. Using embedding matrix seemed to produce much better results. 

Even though, I included an option to use randomzied teacher's forcing, I kept teacher forcing off through out this training to check if it can still fit on the training data. 


### TO DO

* Evaluation (BLEU\METEOR etc.)
* Validation
* Testing 

(For now, I was just checking if the model can at least fit on the training data- whether it overfits or not is to yet to be checked)
    
