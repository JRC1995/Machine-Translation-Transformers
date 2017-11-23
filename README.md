
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

### Function for generating a sequence of positional codes that will be used for positional encoding


```python
def positional_encoding(seq_len,model_dimensions):
    pe = np.zeros((seq_len,model_dimensions,),np.float32)
    for pos in xrange(0,seq_len):
        for i in xrange(0,model_dimensions):
            pe[pos][i] = math.sin(pos/(10000**(2*i/model_dimensions)))
    return pe.reshape((seq_len,model_dimensions))
```

### Hyperparameters.


```python
import tensorflow as tf

h=8 #no. of heads
N=3 #no. of decoder and encoder layers
learning_rate=0.008
iters = 200
max_len = 25
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None,None,word_vec_dim])
y = tf.placeholder(tf.int32, [None,None])
output_len = tf.placeholder(tf.int32)
teacher_forcing = tf.placeholder(tf.bool)
tf_mask = tf.placeholder(tf.float32,[None,None])
```

### Function for Layer Normalization 


```python
#modified version of def LN used here: 
#https://theneuralperspective.com/2016/10/27/gradient-topics/

def layer_norm(inputs,scale,shift,epsilon = 1e-5):

    mean, var = tf.nn.moments(inputs, [1,2], keep_dims=True)

    LN = scale * (inputs - mean) / tf.sqrt(var + epsilon) + shift
 
    return LN
```

### Pre-generating all possible masks (for masked multi-head attention)

These masks are to be used to fill illegal positions with infinity (or a very high value eg. 2^30).

Illegal positions are positions in the decoder input sequence which may be filled in the future timestep,
but currently not filled. These positions shouldn't be allowed to influence the network.

The masks are used to assign the value 2^30 to all positions in the tensor influenced by the illegal ones.
After going through the softmax layer, these positions become close to 0, as it should be.

Dynamically creating masks depending on the current position\timestep (depending on which the program can know which positions are legal and which aren't) is, however,
a bit troublesome with tensorflow tf_while_loop. 

I am pre-generating all possible masks here, and packing them into a tensor such that the network can dynamically
access the required mask using the index of the tensor (the index will be the same as the timestep) 
                                                                    
                                                                   


```python
masks=np.zeros((max_len-1,max_len,max_len),dtype=np.float32)

for i in xrange(1,max_len):
    mask = np.zeros((max_len,max_len),dtype=np.float32)
    mask[i:max_len,:] = -2**30
    mask[:,i:max_len] = -2**30
    masks[i-1] = mask
    
masks = tf.convert_to_tensor(masks) 
```

### Function for Multi-Headed Attention.

Details: https://arxiv.org/pdf/1706.03762.pdf

Q = Query

K = Key

V = Value

d is the dimension for Q, K and V. 


```python

def attention(Q,K,V,d,pos=0,mask=False):

    K = tf.transpose(K,[0,2,1])
    d = tf.cast(d,tf.float32)
    
    softmax_component = tf.div(tf.matmul(Q,K),tf.sqrt(d))
    
    if mask == True:
        softmax_component = softmax_component + masks[pos-1]
        

    result = tf.matmul(tf.nn.dropout(tf.nn.softmax(softmax_component),keep_prob),V)
 
    return result
       

def multihead_attention(Q,K,V,d,weights,pos=0,mask=False):
    
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

        head = attention(Q_w,K_w,V_w,d,pos,mask)
            
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
    
    d=1024
    
    W1 = weights['W1']
    W2 = weights['W2']
    b1 = weights['b1']
    b2 = weights['b2']
    
    scale1 = weights['scale1']
    shift1 = weights['shift1']
    scale2 = weights['scale2']
    shift2 = weights['shift2']
    
    
    sublayer1 = multihead_attention(x,x,x,dqkv,attention_weights)
    sublayer1 = tf.nn.dropout(sublayer1,keep_prob)
    sublayer1 = layer_norm(sublayer1 + x,scale1,shift1)
    
    sublayer1_ = tf.reshape(sublayer1,[-1,word_vec_dim])
    
    sublayer2 = tf.matmul(tf.nn.relu(tf.matmul(sublayer1_,W1)+b1),W2) + b2
    sublayer2 = tf.reshape(sublayer2,tf.shape(x))
    sublayer2 = tf.nn.dropout(sublayer2,keep_prob)
    sublayer2 = layer_norm(sublayer2 + sublayer1,scale2,shift2)
    
    return sublayer2

```

### Function for decoder

More details: https://arxiv.org/pdf/1706.03762.pdf


```python
def decoder(y,enc_out,weights,attention_weights_1,attention_weights_2,dqkv,mask=False,pos=0):
    
    d=1024
    
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

    sublayer1 = multihead_attention(y,y,y,dqkv,attention_weights_1,pos,mask)
    sublayer1 = tf.nn.dropout(sublayer1,keep_prob)
    sublayer1 = layer_norm(sublayer1 + y,scale1,shift1)
    
    sublayer2 = multihead_attention(sublayer1,enc_out,enc_out,dqkv,attention_weights_2)
    sublayer2 = tf.nn.dropout(sublayer2,keep_prob)
    sublayer2 = layer_norm(sublayer2 + sublayer1,scale2,shift2)
    
    sublayer2_ = tf.reshape(sublayer2,[-1,word_vec_dim])
    
    sublayer3 = tf.matmul(tf.nn.relu(tf.matmul(sublayer2_,W1)+b1),W2) + b2
    sublayer3 = tf.reshape(sublayer3,tf.shape(y))
    sublayer3 = tf.nn.dropout(sublayer3,keep_prob)
    sublayer3 = layer_norm(sublayer3 + sublayer2,scale3,shift3)
    
    return sublayer3
```

### Model Definition

It follows the encoder-decoder architecture. The main exception from standard encoder-decoder paradigm, is that it uses 'transformers' instead of Reccurrent networks. 

If teacher forcing is True, the decoder is made to guess the next output from the previous words in the actual target output, else the decoder predicts the next output from the previously predicted output of the decoder.

Details about the model: https://arxiv.org/pdf/1706.03762.pdf


```python
def fn1(out_prob_dist,tf_embd):
    out_index = tf.cast(tf.argmax(out_prob_dist,1),tf.int32)
    return tf.gather(tf_embd,out_index)

def fn2_1(output,out_prob_dist):
    return output,tf.constant(1),tf.reshape(out_prob_dist,[tf.shape(x)[0],1,vocab_len])

def fn2_2(decoderin_part1,output,filled,out_probs,out_prob_dist):
    decoderin_part1 = tf.concat([decoderin_part1,output],1)
    filled += 1
    out_probs = tf.concat([out_probs,tf.reshape(out_prob_dist,[tf.shape(x)[0],1,vocab_len])],1)
    return decoderin_part1,filled,out_probs
    
def model(x,y,teacher_forcing=True):
    
    #dimensions for Q,K and V for attention layers. 
    dqkv = 32 
    
    #Parameters for attention sub-layers for all n encoders
    Wq_enc = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
    Wk_enc = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
    Wv_enc = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
    Wo_enc = tf.Variable(tf.truncated_normal(shape=[N,h*dqkv,word_vec_dim],stddev=0.01))
    
    #Parameters for fully connected layers for n encoders
    d = 1024
    W1_enc = tf.Variable(tf.truncated_normal(shape=[N,word_vec_dim,d],stddev=0.01))
    b1_enc = tf.Variable(tf.truncated_normal(shape=[N,1,d],stddev=0.01))
    W2_enc = tf.Variable(tf.truncated_normal(shape=[N,d,word_vec_dim],stddev=0.01))
    b2_enc = tf.Variable(tf.truncated_normal(shape=[N,1,word_vec_dim],stddev=0.01))
    
    #Parameters for 2 attention sub-layers for all n decoders
    Wq_dec_1 = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
    Wk_dec_1 = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
    Wv_dec_1 = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
    Wo_dec_1 = tf.Variable(tf.truncated_normal(shape=[N,h*dqkv,word_vec_dim],stddev=0.01))
    Wq_dec_2 = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
    Wk_dec_2 = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
    Wv_dec_2 = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
    Wo_dec_2 = tf.Variable(tf.truncated_normal(shape=[N,h*dqkv,word_vec_dim],stddev=0.01))
    
    #Parameters for fully connected layers for n decoders
    d = 1024
    W1_dec = tf.Variable(tf.truncated_normal(shape=[N,word_vec_dim,d],stddev=0.01))
    b1_dec = tf.Variable(tf.truncated_normal(shape=[N,1,d],stddev=0.01))
    W2_dec = tf.Variable(tf.truncated_normal(shape=[N,d,word_vec_dim],stddev=0.01))
    b2_dec = tf.Variable(tf.truncated_normal(shape=[N,1,word_vec_dim],stddev=0.01))
    
    #Layer Normalization parameters for encoder and decoder   
    scale_enc_1 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
    shift_enc_1 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
    scale_enc_2 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
    shift_enc_2 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
    scale_dec_1 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
    shift_dec_1 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
    scale_dec_2 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
    shift_dec_2 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
    scale_dec_3 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
    shift_dec_3 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
    
    #Parameters for the linear layers converting decoder output to probability distibutions.   
    d=1024
    Wpd1 = tf.Variable(tf.truncated_normal(shape=[max_len*word_vec_dim,d],stddev=0.01))
    bpd1 = tf.Variable(tf.truncated_normal(shape=[1,d],stddev=0.01))
    Wpd2 = tf.Variable(tf.truncated_normal(shape=[d,vocab_len],stddev=0.01))
    bpd2 = tf.Variable(tf.truncated_normal(shape=[1,vocab_len],stddev=0.01))
                         
    encoderin = x #should be already positionally encoded 
    encoderin = tf.nn.dropout(encoderin,keep_prob)
    
    #all position encodings for outputs
    pe_out = tf.constant(positional_encoding(max_len,word_vec_dim)) 
    pe_out = tf.reshape(pe_out,[max_len,1,word_vec_dim])
    
    #encoder layers
    
    for i in xrange(0,N):
        
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
        
    encoderout = encoderin
    
    #decoder_layers
    
    #tf.shape(x)[0] == batch_size
    
    decoderin_part1 = tf.ones([tf.shape(x)[0],1,word_vec_dim],dtype=tf.float32)
    filled = tf.constant(1) #no. of output words that are filled
    
    tf_embd = tf.convert_to_tensor(np_embedding_beng)
    
    out_probs = tf.zeros([tf.shape(x)[0],output_len,vocab_len],tf.float32)

    #tf_while_loop since output_len will be dynamically defined during session run
    
    i=tf.constant(0)
    
    def cond(i,filled,decoderin_part1,out_probs):
        return i<output_len
    
    def body(i,filled,decoderin_part1,out_probs):
        
        decoderin_part2 = tf.zeros([tf.shape(x)[0],(max_len-filled),word_vec_dim],dtype=tf.float32)
        decoderin = tf.concat([decoderin_part1,decoderin_part2],1)
        decoderin = tf.nn.dropout(decoderin,keep_prob)
        
        for j in xrange(0,N):
            
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
            
            attention_weights_1 = {
                
                'Wq': Wq_dec_1[j],
                'Wk': Wk_dec_1[j],
                'Wv': Wv_dec_1[j],
                'Wo': Wo_dec_1[j],                       
             }
            
            attention_weights_2 = {
                
                'Wq': Wq_dec_2[j],
                'Wk': Wk_dec_2[j],
                'Wv': Wv_dec_2[j],
                'Wo': Wo_dec_2[j],                       
             }
            
            decoderin = decoder(decoderin,encoderout,
                                decoder_weights,
                                attention_weights_1,
                                attention_weights_2,
                                dqkv,
                                mask=True,pos=filled)
            
        decoderout = decoderin
        #decoderout shape = batch_size x seq_len x word_vec_dim
        
        #converting to probability distributions
        decoderout = tf.reshape(decoderout,[tf.shape(x)[0],max_len*word_vec_dim])
        
        # A two fully connected feed forward layer for transforming dimensions
        # (onverting to probability distributions)
        
        out_prob_dist = tf.nn.relu(tf.matmul(decoderout,Wpd1)+bpd1)
        out_prob_dist = tf.matmul(out_prob_dist,Wpd2)+bpd2
        out_prob_dist = tf.reshape(out_prob_dist,[tf.shape(x)[0],vocab_len])
   
        # if teacher forcing is false, initiate fn1(). It guesses the output embeddings
        # to be those whose vocabulary index has maximum probability in out_prob_dist
        # (the current output probability distribution). The embedding is used in the next
        # iteration. 
        
        # if teacher forcing is true, use the embedding of target index from y for the next 
        # iteration.
        
        output = tf.cond(tf.equal(teacher_forcing,tf.convert_to_tensor(False)),
                         lambda: fn1(out_prob_dist,tf_embd),
                         lambda: tf.gather(tf_embd,y[:,i]))
        
        # Position Encoding the output
        output = output + pe_out[i]
        output = tf.reshape(output,[tf.shape(x)[0],1,word_vec_dim])
                                
        
        #concatenate with previous batch_outputs
        decoderin_part1,filled,out_probs = tf.cond(tf.equal(i,0),
                                        lambda:fn2_1(output,out_prob_dist),
                                        lambda:fn2_2(decoderin_part1,output,filled,out_probs,out_prob_dist))
        
        return i+1,filled,decoderin_part1,out_probs
            
    _,_,_,out_probs = tf.while_loop(cond,body,[i,filled,decoderin_part1,out_probs],
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
cost = tf.multiply(cost,tf_mask) #mask used to remove loss effect due to PADS
cost = tf.reduce_mean(cost)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.98,epsilon=1e-9).minimize(cost)

#wanna add some temperature?

"""temperature = 0.7
scaled_output = tf.log(output)/temperature
softmax_output = tf.nn.softmax(scaled_output)"""

#(^Use it with "#prediction_int = np.random.choice(range(vocab_len), p=array.ravel())")

softmax_output = tf.nn.softmax(output)

```

### Function to create a Mask (to 'mask' the effect of PADs in the target sequence)

The mask will have the same shape as the batch of target sequences (sequence of indices of target Bengali words in Bengali vocabulary) but with the value 0 wherever there is an index corresponding to a PAD. 
The mask will be multipled to the cost (before its averaged), so that any position in the cost tensor that is effected by the pad will be multiplied by 0. This way, the effect of PADs (which we don't need to care about- Only the position of EOS is important) on the cost (and therefore on the gradients) can be nullified. 


```python
def create_Mask(output_batch):
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
Using RNG to determine if the iteration through the network will use Teacher Forcing or not. 


Some comments:

The dataset may not have enough samples for proper training. 

The output contains too many repitition if I take the greedy approach of "argmax(output probability distribution)".


(Same experience with the other transformer model I used for abstractive summarization)

Choosing an output word randomly based on its probability in the probability distribution, does brings some variety, but that's just a cheap workaround.

I don't know if beam search will significantly improve anything. 

Training doesn't look any good. But may require better dataset, with more examples, and much longer training to put out something good. 

For now, this is just a toy implementation.

I will not be completing the training for now. So, I will not be setting up validation and testing either. But that can be done later. No evaluation metrics used for now either. I may think of it, if I intend to seriously train and investigate this later. 

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
    
    while step < iters:
           
        batch_len = len(train_batch_x)
        shuffled_indices = np.arange(batch_len)
        np.random.shuffle(shuffled_indices)
        
        for i in xrange(0,batch_len):
            
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
            
            #print(train_batch_y[shuffled_indices[i]].shape)
            
            rand = random.randint(0,2) #determines chance of using Teacher Forcing
            if rand==1:
                random_bool = True
            else:
                random_bool = False
            
            mask = create_Mask(train_batch_y[shuffled_indices[i]])
            
            # Run optimization operation (backpropagation)
            _,loss,out = sess.run([optimizer,cost,softmax_output],
                                  feed_dict={x: (train_batch_x[shuffled_indices[i]]+pe_in), 
                                             y: train_batch_y[shuffled_indices[i]],
                                             keep_prob: 0.9,
                                             output_len: len(train_batch_y[shuffled_indices[i]][0]),
                                             tf_mask: mask,
                                             teacher_forcing: random_bool
                                             })
            
            if i%display_step==0:
                print("\nPREDICTED TRANSLATION OF THE SAMPLE:\n")
                flag = 0
                for array in out[sample_no]:
                    
                    prediction_int = np.random.choice(range(vocab_len), p=array.ravel()) 
                    #(^use this if you want some variety)
                    #(or use this what's below:)
                    
                    #prediction_int = np.argmax(array)
                    
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
                saver.save(sess, 'Model_Backup/translatipn_model.ckpt')

        step=step+1
    
```

    
    CHOSEN SAMPLE NO.: 43
    
    Epoch: 1 Iteration: 1
    
    SAMPLE TEXT:
    can i eat <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বেডিযে বলছে ঠিকানাটা পালান নিস।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি কি খেতে পারি <EOS> 
    
    loss=6.56739
    
    CHOSEN SAMPLE NO.: 6
    
    Epoch: 1 Iteration: 2
    
    SAMPLE TEXT:
    the terrorists released the hostages <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আতঙকবাদীরা বনদিদের ছেডে দিলো। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=161.808
    
    CHOSEN SAMPLE NO.: 57
    
    Epoch: 1 Iteration: 3
    
    SAMPLE TEXT:
    when is the museum open <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কে কে কে কে কে এখানে কে কে এটা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    জাদঘরটা কখন খোলা থাকে <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=101.757
    
    CHOSEN SAMPLE NO.: 19
    
    Epoch: 1 Iteration: 4
    
    SAMPLE TEXT:
    would you like to come fishing with us <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি আমি আমি আমি আমি আমি আমি আমি আমি আমি আমি আমি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি কি আমাদের সাথে মাছ ধরতে যাবে <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=59.2156
    
    CHOSEN SAMPLE NO.: 38
    
    Epoch: 1 Iteration: 5
    
    SAMPLE TEXT:
    i 'm not at all tired <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমরা আমরা আমরা আমরা আমরা আমরা আমরা আমরা টম টম
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি একদমই কলানত নই। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=39.9769
    
    CHOSEN SAMPLE NO.: 14
    
    Epoch: 1 Iteration: 6
    
    SAMPLE TEXT:
    i believe tom is doing well <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আপনার তমি আপনার কথা আপনার আপনার আপনার আপনার কি আপনার
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার মনে হয টম ভালোই আছে। <EOS> <PAD> <PAD> <PAD> 
    
    loss=15.8207
    
    CHOSEN SAMPLE NO.: 1
    
    Epoch: 1 Iteration: 7
    
    SAMPLE TEXT:
    i teach english <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমার <EOS> আমার <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ইংরাজি পডাই। <EOS> <PAD> <PAD> 
    
    loss=14.7499
    
    CHOSEN SAMPLE NO.: 13
    
    Epoch: 1 Iteration: 8
    
    SAMPLE TEXT:
    are you all ready <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমাকে সেই আছে না। আছে গত গত সেই
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনারা সবাই তৈরী <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=7.61839
    
    CHOSEN SAMPLE NO.: 50
    
    Epoch: 1 Iteration: 9
    
    SAMPLE TEXT:
    tom certainly is not as smart as mary thinks he is <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আর আপনাকে মধযে জনয গানে খজে ভাষায একবার তোমাকে করা অনগরহ তাডাতাডি যাচছি। টমের আপনি <EOS> তোমাকে হবে।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম অতটা বদধিমান নয যতটা মেরি মনে করে। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=6.53
    
    CHOSEN SAMPLE NO.: 58
    
    Epoch: 1 Iteration: 10
    
    SAMPLE TEXT:
    eat whatever food you like <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    হচছে মা পেল। নোংরা বল বিভিনন আছে পার
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনার যে খাবার পছনদ হয সেটা খান। <EOS> 
    
    loss=5.36857
    
    CHOSEN SAMPLE NO.: 7
    
    Epoch: 1 Iteration: 11
    
    SAMPLE TEXT:
    tom started coughing <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ডাউনলোড অনধকারকে করে বডো তিকত
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম কাশতে শর করলো। <EOS> 
    
    loss=6.17964
    
    CHOSEN SAMPLE NO.: 39
    
    Epoch: 1 Iteration: 12
    
    SAMPLE TEXT:
    i have two nieces <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ধরতে বডো থেকে পারছি বাজে টপিটা যাব। একজন
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার দটো ভাগনী আছে। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.67193
    
    CHOSEN SAMPLE NO.: 28
    
    Epoch: 1 Iteration: 13
    
    SAMPLE TEXT:
    he speaks french <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আইন। mp3 ফরাসিতে যেতে শতর আমাদের
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    সে ফরাসিতে কথা বলে। <EOS> <PAD> 
    
    loss=5.45646
    
    CHOSEN SAMPLE NO.: 11
    
    Epoch: 1 Iteration: 14
    
    SAMPLE TEXT:
    i made it myself <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বিকলপ। খোলামেলা। ওখানে পারি। গেছে। ভলবেন বসটনে করলো।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি এটা নিজে নিজে বানিযেছি। <EOS> <PAD> <PAD> 
    
    loss=4.58858
    
    CHOSEN SAMPLE NO.: 28
    
    Epoch: 1 Iteration: 15
    
    SAMPLE TEXT:
    i saw him running <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বারতাটা আপলোড আছেন করার হবে। এটাই ঘর পছনদ
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ওনাকে দৌডাতে দেখেছিলাম। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.58135
    
    CHOSEN SAMPLE NO.: 50
    
    Epoch: 1 Iteration: 16
    
    SAMPLE TEXT:
    tom certainly is eloquent <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    সতয। শেষের নাম ও মীমাংসাযোগয। চেচালো। চডানতভাবে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম সতযি বাকযবাগীশ। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.80474
    
    CHOSEN SAMPLE NO.: 46
    
    Epoch: 1 Iteration: 17
    
    SAMPLE TEXT:
    we lost <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    চাও ভরকটি কাচা খেলতাম। সমভাবনা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমরা হেরে গেছি। <EOS> <PAD> 
    
    loss=5.18577
    
    CHOSEN SAMPLE NO.: 11
    
    Epoch: 1 Iteration: 18
    
    SAMPLE TEXT:
    i 'd like to check out tomorrow morning <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি উনি হবে। বাডিটাকে সঠিক। আছে পারে। ছিলাম। বাডিটাকে আমি দেখতে মারলো।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি কাল সকালে ঘরটা ছেরে দিতে চাই। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.59181
    
    CHOSEN SAMPLE NO.: 23
    
    Epoch: 1 Iteration: 19
    
    SAMPLE TEXT:
    you look sick <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    থেকেই কমান। দিন। সে টম বসটনে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনাকে অসসথ বলে মনে হচছে। <EOS> 
    
    loss=4.68394
    
    CHOSEN SAMPLE NO.: 24
    
    Epoch: 1 Iteration: 20
    
    SAMPLE TEXT:
    i give you my word <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    করলো। বঝতে এক ভালোবাসে। গরতবপরণ। বযরথ পডল। পালটে আমাকে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি তোমাকে কথা দিলাম। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.33229
    
    CHOSEN SAMPLE NO.: 53
    
    Epoch: 1 Iteration: 21
    
    SAMPLE TEXT:
    keep quiet <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ছটির ওকে পারবেন বেডাতে টেপ এনো।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    চপ করো <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.41801
    
    CHOSEN SAMPLE NO.: 46
    
    Epoch: 1 Iteration: 22
    
    SAMPLE TEXT:
    what team does tom play for <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    অসবিধা যাক। থাকাটাও বষটি জিরোচছি। শানত আমাকে তেষটা বডো
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম কোন দলের হযে খেলে <EOS> <PAD> <PAD> <PAD> 
    
    loss=5.15387
    
    CHOSEN SAMPLE NO.: 57
    
    Epoch: 1 Iteration: 23
    
    SAMPLE TEXT:
    tom now knows everything <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ঘরটা আমি ভাষা ঠিক কাজ পারব বসো।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম এখন সবকিছই জানে। <EOS> <PAD> <PAD> 
    
    loss=4.64027
    
    CHOSEN SAMPLE NO.: 25
    
    Epoch: 1 Iteration: 24
    
    SAMPLE TEXT:
    you ought to ask him for advice <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমার কি খব পরেছিস দিন। খায সেটা তমি আমরা আমার করছে।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তোমার তাকে পরামরশের জনয জিজঞাসা করা উচিৎ। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.74578
    
    CHOSEN SAMPLE NO.: 57
    
    Epoch: 1 Iteration: 25
    
    SAMPLE TEXT:
    the baby is screaming <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তোদের এসেছিলো। বাজে। পালটিও আটটার একবার দিশেহারা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    বাচচাটা চেচাচছে। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.48393
    
    CHOSEN SAMPLE NO.: 46
    
    Epoch: 1 Iteration: 26
    
    SAMPLE TEXT:
    tom has decided to keep a diary <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ওনার যেত আপনি নিরবাচিত বযসত। টম পেছনে কি নিযে ফরাসি সতরী।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম একটা ডাইরি রাখার কথা ঠিক করেছে। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.51531
    
    CHOSEN SAMPLE NO.: 28
    
    Epoch: 1 Iteration: 27
    
    SAMPLE TEXT:
    i drink beer <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এই উচিৎ। কি ধনী। বাডিতে শোনো। না।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি বিযার খাই। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.85868
    
    CHOSEN SAMPLE NO.: 42
    
    Epoch: 1 Iteration: 28
    
    SAMPLE TEXT:
    sit down <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    থেকে আমার আমি না। আমি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    বসো <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.48524
    
    CHOSEN SAMPLE NO.: 13
    
    Epoch: 1 Iteration: 29
    
    SAMPLE TEXT:
    tom used to play guitar <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পাযনি। সমভব চকোলেট পডলো এটা আমি আরমভ ঠিক পেলো
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম গীটার বাজাতো। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.00684
    
    CHOSEN SAMPLE NO.: 3
    
    Epoch: 1 Iteration: 30
    
    SAMPLE TEXT:
    you may swim <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    দিযেছে। কে দেখে আপনি সপারিশ ফরাসি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    সাতার কাটতে পারেন। <EOS> <PAD> <PAD> 
    
    loss=3.85715
    
    CHOSEN SAMPLE NO.: 43
    
    Epoch: 1 Iteration: 31
    
    SAMPLE TEXT:
    do you live in this neighborhood <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    চেষটা আমি উনি টম সঙগে ভাষায ফোন বনধ কী
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি কি এই পাডাতেই থাকো <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.90109
    
    CHOSEN SAMPLE NO.: 54
    
    Epoch: 1 Iteration: 32
    
    SAMPLE TEXT:
    take tom to the hospital <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কেযার গরবিত রাখব। পছনদ। পারে। ধান উদবিগন বলন। চাবি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টমকে হাসপাতালে নিযে যাও। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.01311
    
    CHOSEN SAMPLE NO.: 47
    
    Epoch: 1 Iteration: 33
    
    SAMPLE TEXT:
    may i have your phone number <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    সকতলা আকরষণীয সামলাতে গরতবপরণ হযে যা আমার জতো উনি না
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি কি আপনার ফোন নামবারটি পেতে পারি <EOS> <PAD> <PAD> 
    
    loss=4.25953
    
    CHOSEN SAMPLE NO.: 21
    
    Epoch: 1 Iteration: 34
    
    SAMPLE TEXT:
    do you understand what i 'm saying <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> <EOS> আমার <EOS> <EOS> <EOS> বাইকটা এই দেবো। এই
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি যা বলছি আপনি কি তা বঝতে পারছেন <EOS> <PAD> 
    
    loss=4.64283
    
    CHOSEN SAMPLE NO.: 53
    
    Epoch: 1 Iteration: 35
    
    SAMPLE TEXT:
    do you live here <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    করতে পারেন। কোথায <EOS> সি তারা না। এটা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি কি এখানে থাকেন <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.70672
    
    CHOSEN SAMPLE NO.: 27
    
    Epoch: 1 Iteration: 36
    
    SAMPLE TEXT:
    that is because you are a girl <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    সে নই। বোন। আমরা দিলাম। বলে ফরাসিতে আমি চাই হচচে। গেছে কোথা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তার কারণ তই একজন মেযে। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.65005
    
    CHOSEN SAMPLE NO.: 58
    
    Epoch: 1 Iteration: 37
    
    SAMPLE TEXT:
    tom rides a scooter <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    থেকে এটা কফি কত ঠিকমতো তিনি আমাকে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম সকটার চডে। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.16478
    
    CHOSEN SAMPLE NO.: 11
    
    Epoch: 1 Iteration: 38
    
    SAMPLE TEXT:
    i came back <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বসটনে পডাতে সাথে এসেছিলো। পারব দাতের
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ফিরে এলাম। <EOS> <PAD> <PAD> 
    
    loss=3.98497
    
    CHOSEN SAMPLE NO.: 0
    
    Epoch: 1 Iteration: 39
    
    SAMPLE TEXT:
    how are you did you have a good trip <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বযসত। পারি খশি <EOS> ওনাকে বাকযাংশটির খাচছেন। <EOS> দযা আমার সহজেই বরফ
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    কেমন আছেন যাতরা ভালো ছিল তো <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=5.24832
    
    CHOSEN SAMPLE NO.: 46
    
    Epoch: 1 Iteration: 40
    
    SAMPLE TEXT:
    do not cry <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    হাসলো সাবধানে খলন লাল অংশে ওই খেযেছো
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    কেদো না। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.94055
    
    CHOSEN SAMPLE NO.: 19
    
    Epoch: 1 Iteration: 41
    
    SAMPLE TEXT:
    she is asking how that is possible <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কাশতে সকতলা দেখেছি। পারবেন বানাতে কোথা করন যেতে সংজত দৌডান। মনোযোগ
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    উনি জিজঞাসা করছেন এটা কি করে সমভব। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.07073
    
    CHOSEN SAMPLE NO.: 44
    
    Epoch: 1 Iteration: 42
    
    SAMPLE TEXT:
    tom knew <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ওটা <EOS> থেকে চেচালো। যা টম
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম জানতো। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.41441
    
    CHOSEN SAMPLE NO.: 63
    
    Epoch: 1 Iteration: 43
    
    SAMPLE TEXT:
    tom laughed <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমার বলতে আমি <EOS> কি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম হাসলো। <EOS> <PAD> <PAD> 
    
    loss=4.00027
    
    CHOSEN SAMPLE NO.: 27
    
    Epoch: 1 Iteration: 44
    
    SAMPLE TEXT:
    eat anything you like <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    রাখতে খান মেরি <EOS> আর হাটে। <EOS> চিৎকার
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনার যা কিছ পছনদ হয তাই খান। <EOS> 
    
    loss=3.86403
    
    CHOSEN SAMPLE NO.: 19
    
    Epoch: 1 Iteration: 45
    
    SAMPLE TEXT:
    close the door when you leave <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কাদছিলো। রহসয বল বলতে তোমার কষমাপরারথনা দাও। খান আমরা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    বেডোবার সময দরজাটা বনধ করে দিও। <EOS> <PAD> <PAD> 
    
    loss=4.40852
    
    CHOSEN SAMPLE NO.: 41
    
    Epoch: 1 Iteration: 46
    
    SAMPLE TEXT:
    did you read it all <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বঝি। পডন। ছাডন। ওদের আছো আছো এমনিতে খেলি।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি কি পরোটা পরেছো <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.93985
    
    CHOSEN SAMPLE NO.: 0
    
    Epoch: 1 Iteration: 47
    
    SAMPLE TEXT:
    listen carefully <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বাডিটা ও <EOS> ভালো এটা বলেছিল
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    মন দিযে শনবে। <EOS> <PAD> <PAD> 
    
    loss=3.44873
    
    CHOSEN SAMPLE NO.: 58
    
    Epoch: 1 Iteration: 48
    
    SAMPLE TEXT:
    tom has won many races <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> টম
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম অনেক দৌড জিতেছে। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.87674
    
    CHOSEN SAMPLE NO.: 4
    
    Epoch: 1 Iteration: 49
    
    SAMPLE TEXT:
    call the police <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> সবাই তারা <EOS> ভলে বোঝে সেটা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    পলিশ ডাকো <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.42726
    
    CHOSEN SAMPLE NO.: 20
    
    Epoch: 1 Iteration: 50
    
    SAMPLE TEXT:
    this is not important <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    থাকতে চান ঘোডায কাল বিবাহবিচছেদ পারে। সতরী। খাবো।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এটি গরতবপরণ না। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.79798
    
    CHOSEN SAMPLE NO.: 15
    
    Epoch: 1 Iteration: 51
    
    SAMPLE TEXT:
    i would like to be a pilot in the future <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    যান। আমরা অংশে গেছিলাম। গাছটা এখন পারছি দাডাও। আমি করো। কিনলাম। ভেবছিলাম আমি ইনি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ভবিষযতে পাইলট হতে চাই। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.60735
    
    CHOSEN SAMPLE NO.: 18
    
    Epoch: 2 Iteration: 1
    
    SAMPLE TEXT:
    you should apologize <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কেমন কর। দেখবো। গেছি ছিলে গেছিলো।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনার কষমা চাওযা উচিৎ। <EOS> <PAD> 
    
    loss=3.6305
    
    CHOSEN SAMPLE NO.: 50
    
    Epoch: 2 Iteration: 2
    
    SAMPLE TEXT:
    you called <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ধরন। কী জাপানি সে সবাই
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি ডেকেছিলে <EOS> <PAD> <PAD> 
    
    loss=3.72863
    
    CHOSEN SAMPLE NO.: 34
    
    Epoch: 2 Iteration: 3
    
    SAMPLE TEXT:
    i saw mary sitting in front of a mirror brushing her hair <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> <EOS> <EOS> <EOS> তিনি <EOS> <EOS> <EOS> <EOS> <EOS> তোমরা ভালো <EOS> <EOS> <EOS> কর। <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    মেরিকে আযনার সামনে বসে চল আচডাতে দেখলাম। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=6.1713
    
    CHOSEN SAMPLE NO.: 10
    
    Epoch: 2 Iteration: 4
    
    SAMPLE TEXT:
    i was alone <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    দেখে <EOS> টম অকটোবর। কেউ
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি একা ছিলাম। <EOS> <PAD> 
    
    loss=4.39774
    
    CHOSEN SAMPLE NO.: 46
    
    Epoch: 2 Iteration: 5
    
    SAMPLE TEXT:
    take us there <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টম উপনযাস খায ওটা মানষ। আপনার মগ।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমাদের ওখানে নিযে চলো। <EOS> <PAD> <PAD> 
    
    loss=3.59155
    
    CHOSEN SAMPLE NO.: 31
    
    Epoch: 2 Iteration: 6
    
    SAMPLE TEXT:
    i just want tom to be happy <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এখানে। গেছিলাম। আছেন। টম <EOS> ওনারা চাইছি কটার পারো কিনতে তোমরা মেযে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি খালি চাই টম সখে থাকক। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.63375
    
    CHOSEN SAMPLE NO.: 47
    
    Epoch: 2 Iteration: 7
    
    SAMPLE TEXT:
    i would like an air-conditioned room <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বারতাটা <EOS> <EOS> না। না। আসন। চেডে সকালে <EOS> টম যাচছি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার একটী শীততাপ নিযনতরিত ঘর চাই। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.81769
    
    CHOSEN SAMPLE NO.: 10
    
    Epoch: 2 Iteration: 8
    
    SAMPLE TEXT:
    i would like you to come with me <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    দেখো। চাই। রাখন। টমের শনিনি। চপ না। অনেক গেলো। পারি আগে যাবেন
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি চাই তমি আমার সাথে আসো। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.25251
    
    CHOSEN SAMPLE NO.: 48
    
    Epoch: 2 Iteration: 9
    
    SAMPLE TEXT:
    see above <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    একবার। <EOS> যান। এটা এটি ছিলো।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    উপরে দেখন। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.36901
    
    CHOSEN SAMPLE NO.: 5
    
    Epoch: 2 Iteration: 10
    
    SAMPLE TEXT:
    i 'm still your friend <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    যাক। থেকে পরতযেকদিন একটা ছোট। পেযেছিলাম। সে এখানে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি এখনো আপনার বনধ আছি। <EOS> <PAD> <PAD> 
    
    loss=3.68839
    
    CHOSEN SAMPLE NO.: 24
    
    Epoch: 2 Iteration: 11
    
    SAMPLE TEXT:
    just staying alive in these times is hard enough <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    না। ওর পারবে দরকার। <EOS> দিলাম। না। নিরবোধ। টম যাবেন। করি। কর।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এখনকার সময শধ বেচে থাকাটাও যথেষট কঠিন। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.53596
    
    CHOSEN SAMPLE NO.: 32
    
    Epoch: 2 Iteration: 12
    
    SAMPLE TEXT:
    i 'm not proud of this <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    সময গান টমের তমি এমনিতে চেচাচছে সময বযপারে সবাই হারিযে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি এটায গরবিত নই। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.00889
    
    CHOSEN SAMPLE NO.: 33
    
    Epoch: 2 Iteration: 13
    
    SAMPLE TEXT:
    i 'm busy <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পারব দাডাও আপনাকে পরেন <EOS> সময পারছেন
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি বযসত। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.35917
    
    CHOSEN SAMPLE NO.: 9
    
    Epoch: 2 Iteration: 14
    
    SAMPLE TEXT:
    is there any mail for me <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এটা <EOS> <EOS> টমের <EOS> <EOS> করলো। মখ চায না
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার জনয কোনো ডাক আছে <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.33615
    
    CHOSEN SAMPLE NO.: 40
    
    Epoch: 2 Iteration: 15
    
    SAMPLE TEXT:
    you are not paying attention <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    রবিবার ওঠো মাংস চাই একমাতর নাও। কি চাই বনধ
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি কিনত মনোযোগ দিচছো না। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.56014
    
    CHOSEN SAMPLE NO.: 14
    
    Epoch: 2 Iteration: 16
    
    SAMPLE TEXT:
    eat something <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বযকতি। বসবাস থেকে <EOS> আছি।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    কিছ খেযে নিন। <EOS> <PAD> 
    
    loss=4.05344
    
    CHOSEN SAMPLE NO.: 52
    
    Epoch: 2 Iteration: 17
    
    SAMPLE TEXT:
    she did not tell me her name <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টমের করেছে বলতে টমকে রাখন। বলে আপনার দাডালেন। একলা চিৎকার দেখে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ও আমাকে ওর নাম বলেনি। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.983
    
    CHOSEN SAMPLE NO.: 43
    
    Epoch: 2 Iteration: 18
    
    SAMPLE TEXT:
    where would tom go <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    না। যা টম দিন। আমি <EOS> আমি করব।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম আর কোথায যাবে <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.51351
    
    CHOSEN SAMPLE NO.: 39
    
    Epoch: 2 Iteration: 19
    
    SAMPLE TEXT:
    tom 's deaf <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টমের আপনার বাডি এটা আমি কি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম বধির। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.63925
    
    CHOSEN SAMPLE NO.: 24
    
    Epoch: 2 Iteration: 20
    
    SAMPLE TEXT:
    i give you my word <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    খালি আছে। <EOS> করা এই আপনি আমি আমাদের নেই।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি তোমাকে কথা দিলাম। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.8243
    
    CHOSEN SAMPLE NO.: 26
    
    Epoch: 2 Iteration: 21
    
    SAMPLE TEXT:
    you can study here <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বাডিতে তিনি আমার <EOS> আমার মারলাম। কাছে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি এখানে পডাশোনা করতে পার। <EOS> <PAD> 
    
    loss=4.09468
    
    CHOSEN SAMPLE NO.: 59
    
    Epoch: 2 Iteration: 22
    
    SAMPLE TEXT:
    do not come in <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ধরে হাসলো। যাও। নাম ভেবে লডবো। যা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ভেতরে আসবেন না। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.10425
    
    CHOSEN SAMPLE NO.: 33
    
    Epoch: 2 Iteration: 23
    
    SAMPLE TEXT:
    i 'm genuinely happy for tom <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টম ভাগ আমার চাই। কার চাই। আমি বলে না।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি পরকতই টমের জনয খশি হযেছি। <EOS> <PAD> <PAD> 
    
    loss=4.53609
    
    CHOSEN SAMPLE NO.: 45
    
    Epoch: 2 Iteration: 24
    
    SAMPLE TEXT:
    no one knew it <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    জিজঞাসা আছে। টম আপনাকে উনি গাডি আমার কি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    কেউই এটা জানতো না। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.51969
    
    CHOSEN SAMPLE NO.: 59
    
    Epoch: 2 Iteration: 25
    
    SAMPLE TEXT:
    hello <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তমি ফোন খায। আছে। যাবে তোমার
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    নমসকার <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.11391
    
    CHOSEN SAMPLE NO.: 22
    
    Epoch: 2 Iteration: 26
    
    SAMPLE TEXT:
    i like him <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    হাসলো। চিৎকার বাডি। পারবে <EOS> চাই
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার ওকে ভালো লাগে। <EOS> <PAD> 
    
    loss=4.05172
    
    CHOSEN SAMPLE NO.: 50
    
    Epoch: 2 Iteration: 27
    
    SAMPLE TEXT:
    tom is living in boston <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কেন চাই। <EOS> ভালো ফল আমি <EOS> আমি চর
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম বসটনে থাকছে। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.79604
    
    CHOSEN SAMPLE NO.: 40
    
    Epoch: 2 Iteration: 28
    
    SAMPLE TEXT:
    this is not a good sign <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    যারা <EOS> বইটা ওদের আপনি <EOS> টম দটো <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এটা ভালো লকষণ নয। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.27584
    
    CHOSEN SAMPLE NO.: 61
    
    Epoch: 2 Iteration: 29
    
    SAMPLE TEXT:
    keep it <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পারেন। সময দিন। এখানে। কোরো।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ওটা রাখন। <EOS> <PAD> <PAD> 
    
    loss=3.62392
    
    CHOSEN SAMPLE NO.: 4
    
    Epoch: 2 Iteration: 30
    
    SAMPLE TEXT:
    your time is up <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> <EOS> এই পারি একমত। এখনো আর <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তোমার সময শেষ। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.58457
    
    CHOSEN SAMPLE NO.: 47
    
    Epoch: 2 Iteration: 31
    
    SAMPLE TEXT:
    the station is pretty far <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পরতি <EOS> আজ <EOS> দ <EOS> গেছিলেন। আছেন।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    সটেশনটা বেশ কিছটা দরে। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.01864
    
    CHOSEN SAMPLE NO.: 48
    
    Epoch: 2 Iteration: 32
    
    SAMPLE TEXT:
    fill out this form please <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বোঝা দখল চেচালেন। এটা আমাকে বঝতে তিনি কযেক তেশরা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    অনগরহ করে এই ফরমটি পরণ করন। <EOS> <PAD> <PAD> 
    
    loss=3.7789
    
    CHOSEN SAMPLE NO.: 3
    
    Epoch: 2 Iteration: 33
    
    SAMPLE TEXT:
    what is your home address <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বিশবাস টম বনধ দখতে অসফল বলতে <EOS> বাডির
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তোমার বাডির ঠিকানাটা কী <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.34493
    
    CHOSEN SAMPLE NO.: 55
    
    Epoch: 2 Iteration: 34
    
    SAMPLE TEXT:
    do not let it get soiled <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    শর নাম খেযে আগে কথা ওনাকে <EOS> অনগরহ ভালো
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এটাকে নোংরা হতে দেবেন না। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.15138
    
    CHOSEN SAMPLE NO.: 16
    
    Epoch: 2 Iteration: 35
    
    SAMPLE TEXT:
    i agree with you <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তমি তমি কষমা সবাই <EOS> বললেন। করছে।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি আপনার সাথে একমত। <EOS> <PAD> <PAD> 
    
    loss=3.79577
    
    CHOSEN SAMPLE NO.: 35
    
    Epoch: 2 Iteration: 36
    
    SAMPLE TEXT:
    tom can read french <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> বসটন আমার আপনার আর গানটা <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম ফরাসি পডতে পারে। <EOS> <PAD> <PAD> 
    
    loss=3.9028
    
    CHOSEN SAMPLE NO.: 17
    
    Epoch: 2 Iteration: 37
    
    SAMPLE TEXT:
    turn left here <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ওটার ভলে টম মযাকডোনালডসের <EOS> রাখতে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এখান থেকে বাদিকে নিন। <EOS> <PAD> 
    
    loss=3.68729
    
    CHOSEN SAMPLE NO.: 39
    
    Epoch: 2 Iteration: 38
    
    SAMPLE TEXT:
    who is speaking <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কি অনগরহ <EOS> যা <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    কে কথা বলছে <EOS> <PAD> 
    
    loss=4.41564
    
    CHOSEN SAMPLE NO.: 3
    
    Epoch: 2 Iteration: 39
    
    SAMPLE TEXT:
    you you will need a ticket to travel by bus <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টম আমি <EOS> নাম না। মালপতরটা সঠিক। <EOS> আমি <EOS> <EOS> <EOS> কমই আমার
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    বাসে করে ঘরতে হলে আপনাকে টিকিট কাটতে হবে। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.66935
    
    CHOSEN SAMPLE NO.: 0
    
    Epoch: 2 Iteration: 40
    
    SAMPLE TEXT:
    do not speak <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ভালো তাইতো তাই এখান বযস। বাকি।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    কথা বলবেন না। <EOS> <PAD> <PAD> 
    
    loss=4.18294
    
    CHOSEN SAMPLE NO.: 11
    
    Epoch: 2 Iteration: 41
    
    SAMPLE TEXT:
    you ought to ask him for advice <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বঝি আঙর আসবেন অবসথাটা হচছে আমাকে আজ ইনি আর ভাষা সেটা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনার ওনাকে পরামরশের জনয জিজঞাসা করা উচিৎ। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.1447
    
    CHOSEN SAMPLE NO.: 1
    
    Epoch: 2 Iteration: 42
    
    SAMPLE TEXT:
    i teach english <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    দিচছেন করে। জিজঞাসা বাডিতে কি আমাদের
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ইংরাজি পডাই। <EOS> <PAD> <PAD> 
    
    loss=4.09819
    
    CHOSEN SAMPLE NO.: 39
    
    Epoch: 2 Iteration: 43
    
    SAMPLE TEXT:
    what time do you usually get up <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ছিলো। ভালো করি। টম <EOS> <EOS> কি <EOS> টম <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি এমনিতে কটার সময ঘম থেকে ওঠো <EOS> <PAD> <PAD> 
    
    loss=4.50791
    
    CHOSEN SAMPLE NO.: 14
    
    Epoch: 2 Iteration: 44
    
    SAMPLE TEXT:
    i used to play tennis <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কথা খাচছিল। গেছিলো। বধির। তারিফ <EOS> করছিলো। কর। দীরঘতম
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি টেনিস খেলতাম। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.87848
    
    CHOSEN SAMPLE NO.: 24
    
    Epoch: 2 Iteration: 45
    
    SAMPLE TEXT:
    i do not want to go with tom <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এটা পাবো। যথেষট আজ তোমরা যান। পারি হাসানোর বলছেন কথা আছি। পরেছিল।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি টমের সাথে যেতে চাই না। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.22809
    
    CHOSEN SAMPLE NO.: 5
    
    Epoch: 2 Iteration: 46
    
    SAMPLE TEXT:
    where do you live <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    শেষ। <EOS> <EOS> <EOS> সময সটেশন আল <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি কোথায থাকো <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.53752
    
    CHOSEN SAMPLE NO.: 31
    
    Epoch: 2 Iteration: 47
    
    SAMPLE TEXT:
    turn left at the corner <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পছনদ <EOS> করে করলো। <EOS> <EOS> টম ওনাকে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    মোডে গিযে বাদিকে বেকে যাবেন। <EOS> <PAD> <PAD> 
    
    loss=4.91523
    
    CHOSEN SAMPLE NO.: 29
    
    Epoch: 2 Iteration: 48
    
    SAMPLE TEXT:
    that is my cat <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কাজই থাকো। ঘমাতে ওর না। <EOS> <EOS> করি।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ওটা আমার বিডাল। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.53141
    
    CHOSEN SAMPLE NO.: 2
    
    Epoch: 2 Iteration: 49
    
    SAMPLE TEXT:
    this will cost €30 <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বনধ। পেযেছিলাম। আসেনি। <EOS> তমি পরথম দেখেছি। বযাথা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এটার দাম পডবে €৩০। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.54887
    
    CHOSEN SAMPLE NO.: 55
    
    Epoch: 2 Iteration: 50
    
    SAMPLE TEXT:
    tom 's alone <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আছেন একটা চেচালো। ফোন টম বেকে গরনথাগারে।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম একলা আছে। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.6899
    
    CHOSEN SAMPLE NO.: 63
    
    Epoch: 2 Iteration: 51
    
    SAMPLE TEXT:
    i do not agree with him <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> করে থাকে। কি তমি রাখেন। কাছ আটটার খশি বাবা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ওনার সাথে একমত নই। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.25289
    
    CHOSEN SAMPLE NO.: 59
    
    Epoch: 3 Iteration: 1
    
    SAMPLE TEXT:
    they have not done anything wrong <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এটা সবাই এটা ছিলে। দাডিযেছিলো। আওযাজ আমি কি আপনি দেখতে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ওনারা কোনো ভল কিছ তো করেননি। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.03251
    
    CHOSEN SAMPLE NO.: 10
    
    Epoch: 3 Iteration: 2
    
    SAMPLE TEXT:
    when did you begin studying english <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> <EOS> কোথায তাই কি মারলাম। ওটার দেখা আমি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি ইংরাজি পডা কবে থেকে শর করলেন <EOS> <PAD> 
    
    loss=4.20393
    
    CHOSEN SAMPLE NO.: 11
    
    Epoch: 3 Iteration: 3
    
    SAMPLE TEXT:
    what time do you usually go to bed <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    দিন খান। না। এটা তোর দিযেছে। ছেডে সাথে দাত পরশনের করছিস করে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি এমনিতে কটার সময ঘমাতে যাও <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.90957
    
    CHOSEN SAMPLE NO.: 2
    
    Epoch: 3 Iteration: 4
    
    SAMPLE TEXT:
    some of my friends can speak french well <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    সটেশন শেষ দিযেছি। যাবে। শিখতে খেলোযার। এসে থাক। ওনার <EOS> ওঠো।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার বনধদের মধযে কেউ কেউ ভালো ফরাসি বলতে পারে। <EOS> <PAD> 
    
    loss=4.24333
    
    CHOSEN SAMPLE NO.: 18
    
    Epoch: 3 Iteration: 5
    
    SAMPLE TEXT:
    we won <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    নয। ভালো কষমাপরারথনা চান বাস
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমরা জিতে গেছে। <EOS> <PAD> 
    
    loss=4.04047
    
    CHOSEN SAMPLE NO.: 19
    
    Epoch: 3 Iteration: 6
    
    SAMPLE TEXT:
    my stomach hurts after meals <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    না। বাবার ভাবে <EOS> জানি। ফরাসি করবেন কথা টম
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    খাওযার পর আমার পেট বযাথা করে। <EOS> <PAD> <PAD> 
    
    loss=3.96237
    
    CHOSEN SAMPLE NO.: 13
    
    Epoch: 3 Iteration: 7
    
    SAMPLE TEXT:
    i like him <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    মিথযা কর। আমি কাটতে আমার আর
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি তাকে পছনদ করি। <EOS> <PAD> 
    
    loss=3.37837
    
    CHOSEN SAMPLE NO.: 35
    
    Epoch: 3 Iteration: 8
    
    SAMPLE TEXT:
    i 'm not a doctor <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    সময অনয আমি দরজাটা <EOS> <EOS> সবাই <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ডাকতার নই। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.29195
    
    CHOSEN SAMPLE NO.: 58
    
    Epoch: 3 Iteration: 9
    
    SAMPLE TEXT:
    what will you be doing at this time tomorrow <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পরোপরি আমার টম ঠিক মা শতরর <EOS> শিখতে চিৎকার সাথে খেতে হাসপাতালে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    কাল তমি এই সময কি করবে <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.69243
    
    CHOSEN SAMPLE NO.: 56
    
    Epoch: 3 Iteration: 10
    
    SAMPLE TEXT:
    do not call me <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বডো আসছেন পডে। খরগোশ বাইরে পেতে তোমার হবে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমাকে ডেকো না। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.56516
    
    CHOSEN SAMPLE NO.: 57
    
    Epoch: 3 Iteration: 11
    
    SAMPLE TEXT:
    who is he <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> নডলস দেখাবেন জানি আজ চটপটে।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ও কে <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.09663
    
    CHOSEN SAMPLE NO.: 53
    
    Epoch: 3 Iteration: 12
    
    SAMPLE TEXT:
    we will try <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কি না। মজা করি। সকালে আমি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমরা চেষটা করব। <EOS> <PAD> <PAD> 
    
    loss=3.98838
    
    CHOSEN SAMPLE NO.: 9
    
    Epoch: 3 Iteration: 13
    
    SAMPLE TEXT:
    she can say whatever she wants <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> করা এমনিতে <EOS> কিছ <EOS> <EOS> আমার আজ
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ওনার যা ইচছে উনি বলতে পারেন। <EOS> <PAD> <PAD> 
    
    loss=4.78535
    
    CHOSEN SAMPLE NO.: 48
    
    Epoch: 3 Iteration: 14
    
    SAMPLE TEXT:
    how can i get to gate a-1 <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এই ইচছা কষমা <EOS> ওইগলো পেলাম। দেখলো। তার আপনার সাহাযয
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি a-1 গেটে কিভাবে যাব <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.21944
    
    CHOSEN SAMPLE NO.: 32
    
    Epoch: 3 Iteration: 15
    
    SAMPLE TEXT:
    why are you shouting <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    খবর সাবধান লাগে। চালাবার দ দাত ফরাসিতেও দিলাম।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তোমরা চিৎকার করছো কেন <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.64164
    
    CHOSEN SAMPLE NO.: 12
    
    Epoch: 3 Iteration: 16
    
    SAMPLE TEXT:
    these are birds <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বরণানকরমে করবেন। শেষের পারে। বললেন সতযি পারটিতে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এইগলো পাখি। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.55749
    
    CHOSEN SAMPLE NO.: 39
    
    Epoch: 3 Iteration: 17
    
    SAMPLE TEXT:
    it looks like tom has broken a couple of ribs <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ওটা আর করি। তিনি কি আমরা এসে <EOS> বেকে ঘাড আমি শনলো। যাবেন খাই।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    মনে হচছে টমের কযেকটা পাজর ভেঙগেছে। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.2831
    
    CHOSEN SAMPLE NO.: 31
    
    Epoch: 3 Iteration: 18
    
    SAMPLE TEXT:
    i do not want to be like that <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    থেকে ভেতরে <EOS> ফরাসি আমার কষণিকের বলতে <EOS> <EOS> কি <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ওইরকম হতে চাই না। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.93504
    
    CHOSEN SAMPLE NO.: 8
    
    Epoch: 3 Iteration: 19
    
    SAMPLE TEXT:
    me too <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এ <EOS> <EOS> <EOS> কি মেরিকে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমিও। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.31524
    
    CHOSEN SAMPLE NO.: 32
    
    Epoch: 3 Iteration: 20
    
    SAMPLE TEXT:
    why do you want to hurt tom <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> পারি। আর করেছে। কাছে দিযে কঠিন। মডতে কোনো পরেমে লিখলো।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি টমকে আঘাত দিতে চান কেন <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.9945
    
    CHOSEN SAMPLE NO.: 46
    
    Epoch: 3 Iteration: 21
    
    SAMPLE TEXT:
    tom was shouting <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    নিযে টমকে হাসলো। করে করি।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম চেচাচছিলো। <EOS> <PAD> <PAD> 
    
    loss=4.44068
    
    CHOSEN SAMPLE NO.: 51
    
    Epoch: 3 Iteration: 22
    
    SAMPLE TEXT:
    i 'm exhausted <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তোদের যে বনধ টম কাছে থেকে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি কলানত। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.73241
    
    CHOSEN SAMPLE NO.: 10
    
    Epoch: 3 Iteration: 23
    
    SAMPLE TEXT:
    i do not deny it <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি সেটা পারব <EOS> <EOS> <EOS> যে তো <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি এটা অসবীকার করি না। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.9017
    
    CHOSEN SAMPLE NO.: 20
    
    Epoch: 3 Iteration: 24
    
    SAMPLE TEXT:
    stop tom <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> <EOS> টমকে সে আমি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টমকে আটকান। <EOS> <PAD> <PAD> 
    
    loss=3.58409
    
    CHOSEN SAMPLE NO.: 28
    
    Epoch: 3 Iteration: 25
    
    SAMPLE TEXT:
    maybe it will snow <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    চেষটা থাকব। বলবেন জিজঞাসা লাঞচ আগে <EOS> হযেছে।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    হযতো বরফ পরবে। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.66221
    
    CHOSEN SAMPLE NO.: 10
    
    Epoch: 3 Iteration: 26
    
    SAMPLE TEXT:
    tom visited boston <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    খালি চাও ভাগনা থাকলো। চলে না।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম বসটন গেছিলেন। <EOS> <PAD> <PAD> 
    
    loss=3.76476
    
    CHOSEN SAMPLE NO.: 17
    
    Epoch: 3 Iteration: 27
    
    SAMPLE TEXT:
    tom is definitely not happy <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> <EOS> আছেন। হযেগেছিলাম। তার আমার টম আমার দেখে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম সপষটতই খশি নয। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.75236
    
    CHOSEN SAMPLE NO.: 62
    
    Epoch: 3 Iteration: 28
    
    SAMPLE TEXT:
    i 'd like to change my room <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টম ফরাসি নটার <EOS> টম আমি অপেকষা তিনি টমের <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি আমার ঘরটা পালটাতে চাই। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.61722
    
    CHOSEN SAMPLE NO.: 18
    
    Epoch: 3 Iteration: 29
    
    SAMPLE TEXT:
    why do you want to leave today <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এই লোকটা কেউ <EOS> ওটা আওযাজ বযস গেলাম। <EOS> আছেন <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনারা আজকেই যেতে চাইছেন কেন <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.86553
    
    CHOSEN SAMPLE NO.: 2
    
    Epoch: 3 Iteration: 30
    
    SAMPLE TEXT:
    what does your son do <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তার গাইতে তই হোককাইডো বনধ অপেকষা কেউ আপনি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তোমার ছেলে কি করে <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.32078
    
    CHOSEN SAMPLE NO.: 10
    
    Epoch: 3 Iteration: 31
    
    SAMPLE TEXT:
    have you finished it <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    চায যোগয। বযবহার করতে কি করব। দর
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তোমাদের ওটা করা হযে গেছে <EOS> <PAD> 
    
    loss=3.97464
    
    CHOSEN SAMPLE NO.: 52
    
    Epoch: 3 Iteration: 32
    
    SAMPLE TEXT:
    i 'm still your friend <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টম মা <EOS> <EOS> করে। টমের কাজ <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি এখনো তোমার বনধ আছি। <EOS> <PAD> <PAD> 
    
    loss=4.03356
    
    CHOSEN SAMPLE NO.: 63
    
    Epoch: 3 Iteration: 33
    
    SAMPLE TEXT:
    i do not agree with him <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমাদের গবেষণাটি তিনি ছাডে ফরাসিতে <EOS> <EOS> অপেকষা <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ওনার সাথে একমত নই। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.01831
    
    CHOSEN SAMPLE NO.: 12
    
    Epoch: 3 Iteration: 34
    
    SAMPLE TEXT:
    here 's your mug <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> কার হযে টম তমি টম অঙক বনধ
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এই নিন আপনার মগ। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.55271
    
    CHOSEN SAMPLE NO.: 5
    
    Epoch: 3 Iteration: 35
    
    SAMPLE TEXT:
    tom screamed loudly <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পারবেন দিকে <EOS> আপনাকে তোমার
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম জোরে চেচালো। <EOS> <PAD> 
    
    loss=4.35615
    
    CHOSEN SAMPLE NO.: 26
    
    Epoch: 3 Iteration: 36
    
    SAMPLE TEXT:
    eat anything you like <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পৌনে <EOS> <EOS> আমি জযাকসন। মাছ অসফল খরগোশ
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তোমাদের যা কিছ পছনদ হয তাই খাও। <EOS> 
    
    loss=3.52522
    
    CHOSEN SAMPLE NO.: 1
    
    Epoch: 3 Iteration: 37
    
    SAMPLE TEXT:
    my father is busy <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> তার চিৎকার <EOS> আপনার <EOS> <EOS> কাছে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার বাবা বযসত আছে। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.57193
    
    CHOSEN SAMPLE NO.: 62
    
    Epoch: 3 Iteration: 38
    
    SAMPLE TEXT:
    wonderful <EOS> <PAD> 
    
