
# Preprocessing Translation Data

### Function for expanding english contractions

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

The index of vocabulary will represent the numerical representation of the word which is the value of vocabulary at that index. 



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

In this way, first, from each sentence, I am creating a list of words, and corresponding list of context words.
Doing the same thing for


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
then, here I will reconstrcut the data as:

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

for details of word2vec and code description


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

See: https://www.tensorflow.org/tutorials/word2vec

for details of word2vec and code description


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
```.....
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

Randomly shuffling the complete dataset, and then splitting it into train, validation and test set


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

Solution is to fill shorter sentences with PADs so that length of all sentences become equal.
But, if one sentence in a batch has 20 words, and the same batch has another sentence with one word, then the latter sentence will have to be filled in by at least 19 pads. If most of the sentences start to have more PADs than actual content, training will become troublesome.

The solution to that is bucketing. First the sentences in the total list are sorted. After that sentences of similar lengths are closer to each other. Batches are then formed with sentences of similar lengths. Much less padding will be required to turning sentences of similar lengths into senetences of equal lengths. 


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

### Creating train, validation and test batches


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

### Function for converting vector of size word_vec_dim into the closest reprentative english word. 


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

### Function for generating a sequence of positional codes for positional encoding.


```python
def positional_encoding(seq_len,model_dimensions):
    pe = np.zeros((seq_len,model_dimensions,),np.float32)
    for pos in xrange(0,seq_len):
        for i in xrange(0,model_dimensions):
            pe[pos][i] = math.sin(pos/(10000**(2*i/model_dimensions)))
    return pe.reshape((seq_len,model_dimensions))
```

### Hyperparametes.


```python
import tensorflow as tf

h=8 #no. of heads
N=3 #no. of decoder and encoder layers
learning_rate=0.002
iters = 200
max_len = 25
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None,None,word_vec_dim])
y = tf.placeholder(tf.int32, [None,None])
output_len = tf.placeholder(tf.int32)
teacher_forcing = tf.placeholder(tf.bool)
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

### Pre-generating all possible masks

These masks are to be used to fill illegal positions with infinity (or a very high value eg. 2^30).

Illegal positions are positions in the decoder input sequence which may be filled in the future timestep,
but currently not filled. These positions shouldn't be allowed to influence the network.

The masks are used to assign the value 2^30 to all positions in the tensor influenced by the illegal ones.
After going through the softmax layer, these positions become close to 0, as it should be.

Dynamically creating masks depending on the current position\timestep (depending on which the program can know which positions are legal and which aren't) is, however,
a bit troublesome with tensorflow tf_while_loop. 

I am pre-generating all possible masks here, and pack them into a tensor such that the network can dynamically
access the required mask using the index of the tensor (the index will same be the timestep) 
                                                                    
                                                                   


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
    scale3 = weights['scale2']
    shift3 = weights['shift2']

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

It follows the encoder-decoder paradigm. The main exception from standard encoder-decoder paradigm, is that it uses 'transformers' instead of Reccurrent networks. 

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
    shift_enc_2 = tf.Variable(tf.ones([N,1,word_vec_dim]),dtype=tf.float32)
    scale_dec_1 = tf.Variable(tf.ones([N,1,word_vec_dim]),dtype=tf.float32)
    shift_dec_1 = tf.Variable(tf.ones([N,1,word_vec_dim]),dtype=tf.float32)
    scale_dec_2 = tf.Variable(tf.ones([N,1,word_vec_dim]),dtype=tf.float32)
    shift_dec_2 = tf.Variable(tf.ones([N,1,word_vec_dim]),dtype=tf.float32)
    scale_dec_3 = tf.Variable(tf.ones([N,1,word_vec_dim]),dtype=tf.float32)
    shift_dec_3 = tf.Variable(tf.ones([N,1,word_vec_dim]),dtype=tf.float32)
    
    #Parameters for the linear layers converting decoder output to probability distibutions.   
    d=1024
    Wpd1 = tf.Variable(tf.truncated_normal(shape=[max_len*word_vec_dim,d],stddev=0.01))
    bpd1 = tf.Variable(tf.truncated_normal(shape=[1,d],stddev=0.01))
    Wpd2 = tf.Variable(tf.truncated_normal(shape=[d,vocab_len],stddev=0.01))
    bpd2 = tf.Variable(tf.truncated_normal(shape=[1,vocab_len],stddev=0.01))
                         
    encoderin = x #should be already positionally encoded 
    
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
        #decoderin = tf.nn.dropout(decoderin,keep_prob)
        
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
                'scale3': scale_dec_2[j],
                'shift3': shift_dec_2[j],
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

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.98,epsilon=1e-9).minimize(cost)

#wanna add some temperature?

"""temperature = 0.7
scaled_output = tf.log(output)/temperature
softmax_output = tf.nn.softmax(scaled_output)"""

#(^Use it with "#prediction_int = np.random.choice(range(vocab_len), p=array.ravel())")

softmax_output = tf.nn.softmax(output)

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
            
            # Run optimization operation (backpropagation)
            _,loss,out = sess.run([optimizer,cost,softmax_output],
                                  feed_dict={x: (train_batch_x[shuffled_indices[i]]+pe_in), 
                                             y: train_batch_y[shuffled_indices[i]],
                                             keep_prob: 0.9,
                                             output_len: len(train_batch_y[shuffled_indices[i]][0]),
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

    
    CHOSEN SAMPLE NO.: 33
    
    Epoch: 1 Iteration: 1
    
    SAMPLE TEXT:
    tom continued yelling <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পডোতে ভযাবহ অপদসত করে। এলেন। নাচে।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম চেচাতেই থাকলো। <EOS> <PAD> <PAD> 
    
    loss=8.20932
    
    CHOSEN SAMPLE NO.: 45
    
    Epoch: 1 Iteration: 2
    
    SAMPLE TEXT:
    now listen carefully <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এবার মন দিযে শোনো। <EOS> 
    
    loss=23.314
    
    CHOSEN SAMPLE NO.: 17
    
    Epoch: 1 Iteration: 3
    
    SAMPLE TEXT:
    we do not care what he does <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ও কী করে না করে আমরা তা কেযার করি না। <EOS> 
    
    loss=30.1632
    
    CHOSEN SAMPLE NO.: 52
    
    Epoch: 1 Iteration: 4
    
    SAMPLE TEXT:
    call tom <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টম আমি টম আমার বাডিতে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টমকে ডাকো। <EOS> <PAD> <PAD> 
    
    loss=15.766
    
    CHOSEN SAMPLE NO.: 18
    
    Epoch: 1 Iteration: 5
    
    SAMPLE TEXT:
    bring it closer <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এটা দিন। চিৎকার আমরা এটা সে এখানে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ওটাকে কাছে নিযে আসো। <EOS> <PAD> <PAD> 
    
    loss=10.3252
    
    CHOSEN SAMPLE NO.: 37
    
    Epoch: 1 Iteration: 6
    
    SAMPLE TEXT:
    success depends mostly on effort <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কটিল। বাডি শধ দেখতে দর তিনি হযেছি। করিযে পযানকেক
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    সাফলয অধিকাংশ কষেতরে পরচেষটার উপর নিরভর করে। <EOS> <PAD> 
    
    loss=7.04045
    
    CHOSEN SAMPLE NO.: 32
    
    Epoch: 1 Iteration: 7
    
    SAMPLE TEXT:
    you are always complaining <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> <PAD> <PAD> চাই। <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনারা সবসময অভিযোগ করেন। <EOS> <PAD> <PAD> <PAD> 
    
    loss=6.53417
    
    CHOSEN SAMPLE NO.: 61
    
    Epoch: 1 Iteration: 8
    
    SAMPLE TEXT:
    where is the ticket counter <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> <PAD> <PAD> <PAD> ভিটামিন <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টিকিট কাউনটারটা কোথায <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=5.41661
    
    CHOSEN SAMPLE NO.: 58
    
    Epoch: 1 Iteration: 9
    
    SAMPLE TEXT:
    eat everything <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    দিনে অনমতি জিজঞাসা সকালের দযা ও
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    সবকিছই খাও। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.97987
    
    CHOSEN SAMPLE NO.: 26
    
    Epoch: 1 Iteration: 10
    
    SAMPLE TEXT:
    tom has been injured <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> আসতে পদবিটি সেটা রাখন। শর আপনাকে <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম আহত হযেছে। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.75854
    
    CHOSEN SAMPLE NO.: 40
    
    Epoch: 1 Iteration: 11
    
    SAMPLE TEXT:
    i wonder what i should get you for your birthday <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> এটে <PAD> <PAD> কত <PAD> <PAD> কথা দেবে। <EOS> <PAD> <PAD> <PAD> জনযে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ভাবছি তোমাকে জনমদিনে কী দেবো। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=5.83102
    
    CHOSEN SAMPLE NO.: 12
    
    Epoch: 1 Iteration: 12
    
    SAMPLE TEXT:
    he is eating lunch now <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> <EOS> <PAD> <PAD> আমাকে <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    উনি এখন লাঞচ করছেন। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=5.0865
    
    CHOSEN SAMPLE NO.: 62
    
    Epoch: 1 Iteration: 13
    
    SAMPLE TEXT:
    i 'm going to take a bath <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    চেচালো। আমাদের <PAD> কেন <PAD> <EOS> <EOS> <EOS> <EOS> দেরী
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি সনান করতে যাবো। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=5.57745
    
    CHOSEN SAMPLE NO.: 39
    
    Epoch: 1 Iteration: 14
    
    SAMPLE TEXT:
    where can i buy that magazine <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    হযে একলা জিজঞাসা কাশলেন। এসি অফিসে আজ থাকবেন <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ওই পতরিকাটা কোথা থেকে কিনতে পারব <EOS> <PAD> 
    
    loss=5.6702
    
    CHOSEN SAMPLE NO.: 61
    
    Epoch: 1 Iteration: 15
    
    SAMPLE TEXT:
    children love doing this <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> দদিনে না খোলে হযেছি। কারণ না।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    বাচচারা এগলো করতে ভালোবাসে। <EOS> <PAD> <PAD> 
    
    loss=5.06569
    
    CHOSEN SAMPLE NO.: 9
    
    Epoch: 1 Iteration: 16
    
    SAMPLE TEXT:
    is this seat empty <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কিনত <PAD> তেমন সাথে <EOS> চাবি। আছি।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এই সীটটা কি ফাকা আছে <EOS> <PAD> 
    
    loss=4.98385
    
    CHOSEN SAMPLE NO.: 24
    
    Epoch: 1 Iteration: 17
    
    SAMPLE TEXT:
    he eats nothing but fruit <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> <PAD> <EOS> খব ফেলেছি এলেন। গেছে। <PAD> কম
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    সে ফল ছাডা আর কিছই খায না। <EOS> <PAD> 
    
    loss=4.68397
    
    CHOSEN SAMPLE NO.: 33
    
    Epoch: 1 Iteration: 18
    
    SAMPLE TEXT:
    how much is this <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    যাব। কোনো শর <EOS> আমি আমি কেন চান
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এটা কত <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.43482
    
    CHOSEN SAMPLE NO.: 0
    
    Epoch: 1 Iteration: 19
    
    SAMPLE TEXT:
    listen carefully <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> <PAD> <PAD> টম আমি <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    মন দিযে শনবে। <EOS> <PAD> <PAD> 
    
    loss=4.56151
    
    CHOSEN SAMPLE NO.: 11
    
    Epoch: 1 Iteration: 20
    
    SAMPLE TEXT:
    tom coughs a lot <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> <PAD> <PAD> আমি এটা <PAD> টরেনে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম খব কাশে। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.96252
    
    CHOSEN SAMPLE NO.: 59
    
    Epoch: 1 Iteration: 21
    
    SAMPLE TEXT:
    i am busy <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তারা যাইনি। <PAD> <PAD> আমরা করতে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি বযসত। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.53902
    
    CHOSEN SAMPLE NO.: 8
    
    Epoch: 1 Iteration: 22
    
    SAMPLE TEXT:
    i live here <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    সতরী। কোথায বযবহার চযানেলটা চাই। বোতাম
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি এখানে থাকি। <EOS> <PAD> <PAD> 
    
    loss=4.7305
    
    CHOSEN SAMPLE NO.: 55
    
    Epoch: 1 Iteration: 23
    
    SAMPLE TEXT:
    is your watch correct <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    চেষটা <PAD> ডাকন। <EOS> টম চেচাচছো বেশীই বিভরানত
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনার ঘডি ঠিক আছে <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.47656
    
    CHOSEN SAMPLE NO.: 9
    
    Epoch: 1 Iteration: 24
    
    SAMPLE TEXT:
    would you like to come <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তমি ওর টম <PAD> ফল কে <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি কি আসতে চান <EOS> <PAD> <PAD> <PAD> 
    
    loss=5.20824
    
    CHOSEN SAMPLE NO.: 21
    
    Epoch: 1 Iteration: 25
    
    SAMPLE TEXT:
    no problem <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ছাড তমি তিনি <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    কোনো অসবিধা নেই <EOS> <PAD> 
    
    loss=4.89407
    
    CHOSEN SAMPLE NO.: 49
    
    Epoch: 1 Iteration: 26
    
    SAMPLE TEXT:
    tom is as big as i am <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> ছিলাম। <PAD> ওটা <PAD> <PAD> না। পরাতঃরাশের <PAD> <EOS> <EOS> কথা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম আমার মতই বডো। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.58395
    
    CHOSEN SAMPLE NO.: 20
    
    Epoch: 1 Iteration: 27
    
    SAMPLE TEXT:
    you do not understand <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> <PAD> ওখানে শিকষক নাম দাও। <EOS> সেটা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনারা বোঝেন না। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.36749
    
    CHOSEN SAMPLE NO.: 16
    
    Epoch: 1 Iteration: 28
    
    SAMPLE TEXT:
    do you want to eat now or later <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> <PAD> ফিরে ভাল <PAD> বনধ পডেছি। <PAD> <PAD> <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি কি এখন খেতে চাও না পরে <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.85434
    
    CHOSEN SAMPLE NO.: 31
    
    Epoch: 1 Iteration: 29
    
    SAMPLE TEXT:
    turn left at the corner <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বাইকটা <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    মোডে গিযে বাদিকে বেকে যাবেন। <EOS> <PAD> <PAD> 
    
    loss=5.1344
    
    CHOSEN SAMPLE NO.: 5
    
    Epoch: 1 Iteration: 30
    
    SAMPLE TEXT:
    how many museums did you visit <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    চযানেলটা <PAD> লাগেনি। ভালো বাস আমাকে খাওযার কম করছে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি কতগলো জাদঘর ঘরেছো <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=5.30194
    
    CHOSEN SAMPLE NO.: 55
    
    Epoch: 1 Iteration: 31
    
    SAMPLE TEXT:
    i did my work <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> <PAD> <PAD> গেছে নতন করন। <PAD> আমরা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি আমার কাজ করলাম। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.44533
    
    CHOSEN SAMPLE NO.: 63
    
    Epoch: 1 Iteration: 32
    
    SAMPLE TEXT:
    i 'd be grateful <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> চেষটা দেখক। ছবি হযেছে। বললাম। <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি কতজঞ থাকবো। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.88166
    
    CHOSEN SAMPLE NO.: 52
    
    Epoch: 1 Iteration: 33
    
    SAMPLE TEXT:
    i 've been hoping you would drop in <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পরিষকার <EOS> <PAD> কি <PAD> এটা আগে কোনো যেতে <PAD> <PAD> এই
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ভাবছিলাম আপনি এসে উপসথিত হবেন। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.88528
    
    CHOSEN SAMPLE NO.: 25
    
    Epoch: 1 Iteration: 34
    
    SAMPLE TEXT:
    may i eat <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এই দিন। <PAD> <PAD> না। <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি কি খেতে পারি <EOS> <PAD> 
    
    loss=4.3566
    
    CHOSEN SAMPLE NO.: 22
    
    Epoch: 1 Iteration: 35
    
    SAMPLE TEXT:
    she is younger than me <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> চেচালো। খেযে চম <PAD> কাজটা কত <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    সে আমার থেকে ছোট। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.79859
    
    CHOSEN SAMPLE NO.: 24
    
    Epoch: 1 Iteration: 36
    
    SAMPLE TEXT:
    tom does not look very happy <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    দিলাম। খেলো। লিফট <PAD> <EOS> কষমা এসেছিলো। শর কেমন <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টমকে দেখে খব খশি বলে মনে হচচে না। <EOS> <PAD> 
    
    loss=4.83564
    
    CHOSEN SAMPLE NO.: 36
    
    Epoch: 1 Iteration: 37
    
    SAMPLE TEXT:
    may i use a credit card <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    খাচছেন। <PAD> পিছ <EOS> কোন ভালো ওরা ২০১৩ পরেছিলাম। এটা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি কি করেডিট কারড বযবহার করতে পারি <EOS> <PAD> <PAD> 
    
    loss=4.62498
    
    CHOSEN SAMPLE NO.: 46
    
    Epoch: 1 Iteration: 38
    
    SAMPLE TEXT:
    take us there <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> এক <PAD> ছবিটার রাখন। বঝতে রাখ।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমাদের ওখানে নিযে চলো। <EOS> <PAD> <PAD> 
    
    loss=4.12933
    
    CHOSEN SAMPLE NO.: 29
    
    Epoch: 1 Iteration: 39
    
    SAMPLE TEXT:
    tom tried to keep from smiling <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> <PAD> দিযেছে। <PAD> আমি <PAD> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম না হাসার চেষটা করছিলেন। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.68603
    
    CHOSEN SAMPLE NO.: 37
    
    Epoch: 1 Iteration: 40
    
    SAMPLE TEXT:
    it is rather cold today <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    একবার জানেন <PAD> <PAD> চিনতা ইচছে <PAD> করন <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আজ বেশ ঠানডা আছে। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.37035
    
    CHOSEN SAMPLE NO.: 3
    
    Epoch: 1 Iteration: 41
    
    SAMPLE TEXT:
    how can i get to the hospital by bus <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    অনগরহ ওখানেই ওযাইকিকি কাছে ডানদিকে <EOS> আমরা <PAD> <EOS> <PAD> যোগাযোগ টম
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি বাসে করে হাসপাতালে কিভাবে যাব <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=5.38092
    
    CHOSEN SAMPLE NO.: 37
    
    Epoch: 1 Iteration: 42
    
    SAMPLE TEXT:
    i 'm sorry but it is impossible <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পরকতই টম উচ লাগছে। পারেন। <PAD> না। <PAD> ডেকেছিলে <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমায কষমা করবেন কিনত এটা সমভব নয। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.45662
    
    CHOSEN SAMPLE NO.: 48
    
    Epoch: 1 Iteration: 43
    
    SAMPLE TEXT:
    call me <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তোমারে করলেন। পাশ দৌডাল গেছিলাম।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমাকে ডেক। <EOS> <PAD> <PAD> 
    
    loss=5.16164
    
    CHOSEN SAMPLE NO.: 63
    
    Epoch: 1 Iteration: 44
    
    SAMPLE TEXT:
    you are always complaining <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> <PAD> অবধি <PAD> <PAD> <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি সবসময অভিযোগ করেন। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.31976
    
    CHOSEN SAMPLE NO.: 36
    
    Epoch: 1 Iteration: 45
    
    SAMPLE TEXT:
    go wait outside <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> <PAD> <EOS> <EOS> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    বাইরে গিযে অপেকষা করো। <EOS> <PAD> <PAD> 
    
    loss=4.25997
    
    CHOSEN SAMPLE NO.: 40
    
    Epoch: 1 Iteration: 46
    
    SAMPLE TEXT:
    there is no way i 'm leaving you here alone with tom <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> <PAD> আমি <PAD> <EOS> টম <PAD> <PAD> <PAD> টম <EOS> <PAD> কিনতে <EOS> <EOS> <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি কোনমতেই তোমাকে এখানে টমের সাথে একলা ছেডে রেখে যেতে পারবো না। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=5.66735
    
    CHOSEN SAMPLE NO.: 44
    
    Epoch: 1 Iteration: 47
    
    SAMPLE TEXT:
    what you said is not true <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> বলবেন <EOS> কিছ <EOS> যাওযা <PAD> <PAD> আমি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি যা বললে তা সতযি নয। <EOS> <PAD> <PAD> 
    
    loss=4.76324
    
    CHOSEN SAMPLE NO.: 57
    
    Epoch: 1 Iteration: 48
    
    SAMPLE TEXT:
    i play football <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পারি। মজার চিনী বযবসথা মত <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ফটবল খেলি। <EOS> <PAD> <PAD> 
    
    loss=4.73921
    
    CHOSEN SAMPLE NO.: 14
    
    Epoch: 1 Iteration: 49
    
    SAMPLE TEXT:
    write your address here <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কোরো আমার চপচাপ চপ <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এখানে আপনার ঠিকানা লিখন। <EOS> <PAD> 
    
    loss=5.06292
    
    CHOSEN SAMPLE NO.: 55
    
    Epoch: 1 Iteration: 50
    
    SAMPLE TEXT:
    we are desperate to find a solution <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> ছিল আপনার <EOS> তমি <EOS> <PAD> কত ফরাসি কি এসে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমরা একটি সমাধান খোজার জনয মরিযা হযে আছি। <EOS> <PAD> <PAD> 
    
    loss=4.82761
    
    CHOSEN SAMPLE NO.: 14
    
    Epoch: 1 Iteration: 51
    
    SAMPLE TEXT:
    he got angry <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    চান টম কলানত। বইটা বরেকফাসট
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    উনি রেগে গেছিলেন। <EOS> <PAD> 
    
    loss=4.98967
    
    CHOSEN SAMPLE NO.: 0
    
    Epoch: 2 Iteration: 1
    
    SAMPLE TEXT:
    i 'm tom 's wife <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কিনতে এখানে। <PAD> কি কে <EOS> না। মন এটা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি টমের সতরী। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.19294
    
    CHOSEN SAMPLE NO.: 8
    
    Epoch: 2 Iteration: 2
    
    SAMPLE TEXT:
    what exactly has tom done <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> <PAD> <PAD> <PAD> <EOS> <EOS> <EOS> গেছে।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম ঠিক কী করেছে <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.53906
    
    CHOSEN SAMPLE NO.: 33
    
    Epoch: 2 Iteration: 3
    
    SAMPLE TEXT:
    we are trapped <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> খব চিৎকার <PAD> <PAD> আমরা <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমরা আটকে পরেছি। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.86885
    
    CHOSEN SAMPLE NO.: 32
    
    Epoch: 2 Iteration: 4
    
    SAMPLE TEXT:
    i screamed <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    শর করছি। <PAD> পছনদ <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি চিৎকার করলাম। <EOS> <PAD> 
    
    loss=4.40573
    
    CHOSEN SAMPLE NO.: 21
    
    Epoch: 2 Iteration: 5
    
    SAMPLE TEXT:
    can you help me <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ভারা অনগরহ <PAD> <EOS> <PAD> সে এসেছে। টমের
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি কি আমাকে সাহাযয করতে পারবেন <EOS> <PAD> 
    
    loss=3.98898
    
    CHOSEN SAMPLE NO.: 57
    
    Epoch: 2 Iteration: 6
    
    SAMPLE TEXT:
    i was away <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> <PAD> রাসতা হচছে <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি বাইরে ছিলাম। <EOS> <PAD> 
    
    loss=4.78602
    
    CHOSEN SAMPLE NO.: 47
    
    Epoch: 2 Iteration: 7
    
    SAMPLE TEXT:
    there is room inside <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কথা করলো। সীটটা হতে সময ভেতরে ডাকতার। <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ভেতরে একটা ঘর আছে। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.99248
    
    CHOSEN SAMPLE NO.: 10
    
    Epoch: 2 Iteration: 8
    
    SAMPLE TEXT:
    i 'm not denying that <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> নডলস বাপ <PAD> <PAD> <PAD> <PAD> আপনাকে <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ওটা অসবীকার করছি না। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.1744
    
    CHOSEN SAMPLE NO.: 47
    
    Epoch: 2 Iteration: 9
    
    SAMPLE TEXT:
    where will tom go <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পরেছিলাম। <EOS> সমভব করলো। এটা <EOS> <PAD> তারাতারি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম কোথায যাবে <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.09275
    
    CHOSEN SAMPLE NO.: 38
    
    Epoch: 2 Iteration: 10
    
    SAMPLE TEXT:
    why do you want to leave today <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ডাকতার। <EOS> করতে দৌডে ঠিক সঙগে <PAD> <EOS> <EOS> পছনদ <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি আজকেই যেতে চাইছেন কেন <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.20466
    
    CHOSEN SAMPLE NO.: 54
    
    Epoch: 2 Iteration: 11
    
    SAMPLE TEXT:
    open your mouth <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    চলন। বসটনে দাডি করত। আপনাদেরকে <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    মখ খোলো <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.13812
    
    CHOSEN SAMPLE NO.: 39
    
    Epoch: 2 Iteration: 12
    
    SAMPLE TEXT:
    we are not yet sure what the problem is <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    করে চিঠি রাতরি গেছিল। তৈরি <PAD> <PAD> <PAD> <EOS> <PAD> ওর ছিলাম।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমারা সমসযাটার বযাপারে এখনো নিশচিত হইনি। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=5.20972
    
    CHOSEN SAMPLE NO.: 27
    
    Epoch: 2 Iteration: 13
    
    SAMPLE TEXT:
    what he said is not true <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এখানেই সব কেন <PAD> আমি <EOS> <PAD> ওই <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ও যা বললো তা সতযি নয। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.08967
    
    CHOSEN SAMPLE NO.: 1
    
    Epoch: 2 Iteration: 14
    
    SAMPLE TEXT:
    what do you want to be when you grow up <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টমকে শহর সাহস তমি নিচে <EOS> <EOS> পছনদ <EOS> <EOS> <EOS> কর। <EOS> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি বডো হযে কী হতে চাও <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=5.01999
    
    CHOSEN SAMPLE NO.: 36
    
    Epoch: 2 Iteration: 15
    
    SAMPLE TEXT:
    i 've finished my work <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    রেখে বঝতে অপদসত ডাকতার কেন কোথায <PAD> <PAD> কেউ
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি আমার কাজ শেষ করেছি। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.15386
    
    CHOSEN SAMPLE NO.: 41
    
    Epoch: 2 Iteration: 16
    
    SAMPLE TEXT:
    they screamed <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    যদধটা দিন। পারেন। পারেন। সব
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ওনারা চিৎকার করলেন। <EOS> <PAD> 
    
    loss=5.0103
    
    CHOSEN SAMPLE NO.: 47
    
    Epoch: 2 Iteration: 17
    
    SAMPLE TEXT:
    which hat is yours <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ভালোবাসা একবার করে <EOS> চেচালাম। পডতে তমি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    কোন টপিটা তোমার <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.33538
    
    CHOSEN SAMPLE NO.: 30
    
    Epoch: 2 Iteration: 18
    
    SAMPLE TEXT:
    i 'm still your friend <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> <PAD> <PAD> খায <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি এখনো তোমাদের বনধ আছি। <EOS> <PAD> <PAD> 
    
    loss=5.00796
    
    CHOSEN SAMPLE NO.: 7
    
    Epoch: 2 Iteration: 19
    
    SAMPLE TEXT:
    who is screaming <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> <EOS> <EOS> হযেছে। <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    কে চেচাচছে <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.20484
    
    CHOSEN SAMPLE NO.: 4
    
    Epoch: 2 Iteration: 20
    
    SAMPLE TEXT:
    you should eat vegetables <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বারে <PAD> আপনাকে <PAD> <EOS> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তোমার শাকসবজি খাওযা উচিত। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.18169
    
    CHOSEN SAMPLE NO.: 14
    
    Epoch: 2 Iteration: 21
    
    SAMPLE TEXT:
    tom is a bit shorter than mary <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> <EOS> <PAD> হযেছে। আমি টম হয। জীবন <EOS> একলা করো। এটা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম মেরির থেকে একট বেটে। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.18676
    
    CHOSEN SAMPLE NO.: 56
    
    Epoch: 2 Iteration: 22
    
    SAMPLE TEXT:
    i am going to osaka station <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    হাত আজ আমাকে কি <PAD> <PAD> সতযি দেখে <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ওসাকা সটেশনে যাচছি। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.69879
    
    CHOSEN SAMPLE NO.: 8
    
    Epoch: 2 Iteration: 23
    
    SAMPLE TEXT:
    why are you shouting <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আপনার আমি চযানেলটা <PAD> <PAD> বযাপারে <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি চেচাচছো কেন <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.47645
    
    CHOSEN SAMPLE NO.: 42
    
    Epoch: 2 Iteration: 24
    
    SAMPLE TEXT:
    do you still read books <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কে বসটনে মোটামটি অরথ পরের গাডি <PAD> বেচে ওখানে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি কি এখনো বই পডেন <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.50996
    
    CHOSEN SAMPLE NO.: 24
    
    Epoch: 2 Iteration: 25
    
    SAMPLE TEXT:
    tom does not look very happy <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বলছে সাথে অবসথাটা আপনি বেকারিতে <EOS> আমি <PAD> <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টমকে দেখে খব খশি বলে মনে হচচে না। <EOS> <PAD> 
    
    loss=4.5684
    
    CHOSEN SAMPLE NO.: 18
    
    Epoch: 2 Iteration: 26
    
    SAMPLE TEXT:
    tom is dreadfully wrong <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    অপেকষা <PAD> <PAD> <EOS> <PAD> শেখো <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম মারাতমক ভাবে ভল। <EOS> <PAD> <PAD> 
    
    loss=4.48781
    
    CHOSEN SAMPLE NO.: 4
    
    Epoch: 2 Iteration: 27
    
    SAMPLE TEXT:
    we will help <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> গরামটাকে <EOS> <PAD> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমরা সাহাযয করব। <EOS> <PAD> <PAD> 
    
    loss=4.63347
    
    CHOSEN SAMPLE NO.: 23
    
    Epoch: 2 Iteration: 28
    
    SAMPLE TEXT:
    we are going the wrong way <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    না আপনার <PAD> খোলে <PAD> আছে <EOS> <EOS> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমরা ভল রাসতায যাচছি। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.96231
    
    CHOSEN SAMPLE NO.: 43
    
    Epoch: 2 Iteration: 29
    
    SAMPLE TEXT:
    she is eating fruit <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    নয। কষেতরে ফেলেছি মাতভাষা <PAD> <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তিনি ফল খাচচেন। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.98182
    
    CHOSEN SAMPLE NO.: 53
    
    Epoch: 2 Iteration: 30
    
    SAMPLE TEXT:
    she went to paris to see her aunt <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> বইটা <PAD> মেরি খাবেন <PAD> <EOS> <PAD> সাতটার সখী করতে <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    সে পযারিস গেছে তার কাকিমার সঙগে দেখা করতে। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.45319
    
    CHOSEN SAMPLE NO.: 52
    
    Epoch: 2 Iteration: 31
    
    SAMPLE TEXT:
    we are arabs <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আপনার নিন। নই। তোমার করবেন <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমরা আরব। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.30661
    
    CHOSEN SAMPLE NO.: 37
    
    Epoch: 2 Iteration: 32
    
    SAMPLE TEXT:
    tom does not want to see you <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    চাবি <PAD> আমাকে বাস কখনো করে কি না আপনি আমি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম তোমার সঙগে দেখা করতে চায না। <EOS> <PAD> <PAD> 
    
    loss=4.88417
    
    CHOSEN SAMPLE NO.: 28
    
    Epoch: 2 Iteration: 33
    
    SAMPLE TEXT:
    he decided not to go to the party <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কল আমেরিকান <PAD> <PAD> শিখেছি। <PAD> <PAD> <PAD> <PAD> আমি <PAD> টম
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তিনি পারটিতে না যাওযাই ঠিক করলেন। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.54944
    
    CHOSEN SAMPLE NO.: 61
    
    Epoch: 2 Iteration: 34
    
    SAMPLE TEXT:
    my shoulders hurt <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    খেলি। ইচছা কথা <PAD> <PAD> ওনার
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার কাধ বযাথা করছে। <EOS> <PAD> 
    
    loss=4.51995
    
    CHOSEN SAMPLE NO.: 21
    
    Epoch: 2 Iteration: 35
    
    SAMPLE TEXT:
    have you ever visited boston <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    শর তার কে নতন জাপান আমার <PAD> জমা <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি কি কখনো বসটন দেখতে গেছো <EOS> <PAD> <PAD> 
    
    loss=4.39666
    
    CHOSEN SAMPLE NO.: 47
    
    Epoch: 2 Iteration: 36
    
    SAMPLE TEXT:
    i fell in love with him <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আজ <PAD> বডড <EOS> আর <PAD> আছে। আমি <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি তার পরেমে পডে গেলাম। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.97433
    
    CHOSEN SAMPLE NO.: 34
    
    Epoch: 2 Iteration: 37
    
    SAMPLE TEXT:
    go slow <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কথা লকষ <PAD> মত করে। ঘমানো
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আসতে যা। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.85364
    
    CHOSEN SAMPLE NO.: 18
    
    Epoch: 2 Iteration: 38
    
    SAMPLE TEXT:
    tom stopped talking as soon as he noticed mary was not listening anymore <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি তমি তারা হযতো ককরকে <PAD> সে <PAD> <PAD> করেন <PAD> জতো <PAD> তাডাতাডি আসো। পরে <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম যখন দেখলো যে মেরি আর কথা শনছে না তখন সে সঙগে সঙগে কথা বনধ করে দিলো। <EOS> 
    
    loss=5.12669
    
    CHOSEN SAMPLE NO.: 19
    
    Epoch: 2 Iteration: 39
    
    SAMPLE TEXT:
    why me <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    করব। আমি সবাই <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমিই কেন <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.79779
    
    CHOSEN SAMPLE NO.: 43
    
    Epoch: 2 Iteration: 40
    
    SAMPLE TEXT:
    i need new soles on these shoes <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    খাচছি। ভত <PAD> <PAD> <PAD> <PAD> আর <PAD> এই আমি <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এই জতোগলোর জনয নতন সকতলা চাই। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.66124
    
    CHOSEN SAMPLE NO.: 58
    
    Epoch: 2 Iteration: 41
    
    SAMPLE TEXT:
    eat whatever food you like <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টমকে এসে <PAD> জামা দেখা <PAD> ঠিক <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনার যে খাবার পছনদ হয সেটা খান। <EOS> 
    
    loss=4.67541
    
    CHOSEN SAMPLE NO.: 29
    
    Epoch: 2 Iteration: 42
    
    SAMPLE TEXT:
    be quiet <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    সময এখনও ভেবে <PAD> কোনো
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    চপ কর। <EOS> <PAD> <PAD> 
    
    loss=4.21478
    
    CHOSEN SAMPLE NO.: 43
    
    Epoch: 2 Iteration: 43
    
    SAMPLE TEXT:
    who would believe me <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আজকের আমার না। অসফল <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি কাকে বিশবাস করবেন <EOS> <PAD> <PAD> 
    
    loss=4.37287
    
    CHOSEN SAMPLE NO.: 29
    
    Epoch: 2 Iteration: 44
    
    SAMPLE TEXT:
    i love football <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এখানে এখন <EOS> টমের <EOS> ইচছে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ফটবল ভালোবাসি। <EOS> <PAD> <PAD> 
    
    loss=4.11787
    
    CHOSEN SAMPLE NO.: 41
    
    Epoch: 2 Iteration: 45
    
    SAMPLE TEXT:
    what time does the ship leave <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> জিনিস না <PAD> যেতে <PAD> <PAD> <PAD> এলো। পেযেছেন।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    জাহাজটা কটার সময ছাডে <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.48653
    
    CHOSEN SAMPLE NO.: 32
    
    Epoch: 2 Iteration: 46
    
    SAMPLE TEXT:
    why do you want to hurt tom <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> ঘমানো <EOS> <EOS> <PAD> <EOS> <PAD> <EOS> <PAD> <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি টমকে আঘাত দিতে চান কেন <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.91589
    
    CHOSEN SAMPLE NO.: 33
    
    Epoch: 2 Iteration: 47
    
    SAMPLE TEXT:
    tom will speak <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বলে। হবে ডান <EOS> আমার সাথে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম কথা বলবে। <EOS> <PAD> <PAD> 
    
    loss=3.93453
    
    CHOSEN SAMPLE NO.: 50
    
    Epoch: 2 Iteration: 48
    
    SAMPLE TEXT:
    a car hit tom <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আনদাজ টম <EOS> দ <PAD> <PAD> <PAD> ছাডে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    একটা গাডী টমকে ধাককা মারল। <EOS> <PAD> <PAD> 
    
    loss=3.95405
    
    CHOSEN SAMPLE NO.: 5
    
    Epoch: 2 Iteration: 49
    
    SAMPLE TEXT:
    who is that <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    দেখেছিলাম। <EOS> খশি <EOS> করছে।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ওটা কে <EOS> <PAD> <PAD> 
    
    loss=4.71445
    
    CHOSEN SAMPLE NO.: 58
    
    Epoch: 2 Iteration: 50
    
    SAMPLE TEXT:
    i have not tried <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আছে খশি <PAD> <PAD> চেষটা <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি চেষটা করিনি। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.81262
    
    CHOSEN SAMPLE NO.: 54
    
    Epoch: 2 Iteration: 51
    
    SAMPLE TEXT:
    leave it there <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> আছেন <EOS> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ওটা ওখানেই রাখন। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.60077
    
    CHOSEN SAMPLE NO.: 43
    
    Epoch: 3 Iteration: 1
    
    SAMPLE TEXT:
    please speak slowly <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> ফাকা সেটা কি <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    অনগরহ করে আসতে কথা বলন। <EOS> <PAD> 
    
    loss=3.69819
    
    CHOSEN SAMPLE NO.: 9
    
    Epoch: 3 Iteration: 2
    
    SAMPLE TEXT:
    would you like to come <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বাচচাদের তিনি করতে <PAD> <EOS> আমি <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি কি আসতে চান <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.67028
    
    CHOSEN SAMPLE NO.: 24
    
    Epoch: 3 Iteration: 3
    
    SAMPLE TEXT:
    tom grew up in boston <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পেযেছে যাবো। <PAD> <EOS> <EOS> <PAD> <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম বসটনে বডো হযেছে। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.43666
    
    CHOSEN SAMPLE NO.: 59
    
    Epoch: 3 Iteration: 4
    
    SAMPLE TEXT:
    nobody is speaking <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    সথির এক যদধে থাকেন। <PAD> আছে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    কেউ কথা বলছেন না। <EOS> <PAD> 
    
    loss=4.25782
    
    CHOSEN SAMPLE NO.: 49
    
    Epoch: 3 Iteration: 5
    
    SAMPLE TEXT:
    we are all scared <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    একজন খোজা সেটা না <EOS> <PAD> যাও।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমরা সবাই ভিত। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.17715
    
    CHOSEN SAMPLE NO.: 3
    
    Epoch: 3 Iteration: 6
    
    SAMPLE TEXT:
    stop there <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বাডি করলেন। <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ওখানেই থামো। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.53829
    
    CHOSEN SAMPLE NO.: 3
    
    Epoch: 3 Iteration: 7
    
    SAMPLE TEXT:
    what time does the train reach osaka <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    হরদটা <EOS> ও হাত-ঘডিটা <PAD> <PAD> <EOS> <EOS> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টরেনটা কটার সময ওসাকা পৌছায <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.41385
    
    CHOSEN SAMPLE NO.: 26
    
    Epoch: 3 Iteration: 8
    
    SAMPLE TEXT:
    tom ate breakfast alone <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমাকে টম কেন <PAD> <EOS> <PAD> <PAD> তার
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম একলা একলা বরেকফাসট খেলো। <EOS> <PAD> <PAD> 
    
    loss=3.97969
    
    CHOSEN SAMPLE NO.: 7
    
    Epoch: 3 Iteration: 9
    
    SAMPLE TEXT:
    tom brought this <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    হচছে। পরতযেকে ওটা আমার <PAD> একলা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম এটা নিযে এসেছিল। <EOS> <PAD> 
    
    loss=3.90411
    
    CHOSEN SAMPLE NO.: 45
    
    Epoch: 3 Iteration: 10
    
    SAMPLE TEXT:
    he does not eat anything other than fruit <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কেদেছিলেন। বাইরে কেউ ভালো <PAD> <PAD> <PAD> <PAD> দর <PAD> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    উনি ফল ছাডা অনয কিছ খান না। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.59867
    
    CHOSEN SAMPLE NO.: 16
    
    Epoch: 3 Iteration: 11
    
    SAMPLE TEXT:
    i 'll buy it <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    সহজ। নাম করতে <PAD> <EOS> <EOS> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি এটা কিনবো। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.10658
    
    CHOSEN SAMPLE NO.: 26
    
    Epoch: 3 Iteration: 12
    
    SAMPLE TEXT:
    he spoke <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বলন <PAD> টমের <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তিনি বললেন। <EOS> <PAD> <PAD> 
    
    loss=4.34343
    
    CHOSEN SAMPLE NO.: 58
    
    Epoch: 3 Iteration: 13
    
    SAMPLE TEXT:
    do you think we can find someone to replace tom <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পৌছাননি। যেত কি এতে <PAD> করি। <PAD> আমার <PAD> <EOS> <PAD> <PAD> <PAD> <PAD> যাবো। <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি কি মনে করো যে টমের বদলে আর কাউকে পাওযা যাবে <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=5.07371
    
    CHOSEN SAMPLE NO.: 26
    
    Epoch: 3 Iteration: 14
    
    SAMPLE TEXT:
    stop right here <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    যাও। পালটাতে খশি <PAD> পারি <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এখানেই থামন। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.78698
    
    CHOSEN SAMPLE NO.: 31
    
    Epoch: 3 Iteration: 15
    
    SAMPLE TEXT:
    is mary your daughter <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    হারিযে <EOS> খোলে <EOS> এইসমসত বলছিলো। যেতে <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    মেরি কি আপনার মেযে <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.75275
    
    CHOSEN SAMPLE NO.: 41
    
    Epoch: 3 Iteration: 16
    
    SAMPLE TEXT:
    i promised <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    চান অতিবেগনী আমাদের <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি পরতিশরতি দিলাম। <EOS> <PAD> 
    
    loss=4.17088
    
    CHOSEN SAMPLE NO.: 9
    
    Epoch: 3 Iteration: 17
    
    SAMPLE TEXT:
    meat should not be eaten raw <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    যাবো। করো। <PAD> <PAD> <PAD> <PAD> <PAD> কি <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    মাংস কাচা খাওযা উচিৎ নয। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.90786
    
    CHOSEN SAMPLE NO.: 1
    
    Epoch: 3 Iteration: 18
    
    SAMPLE TEXT:
    the ambassador returned <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    গলপ <PAD> <EOS> টম <EOS> <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    রাষটরদতটি ফিরে এলেন। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.3319
    
    CHOSEN SAMPLE NO.: 24
    
    Epoch: 3 Iteration: 19
    
    SAMPLE TEXT:
    she did not tell me her name <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি মজা <EOS> <PAD> <PAD> সঙগে <PAD> <PAD> <EOS> <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তিনি আমাকে তার নাম বলেননি। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.73823
    
    CHOSEN SAMPLE NO.: 22
    
    Epoch: 3 Iteration: 20
    
    SAMPLE TEXT:
    i 'm screaming <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ডাকবেন। হাটো। ফোন <PAD> <PAD> বলন।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি চেচাচছি। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.80593
    
    CHOSEN SAMPLE NO.: 48
    
    Epoch: 3 Iteration: 21
    
    SAMPLE TEXT:
    you deserve the prize <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> আগে না। পারছি। <PAD> <PAD> <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি এই পরষকারটির যোগয। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.61348
    
    CHOSEN SAMPLE NO.: 54
    
    Epoch: 3 Iteration: 22
    
    SAMPLE TEXT:
    your book is on the desk <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    মনে সতেরোতম <PAD> <PAD> <EOS> <PAD> <PAD> তোমার <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তোমার বই ডেসকের উপর রযেছে। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.52503
    
    CHOSEN SAMPLE NO.: 42
    
    Epoch: 3 Iteration: 23
    
    SAMPLE TEXT:
    accidents will happen <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    যান। নেই। কযাশার করন। <PAD> কি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    দরঘটনা ঘটবেই। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.91224
    
    CHOSEN SAMPLE NO.: 5
    
    Epoch: 3 Iteration: 24
    
    SAMPLE TEXT:
    why are not you coming with us <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    মগ। তোমরা এখানে <EOS> <PAD> <PAD> টম <EOS> <EOS> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমাদের সাথে আসছেন না কেন <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.70056
    
    CHOSEN SAMPLE NO.: 51
    
    Epoch: 3 Iteration: 25
    
    SAMPLE TEXT:
    i 'm never wrong <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আছেন বলেছিল সাধারণ <PAD> <EOS> <PAD> <EOS> আছে।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি কখনই ভল করি না। <EOS> <PAD> <PAD> 
    
    loss=3.61648
    
    CHOSEN SAMPLE NO.: 5
    
    Epoch: 3 Iteration: 26
    
    SAMPLE TEXT:
    there is so much i want to say to you <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    চেচাচছে। সে <PAD> <EOS> <PAD> <PAD> <PAD> একজন করি <PAD> বলতে টম <PAD> নাও।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনাকে আমার কত কিছ বলার ইচছা আছে। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.96401
    
    CHOSEN SAMPLE NO.: 43
    
    Epoch: 3 Iteration: 27
    
    SAMPLE TEXT:
    who would believe me <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি দিকে <PAD> <EOS> <PAD> <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি কাকে বিশবাস করবেন <EOS> <PAD> <PAD> 
    
    loss=4.00679
    
    CHOSEN SAMPLE NO.: 50
    
    Epoch: 3 Iteration: 28
    
    SAMPLE TEXT:
    i can only speak french and a little english <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    গরহণ গাডি হযেছে রেখে <PAD> <PAD> <PAD> <PAD> <EOS> <PAD> <PAD> করছো
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি খালি ফরাসি আর একট ইংরাজিতে কথা বলতে পারি। <EOS> <PAD> <PAD> 
    
    loss=5.09598
    
    CHOSEN SAMPLE NO.: 63
    
    Epoch: 3 Iteration: 29
    
    SAMPLE TEXT:
    tom certainly is good-looking <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    খেলতাম। ভালোভাবে করতে করছিলাম আছে। পেলো। <PAD> সময
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম সতযিই খব সদরশন। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.96998
    
    CHOSEN SAMPLE NO.: 37
    
    Epoch: 3 Iteration: 30
    
    SAMPLE TEXT:
    is everybody ready <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    একটা বাইরে শনেছেন করলো। পরায
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    সবাই তৈরি <EOS> <PAD> <PAD> 
    
    loss=4.76945
    
    CHOSEN SAMPLE NO.: 53
    
    Epoch: 3 Iteration: 31
    
    SAMPLE TEXT:
    i can not find my luggage <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমার। খালি জবাব যাক। <EOS> সঙগে <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি আমার মালপতর খজে পাচছি না। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.04935
    
    CHOSEN SAMPLE NO.: 55
    
    Epoch: 3 Iteration: 32
    
    SAMPLE TEXT:
    we are desperate to find a solution <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তমি ফরাসিতে টমের দেখছে আমার <PAD> <PAD> <PAD> <PAD> <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমরা একটি সমাধান খোজার জনয মরিযা হযে আছি। <EOS> <PAD> <PAD> 
    
    loss=4.41382
    
    CHOSEN SAMPLE NO.: 53
    
    Epoch: 3 Iteration: 33
    
    SAMPLE TEXT:
    he tries <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এটাকে পরে পারি। <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ও চেষটা করে। <EOS> <PAD> <PAD> 
    
    loss=3.69723
    
    CHOSEN SAMPLE NO.: 60
    
    Epoch: 3 Iteration: 34
    
    SAMPLE TEXT:
    what is today 's date <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> আমি টম। <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আজকের তারিখ কত <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.10313
    
    CHOSEN SAMPLE NO.: 43
    
    Epoch: 3 Iteration: 35
    
    SAMPLE TEXT:
    got it <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কি <EOS> পৌছালাম। <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    বঝেছিস <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.92349
    
    CHOSEN SAMPLE NO.: 14
    
    Epoch: 3 Iteration: 36
    
    SAMPLE TEXT:
    tom waited outside the gate <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    সকলে তার কথা জানে। <PAD> <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম গেটের বাইরে অপেকষা করলো। <EOS> <PAD> <PAD> 
    
    loss=4.48869
    
    CHOSEN SAMPLE NO.: 62
    
    Epoch: 3 Iteration: 37
    
    SAMPLE TEXT:
    tom has time <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বযগ। গেলো। অপেকষা <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টমের হাতে সময আছে। <EOS> <PAD> <PAD> 
    
    loss=3.43815
    
    CHOSEN SAMPLE NO.: 25
    
    Epoch: 3 Iteration: 38
    
    SAMPLE TEXT:
    arabic is a very important language <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এ এর <PAD> পারো <PAD> <PAD> <PAD> <PAD> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আরবী খব গরতবপরণ ভাষা। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.40109
    
    CHOSEN SAMPLE NO.: 14
    
    Epoch: 3 Iteration: 39
    
    SAMPLE TEXT:
    she was wearing a black hat <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি কোথায করেন বাজারে <EOS> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    সে একটা কালো টপি পরেছিল। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.80626
    
    CHOSEN SAMPLE NO.: 41
    
    Epoch: 3 Iteration: 40
    
    SAMPLE TEXT:
    the soldiers occupied the building <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তিনি আসবে সঙগে আমেরিকান <PAD> <EOS> বনধ <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    সৈনয বাডিটাকে দখল করলো। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.01123
    
    CHOSEN SAMPLE NO.: 21
    
    Epoch: 3 Iteration: 41
    
    SAMPLE TEXT:
    i am starting this evening <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এসেছি। আমি বসো করেছেন ছোট। <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি আজ সনধযেতে শর করব। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.92855
    
    CHOSEN SAMPLE NO.: 35
    
    Epoch: 3 Iteration: 42
    
    SAMPLE TEXT:
    i want to talk to tom alone <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ডাকতার। ভালো বনধ না। সঙগীত <PAD> <PAD> <PAD> <PAD> নয। <EOS> চলো।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি টমের সঙগে একলা কথা বলতে চাই। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.77554
    
    CHOSEN SAMPLE NO.: 25
    
    Epoch: 3 Iteration: 43
    
    SAMPLE TEXT:
    tom is wounded <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টম এটা থেকে <PAD> বাডি <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম আহত। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.28968
    
    CHOSEN SAMPLE NO.: 14
    
    Epoch: 3 Iteration: 44
    
    SAMPLE TEXT:
    i believe tom is doing well <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টমকে আমি রঙ এটা <EOS> <EOS> <EOS> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার মনে হয টম ভালোই আছে। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.0224
    
    CHOSEN SAMPLE NO.: 35
    
    Epoch: 3 Iteration: 45
    
    SAMPLE TEXT:
    i feel well <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আছি। আমি রাতরি আপনি <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার ভাল লাগছে। <EOS> <PAD> 
    
    loss=4.53445
    
    CHOSEN SAMPLE NO.: 5
    
    Epoch: 3 Iteration: 46
    
    SAMPLE TEXT:
    i 'll take those <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তে এই বাডি। <EOS> <EOS> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ওইগলো নেবো। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.59797
    
    CHOSEN SAMPLE NO.: 61
    
    Epoch: 3 Iteration: 47
    
    SAMPLE TEXT:
    i 'm your friend <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এই হযনি। শিকষক <EOS> <EOS> <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি তোদের বনধ। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.89092
    
    CHOSEN SAMPLE NO.: 49
    
    Epoch: 3 Iteration: 48
    
    SAMPLE TEXT:
    tom is 100 % correct <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    করেছে। এখনো <PAD> <EOS> <PAD> <PAD> <PAD> <EOS> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম ১০০ % ঠিক। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.14616
    
    CHOSEN SAMPLE NO.: 38
    
    Epoch: 3 Iteration: 49
    
    SAMPLE TEXT:
    would you like to come fishing with us <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <EOS> করন। আছি। অকটোবরে <PAD> <PAD> <PAD> <PAD> <PAD> আছেন। <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি কি আমাদের সাথে মাছ ধরতে যাবেন <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.38031
    
    CHOSEN SAMPLE NO.: 44
    
    Epoch: 3 Iteration: 50
    
    SAMPLE TEXT:
    you did not understand <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এখানে দোষ চাইছি। কখন <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি বঝতে পারেন নি। <EOS> <PAD> <PAD> 
    
    loss=4.05696
    
    CHOSEN SAMPLE NO.: 39
    
    Epoch: 3 Iteration: 51
    
    SAMPLE TEXT:
    i do not believe it <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বসটনের পারি এর পারি ভাবছিলাম <PAD> <PAD> <PAD> না।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি এটা বিশবাস করতে পারছি না <EOS> <PAD> <PAD> 
    
    loss=4.03759
    
    CHOSEN SAMPLE NO.: 55
    
    Epoch: 4 Iteration: 1
    
    SAMPLE TEXT:
    what station is it <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আপনার ছিলাম ছিলো। <PAD> আটকাও। নাম <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এটা কোন সটেশন <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.8935
    
    CHOSEN SAMPLE NO.: 7
    
    Epoch: 4 Iteration: 2
    
    SAMPLE TEXT:
    spanish is her native language <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    খেযেছে টাকাটা করছি। লাগবে। <PAD> <PAD> <PAD> দিতে করতেই
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    সপযানিশ তার মাতভাষা। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.97828
    
    CHOSEN SAMPLE NO.: 35
    
    Epoch: 4 Iteration: 3
    
    SAMPLE TEXT:
    you are not going to get away with this tom <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ও আমি তার <PAD> কি <PAD> <PAD> <PAD> <PAD> করে <PAD> <PAD> <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম তমি এর থেকে পার পেযে যাবে না। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.69896
    
    CHOSEN SAMPLE NO.: 13
    
    Epoch: 4 Iteration: 4
    
    SAMPLE TEXT:
    who is speaking <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টম আপনি কি পডছে। তাই <EOS> ফরাসি
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    কে কথা বলছেন <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.45669
    
    CHOSEN SAMPLE NO.: 45
    
    Epoch: 4 Iteration: 5
    
    SAMPLE TEXT:
    he got angry <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    গরতবপরণ ওটা যেতে <PAD> দেখেছো <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ও রেগে গেছিলো। <EOS> <PAD> <PAD> 
    
    loss=3.80011
    
    CHOSEN SAMPLE NO.: 1
    
    Epoch: 4 Iteration: 6
    
    SAMPLE TEXT:
    may i open the windows <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কাশতে ওনার ৭ <EOS> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি কি জানলাগলো খলতে পারি <EOS> <PAD> <PAD> 
    
    loss=4.45367
    
    CHOSEN SAMPLE NO.: 9
    
    Epoch: 4 Iteration: 7
    
    SAMPLE TEXT:
    spanish is her native language <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ভেতরে আপনার <EOS> <PAD> মাতাল <PAD> <PAD> <PAD> নাম
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    সপযানিশ তার মাতভাষা। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.02746
    
    CHOSEN SAMPLE NO.: 8
    
    Epoch: 4 Iteration: 8
    
    SAMPLE TEXT:
    tom started yelling <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি তাইতো নিচে একট <EOS> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম চিৎকার আরমভ করলো। <EOS> <PAD> 
    
    loss=3.86933
    
    CHOSEN SAMPLE NO.: 23
    
    Epoch: 4 Iteration: 9
    
    SAMPLE TEXT:
    show me <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    জযাকসনদের আপনার দেখবো। <EOS> পাব
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমাকে দেখান। <EOS> <PAD> <PAD> 
    
    loss=3.97496
    
    CHOSEN SAMPLE NO.: 37
    
    Epoch: 4 Iteration: 10
    
    SAMPLE TEXT:
    the rumor is not true <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি টম নয। বাডিতে বঝতে <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    গজবটা সতযি নয। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.19715
    
    CHOSEN SAMPLE NO.: 52
    
    Epoch: 4 Iteration: 11
    
    SAMPLE TEXT:
    i 've been hoping you would drop in <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    রাখেন। টম আতবীযদের <EOS> থাকার <PAD> <PAD> <PAD> <PAD> <PAD> <EOS> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ভাবছিলাম আপনি এসে উপসথিত হবেন। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.50563
    
    CHOSEN SAMPLE NO.: 53
    
    Epoch: 4 Iteration: 12
    
    SAMPLE TEXT:
    forget him <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পরণ নতন। <EOS> <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তাকে ছাডো। <EOS> <PAD> <PAD> 
    
    loss=4.17571
    
    CHOSEN SAMPLE NO.: 35
    
    Epoch: 4 Iteration: 13
    
    SAMPLE TEXT:
    i feel well <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    সঙগে টম অনযরকম। <PAD> করতে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার ভাল লাগছে। <EOS> <PAD> 
    
    loss=4.26027
    
    CHOSEN SAMPLE NO.: 51
    
    Epoch: 4 Iteration: 14
    
    SAMPLE TEXT:
    what time does this train reach yokohama <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কিছ কি ছিলেন <EOS> <EOS> <PAD> <PAD> <PAD> না। <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টরেনটা কটার সময ইযোকোহামা পৌছায <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.78038
    
    CHOSEN SAMPLE NO.: 49
    
    Epoch: 4 Iteration: 15
    
    SAMPLE TEXT:
    yesterday was my seventeenth birthday <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কি করলেন। <EOS> বযপারটা <EOS> <PAD> ঝামেলায <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    গতকাল আমার সতেরোতম জনমদিন ছিলো। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.89854
    
    CHOSEN SAMPLE NO.: 42
    
    Epoch: 4 Iteration: 16
    
    SAMPLE TEXT:
    you are always complaining <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পরশনের পরতযেক কষেতরে <PAD> <PAD> বলতে <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি সবসময অভিযোগ কর। <EOS> <PAD> <PAD> 
    
    loss=3.88641
    
    CHOSEN SAMPLE NO.: 11
    
    Epoch: 4 Iteration: 17
    
    SAMPLE TEXT:
    try hard <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ফোন নিযম <EOS> টমের রাখো। <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আরও চেষটা কর। <EOS> <PAD> <PAD> 
    
    loss=3.65279
    
    CHOSEN SAMPLE NO.: 25
    
    Epoch: 4 Iteration: 18
    
    SAMPLE TEXT:
    tom seemed happy to see you <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    উপর সে বললাম। <PAD> <PAD> <PAD> <PAD> <PAD> কথা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম আপনাকে দেখে খশি বলে মনে হচচে। <EOS> <PAD> 
    
    loss=4.7744
    
    CHOSEN SAMPLE NO.: 22
    
    Epoch: 4 Iteration: 19
    
    SAMPLE TEXT:
    my mother loves music <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বাডিতে ভিজে যাই। করেন। <PAD> রাখতে <PAD> পারে।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার মাযের সঙগীত ভাল লাগে। <EOS> <PAD> <PAD> 
    
    loss=3.53462
    
    CHOSEN SAMPLE NO.: 18
    
    Epoch: 4 Iteration: 20
    
    SAMPLE TEXT:
    control yourself <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি সঙগেই বলব। হোল। <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    নিজেকে সংজত করো। <EOS> <PAD> <PAD> 
    
    loss=3.5218
    
    CHOSEN SAMPLE NO.: 37
    
    Epoch: 4 Iteration: 21
    
    SAMPLE TEXT:
    i like yellow <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    খাচছিল। যাবেন <EOS> <EOS> খেতে <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার হলদ রঙ পছনদ। <EOS> <PAD> <PAD> 
    
    loss=3.50285
    
    CHOSEN SAMPLE NO.: 6
    
    Epoch: 4 Iteration: 22
    
    SAMPLE TEXT:
    i 'll call first <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমরা মডতে <EOS> <PAD> <PAD> <PAD> <PAD> না।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি আগে ফোন করবো। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.46628
    
    CHOSEN SAMPLE NO.: 37
    
    Epoch: 4 Iteration: 23
    
    SAMPLE TEXT:
    tom does not want to see you <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টম উততর ইংরাজি <PAD> <PAD> <PAD> যে <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম তোমার সঙগে দেখা করতে চায না। <EOS> <PAD> <PAD> 
    
    loss=4.8996
    
    CHOSEN SAMPLE NO.: 61
    
    Epoch: 4 Iteration: 24
    
    SAMPLE TEXT:
    i 'm your friend <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    থাকলো। বাডিতে তিনি <EOS> <EOS> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি তোদের বনধ। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.52805
    
    CHOSEN SAMPLE NO.: 26
    
    Epoch: 4 Iteration: 25
    
    SAMPLE TEXT:
    are you sure <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    নীচে তমি কাল <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি নিশচিত তো <EOS> <PAD> <PAD> 
    
    loss=3.99342
    
    CHOSEN SAMPLE NO.: 6
    
    Epoch: 4 Iteration: 26
    
    SAMPLE TEXT:
    do you know a good dentist <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আসতে আমি বলেনি <PAD> <EOS> <EOS> সেটা <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি কি কোনো ভালো দাতের ডাকতারকে চেনো <EOS> <PAD> 
    
    loss=4.52117
    
    CHOSEN SAMPLE NO.: 34
    
    Epoch: 4 Iteration: 27
    
    SAMPLE TEXT:
    are those yours <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তাহলে আমি ঘডিটার ফল <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    সেইগলো কি তোমার <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.51858
    
    CHOSEN SAMPLE NO.: 6
    
    Epoch: 4 Iteration: 28
    
    SAMPLE TEXT:
    how much did you eat <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    না। আপনারা ছাডবে সাথে সটপ টমের <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি কতটা খেযেছো <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.7123
    
    CHOSEN SAMPLE NO.: 48
    
    Epoch: 4 Iteration: 29
    
    SAMPLE TEXT:
    you deserve the prize <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বিষয সবসময মত <EOS> <EOS> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি এই পরষকারটির যোগয। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.41044
    
    CHOSEN SAMPLE NO.: 0
    
    Epoch: 4 Iteration: 30
    
    SAMPLE TEXT:
    she is asking how that is possible <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টম শীত পরামরশের শর <PAD> <PAD> <EOS> <PAD> <PAD> <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ইনি জিজঞাসা করছেন এটা কি করে সমভব। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.43086
    
    CHOSEN SAMPLE NO.: 14
    
    Epoch: 4 Iteration: 31
    
    SAMPLE TEXT:
    could you repeat that please <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ওনারা সে সাহাযয থাকা <EOS> <PAD> <PAD> সতযি <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি কি অনগরহ করে আর একবার বলতে পারবেন <EOS> 
    
    loss=3.69709
    
    CHOSEN SAMPLE NO.: 30
    
    Epoch: 4 Iteration: 32
    
    SAMPLE TEXT:
    i only speak french at home with my parents <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    থাকলো। একবার বলবেন ভাল বলতে <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি বাডিতে মা বাবার সঙগে খালি ফরাসিতে কথা বলি। <EOS> <PAD> <PAD> 
    
    loss=5.15136
    
    CHOSEN SAMPLE NO.: 5
    
    Epoch: 4 Iteration: 33
    
    SAMPLE TEXT:
    do not leave me alone <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টমের টম বারবার <EOS> <PAD> <EOS> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমাকে একলা ছেডে যাবেন না। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.83849
    
    CHOSEN SAMPLE NO.: 63
    
    Epoch: 4 Iteration: 34
    
    SAMPLE TEXT:
    i do not agree with him <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পলিশ খশি। করে ঠিক না। সাহাযয করে শর <EOS> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ওনার সাথে একমত নই। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.99498
    
    CHOSEN SAMPLE NO.: 19
    
    Epoch: 4 Iteration: 35
    
    SAMPLE TEXT:
    i 'm getting off at the next stop <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    যাওযার ওদের ভাজ ভবিষযতে <EOS> <PAD> <PAD> <PAD> <PAD> আরমভ <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি পরের সটপে নেবে যাব। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.24719
    
    CHOSEN SAMPLE NO.: 26
    
    Epoch: 4 Iteration: 36
    
    SAMPLE TEXT:
    is there a discount for children <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ও চিৎকার জানেন <PAD> জনযে <PAD> <PAD> করতে <PAD> করছি।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    বাচচাদের জনয কোনো ছাড আছে <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.08106
    
    CHOSEN SAMPLE NO.: 9
    
    Epoch: 4 Iteration: 37
    
    SAMPLE TEXT:
    tom saw you <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টম। টম পরতযেক বলতে রাখো। <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম আপনাকে দেখলো। <EOS> <PAD> <PAD> 
    
    loss=4.16876
    
    CHOSEN SAMPLE NO.: 22
    
    Epoch: 4 Iteration: 38
    
    SAMPLE TEXT:
    close the door when you leave <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পরযনত টম থাকলাম। <EOS> বনধ <PAD> <PAD> <EOS> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    বেডোবার সময দরজাটা বনধ করে দেবেন। <EOS> <PAD> <PAD> 
    
    loss=4.19795
    
    CHOSEN SAMPLE NO.: 44
    
    Epoch: 4 Iteration: 39
    
    SAMPLE TEXT:
    i 'm a student <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি নেই। পারবেন কিছ <PAD> <EOS> এটা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি একজন ছাতরী। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.90148
    
    CHOSEN SAMPLE NO.: 19
    
    Epoch: 4 Iteration: 40
    
    SAMPLE TEXT:
    i 'll arrange it <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বাইরে চিৎকার গেলো। <EOS> <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি সেটা বযবসথা করে দেবো। <EOS> <PAD> 
    
    loss=3.898
    
    CHOSEN SAMPLE NO.: 61
    
    Epoch: 4 Iteration: 41
    
    SAMPLE TEXT:
    do you understand what i mean <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তমি কি খব <EOS> হযে <PAD> <PAD> <EOS> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি যা বলতে চাইছি তমি কি তা বঝতে পারছো <EOS> <PAD> 
    
    loss=4.21859
    
    CHOSEN SAMPLE NO.: 48
    
    Epoch: 4 Iteration: 42
    
    SAMPLE TEXT:
    tom likes chocolate cake a lot <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আপনি জমা ভাষা লাগছে। করে <PAD> <EOS> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম চকলেট কেক খব পছনদ করে। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.97226
    
    CHOSEN SAMPLE NO.: 19
    
    Epoch: 4 Iteration: 43
    
    SAMPLE TEXT:
    can i change the channel <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    যান। নিযে <EOS> শনলাম। <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি কি চযানেলটা পালটাতে পারি <EOS> <PAD> <PAD> 
    
    loss=3.9458
    
    CHOSEN SAMPLE NO.: 46
    
    Epoch: 4 Iteration: 44
    
    SAMPLE TEXT:
    i am not a doctor but a teacher <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টম কি ভবিষযতে পারি <PAD> <PAD> <PAD> <PAD> <EOS> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ডকতার নই আমি শিকষক। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.04899
    
    CHOSEN SAMPLE NO.: 22
    
    Epoch: 4 Iteration: 45
    
    SAMPLE TEXT:
    that is my brother <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টম টেনিস নিযে চিনলে খান <PAD> এনো। <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ওটা আমার ভাই। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.57186
    
    CHOSEN SAMPLE NO.: 13
    
    Epoch: 4 Iteration: 46
    
    SAMPLE TEXT:
    they screamed <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    সতযিই ধরে দাদা রাখার বাচচা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তারা চিৎকার করলেন। <EOS> <PAD> 
    
    loss=4.99665
    
    CHOSEN SAMPLE NO.: 48
    
    Epoch: 4 Iteration: 47
    
    SAMPLE TEXT:
    i have an old computer that i do not want anymore <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টম কিছ হেটে আই বযাঙকে <PAD> <EOS> <PAD> <PAD> পেযেছি। <PAD> <PAD> <PAD> না। <PAD> <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার একটা পরানো কমপিউটার আছে যেটার আমার আর কোনো পরযোজন নেই। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=5.10239
    
    CHOSEN SAMPLE NO.: 45
    
    Epoch: 4 Iteration: 48
    
    SAMPLE TEXT:
    who was it <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এটার মনে নিশবাস দোষ টেবিল <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    কে ছিলো <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.93889
    
    CHOSEN SAMPLE NO.: 0
    
    Epoch: 4 Iteration: 49
    
    SAMPLE TEXT:
    i will not lie <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বাডিতেই দৌডান। ধারমিক। না। <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি মিথযা বলবো না। <EOS> <PAD> <PAD> 
    
    loss=3.84506
    
    CHOSEN SAMPLE NO.: 20
    
    Epoch: 4 Iteration: 50
    
    SAMPLE TEXT:
    i 'm inside <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমার সঙগে কাপরগলো হয <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ভেতরে আছি। <EOS> <PAD> <PAD> 
    
    loss=3.62606
    
    CHOSEN SAMPLE NO.: 53
    
    Epoch: 4 Iteration: 51
    
    SAMPLE TEXT:
    just a minute <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    নিন একসাথে <PAD> গেছি। <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এক মিনিট। <EOS> <PAD> <PAD> 
    
    loss=4.31503
    
    CHOSEN SAMPLE NO.: 13
    
    Epoch: 5 Iteration: 1
    
    SAMPLE TEXT:
    let us go <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এখানে কি টাকা <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    যাওযা যাক <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.7096
    
    CHOSEN SAMPLE NO.: 29
    
    Epoch: 5 Iteration: 2
    
    SAMPLE TEXT:
    i hear you have friends in the cia <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কষমা যা গান <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি শনলাম আপনার নাকি সি আই এ তে কিছ বনধ আছে। <EOS> 
    
    loss=4.61973
    
    CHOSEN SAMPLE NO.: 54
    
    Epoch: 5 Iteration: 3
    
    SAMPLE TEXT:
    good evening <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    খব <EOS> <EOS> <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    শভ সনধযা। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.56113
    
    CHOSEN SAMPLE NO.: 57
    
    Epoch: 5 Iteration: 4
    
    SAMPLE TEXT:
    she folded her handkerchief neatly <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি <EOS> <EOS> <EOS> <EOS> খশি <PAD> <PAD> না।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ও ওর রমালটা পরিপাটি করে ভাজ করলো। <EOS> <PAD> 
    
    loss=4.05324
    
    CHOSEN SAMPLE NO.: 16
    
    Epoch: 5 Iteration: 5
    
    SAMPLE TEXT:
    do not underestimate us <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কি আপনার পাগল দেখেছেন <PAD> <PAD> <EOS> সনধযেতে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমাদেরকে কোন অংশে কম মনে করবেন না। <EOS> 
    
    loss=3.48413
    
    CHOSEN SAMPLE NO.: 7
    
    Epoch: 5 Iteration: 6
    
    SAMPLE TEXT:
    stay there <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পাশে। আমি অরথহীন। খাও। না <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ওখানেই থাকো। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.43192
    
    CHOSEN SAMPLE NO.: 47
    
    Epoch: 5 Iteration: 7
    
    SAMPLE TEXT:
    tom frowned <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি গাডি উপর <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম ভর কোচকালো। <EOS> <PAD> 
    
    loss=3.85635
    
    CHOSEN SAMPLE NO.: 55
    
    Epoch: 5 Iteration: 8
    
    SAMPLE TEXT:
    my arm hurts <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    নই। পরায ডাকতার। <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার হাত বযাথা করছে। <EOS> <PAD> 
    
    loss=3.75832
    
    CHOSEN SAMPLE NO.: 41
    
    Epoch: 5 Iteration: 9
    
    SAMPLE TEXT:
    he can speak japanese <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কাটতে আমি ফেলেছি। <EOS> <PAD> <PAD> চেচালেন।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    সে জাপানি বলতে পারে। <EOS> <PAD> <PAD> 
    
    loss=3.70636
    
    CHOSEN SAMPLE NO.: 51
    
    Epoch: 5 Iteration: 10
    
    SAMPLE TEXT:
    i want to be more independent <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আবার ওখানে কাটতে <EOS> <PAD> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি আরও সবাধীন হতে চাই। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.7115
    
    CHOSEN SAMPLE NO.: 21
    
    Epoch: 5 Iteration: 11
    
    SAMPLE TEXT:
    do you understand me <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আছে। টম তা <EOS> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি কি আমার কথা বঝতে পারছ <EOS> <PAD> 
    
    loss=3.44831
    
    CHOSEN SAMPLE NO.: 39
    
    Epoch: 5 Iteration: 12
    
    SAMPLE TEXT:
    i know tom is tired <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বলতে টম <EOS> খেতে <EOS> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি জানি টম কলানত। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.07783
    
    CHOSEN SAMPLE NO.: 21
    
    Epoch: 5 Iteration: 13
    
    SAMPLE TEXT:
    my mother does not speak english <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তারাতারি কিছই না। এখানে <PAD> <PAD> <PAD> সঙগে <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার মা ইংরাজি বলতে পারে না। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.97175
    
    CHOSEN SAMPLE NO.: 61
    
    Epoch: 5 Iteration: 14
    
    SAMPLE TEXT:
    i found my mother busy ironing out some shirts <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    চিৎকার একটা করে আছে। <EOS> <PAD> <PAD> পেযেছিলাম। <PAD> <EOS> হোক। <PAD> <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি দেখলাম যে আমার মা কিছ জামা ইসতিরি করতে বযাসত। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.93717
    
    CHOSEN SAMPLE NO.: 9
    
    Epoch: 5 Iteration: 15
    
    SAMPLE TEXT:
    let us go <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ডাকো নিহত আমি <PAD> <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    যাওযা যাক। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.36426
    
    CHOSEN SAMPLE NO.: 49
    
    Epoch: 5 Iteration: 16
    
    SAMPLE TEXT:
    you must take this cough syrup <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আপনারা অভযসত। কারা শনছে। দেখলো। লাগে <PAD> হারিযেছিলেন। <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনাকে অতি অবশযই এই কাশির সিরাপটা খেতে হবে। <EOS> <PAD> 
    
    loss=3.96931
    
    CHOSEN SAMPLE NO.: 62
    
    Epoch: 5 Iteration: 17
    
    SAMPLE TEXT:
    try it <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এখন মাথা কেউ <PAD> <EOS> রাখতে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    চেখে দেখো। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.1625
    
    CHOSEN SAMPLE NO.: 23
    
    Epoch: 5 Iteration: 18
    
    SAMPLE TEXT:
    what does your son do <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তমি চেচানো আননদ যাও। করো। <PAD> মধযের <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনার ছেলে কি করে <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.67302
    
    CHOSEN SAMPLE NO.: 15
    
    Epoch: 5 Iteration: 19
    
    SAMPLE TEXT:
    tom always keeps his promises <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টম আমরা পারো একমাতর <PAD> <PAD> <PAD> <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম সবসময ওনার কথা রাখেন। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.76166
    
    CHOSEN SAMPLE NO.: 30
    
    Epoch: 5 Iteration: 20
    
    SAMPLE TEXT:
    i 'm sorry but it is impossible <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    করো। টম কিছ তো <EOS> <PAD> <PAD> গেছে। <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমায কষমা কোর কিনত এটা সমভব নয। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.29921
    
    CHOSEN SAMPLE NO.: 61
    
    Epoch: 5 Iteration: 21
    
    SAMPLE TEXT:
    where was your daughter <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি যা বিযে <EOS> <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনার মেযে কোথায ছিলো <EOS> <PAD> <PAD> 
    
    loss=3.73
    
    CHOSEN SAMPLE NO.: 57
    
    Epoch: 5 Iteration: 22
    
    SAMPLE TEXT:
    i 'm not jealous <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ভাবে টম এটা থাকছে। <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি ঈরষাপরাযণ নই। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.71536
    
    CHOSEN SAMPLE NO.: 19
    
    Epoch: 5 Iteration: 23
    
    SAMPLE TEXT:
    i can read <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    খাও। তাডাতাডি ওযাশিংটন <EOS> করতে <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি পডতে পারি। <EOS> <PAD> <PAD> 
    
    loss=3.71131
    
    CHOSEN SAMPLE NO.: 58
    
    Epoch: 5 Iteration: 24
    
    SAMPLE TEXT:
    i never listen to tom anyway <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    দিলো আমি সতয। কত <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি এমনিতেও কখনও টমের কথা শনি না। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.05847
    
    CHOSEN SAMPLE NO.: 56
    
    Epoch: 5 Iteration: 25
    
    SAMPLE TEXT:
    i 'm not very patient <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    জিতেছে। একজন শবদ থেকে <PAD> <EOS> <EOS> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি খব একটা ধৈরযশীল নই। <EOS> <PAD> <PAD> 
    
    loss=4.24794
    
    CHOSEN SAMPLE NO.: 55
    
    Epoch: 5 Iteration: 26
    
    SAMPLE TEXT:
    do you understand what he is saying <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি পডাশোনা <PAD> <PAD> <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ও যা বলছে তমি কি তা বঝতে পারছো <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.88211
    
    CHOSEN SAMPLE NO.: 17
    
    Epoch: 5 Iteration: 27
    
    SAMPLE TEXT:
    close that door <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমরা এখন অরথ <EOS> <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ওই দরজাটা বনধ করো। <EOS> <PAD> <PAD> 
    
    loss=3.37673
    
    CHOSEN SAMPLE NO.: 11
    
    Epoch: 5 Iteration: 28
    
    SAMPLE TEXT:
    i was astonished <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    একমত। ওর মরিযা করছেন। <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি বিসমিত হযে গেছিলাম। <EOS> <PAD> <PAD> 
    
    loss=3.50304
    
    CHOSEN SAMPLE NO.: 43
    
    Epoch: 5 Iteration: 29
    
    SAMPLE TEXT:
    where is the boarding gate for ua 111 <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ধরতে ইতালীযতে ঝগডা ভাষায <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ua 111 তে ওঠার দরজাটা কোথায <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.07333
    
    CHOSEN SAMPLE NO.: 2
    
    Epoch: 5 Iteration: 30
    
    SAMPLE TEXT:
    what does your son do <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    চিৎকার <PAD> পরশনের <PAD> <PAD> করা <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তোমার ছেলে কি করে <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.21765
    
    CHOSEN SAMPLE NO.: 55
    
    Epoch: 5 Iteration: 31
    
    SAMPLE TEXT:
    they say this old house is haunted <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমার ওখানে হবে। খেতে <PAD> কাদছে। <PAD> <PAD> <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    সবাই বলে যে এই পরানো বাডিটায ভত আছে। <EOS> <PAD> <PAD> 
    
    loss=4.22835
    
    CHOSEN SAMPLE NO.: 34
    
    Epoch: 5 Iteration: 32
    
    SAMPLE TEXT:
    would you like to come <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমার আমাদের ঠিকঠাক উপর <PAD> পছনদ <PAD> থাকবি <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি কি আসতে চাও <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.77764
    
    CHOSEN SAMPLE NO.: 45
    
    Epoch: 5 Iteration: 33
    
    SAMPLE TEXT:
    is tom here <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমরা কি আটটা পডা না
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম কি এখানে আছে <EOS> 
    
    loss=4.65077
    
    CHOSEN SAMPLE NO.: 18
    
    Epoch: 5 Iteration: 34
    
    SAMPLE TEXT:
    you fainted <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি তমি বাডি। দাও। বই
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি অজঞান হযে গেছিলেন। <EOS> 
    
    loss=4.5705
    
    CHOSEN SAMPLE NO.: 55
    
    Epoch: 5 Iteration: 35
    
    SAMPLE TEXT:
    you can eat anything you want <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কি যাক। চম গেলো। নই। <PAD> পারবে <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তোমার যেটা ইচছে হয সেটা খেতে পারো। <EOS> <PAD> 
    
    loss=4.55468
    
    CHOSEN SAMPLE NO.: 35
    
    Epoch: 5 Iteration: 36
    
    SAMPLE TEXT:
    is this your book <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আপনার ওরা বলবে। <EOS> <EOS> বলতে <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এটা কি আপনার বই <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.42308
    
    CHOSEN SAMPLE NO.: 21
    
    Epoch: 5 Iteration: 37
    
    SAMPLE TEXT:
    keep tom there <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ফরাসি আমি <EOS> <EOS> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টমকে ওখানেই রাখন। <EOS> <PAD> 
    
    loss=4.16807
    
    CHOSEN SAMPLE NO.: 22
    
    Epoch: 5 Iteration: 38
    
    SAMPLE TEXT:
    close the door when you leave <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পছনদ করবে <EOS> <PAD> <PAD> <EOS> <PAD> <EOS> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    বেডোবার সময দরজাটা বনধ করে দেবেন। <EOS> <PAD> <PAD> 
    
    loss=4.46699
    
    CHOSEN SAMPLE NO.: 43
    
    Epoch: 5 Iteration: 39
    
    SAMPLE TEXT:
    do not underestimate me <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    খেলি। শর থেকে <PAD> <EOS> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমাকে কোন অংশে কম মনে কোরো না। <EOS> 
    
    loss=3.36823
    
    CHOSEN SAMPLE NO.: 22
    
    Epoch: 5 Iteration: 40
    
    SAMPLE TEXT:
    tom was humiliated by mary <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি ওই <PAD> করে। <PAD> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম মেরি দবারা অপদসত হযেছিলো। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.76962
    
    CHOSEN SAMPLE NO.: 58
    
    Epoch: 5 Iteration: 41
    
    SAMPLE TEXT:
    i 'm a free man <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পার। যা মাতভাষা। লকষণ। <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি একজন সবাধিন মানষ। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.17471
    
    CHOSEN SAMPLE NO.: 24
    
    Epoch: 5 Iteration: 42
    
    SAMPLE TEXT:
    puzzles are fun <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টমকে পাচছি জিজঞাসা <EOS> <PAD> ফেল।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ধাধা মজার জিনিস। <EOS> <PAD> <PAD> 
    
    loss=3.70663
    
    CHOSEN SAMPLE NO.: 7
    
    Epoch: 5 Iteration: 43
    
    SAMPLE TEXT:
    can we keep it <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    উপেকষা এই ভাল <PAD> <PAD> <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমরা কি এটা রাখতে পারি <EOS> <PAD> <PAD> 
    
    loss=3.4329
    
    CHOSEN SAMPLE NO.: 50
    
    Epoch: 5 Iteration: 44
    
    SAMPLE TEXT:
    tom seldom eats at home <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এটা বনধ আমি <EOS> <EOS> <PAD> <PAD> বঝি ভাজ
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম কমই বাডিতে খায। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.83508
    
    CHOSEN SAMPLE NO.: 4
    
    Epoch: 5 Iteration: 45
    
    SAMPLE TEXT:
    what time do you usually get up <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি ডেকেছিলে জিজঞাসা পৌছেছি। <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি এমনিতে কটার সময ঘম থেকে ওঠেন <EOS> <PAD> <PAD> 
    
    loss=4.4965
    
    CHOSEN SAMPLE NO.: 54
    
    Epoch: 5 Iteration: 46
    
    SAMPLE TEXT:
    i was away <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    করা উনি জনয খাওযা ইচছে <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি বাইরে গেছিলাম। <EOS> <PAD> <PAD> 
    
    loss=3.96786
    
    CHOSEN SAMPLE NO.: 59
    
    Epoch: 5 Iteration: 47
    
    SAMPLE TEXT:
    what will you be doing at this time tomorrow <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি কেউ একা <PAD> <EOS> <PAD> <EOS> <PAD> <PAD> কি তা <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    কাল আপনি এই সময কি করবেন <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.9504
    
    CHOSEN SAMPLE NO.: 61
    
    Epoch: 5 Iteration: 48
    
    SAMPLE TEXT:
    were you born there <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আর ভেবছিলাম <EOS> অনধ। আর গেলাম। <PAD> থাকি।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি কি ওখানে জনমেছিলেন <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.6172
    
    CHOSEN SAMPLE NO.: 54
    
    Epoch: 5 Iteration: 49
    
    SAMPLE TEXT:
    how tall you are <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আসা খব কেন একা <PAD> কিছই <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি কত লমবা <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.88413
    
    CHOSEN SAMPLE NO.: 48
    
    Epoch: 5 Iteration: 50
    
    SAMPLE TEXT:
    i have an old computer that i do not want anymore <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বযবহার এতে করা সংরকষণ হবে। <PAD> <PAD> তাডাতাডি <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার একটা পরানো কমপিউটার আছে যেটার আমার আর কোনো পরযোজন নেই। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.9609
    
    CHOSEN SAMPLE NO.: 35
    
    Epoch: 5 Iteration: 51
    
    SAMPLE TEXT:
    go inside <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এখন পৌনে খায <PAD> গেছি।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ভেতরে যাও। <EOS> <PAD> <PAD> 
    
    loss=4.56059
    
    CHOSEN SAMPLE NO.: 20
    
    Epoch: 6 Iteration: 1
    
    SAMPLE TEXT:
    then what <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    গীটারটা তারা আশাবাদী। <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তাহলে <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.62812
    
    CHOSEN SAMPLE NO.: 8
    
    Epoch: 6 Iteration: 2
    
    SAMPLE TEXT:
    please hurry <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এই হাটেন। খজে <EOS> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    একট তারাতারি করন। <EOS> <PAD> 
    
    loss=3.95788
    
    CHOSEN SAMPLE NO.: 29
    
    Epoch: 6 Iteration: 3
    
    SAMPLE TEXT:
    please hurry up <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    চিডিযাখানাটার খালি <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    একট তারাতারি করন। <EOS> <PAD> 
    
    loss=4.40843
    
    CHOSEN SAMPLE NO.: 62
    
    Epoch: 6 Iteration: 4
    
    SAMPLE TEXT:
    i want this guitar <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    <PAD> এসেছিলো <EOS> <EOS> <EOS> <PAD> করেছিলাম।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার এই গীটারটা চাই। <EOS> <PAD> <PAD> 
    
    loss=4.14236
    
    CHOSEN SAMPLE NO.: 9
    
    Epoch: 6 Iteration: 5
    
    SAMPLE TEXT:
    tom saw you <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমার জমা কর। <EOS> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম আপনাকে দেখলো। <EOS> <PAD> <PAD> 
    
    loss=3.64915
    
    CHOSEN SAMPLE NO.: 22
    
    Epoch: 6 Iteration: 6
    
    SAMPLE TEXT:
    what is your favorite television program <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কে আমার <EOS> <EOS> <PAD> <EOS> <PAD> <EOS> এখনো
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তোমারে পরিয টেলিভিশন কারযকরমটা কী <EOS> <PAD> <PAD> <PAD> 
    
    loss=5.05556
    
    CHOSEN SAMPLE NO.: 52
    
    Epoch: 6 Iteration: 7
    
    SAMPLE TEXT:
    may i look at your passport <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমি আমি <EOS> <PAD> <PAD> <PAD> চম <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি কি আপনার পাসপোরটটি দেখতে পারি <EOS> <PAD> <PAD> 
    
    loss=4.54909
    
    CHOSEN SAMPLE NO.: 50
    
    Epoch: 6 Iteration: 8
    
    SAMPLE TEXT:
    tom is teaching me french <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কাল সবাই <PAD> ছিলে <PAD> <PAD> <PAD> তমি লমবা।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম আমাকে ফরাসি পডাচছে। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.65895
    
    CHOSEN SAMPLE NO.: 41
    
    Epoch: 6 Iteration: 9
    
    SAMPLE TEXT:
    go home <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    মনে চেচানো <EOS> কত মারলো।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    বাডি যা। <EOS> <PAD> <PAD> 
    
    loss=4.15118
    
    CHOSEN SAMPLE NO.: 44
    
    Epoch: 6 Iteration: 10
    
    SAMPLE TEXT:
    my mother is out <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তারাতারি আমার নাম না। <PAD> করছিল। পাবো। <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার মা বাইরে। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.80806
    
    CHOSEN SAMPLE NO.: 9
    
    Epoch: 6 Iteration: 11
    
    SAMPLE TEXT:
    tom has been working here since 2013 <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তার একদম <EOS> করবেন হলো <PAD> <PAD> <PAD> <PAD> বলতে আলাদা তোর
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম এখানে ২০১৩ থেকে কাজ করছে। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.79727
    
    CHOSEN SAMPLE NO.: 35
    
    Epoch: 6 Iteration: 12
    
    SAMPLE TEXT:
    i 'm almost sure <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    যা আপনার বাডিতেই আপনি <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি পরায নিশচিত। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.89258
    
    CHOSEN SAMPLE NO.: 14
    
    Epoch: 6 Iteration: 13
    
    SAMPLE TEXT:
    i used to play tennis <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কি ফরাসিতে <PAD> কি <PAD> <PAD> <PAD> <PAD> বলতে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি টেনিস খেলতাম। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.77934
    
    CHOSEN SAMPLE NO.: 18
    
    Epoch: 6 Iteration: 14
    
    SAMPLE TEXT:
    what time does the first train leave <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তিনি তার টম পারে। কিছই <PAD> <PAD> একদম <PAD> <PAD> পারেন।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    পরথম টরেনটা কটার সময ছাডে <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.36229
    
    CHOSEN SAMPLE NO.: 40
    
    Epoch: 6 Iteration: 15
    
    SAMPLE TEXT:
    tom is inside <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    শীত আমি হযে দেখলো। <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম ভেতরে। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.72071
    
    CHOSEN SAMPLE NO.: 52
    
    Epoch: 6 Iteration: 16
    
    SAMPLE TEXT:
    he spoke <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আসন। কি তারাতারিই ফরাসিতে <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    উনি বললেন। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.55443
    
    CHOSEN SAMPLE NO.: 49
    
    Epoch: 6 Iteration: 17
    
    SAMPLE TEXT:
    yesterday was my seventeenth birthday <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    মেরিকে টম সাহাযয <PAD> চায <PAD> <PAD> <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    গতকাল আমার সতেরোতম জনমদিন ছিলো। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.06852
    
    CHOSEN SAMPLE NO.: 60
    
    Epoch: 6 Iteration: 18
    
    SAMPLE TEXT:
    tom was with me all day <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    খালি আমি <PAD> <EOS> করা <PAD> <PAD> <PAD> আমাকে
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম আমার সঙগে সারা দিন ছিল। <EOS> <PAD> <PAD> 
    
    loss=4.15833
    
    CHOSEN SAMPLE NO.: 27
    
    Epoch: 6 Iteration: 19
    
    SAMPLE TEXT:
    what time do you get up <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমার আপনি বযাঙকে রাতের <EOS> <PAD> জিতলো। <PAD> <PAD> হযেছিলো।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনি কটার সময ঘম থেকে ওঠেন <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.03051
    
    CHOSEN SAMPLE NO.: 39
    
    Epoch: 6 Iteration: 20
    
    SAMPLE TEXT:
    they are doctors <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ককরকে হযেগেছিলাম। করে সতরী। <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ওনারা ডাকতার। <EOS> <PAD> <PAD> 
    
    loss=4.19367
    
    CHOSEN SAMPLE NO.: 37
    
    Epoch: 6 Iteration: 21
    
    SAMPLE TEXT:
    birds fly <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বস। ওখানে নাডলো। <EOS> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    পাখি ওডে। <EOS> <PAD> <PAD> 
    
    loss=4.1448
    
    CHOSEN SAMPLE NO.: 42
    
    Epoch: 6 Iteration: 22
    
    SAMPLE TEXT:
    it is me <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এখন জোরে করি। খব <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.83123
    
    CHOSEN SAMPLE NO.: 28
    
    Epoch: 6 Iteration: 23
    
    SAMPLE TEXT:
    i want a guitar <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    পলেনটাকে তোমাকে ঠিক <PAD> <PAD> <PAD> বযগ।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার একটা গীটার চাই। <EOS> <PAD> <PAD> 
    
    loss=3.86247
    
    CHOSEN SAMPLE NO.: 28
    
    Epoch: 6 Iteration: 24
    
    SAMPLE TEXT:
    what time does the dining room open <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    করলো। ওখানেই বেশি <EOS> <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    খাবার ঘরটা কখন খোলে <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.08265
    
    CHOSEN SAMPLE NO.: 30
    
    Epoch: 6 Iteration: 25
    
    SAMPLE TEXT:
    tom stopped screaming <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    দৌডান। টম <EOS> <PAD> <EOS> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম চিৎকার করা বনধ করলো। <EOS> 
    
    loss=3.61127
    
    CHOSEN SAMPLE NO.: 1
    
    Epoch: 6 Iteration: 26
    
    SAMPLE TEXT:
    mistakes like these are easily overlooked <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ধরে আহত। করে <PAD> <EOS> <PAD> <PAD> <PAD> <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এই ধরনের ভলগলো খব সহজেই উপেকষা করা হয। <EOS> <PAD> 
    
    loss=4.12171
    
    CHOSEN SAMPLE NO.: 27
    
    Epoch: 6 Iteration: 27
    
    SAMPLE TEXT:
    i forgot your phone number <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    এটার টম ছাডা <EOS> <PAD> <EOS> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি তোমার ফোন নামবারটা ভলে গেছি। <EOS> <PAD> 
    
    loss=4.31328
    
    CHOSEN SAMPLE NO.: 30
    
    Epoch: 6 Iteration: 28
    
    SAMPLE TEXT:
    everything is over <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    মেযে তিনজন সারা সনদর। নিতে <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    সব শেষ। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.26358
    
    CHOSEN SAMPLE NO.: 17
    
    Epoch: 6 Iteration: 29
    
    SAMPLE TEXT:
    where are your things <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তোমাকে আবার আই <EOS> তা <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তোমাদের জিনিসপতর কোথায <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.31097
    
    CHOSEN SAMPLE NO.: 20
    
    Epoch: 6 Iteration: 30
    
    SAMPLE TEXT:
    i 'm bad at sports <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    তার টম ছাডো। ভালো <EOS> <EOS> <PAD> তা
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি খেলাধলায খারাপ। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.1637
    
    CHOSEN SAMPLE NO.: 27
    
    Epoch: 6 Iteration: 31
    
    SAMPLE TEXT:
    how old is this tree <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    লোকেরা আমার আপনি বলেননি। <PAD> <PAD> করতে <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এই গাছটা বযস কত <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.77163
    
    CHOSEN SAMPLE NO.: 44
    
    Epoch: 6 Iteration: 32
    
    SAMPLE TEXT:
    do you understand what i want to say <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    চারচটার কতো চেচাচছে । <PAD> <PAD> <PAD> <PAD> <PAD> <EOS> হযেছিলো। <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি কী বলতে চাইছি আপনি বঝতে পারছেন <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.09386
    
    CHOSEN SAMPLE NO.: 13
    
    Epoch: 6 Iteration: 33
    
    SAMPLE TEXT:
    tom is sick <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    মা আপনার চাই। আছেন না। <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম অসসথ। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.87569
    
    CHOSEN SAMPLE NO.: 43
    
    Epoch: 6 Iteration: 34
    
    SAMPLE TEXT:
    how do you write your last name <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    নে চেষটা বযস ভাবে <PAD> <PAD> <PAD> <PAD> করতে <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনার পদবিটি কিভাবে লেখেন <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.3604
    
    CHOSEN SAMPLE NO.: 27
    
    Epoch: 6 Iteration: 35
    
    SAMPLE TEXT:
    valentine 's day is celebrated all around the world <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কি শিরনামটা কতো <EOS> থাকেন। <PAD> <PAD> <PAD> সসতা <PAD> <EOS> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ভযালেনটাইন ডে সারা পথিবী জডে পালন করা হয। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.74226
    
    CHOSEN SAMPLE NO.: 63
    
    Epoch: 6 Iteration: 36
    
    SAMPLE TEXT:
    i know all my neighbors <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আসবো টম যাচছি। আলোটা করেছি। <PAD> <EOS> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি আমার সব পরতিবেশীকে চিনি। <EOS> <PAD> <PAD> 
    
    loss=3.90807
    
    CHOSEN SAMPLE NO.: 11
    
    Epoch: 6 Iteration: 37
    
    SAMPLE TEXT:
    is this your car <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টম আমার সাডে <EOS> ছিলাম। <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এটা কি আপনার গাডি <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.49463
    
    CHOSEN SAMPLE NO.: 61
    
    Epoch: 6 Iteration: 38
    
    SAMPLE TEXT:
    no one laughed <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    অনগরহ একটা <EOS> ওকে বযগ। সে <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    কেউ হাসলো না। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.64158
    
    CHOSEN SAMPLE NO.: 41
    
    Epoch: 6 Iteration: 39
    
    SAMPLE TEXT:
    i 've never told anyone about this <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    জিনিসপতর দিকে জডে <EOS> <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি কোনদিন কাউকে এই বযপারে বলিনি। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.23629
    
    CHOSEN SAMPLE NO.: 47
    
    Epoch: 6 Iteration: 40
    
    SAMPLE TEXT:
    i do not blame you <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমার আমি খেযে গেছি। রাসতা <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি আপনাকে দোষ দিচছি না। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.72207
    
    CHOSEN SAMPLE NO.: 60
    
    Epoch: 6 Iteration: 41
    
    SAMPLE TEXT:
    do you want to eat noodles or rice <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কি এখানে কলানত অষধ <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তমি কি নডলস খেতে চাও না ভাত খেতে চাও <EOS> <PAD> <PAD> 
    
    loss=4.2803
    
    CHOSEN SAMPLE NO.: 23
    
    Epoch: 6 Iteration: 42
    
    SAMPLE TEXT:
    he can read <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    নাক বাডিতে চেযেছিলাম। <EOS> চাই <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    সে পডতে পারে। <EOS> <PAD> <PAD> 
    
    loss=3.65548
    
    CHOSEN SAMPLE NO.: 13
    
    Epoch: 6 Iteration: 43
    
    SAMPLE TEXT:
    tom can not read all these books in one day <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আমাদের শনি জন করছে। <PAD> <PAD> <PAD> <PAD> <PAD> <EOS> <PAD> <PAD> <PAD> রাখে।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম একদিনে এতগলো বই পডতে পারবে না। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=5.02744
    
    CHOSEN SAMPLE NO.: 36
    
    Epoch: 6 Iteration: 44
    
    SAMPLE TEXT:
    i 've seen it <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    টমের বনধ দিতে করতো। <EOS> একলা না। <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি এটা দেখেছি। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.54768
    
    CHOSEN SAMPLE NO.: 11
    
    Epoch: 6 Iteration: 45
    
    SAMPLE TEXT:
    there are no problems <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আগে এই তার অভিজঞ <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    কোনো অসবিধা নেই। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.9008
    
    CHOSEN SAMPLE NO.: 5
    
    Epoch: 6 Iteration: 46
    
    SAMPLE TEXT:
    are those yours <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    হল খব তিনটে সোমবার কল বলতে শনন।
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    সেইগলো কি আপনার <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.36358
    
    CHOSEN SAMPLE NO.: 18
    
    Epoch: 6 Iteration: 47
    
    SAMPLE TEXT:
    do you speak english <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    ওর শবদটার <EOS> <EOS> কেন <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আপনারা ইংরাজি বলতে পারেন <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.57734
    
    CHOSEN SAMPLE NO.: 62
    
    Epoch: 6 Iteration: 48
    
    SAMPLE TEXT:
    my father is in <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কি তিনি উপনযাস <EOS> <PAD> <PAD> <EOS> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমার বাবা ভেতরে। <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.39322
    
    CHOSEN SAMPLE NO.: 18
    
    Epoch: 6 Iteration: 49
    
    SAMPLE TEXT:
    do you have time <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    যে আমার পরিষকার <EOS> <PAD> <PAD> <EOS> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    তোমার হাতে সময আছে <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.44904
    
    CHOSEN SAMPLE NO.: 45
    
    Epoch: 6 Iteration: 50
    
    SAMPLE TEXT:
    tom is one of the most generous people i ever met <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আবার সারা ভালো <EOS> <PAD> <PAD> <PAD> <EOS> করব। <PAD> <PAD> <PAD> কথা <PAD> <PAD> <PAD> কি <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম আমার দেখা সবচেযে উদার মানযদের মধযে একজন। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=5.26957
    
    CHOSEN SAMPLE NO.: 40
    
    Epoch: 6 Iteration: 51
    
    SAMPLE TEXT:
    tom hardly ever keeps his word <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    হাই সমসযা বাডি <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    টম খব কমই তার কথা রাখে। <EOS> <PAD> <PAD> <PAD> 
    
    loss=4.09359
    
    CHOSEN SAMPLE NO.: 14
    
    Epoch: 7 Iteration: 1
    
    SAMPLE TEXT:
    why did this occur <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    নাকি ওনাদের গেছিলাম। <EOS> তাই <PAD> বলি। <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    এটা কেনো ঘটলো <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=3.38648
    
    CHOSEN SAMPLE NO.: 8
    
    Epoch: 7 Iteration: 2
    
    SAMPLE TEXT:
    they have two daughters <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    বিশবাস কী বেহালা দ <PAD> <PAD> <PAD> <EOS>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    ওনাদের দটো মেযে আছে। <EOS> <PAD> <PAD> <PAD> 
    
    loss=3.39317
    
    CHOSEN SAMPLE NO.: 17
    
    Epoch: 7 Iteration: 3
    
    SAMPLE TEXT:
    where can i buy a ticket <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    কে দাডাও। যাস <EOS> <PAD> <PAD> <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি টিকিট কোথা থেকে কিনতে পারব <EOS> <PAD> <PAD> 
    
    loss=4.11485
    
    CHOSEN SAMPLE NO.: 22
    
    Epoch: 7 Iteration: 4
    
    SAMPLE TEXT:
    i 'm looking for a room for rent <EOS> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    একলা আপনার হযেছে। করছেন নামাইনি। <PAD> লাগে <PAD> পালন <PAD> <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি একটা ঘর ভারা খজছি। <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=4.02293
    
    CHOSEN SAMPLE NO.: 61
    
    Epoch: 7 Iteration: 5
    
    SAMPLE TEXT:
    do you understand what i mean <EOS> <PAD> 
    
    
    PREDICTED TRANSLATION OF THE SAMPLE:
    
    আপনার আমি <PAD> কিছই আলর ঘমাতে <PAD> ঘমাতে করন। <PAD> <PAD>
    
    ACTUAL TRANSLATION OF THE SAMPLE:
    
    আমি যা বলতে চাইছি তমি কি তা বঝতে পারছো <EOS> <PAD> 
    
    loss=3.84249
    
    CHOSEN SAMPLE NO.: 8
    
    Epoch: 7 Iteration: 6
    
    SAMPLE TEXT:
    i changed my name to tom jackson <EOS> 
    
