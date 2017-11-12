
# coding: utf-8

# ### Loading Pre-processed Data

# In[1]:


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
    


# ### Function for converting vector of size word_vec_dim into the closest reprentative english word. 

# In[2]:


def most_similar_eucli_eng(x):
    xminusy = np.subtract(np_embedding_eng,x)
    sq_xminusy = np.square(xminusy)
    sum_sq_xminusy = np.sum(sq_xminusy,1)
    eucli_dists = np.sqrt(sum_sq_xminusy)
    return np.argsort(eucli_dists)
    
def vec2word_eng(vec):   # converts a given vector representation into the represented word 
    most_similars = most_similar_eucli_eng(np.asarray(vec,np.float32))
    return vocab_eng[most_similars[0]]
    


# ### Function for generating a sequence of positional codes for positional encoding.

# In[3]:


def positional_encoding(seq_len,model_dimensions):
    pe = np.zeros((seq_len,model_dimensions,),np.float32)
    for pos in xrange(0,seq_len):
        for i in xrange(0,model_dimensions):
            pe[pos][i] = math.sin(pos/(10000**(2*i/model_dimensions)))
    return pe.reshape((seq_len,model_dimensions))


# ### Hyperparametes.

# In[4]:


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


# ### Function for Layer Normalization 

# In[5]:


#modified version of def LN used here: 
#https://theneuralperspective.com/2016/10/27/gradient-topics/

def layer_norm(inputs,scale,shift,epsilon = 1e-5):

    mean, var = tf.nn.moments(inputs, [1,2], keep_dims=True)

    LN = scale * (inputs - mean) / tf.sqrt(var + epsilon) + shift
 
    return LN


# ### Pre-generating all possible masks
# 
# These masks are to be used to fill illegal positions with infinity (or a very high value eg. 2^30).
# 
# Illegal positions are positions in the decoder input sequence which may be filled in the future timestep,
# but currently not filled. These positions shouldn't be allowed to influence the network.
# 
# The masks are used to assign the value 2^30 to all positions in the tensor influenced by the illegal ones.
# After going through the softmax layer, these positions become close to 0, as it should be.
# 
# Dynamically creating masks depending on the current position\timestep (depending on which the program can know which positions are legal and which aren't) is, however,
# a bit troublesome with tensorflow tf_while_loop. 
# 
# I am pre-generating all possible masks here, and pack them into a tensor such that the network can dynamically
# access the required mask using the index of the tensor (the index will same be the timestep) 
#                                                                     
#                                                                    

# In[6]:


masks=np.zeros((max_len-1,max_len,max_len),dtype=np.float32)

for i in xrange(1,max_len):
    mask = np.zeros((max_len,max_len),dtype=np.float32)
    mask[i:max_len,:] = -2**30
    mask[:,i:max_len] = -2**30
    masks[i-1] = mask
    
masks = tf.convert_to_tensor(masks) 


# ### Function for Multi-Headed Attention.
# 
# Details: https://arxiv.org/pdf/1706.03762.pdf
# 
# Q = Query
# 
# K = Key
# 
# V = Value
# 
# d is the dimension for Q, K and V. 

# In[7]:



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
    


# ### Function for encoder
# 
# More details: https://arxiv.org/pdf/1706.03762.pdf

# In[8]:


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


# ### Function for decoder
# 
# More details: https://arxiv.org/pdf/1706.03762.pdf

# In[9]:


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


# ### Model Definition
# 
# It follows the encoder-decoder paradigm. The main exception from standard encoder-decoder paradigm, is that it uses 'transformers' instead of Reccurrent networks. 
# 
# If teacher forcing is True, the decoder is made to guess the next output from the previous words in the actual target output, else the decoder predicts the next output from the previously predicted output of the decoder.
# 
# Details about the model: https://arxiv.org/pdf/1706.03762.pdf

# In[10]:


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


# ### Setting up cost function and optimizer

# In[11]:


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


# ### Training .....
# 
# The input batch is positionally encoded before its fed to the network.

# In[ ]:


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
    

