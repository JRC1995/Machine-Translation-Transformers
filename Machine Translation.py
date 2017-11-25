
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
    


# ### Function for converting vector of size word_vec_dim into the closest representative english word. 

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
    


# ### Hyperparameters and Placeholders.

# In[3]:


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


# ### Model Parameters.

# In[4]:



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
 
#Layer Normalization parameters for encoder and decoder   

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


# ### Function for generating a sequence of positional codes for positional encoding.

# In[5]:


def positional_encoding(seq_len,model_dimensions):
    pe = np.zeros((seq_len,model_dimensions,),np.float32)
    for pos in xrange(0,seq_len):
        for i in xrange(0,model_dimensions):
            pe[pos][i] = math.sin(pos/(10000**(2*i/model_dimensions)))
    return pe.reshape((seq_len,model_dimensions))


# ### Function for Layer Normalization 
# 
# https://arxiv.org/abs/1607.06450

# In[6]:


#modified version of def LN used here: 
#https://theneuralperspective.com/2016/10/27/gradient-topics/

def layer_norm(inputs,scale,shift,epsilon = 1e-5):

    mean, var = tf.nn.moments(inputs, [1,2], keep_dims=True)

    LN = tf.multiply((scale / tf.sqrt(var + epsilon)),(inputs - mean)) + shift
 
    return LN


# ### Function to pre-generate masks for illegal positions. 
# 
# These masks are to be used to fill illegal positions with -infinity (or a very low value eg. -2^30).
# 
# Illegal positions are positions of the decoder input tokens that aren't predicted at a given timestep.
# 
# { In a transformer, the decoder input is of the same shape as the WHOLE decoder output sequence. One word for the sequence is predicted at each timestep (from left to right). So in most timesteps, the left side of the decoder input sequence will contain valid previously predicted output words, but the right side -the yet to be predicted side should contain some values that should be ignored and never attended. We make sure that they're ignored by masking it }
# 
# So, the illegal positions depends on the total output length and the no. of predicted output tokens.
# 
# The appropriate mask when i output tokens are predicted can be retrieved from mask[i-1] where mask is the return value from this function. The argument out_len that function takes, signifies the total length of the output. 
# 
# The masks are used to assign the value -2^30 to all positions in the tensor influenced by the illegal ones.
# After going through the softmax layer, these positions become close to 0, as it should be.
# 
# Dynamically creating masks depending on the current position\timestep (depending on which the program can know which positions are legal and which aren't) is, however,
# a bit troublesome with tensorflow tf_while_loop. 
# 
# I will be pre-generating all the masks with Python native code and feed the list of all required masks to the network at each training step (output length can be different at different training steps). 
#                                                                  

# In[7]:


def generate_masks_for_illegal_positions(out_len):
    
    masks=np.zeros((out_len-1,out_len,out_len),dtype=np.float32)
    
    for i in xrange(1,out_len):
        mask = np.zeros((out_len,out_len),dtype=np.float32)
        mask[i:out_len,:] = -2**30
        mask[:,i:out_len] = -2**30
        masks[i-1] = mask
        
    return masks


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

# In[8]:



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
    


# ### Function for encoder
# 
# More details: https://arxiv.org/pdf/1706.03762.pdf

# In[9]:


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


# ### Function for decoder
# 
# More details: https://arxiv.org/pdf/1706.03762.pdf

# In[10]:


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


# ### Function for Stacking Encoders.

# In[11]:


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
    


# ### Function for Stacking Decoders.

# In[12]:


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
    


# ### predicted_embedding():
# 
# Given a probability distribution and an embedding matrix, this function returns the embedding of the word with the maximum probability in the given distribution.
# 
# ### replaceSOS():
# 
# SOS signifies the start of sentence for the decoder. Also often represented as 'GO'. I am using an all ones vector as the first decoder input token. 
# In the next time step, the SOS will be forgotten, and only the context of the previously predicted output (or the target output at the previous timestep, if teacher forcing is on) will be used.
# 
# ### add_pred_to_output_lists():
# 
# This function will concatenate the last predicted output into a tensor of concatenated sequence of output tokens. 

# In[13]:


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


# ### Model Definition
# 
# It follows the encoder-decoder paradigm. The main exception from standard encoder-decoder paradigm, is that it uses 'transformers' instead of Reccurrent networks. The decoder undergoes a sequential processing, though. 
# 
# If teacher forcing is True, the decoder is made to guess the next output from the previous words in the actual target output, else the decoder predicts the next output from the previously predicted output of the decoder.
# 
# Details about the model: https://arxiv.org/pdf/1706.03762.pdf

# In[14]:


def model(x,y,teacher_forcing=True):
    
        
    # NOTE: tf.shape(x)[0] == batch_size
    
    encoderin = x # (should be already positionally encoded) 
    encoderin = tf.nn.dropout(encoderin,keep_prob)

    
    # ENCODER LAYERS

    encoderout = stacked_encoders(N,encoderin)
    

    # DECODER LAYERS

    decoderin_part_1 = tf.ones([tf.shape(x)[0],1,word_vec_dim],dtype=tf.float32) #represents SOS
    
    filled = tf.constant(1) 
    # no. of output words that are filled
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


# ### Setting up cost function and optimizer

# In[15]:


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


# ### Function to create a Mask for pads' effect on cost. 
# 
# The mask will have the same shape as the batch of labels but with the value 0 wherever there is a PAD.
# The mask will be element-wise multipled to the cost (before its averaged), so that any position in the cost tensor that is effected by the PAD will be multiplied by 0. This way, the effect of PADs (which we don't need to care about) on the cost (and therefore on the gradients) can be nullified. 

# In[16]:


def create_pad_Mask(output_batch):
    pad_index = vocab_beng.index('<PAD>')
    mask = np.ones_like((output_batch),np.float32)
    for i in xrange(len(mask)):
        for j in xrange(len(mask[i])):
            if output_batch[i,j]==pad_index:
                mask[i,j]=0
    return mask


# ### Training .....
# 
# The input batch is positionally encoded before its fed to the network.

# In[17]:


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
    


# In[18]:


def word2vec(word):
    if word in vocab_eng:
        return np_embedding_eng[vocab_eng.index(word)]
    else:
        return np_embedding_eng[vocab_eng.index('<PAD>')]


# ### Prediction.

# In[21]:


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
    

