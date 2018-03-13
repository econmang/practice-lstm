# Set of imports for the model
import numpy as np
import matplotlib
import tensorflow as tf
# to create embeddings
from gensim.scripts.glove2word2vec import glove2word2vec
# import to check file system
from pathlib import Path
from gensim.models import KeyedVectors

filename = 'glove.6b.50d.txt'
def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename,'r', encoding='utf-8')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab,embd

print("Loading Embedding Matrix: GloVe")
vocab,embd = loadGloVe(filename)
vocab_size = len(vocab)
embedding_dim = len(embd[0])
embedding = np.asarray(embd)

# Details of the embedding
print("Length of vocabulary: %s" % vocab_size)
print("Dimensions associated with each embedding: %s" % embedding_dim)
print("Shape of embedding matrix:")
print(embedding.shape)

#W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
#                trainable=False, name="W")
#embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
#embedding_init = W.assign(embedding_placeholder)
#sess = tf.Session()
#sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

# Developing the Model

#Hyperparameters
maxSentence = 12 #Maximum sentence length
dimensions = 100 #dimensions related with each word vector
numClasses = 2
batchSize = 32
numUnits = 128
iterations = 100000
