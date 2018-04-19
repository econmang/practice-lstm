# CNN

from __future__ import print_function

# Imports for keras and sklearn
# keras: API Build on top of Tensorflow
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Concatenate, Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.datasets import imdb
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


max_words = 8000
max_sequence_length = 500
batch_size = 64

print("Loading IMDB Sentiment Analysis data...\n\n")

# Splitting initial dataset into test and train
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_words)
# Splitting out 50% of test data for validation
x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size = 0.5, random_state = 1)
print("Data loaded...")
#Printing length of each set 
print(len(x_train),"training sequences")
print(len(x_valid),"validation sequences")
print(len(x_test),"testing sequences\n")

#Padding sequences to make sure all inputs are of constant size
print("Padding sequences so they are of the same length...\n\n")
x_train = sequence.pad_sequences(x_train, maxlen = max_sequence_length)
x_valid = sequence.pad_sequences(x_valid, maxlen = max_sequence_length)
x_test =  sequence.pad_sequences(x_test, maxlen = max_sequence_length)

#Converting output data to categorical for use with softmax/categorical cross entropy
y_train = to_categorical(y_train, num_classes = 2)
y_valid = to_categorical(y_valid, num_classes = 2)
y_test =  to_categorical(y_test, num_classes = 2)

print("Data shape:")
print("Training input shape:", x_train.shape)
print("Trainint output shape:", y_train.shape,"\n")

print("Validation input shape:", x_valid.shape)
print("Validation output shape:",y_valid.shape,"\n")

print("Testing input shape:", x_test.shape)
print("Testing output shape:", y_test.shape,"\n")

print("Developing the model...\n\n")

#development of sequential model
#model = Sequential()

#input layer
input_layer = Input(shape=(500,))

#embedding layer
lr = (Embedding(max_words, 500))(input_layer)
lr = Dropout(0.2)(lr)

#Creating convolutional layer to perfrom concatenation on
conv_layer = []
for i in (3,8):
    conv = Convolution1D(filters = 10, kernel_size = i, padding = "valid", activation = "relu",strides = 1)(lr)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_layer.append(conv)

lr = Concatenate()(conv_layer) if len(conv_layer) > 1 else conv_layer[0]

#hidden dims and output layer
lr = Dropout(0.2)(lr)
lr = Dense(50,activation="relu")(lr)
output = (Dense(2,activation='softmax'))(lr)

#compiling cnn model
model = Model(input_layer, output)
model.compile(loss='binary_crossentropy',optimizer='adam', metric=['accuracy'])

print("Model constructed..\n")
print("Training...\n")

model.fit(x_train, y_train, batch_size = batch_size, epochs = 4, validation_data = (x_valid, y_valid))
#print("Model finished training...\n\n")

#print("Testing model...\n")
#metric, accuracy = model.evaluate(x_test, y_test, batch_size = batch_size)
#print('Test loss:', metric)
#print('Test accuracy:', accuracy)

#print('\n\n')
#print("Development of model complete.")
#print("Savid model...")
#model.save("../models/cnn_model.h5")
