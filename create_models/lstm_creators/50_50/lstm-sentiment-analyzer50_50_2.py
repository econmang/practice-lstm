# LSTM

# Imports for keras and sklearn
# keras: API Built on top of Tensorflow
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.datasets import imdb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

max_words = 20000
max_sequence_length = 250
batch_size = 32

print("Loading IMDB Sentiment Analysis data...\n\n")

# Splitting initial dataset in half into training and testing set
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.5, random_state=1)
print("Data loaded...")
# Printing out length of each data sample
print(len(x_train), 'training sequences')
print(len(x_test), 'testing sequences\n')

# Padding sequences to keep input of constant size
print('Padding sequences so they are of the same length...\n\n')
x_train = sequence.pad_sequences(x_train, maxlen=max_sequence_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_sequence_length)

# Converting Output data to categorical for use with softmax/categorical cross entropy
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

print("Data shape:")
print('Training input shape:', x_train.shape)
print('Training output shape:', y_train.shape,"\n")

print('Testing input shape:', x_test.shape)
print('Testing output shape:', y_test.shape,"\n")

print('Developing the model...\n\n')
""" Developing a sequential model
    with an embedding matrix feeding into
    128 LSTM units (dropout of neurons is
    set to 0.2 as well as dropout for
    connections to recurrent layers).

    Final layer is softmax output
    layer to determine sentiment"""
model = Sequential()
model.add(Embedding(max_words, 128, input_length=max_sequence_length))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Model constructed..\n")
print("Training...\n")
# Model will work to fit training data in batch sizes of 32
# 2 Epochs (Training iterations) will be performed
# Validation sets will be used to test validity after each epoch, ending training
# if accuracy is within a small enough value
model.fit(x_train, y_train, batch_size = batch_size, epochs=2)
print("Model finished training...\n\n")

print("Testing model...\n")
metric, accuracy = model.evaluate(x_test,y_test, batch_size=batch_size)
y_test_pred = model.predict(x_test)
print('Test loss:',metric)
print('Test accuracy:',accuracy)

y_test_pred = model.predict(x_test)
cmtest = confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))
tp, fp = cmtest[0]
fn, tn = cmtest[1]

accuracy = (tp + tn)/(tp+tn+fn+fp)
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fn)
miss_rate = 1 - sensitivity
fall_out = 1 - specificity

print("\n")
print("Accuracy",accuracy)
print("Sensitivity:",sensitivity)
print("Specificity:",specificity)
print("Miss Rate:", miss_rate)
print("Fall-out:",fall_out)

print('\n\n')
print("Development of model complete.")
print("Saving model...")
model.save("../../../models/lstm_5050epoch2model.h5")
