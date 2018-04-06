#Defining inputs for imdb lstm model
from keras.models import load_model
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np

#Defining max sequence length to pad input
#Same as was used when training
max_seq_len = 500

#initial greeting to user
print("\n\n\nHello, welcome to the movie review sentiment analyzer.\nWe will be taking some data from you and then analyzing the\nsentiment you hold towards the film you are seeking to review.\n\n\n")

#Gathering input from user
sent_input = input("Enter how you felt about a movie:\n")


#Formatting input to be used by the model
print("\n\nCleaning and formatting text input for word lookup.\nAdding IMDB word index to convert text input to numpy array.")
#Cleaning text input for lookup
sent_input = sent_input.lower()
#Adding imdb word index
word_index = imdb.get_word_index()

formatted_input = [[word_index[w] for w in sent_input if w in word_index]]
formatted_input = sequence.pad_sequences(formatted_input, maxlen=max_seq_len)
vector_input = np.array([formatted_input.flatten()])

print("Converted input to word indices for LSTM model...")
#Loading LSTM model to analyze sentiment of review
print("\nThank you for your input.\nLoading LSTM classification model...")
model = load_model("../models/lstm_model.h5")

#Using model to classify sentiment of input
print("\n\nModel Loaded. Analyzing input...\n")
classification = model.predict(vector_input)

print("Classification of your sentiment:")

print("Unformatted network output:",classification)
if classification[0][0] > classification[0][1]:
    print("Actual classification label:\n\nNegative Sentiment!")
else:
    print("Actual classification label:\n\nPositive Sentiment!")

print("\n\nThanks for using Evan's LSTM Movie Review Sentiment Analyzer\n")
