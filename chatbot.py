import nltk
import tensorflow as tf
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from tensorflow.keras.layers import LSTM
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Embedding
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json', encoding='utf-8').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        w = nltk.word_tokenize(pattern)
        words.extend(w)

        documents.append((w, intent['tag']))


        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:

    bag = []

    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower(),pos="v") for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)


    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")





train_x=np.array(train_x)
train_y=np.array(train_y)
training=np.array(training)


print('Training')
print(training)
print(type(training))
print('Train X')
print(train_x)
print('Train Y\n')
print(train_y)
print('The shape of training is'+str(training.shape))
print('The shape of train_x is'+str(train_x.shape))
print('The shape of train_y is'+str(train_y.shape))


#look back is the number of times you want to feed the
#dataset to the model
# look_back=10
# num_features=132


# nb_samples = train_x.shape[0] - look_back

# train_x_reshaped = np.zeros((nb_samples, look_back, num_features))
# train_y_reshaped = np.zeros((nb_samples))

# for i in range(nb_samples):
#     y_position = i + look_back
#     train_x_reshaped[i] = train_x[i:y_position]
#     train_y_reshaped[i] =train_y[y_position]


# train_x=train_x.reshape(85,1,132)
# #train_y.reshape(85)



# print(train_x.shape)


#This is the backup of the code from lines 80-85 in chatbot.py
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
# model = Sequential()
# model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation='softmax'))
# embed_dim=128
# lstm_out=200
# model = Sequential()
# model.add(Embedding(2500, embed_dim,input_length = len(train_x)))
# model.add(LSTM(lstm_out))
# model.add(Dense(2,activation='softmax'))

# # these are the commented ones
# model = Sequential()
# model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation='softmax'))


#these are the lines for the LSTM implementation
# model=Sequential()
# model.add(LSTM(50,return_sequences=True,input_shape=(1,132)))
# model.add(LSTM(50,return_sequences=True))
# model.add(LSTM((50)))
# model.add(Dense(1))
# model.compile(loss="mean_squared_error",optimizer="adam")

# Compiling the model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=555, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")