#importing library
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

words=[]
classes = []
documents = []
ignore_words = ['?', '!',',','.']

#load the json file
data_file = open('intents.json', encoding='utf-8').read()
intents = json.loads(data_file)

#get all the words, tags and responses+tag .
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

#remove the duplicate words and tags
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# initializing training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# shuffle the data and convert it to an array
random.shuffle(training)
training = np.array(training)

# split the features and target labels
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu')) # input = 1st Dense layer's output(128)
model.add(Dropout(0.4))
model.add(Dense(len(train_y[0]), activation='softmax')) # input = 2nd Dense layer's output(64)

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
Hist = model.fit(train_x,train_y, epochs=200, batch_size=5, verbose=1)
model.save('ChatBot_model.h5', Hist)
