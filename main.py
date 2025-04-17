import json


file_path = "intents.json"  
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)
print(json.dumps(data, indent=4, ensure_ascii=False))
intents = data['intents']

for intent in intents:
    tag = intent['tag']
    patterns = intent['patterns']
    responses = intent['responses']
    print(f"Tag: {tag}")
    print(f"Patterns: {patterns}")
    print(f"Responses: {responses}")
    print("-" * 50)


import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('wordnet')

lemma=WordNetLemmatizer()
words=[]
classes=[]
documents=[]
ignore_words = ['?', '!', '.', ',', ':', ';']

for intent in intents:
    for pattern in intent['patterns']:
         word_list=word_tokenize(pattern)
         words.extend(word_list)
         documents.append((pattern, intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

words=[lemma.lemmatize(w.lower()) for w in words if w not in ignore_words]
words=sorted(list(set(words)))

classes=sorted(list(set(classes)))
print("words:", words)
print('classes:',classes)



def bow(sentence, words):
    sentence_words=word_tokenize(sentence)
    sentence_words=[lemma.lemmatize(w.lower()) for w in sentence_words]
    bag=[1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

tr_sentences=[]
tr_labels=[]

for doc in documents:
    tr_sentences.append(bow(doc[0],words))
    tr_labels.append(doc[1])

encode=LabelEncoder()
tr_labels=encode.fit_transform(tr_labels)
print("Training sentences:", len(training_sentences))
print("Training labels:", len(training_labels))



from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

tr_labels=to_categorical(tr_labels, num_classes=len(classes))

model=Sequential()
model.add(Dense(128, input_dim=len(tr_sentences[0]), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(np.array(tr_sentences), np.array(tr_labels), epochs=200, batch_size=16)

import random
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from keras.models import load_model

nltk.download('punkt')
model=load_model('chatbotmodel.h5')

with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

words = []
classes = []
patterns = []
ignore_words = ['?', '!', '.', ',', ':', 'и', 'в', 'на', 'с', 'по', 'для']

for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = word_tokenize(pattern)
        words.extend(word_list)
    classes.append(intent['tag'])

words=[lemma.lemmatize(w.lower()) for w in words if w not in ignore_words]
words=sorted(list(set(words)))
classes=sorted(list(set(classes)))

def bow(sentence,words):
    sentence_words=word_tokenize(sentence)
    sentence_words=[lemma.lemmatize(w.lower()) for w in sentence_words]
    bag=[1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    bow_sentence = bow(sentence, words)
    pred = model.predict(np.array([bow_sentence]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(pred) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

def chat():
    print('Hi, i am bot. Print "exit" to close chat')
    while True:
        userinput=input('You:')
        if userinput.lower()== "exit":
            print("Good bye")
            break
        intents_list=predict_class(userinput)
        response=get_response(intents_list,data)
        print('BOT:', response)

chat()
