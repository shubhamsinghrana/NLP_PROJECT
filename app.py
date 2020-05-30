from flask import Flask, render_template, request,url_for
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq  
import numpy as np 
import pandas as pd 


###############################
# import itertools
# import os
# import tensorflow as tf
# from sklearn.preprocessing import LabelBinarizer, LabelEncoder
# from sklearn.metrics import confusion_matrix
# from tensorflow import keras
# layers = keras.layers
# models = keras.models


app = Flask(__name__)






# def categorizer(text):
# 	data=pd.read_csv("bbc-text.csv")
# 	train_size = int(len(data) * .99)
# 	def train_test_split(data, train_size):
# 	    train = data[:train_size]
# 	    test = data[train_size:]
# 	    return train, test
# 	train_cat, test_cat = train_test_split(data['category'], train_size)
# 	train_text, test_text = train_test_split(data['text'], train_size)
# 	max_words = 1000
# 	tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words, 
#                                               char_level=False)
# 	tokenize.fit_on_texts(train_text) # fit tokenizer to our training text data
# 	x_train = tokenize.texts_to_matrix(train_text)
# 	encoder = LabelEncoder()
# 	encoder.fit(train_cat)
# 	y_train = encoder.transform(train_cat)
# 	y_test = encoder.transform(test_cat)
# 	num_classes = np.max(y_train) + 1
# 	y_train = keras.utils.to_categorical(y_train, num_classes)
# 	y_test = keras.utils.to_categorical(y_test, num_classes)
# 	batch_size = 32
# 	epochs = 2
# 	drop_ratio = 0.5
# 	model = models.Sequential()
# 	model.add(layers.Dense(512, input_shape=(max_words,)))
# 	model.add(layers.Activation('relu'))
# 	# model.add(layers.Dropout(drop_ratio))
# 	model.add(layers.Dense(num_classes))
# 	model.add(layers.Activation('softmax'))

# 	model.compile(loss='categorical_crossentropy',
# 	              optimizer='adam',
# 	              metrics=['accuracy'])
# 	history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_split=0.1)
# 	l=[text]
# 	x_test2 = tokenize.texts_to_matrix(l)
# 	print(np.array([l]).shape)
# 	text_labels = encoder.classes_
# 	prediction = model.predict(np.array([x_test2[0]]))
# 	predicted_label = text_labels[np.argmax(prediction)]
# 	print("Predicted label: " + predicted_label + "\n")  

# 	return predicted_label




def nltk_summarizer(raw_text):
    stopWords = set(stopwords.words("english"))
    word_frequencies = {}  
    for word in nltk.word_tokenize(raw_text):  
        if word not in stopWords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    sentence_list = nltk.sent_tokenize(raw_text)
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 20:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]



    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)  
    return summary


def analyse(text1="",text2=""):
    res=[]
    res.append(len(sent_tokenize(text1)))
    res.append(len(sent_tokenize(text2)))
    res.append(len(word_tokenize(text1)))
    res.append(len(word_tokenize(text2)))
    s1=str(res[0])+" sentences and " +str(res[2])+" converted into " +str(res[1]) + " sentences and " +str(res[3])+" words."
    return s1
    
 
# home page
@app.route('/')
@app.route('/nlpsummarize')
def summarize_nlp():
	return render_template('summarize.html')

# spam classification
@app.route('/summarize',methods= ['POST','GET'])
def sum_route():
	if request.method == 'POST':
		message = request.form['message']
		l=sent_tokenize(message)
		nl=[]
		len(l)
		ans=""
		for line in l:
		    ans=ans+" "+line
		    if len(sent_tokenize(ans))==5:
		        nl.append(ans)
		        ans=""
		if len(sent_tokenize(ans)) >0:
		    nl.append(ans)
		    ans=""
		for para in nl:
		        ans=ans+nltk_summarizer(para)
		res=[]
		res=analyse(message,ans)
		label_predict="Other"
		return render_template('summarize.html',original = message, prediction=ans,tag=label_predict,val=res)



if __name__ == '__main__':
	app.run(debug=True)