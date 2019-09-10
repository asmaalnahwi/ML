# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 00:23:32 2019

@author: Asmaa
"""

from sklearn.naive_bayes import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
import matplotlib.pyplot as plt
from sklearn.svm import *
import pandas
import contractions
import nltk as n
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from wordcloud import WordCloud
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score,precision_score, recall_score

sms = pandas.read_csv('C:\\Users\\dell\Downloads\\spam.csv', encoding='latin-1')
pandas.options.mode.chained_assignment = None
############################# preorcessing ####################################\

sms = sms.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)  # drop not needed colmuns
sms.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)  # rename columns
sms['label'] = sms['label'].map({'ham': 0, 'spam': 1})  # change ham to 0 and spam to 1

def Lemma(message):
    result = []
    lemmatizer = WordNetLemmatizer()
    result = (
    [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in message])  ### call the function and lemmatie each word
    return result

def get_wordnet_pos(word):  ### function to know the POS of the word
    """Map POS tag to first character lemmatize() accepts"""
    tag = n.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)



filtered_sentence = []
for i in range(len(sms)):
	filtered_sentence = []
	message = sms['message'][i]
	message = message.lower()
	message = contractions.fix(message) # convert slang to formal 
	message = n.word_tokenize(message)
	message =  [w for w in message if w.isalnum() |( w in ['!','?',':','$','£'])]
#	stopwords = n.corpus.stopwords.words('english')
#	for w in message:
#	 	if w not in stopwords:
#	 		filtered_sentence.append(w)  # add the word to the new list
	
#	message = filtered_sentence
	#message = Lemma (message)
	string_text = ""

	#********* convert series of words to text to tokenize *****#
	for word in message:
		string_text = string_text + word + " "
	sms['message'][i] = string_text

###############################################################################
    
############################ spliting #########################################



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(sms.message, sms.label, test_size=0.3, random_state=0)


###############################################################################

def classify(classifiers, X_train, X_test, y_train, y_test):
    for classifier in classifiers:
      
        string = ''
        string += classifier.__class__.__name__ + ' with TfidfVectorizer()'

        # train
        vectorizer = TfidfVectorizer()
        vectorize_text = vectorizer.fit_transform(X_train) 
        classifier.fit(vectorize_text, y_train)

        # score
        vectorize_text = vectorizer.transform(X_test)
        score = classifier.score(vectorize_text,y_test)
        ynew = classifier.predict(vectorize_text)
        fscore = f1_score(y_test, ynew) 
        p=precision_score(y_test, ynew)
        rec=recall_score(y_test, ynew)
        
        string += '. Accuracy: ' + str(score)+'. F-score: '+ str(fscore)+'. Precision: '+str(p)+'. Recall: '+str(rec)+"\n"
        print(string)
        
        
classify(
    [
        OneVsRestClassifier(SVC(kernel='linear')),
         BernoulliNB(),
         MultinomialNB()
         
    ],
    
    X_train, X_test, y_train, y_test)