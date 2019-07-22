__author__ = 'i20764'
from flask import Flask, render_template,request,json,jsonify
import nltk, string
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
#app = Flask(__name__)
#nltk.download('punkt') # first-time use only
stemmer = nltk.stem.porter.PorterStemmer()

def StemTokens(tokens):
    return [stemmer.stem(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def StemNormalize(text):
    return StemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
     return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
     return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))



dataset = pd.read_csv('ddldata.csv')
X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]

result_target = np.squeeze(np.asarray(X))
resutl_attribut = np.squeeze(np.asarray(y))
print(resutl_attribut)
LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
array_attribute = LemVectorizer.fit_transform(y)

att_result = array_attribute.toarray()

print(att_result)
#Print feature with position
dataset_vocab = LemVectorizer.vocabulary_
print(dataset_vocab)

#get user input
user_input = input("Input:")
words = user_input.split(" ")
print(words)
print(len(words))
LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
LemVectorizer.fit_transform([user_input])

Input_vocab = LemVectorizer.vocabulary_
print(Input_vocab)


#Transform the calculated vector into array form
#tf_matrix = LemVectorizer.transform(y).toarray()
# print(tf_matrix)

#Geting input from user


# print(LemVectorizer.vocabulary_)
# input_vect = LemVectorizer.transform([user_input]).toarray()
# print(input_vect)

#Calculate the cosine similarity between train data and input data
#Convert the result to array
# result = cosine_similarity(tf_matrix,input_vect)
# # print(result)
# result_array = np.squeeze(np.asanyarray(result))
# # print(result_array)
#
# #Find out  the index from the final array to find out the class
# #Check the index value in array. if index values are 0, we do not consider those values
# #We only consider the among those values which are not equal to zero
# get_indexes = [n for n,x in enumerate(result_array) if x!=0]
# #Assign the index value from above and find the class name
# final_table = ''
# for i in range(len(get_indexes)):
#         # print(get_indexes[i])
#         final_table =final_table +  result_target[get_indexes[i]] +';'





