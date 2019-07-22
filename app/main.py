__author__ = 'i20764'
from flask import Flask, render_template,request,json,jsonify
import nltk, string
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
app = Flask(__name__)
from rivescript import RiveScript
from urllib.parse import unquote
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

# from app.chatbot import get_response

def get_response(usrText):

    bot = ChatBot('Bot',
                  storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        {
            'import_path': 'chatterbot.logic.BestMatch'
        },
        {
            'import_path': 'chatterbot.logic.LowConfidenceAdapter',
            'threshold': 0.7,
            'default_response': 'NoResultFound'
        }
    ],
    trainer='chatterbot.trainers.ListTrainer')
    #trainer='chatterbot.trainers.ChatterBotCorpusTrainer')
    # First, lets train our bot with some data
    #bot.train('chatterbot.corpus.english')
    bot.set_trainer(ListTrainer)
    result = bot.get_response(usrText)
    reply = str(result)
    return reply

rs = RiveScript(utf8=True)
rs.load_directory("./brain")
rs.sort_replies()
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

#Custume function
def get_userinput_from_bot(text):

    print(text)
    x = get_response(unquote(text))
    print(x)
    x=x.replace("- ","")
    if x=="NoResultFound":
        # user_input = text
        # return "hello"
        user_input = rs.reply("localuser", unquote(text))
        print (user_input)
        user_word = user_input.split(" ")
        print(len(user_word))
        if len(user_word)==1:
            dataset = pd.read_csv('ddldata.csv')
            X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]

            result_target = np.squeeze(np.asarray(X))
            resutl_attribut = np.squeeze(np.asarray(y))


            LemVectorizer = CountVectorizer(tokenizer=LemNormalize)
            LemVectorizer.fit_transform(y)

            #Print feature with position
            # print(LemVectorizer.vocabulary_)

            #Transform the calculated vector into array form
            tf_matrix = LemVectorizer.transform(y).toarray()
            # print(tf_matrix)

            #Geting input from user
            input_vect = LemVectorizer.transform([user_input]).toarray()
            #print(input_vect)

            #Calculate the cosine similarity between train data and input data
            #Convert the result to array
            result = cosine_similarity(tf_matrix,input_vect)
            # print(result)
            result_array = np.squeeze(np.asanyarray(result))
            # print(result_array)

            #Find out  the index from the final array to find out the class
            #Check the index value in array. if index values are 0, we do not consider those values
            #We only consider the among those values which are not equal to zero
            get_indexes = [n for n,x in enumerate(result_array) if x!=0]
            print(get_indexes)
            if get_indexes == []:
                print("Please enter the valid text")
            else:
                print("Tables name :")

            #Assign the index value from above and find the class name
            final_table = ''
            for i in range(len(get_indexes)):
                # print(get_indexes[i])
                final_table =final_table +  result_target[get_indexes[i]] +' '

            return "The " + user_input+ " is column name of " + final_table
        else:
            if user_input=="[ERR: No Reply Matched]":
                print(user_input)
                return "Sorry! I can't understand. Will you please explain a bit?"
            else:
                return user_input +' '
    else:
            return x +' '


@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/testurl",methods = ['POST'])

def testurl():
    if request.method == 'POST':
        info = request.data
        res=info.decode('utf-8') #remove the b'
        print(res) #print res without b'
        res=res.replace("text=","")
        print(res)
        x = get_userinput_from_bot(res)
        return x



if __name__ == "__main__":
    app.run()