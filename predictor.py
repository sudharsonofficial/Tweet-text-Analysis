import nltk
import re
import tensorflow as tf
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer 

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")


class Processor():
    def __init__(self):
        self.ps = PorterStemmer()
        self.tfid= TfidfVectorizer() 

        self.corpus = []

        self.X = None


    def pre_process(self,data):
        self.corpus = []

        for i in range(len(data)):
            review = re.sub("[^a-zA-Z]"," ",data[i])

            review = review.lower().split()

            review = [self.ps.stem(r) for r in review if r not in stopwords.words("english")]

            review = " ".join(review)

            self.corpus.append(review)
        
            self.X = self.tfid.fit_transform(self.corpus).toarray()
            return self.X

       
        




class Model():
    def __init__(self):
        self.model =MultinomialNB()

    
    def load_model(self):
        return self.model



    def predict(self,X):

        res = self.model.predict(X)

        res = [round(float(i)) for i in res]

        return res




