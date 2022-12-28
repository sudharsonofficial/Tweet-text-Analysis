import pickle
import streamlit as st
import numpy as np
import time
from PIL import Image
import io
import re
from paddle2 import pad
from nltk.corpus import stopwords
from predictor import Processor,Model
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 

st.set_page_config(
        page_title="Sentiment analysis",
        
    )

tfid=pickle.load(open('vectorizer.pkl',"rb"))
model=pickle.load(open('model.pkl','rb'))
pi=pad()

class processor():
    def __init__(self):
                  self.ps = PorterStemmer()
                  self.tfid=tfid
    def pre_process(self,data):
            self.corpus = []
            
            for i in range(len(data)):
                review = re.sub("[^a-zA-Z]"," ",data[i])

                review = review.lower().split()

                review = [self.ps.stem(r) for r in review if r not in stopwords.words("english")]

                review = " ".join(review)

                self.corpus.append(review)
            try:
                self.X = self.tfid.transform(self.corpus).toarray()
                return self.X
            except ValueError:
                print("Value error")
p=processor()

st.title(":blue[Tweet] Sentiment Analysis")

txt=st.text_input("Enter your text/tweet here:")

if st.button("Predict"):
    
    pre_txt=p.pre_process([txt])
    # vectorized=tfid.transform([pre_txt])
    result=model.predict(pre_txt)
    with st.spinner('Almost there...'):
        time.sleep(5)
        if(result[0]==0):
                # st.markdown(":red[Negative]")
                st.error("Negative")
        else:
            # st.markdown(":green[Positive]")  
            st.success("Positive") 

st.markdown("OR")
upl=st.file_uploader("Upload Your Image:")
# u=st.image(image=upl) 

if upl is not None:
    # bd=upl.getvalue()
    # st.write(type(bd))
    # u=st.image(image=upl)
    img=Image.open(upl)
    img.save('down.png')
    
    # image = np.array(Image.open(io.BytesIO(bd)))#converting bytes to np array
    # image_a=Image#keeping it as array itself
    # st.write(image_a)
    # st.image(image=upl)
    txt1=pi.x_t(r"C:\Users\Sudha\Desktop\Sentiment analysis\down.png")
    st.markdown(":blue[The tweet is:]")
    st.write(txt1)
    pre_txt=p.pre_process([txt1])
     # vectorized=tfid.transform([pre_txt])
    result=model.predict(pre_txt)
    with st.spinner('Almost there...'):
        time.sleep(5)
        if(result[0]==0):
                # st.markdown(":red[Negative]")
                st.error("Negative")
        else:
            # st.markdown(":green[Positive]")  
            st.success("Positive")





