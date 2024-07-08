
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer #steamer le root word patta lagauxa(playing = play)
from sklearn.pipeline import Pipeline #data lai algo ma rakhda kun kun step bata pass hunxa vanera vanne
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

st.title("Emotion classification")
df = pd.read_csv('Emotion_classify_Data.csv')
df
# --------------------------------model creation----------------------------------------
# Training model
from sklearn.linear_model import LogisticRegression
log_regression = LogisticRegression()

vectorizer = TfidfVectorizer()
X = df['Comment']
Y = df['Emotion']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15) #Splitting dataset


# #Creating Pipeline
#pipeline vaneko architecture nai ho
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=1000)),
                     ('clf', LogisticRegression())])


# #Training model
model = pipeline.fit(X_train, y_train)
# ---------------------------------model creation end-----------------------------------------------


text = st.text_area("Enter text here")
if st.button("Submit"):

    data = {'predict_emotion':[text]}
    data_df = pd.DataFrame(data)
    emotion_prediction = model.predict(data_df['predict_emotion']) #dataframe name ani kun key ni lekhne
    st.write("Predicted emotion category = ",emotion_prediction[0])
else:
    st.write("please enter text")