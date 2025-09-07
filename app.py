import numpy as np
import tensorflow
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

## RNN project

# load imdb daaset
word_index=imdb.get_word_index()
rev_word_index={value:key for key, value in word_index.items()}



# load model in h5 file
model=tensorflow.keras.models.load_model('simple_rnn_imdb.h5')

# 2 functions to convert text to vector so model can use another function for reversing vector to text

def decode_review(encoded_review):
    return ' '.join([rev_word_index.get(i-3,'?') for i in encoded_review])


def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word, 2) + 3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# prediction function

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment='positive' if prediction[0][0] >0.5 else 'negative'
    
    return sentiment, prediction[0][0]


# streamlit app

st.title('Movie review center')
st.write('Please submit your review')


# capture user input
user_input=st.text_area('Movie review')

if st.button('Classify'):
    sentiment, score=predict_sentiment(user_input)
    st.write(f"Review : {user_input}")
    st.write(f"sentiment : {sentiment}")
    st.write(f"score : {score}")
    
else:

    st.write('Please enter review')
