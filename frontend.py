import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

model=pk.load(open('model.pkl','rb'))
scaler=pk.load(open('scaler.pkl','rb'))
review=st.text_input('Enter movie Review')

if st.button('Predict'):
    # scaler=TfidfVectorizer(max_features=2500)
    review_scale=scaler.transform([review]).toarray()
    result=model.predict(review_scale)
    if result[0]==0:
        st.write('Negetive Review')
    else:
        st.write('Positive Review')
   