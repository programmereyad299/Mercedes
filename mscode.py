import numpy as np
import streamlit as st
import joblib
import tensorflow as tf

st.set_page_config(page_title = "Mercedes Benz Stock Refined" , page_icon="🚗")
st.title("Mercedes Benz Stock Refined 🚗💵")
st.title("")

model = tf.keras.models.load_model('msmodel.h5')
sc = joblib.load('mssc.pkl')
le12 = joblib.load('msle12.pkl')


m0 = st.slider("Open" , 5.97 , 65.7)

m1 = st.slider("High" , 6.14 , 66.1)

m2 = st.slider("Low" , 5.81 , 64.8)

m3 = st.slider("Price" , 5.89 , 65.4)

m4 = st.slider("Volume" , 8416 , 74000000)

m5 = st.slider("Daily Return" , -18.9 , 27.3)

m6 = st.slider("Price Change" , -3.35 , 3.97)

m7 = st.slider("Price Range" , 0.0 , 5.39)

m8 = st.slider("MA 7" , 6.19 , 64.8)

m9 = st.slider("MA 30" , 6.77 , 63.5)

m10 = st.slider("MA 90" , 7.25 , 61.9)

m11 = st.slider("Volatility 30" , 0.44 , 8.27)

m12  = st.selectbox("Day Of Week",['Wednesday','Thursday','Friday','Monday','Tuesday'])

m13 = st.slider("Month" , 1 , 12)

m14 = st.slider("Year" , 1996 , 2026)


m12 = le12.transform([m12])[0]


btn = st.sidebar.button("pred")

if btn :
    input_data = np.array([[m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14]])
    sc.transform(input_data)[0]
    pred = model.predict(input_data)
    pred = np.argmax(pred)  
    st.sidebar.info("Quarter :")
    st.sidebar.info(pred)







