import streamlit as st
import joblib
import numpy as np

#model load 
model=joblib.load("iris_model.pkl")

st.title("Iris Flower Prediction App")

st.header("Enter the measurement of the iris flower")

sepal_length=st.number_input("Sepal length (cm)",min_value=0.0,max_value=10.0,value=5.0,step=0.1)
sepal_width=st.number_input("Sepal width (cm)",min_value=0.0,max_value=10.0,value=3.0,step=0.1)
petal_length=st.number_input("Patal length (cm)",min_value=0.0,max_value=8.0,value=4.0,step=0.1)
petal_width=st.number_input("Patal width (cm)",min_value=0.0,max_value=10.0,value=1.0,step=0.1)

#prediction
if st.button('Predict'):
  input_data = np.array([[sepal_length,
                          sepal_width,
                          petal_length,
                          petal_width]]).astype(np.float64)
  prediction = model.predict(input_data)
  st.success(f'The predicted Iris species is: {prediction[0]}')
