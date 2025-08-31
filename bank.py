import streamlit as st
import joblib
model=joblib.load("model")
label=joblib.load("label")
st.title("Bank Customer Segmentation Prediction")
score=st.number_input("Enter Credit Score")
gender=st.selectbox("Select gender",options=["Male","Female"])
age=st.number_input("Enter Age")
balance=st.number_input("Enter Balance")
credit=st.selectbox("Credit Card",options=["Yes","No"])
salary=st.number_input("Enter Estimated Salary")
if credit=="Yes":
    credit=1
else:
    credit=0
gen=label.transform([gender])[0]
#st.write(gen)    
if st.button("Predict"):
    result=model.predict([[score,gen,age,balance,credit,salary]])
    if result[0]==0:
        st.success("Customer will not churn")
    else:
        st.error("Customer will churn")
