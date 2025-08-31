import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import joblib
sv=SVC()
data=pd.read_csv(r"C:\Users\payal\Desktop\Bank Customer Churn Prediction.csv")

data=data[["credit_score","gender","age","balance","credit_card","estimated_salary","churn"]]
from sklearn.preprocessing import LabelEncoder 
la=LabelEncoder()

data["gender"]=la.fit_transform(data[["gender"]])
x=data.iloc[:,:-1]
y=data[["churn"]]
x_train,x_test,y_train,y_test=train_test_split(x,y)
sv.fit(x_train,y_train)
joblib.dump(sv,"model")
joblib.dump(la,"label")
