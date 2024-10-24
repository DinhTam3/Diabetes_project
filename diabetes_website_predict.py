# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:32:29 2024

@author: LENOVO
"""

import pickle 
import streamlit as st 
from streamlit_option_menu import option_menu
import numpy as np
from sklearn.preprocessing import StandardScaler

# loading the saved model
filepath = 'C:\\Users\\LENOVO\\Data_Analysis\\Diabetes_prediction\\diabetes_model.sav'
diabetes_model = pickle.load(open(filepath, 'rb'))


scaler = StandardScaler()
def sysPredict(input_data):
    input_data = np.asarray(input_data)
    reshaped_data = input_data.reshape(1, -1)
    standard = scaler.fit_transform(reshaped_data)
    
    y_pred = diabetes_model.predict(standard)
    
    if y_pred[0] == 0:
        return 'This patient does not have diabetes'
    else:
        return 'This patient has diabetes'    

    

def main():
        # page title
        st.title('Diabetes Prediction using Machine Learning')
        
        Pregnancies = st.number_input("Number of Pregnancies", step=1)

        Glucose = st.number_input("Glucose level", step=1)

        BloodPressure = st.number_input("Blood Pressure value", step=1)

        SkinThickness = st.number_input("Skin Thickness value", step=1)

        Insulin = st.number_input("Insulin level", step=1)

        BMI = st.number_input("BMI value")    

        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function value", format="%0.3f")

        Age = st.number_input("Age of the Person", step=1)
        
        # code for prediction
        diabetes_diagnosis = ''
        
        # creating a button for prediction
        if st.button("Diabetes test result"):
            input_data = (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
            diabetes_diagnosis = sysPredict(input_data)
        st.success(diabetes_diagnosis)

if __name__ == "__main__":
    main()

