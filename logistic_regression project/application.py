from flask import Flask,request,render_template
from flask import Response
import pandas as pd 
import numpy as np 
import pickle

application=Flask(__name__)
app=application
scaler=pickle.load(open('scaler.pkl','rb'))
model=pickle.load(open('logit.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', "POST"])
def data():
    result = ''
    if request.method == 'POST':
        # Debugging lines to print form data
        print("Form Data:", request.form)

        Pregnancies = float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        age = float(request.form.get('age'))

        new_data = scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, age]])
        predict = model.predict(new_data)

        if predict[0] == 1:
            result = 'Diabetic'
        else:
            result = 'Non-Diabetic'
        return render_template('home.html', result=result)
    else:
        return render_template('home.html')
    
if __name__=='__main__':
    app.run(debug=True)