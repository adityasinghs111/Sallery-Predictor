from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained pipeline model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'Age': int(request.form['age']),
        'Gender': request.form['gender'],
        'Department': request.form['department'],
        'Job_Title': request.form['job_title'],
        'Experience_Years': int(request.form['experience']),
        'Education_Level': request.form['education'],
        'Location': request.form['location']
    }

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    return render_template('index.html', prediction_text=f"Predicted Salary: ${round(prediction, 2)}")

if __name__ == '__main__':
    app.run(debug=True)
