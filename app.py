import pandas as pd
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and LabelEncoder
model = joblib.load('model_no_pca.pkl')
le = joblib.load('label_encoder.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    if request.method == 'POST':
        # Get form data
        input_data = {
            'Age': float(request.form['Age']),
            'Gender': request.form['Gender'],
            'Neighbourhood': request.form['Neighbourhood'],
            'Scholarship': int(request.form['Scholarship']),
            'Hipertension': int(request.form['Hipertension']),
            'Diabetes': int(request.form['Diabetes']),
            'Alcoholism': int(request.form['Alcoholism']),
            'Handcap': int(request.form['Handcap']),
            'SMS_received': int(request.form['SMS_received']),
            'WaitingDays': float(request.form['WaitingDays']),
            'ScheduledHour': float(request.form['ScheduledHour'])
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode Gender
        input_df['Gender'] = le.transform(input_df['Gender'])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100
        
    return render_template('index.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)