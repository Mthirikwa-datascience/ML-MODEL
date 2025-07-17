import pandas as pd
import joblib
from flask import Flask, request, render_template
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Custom transformer for outlier handling
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
    
    def fit(self, X, y=None):
        self.bounds_ = []
        for col_idx in range(X.shape[1]):
            q1 = np.percentile(X[:, col_idx], 25)
            q3 = np.percentile(X[:, col_idx], 75)
            iqr = q3 - q1
            self.bounds_.append([q1 - self.factor * iqr, q3 + self.factor * iqr])
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col_idx in range(X.shape[1]):
            lower, upper = self.bounds_[col_idx]
            X_copy[:, col_idx] = np.clip(X_copy[:, col_idx], lower, upper)
        return X_copy

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