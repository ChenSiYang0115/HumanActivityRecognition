from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import the CORS module
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from scipy.signal import find_peaks

app = Flask(__name__)
CORS(app)  # Apply CORS to your Flask app

# Load the machine learning model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def receive_accelerometer_data():
    data = request.get_json()

    x_data = preprocess_data(data)

    # Make predictions using the loaded machine learning model
    prediction = model.predict(x_data)

    # Convert prediction to a Python list
    prediction = prediction.tolist()

    # Render the index.html
    return prediction[0]

# Data Preprocessing function
def preprocess_data(data):
    # Assuming data is a dictionary with keys 'x', 'y', and 'z'
    # Convert data into a pandas DataFrame
    df = pd.DataFrame(data, index=[0])

    # Average acceleration (for each axis)
    df['XAVG'] = df['x'].mean()
    df['YAVG'] = df['y'].mean()
    df['ZAVG'] = df['z'].mean()

    # Standard Deviation for each axis
    df['XSTANDDEV'] = df['x'].std()
    df['YSTANDDEV'] = df['y'].std()
    df['ZSTANDDEV'] = df['z'].std()

    # Calculate Average Absolute Difference for each axis
    df['XABSOLDEV'] = np.mean(np.absolute(df['x'] - df['x'].mean()))
    df['YABSOLDEV'] = np.mean(np.absolute(df['y'] - df['y'].mean()))
    df['ZABSOLDEV'] = np.mean(np.absolute(df['z'] - df['z'].mean()))

    # Calculate Average Resultant Acceleration
    df['RESULTANT'] = np.mean(np.sqrt(df['x']**2 + df['y']**2 + df['z']**2))

    # Calculate Time Between Peaks for each axis
    fs = 20
    time_values = np.arange(len(df)) / fs

    # Find peaks for each axis
    # height = 0 -> Find local maxima
    x_peaks, _ = find_peaks(df['x'], height=0)
    y_peaks, _ = find_peaks(df['y'], height=0)
    z_peaks, _ = find_peaks(df['z'], height=0)

    # Calculate Time Between Peaks for each axis
    df['XPEAK'] = np.mean(np.diff(time_values[x_peaks]))
    df['YPEAK'] = np.mean(np.diff(time_values[y_peaks]))
    df['ZPEAK'] = np.mean(np.diff(time_values[z_peaks]))

    # Calculate Binned Distribution for each axis
    num_bins = 10
    # Calculate Binned Distribution for each axis
    num_bins = 10
    x_bins = calculate_binned_distribution(df['x'], num_bins)
    y_bins = calculate_binned_distribution(df['y'], num_bins)
    z_bins = calculate_binned_distribution(df['z'], num_bins)

    # Create separate columns for each binned distribution feature for each axis
    for i in range(num_bins):
        df[f'X{i}'] = x_bins[i]
        df[f'Y{i}'] = y_bins[i]
        df[f'Z{i}'] = z_bins[i]

    # Drop raw data
    df = df.drop(['x', 'y', 'z'], axis=1) 

    # Data Preprocessing 
    scaler = StandardScaler()
    x = df.values
    x = scaler.fit_transform(x)

    print(df.info())
    print(df[["XPEAK", "YPEAK", "ZPEAK", "XSTANDDEV", "YSTANDDEV", "ZSTANDDEV"]])

    # Convert the NumPy array to a Python list
    return x.tolist()  

# Calculate Binned Distribution for each axis
def calculate_binned_distribution(data, num_bins=10):
    min_val = data.min()
    max_val = data.max()
    bin_range = (max_val - min_val) / num_bins

    bins = [min_val + i * bin_range for i in range(num_bins + 1)]
    return np.histogram(data, bins=bins)[0] / len(data)

if __name__ == '__main__':
    app.run(debug=True)