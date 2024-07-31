from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model for diamond price prediction
model_path = "diamondprice (1).pkl"  # Ensure this path is correct
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    carat = float(request.form['carat'])
    cut = request.form['cut']
    color = request.form['color']
    clarity = request.form['clarity']
    depth = float(request.form['depth'])
    table = float(request.form['table'])
    x = float(request.form['x'])
    y = float(request.form['y'])
    z = float(request.form['z'])
    
    # Preprocess categorical features
    # Ordinal Encoding for 'cut'
    cut_mapping = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    cut_encoded = cut_mapping.get(cut, 0)  # Use 0 or a default value if 'cut' is invalid
    
    # Label Encoding for 'color' and 'clarity'
    color_mapping = {'J': 6, 'I': 5, 'H': 4, 'G': 3, 'F': 2, 'E': 1, 'D': 0}
    clarity_mapping = {'I1': 0, 'SI2': 3, 'SI1': 1, 'VS2': 5, 'VS1': 4, 'IF': 2, 'VVS2': 7, 'VVS1': 6}

    color_encoded = color_mapping.get(color, 0)  # Use 0 or a default value if 'color' is invalid
    clarity_encoded = clarity_mapping.get(clarity, 0)  # Use 0 or a default value if 'clarity' is invalid

    # Create feature vector
    features = [carat, cut_encoded, color_encoded, clarity_encoded, depth, table, x, y, z]
    final_features = [np.array(features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    output = f'Estimated Price: RS{prediction[0]:,.2f}'

    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
