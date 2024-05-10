from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get features from the form
    features = [float(x) for x in request.form.values()]

    # Make prediction
    prediction = model.predict([features])[0]

    # Convert prediction to readable format
    if prediction == 0:
        result = 'No stroke risk'
    else:
        result = 'Stroke risk'

    return render_template('index.html', prediction_text='Prediction: {}'.format(result))


# if __name__ == '__main__':
#    app.run(debug=True)
