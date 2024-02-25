from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))  # Load the trained model (pickle file)
scaler = pickle.load(open('minmax_scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route( '/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        feature = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh','exng', 'oldpeak', 'slp', 'caa', 'thall']
        data = [float(request.form[f]) for f in feature]
        data_array = np.array([data]).reshape(1, -1)
        data_normalized = scaler.fit_transform(data_array)
        prediction = model.predict(data_normalized)[0]

        if int(prediction) == 1:
            prediction = 'You have more chance of heart attack'
        else:
            prediction = 'No heart attack expected'
        return render_template('index.html', prediction = prediction)
    
    return render_template('index.html', prediction = '')
    
if __name__ == "__main__":
    app.run(debug=True)