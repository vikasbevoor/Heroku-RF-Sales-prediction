# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'Sales-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        CompPrice = int(request.form['CompPrice'])
        Income = int(request.form['Income'])
        Advertising = int(request.form['Advertising'])
        Population = int(request.form['Population'])
        Price = int(request.form['Price'])
        Age = int(request.form['Age'])
        Education = int(request.form['Education'])
        ShelveLoc_Good = int(request.form['ShelveLoc_Good'])
        ShelveLoc_Medium = int(request.form['ShelveLoc_Medium'])
        Urban_Yes = int(request.form['Urban_Yes'])
        US_Yes = int(request.form['US_Yes'])
        
        data = np.array([[CompPrice, Income, Advertising,Population, Price, Age,Education,ShelveLoc_Good,ShelveLoc_Medium,Urban_Yes, US_Yes]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)