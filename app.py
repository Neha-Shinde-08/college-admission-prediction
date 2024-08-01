from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the pre-trained models
df1 = pd.read_csv("college admission prediction.csv")
df = df1.copy()

colg = np.unique(df['College'])
code = [i+1 for i in range(len(colg))]
df['College'] = df['College'].replace(colg, code)
X = df.drop(columns=["Year", "College"])
y = df['College']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, _, y_train, _ = train_test_split(X, y_encoded, test_size=0.05, random_state=22)

clfxg = XGBClassifier(objective="multi:softmax", n_estimators=50, learning_rate=0.0001)
clfxg.fit(X_train, y_train)

clfdt = DecisionTreeClassifier()
clfdt.fit(X_train, y_train)

# Route to handle the home page
@app.route('/')
def home():
    return render_template('home.html')

# Route to handle the prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Handle the form submission
        # Retrieve input values from the form
        tenth_marks = int(request.form['10th_marks'])
        twelfth_marks = int(request.form['12th_marks'])
        twelfth_division = int(request.form['12th_division'])
        aieee_rank = int(request.form['aieee_rank'])

        # Check if any of the marks are zero or less than 35, if so, return an error message
        if tenth_marks <= 70 or twelfth_marks <= 70 or twelfth_division <= 0 or aieee_rank <= 0:
            return render_template('index.html', error_message="Please enter valid marks.")

        # Generate user input array
        user_input = np.array([tenth_marks, twelfth_marks, twelfth_division, aieee_rank]).reshape(1, -1)
        
        # Make predictions using both models
        predxg = clfxg.predict_proba(user_input)
        preddt = clfdt.predict_proba(user_input)

        # Get the top 3 predicted colleges from each model
        top_colleges_xg = [colg[idx] for idx in predxg.argsort()[0][-6:]][::-1]
        top_colleges_dt = [colg[idx] for idx in preddt.argsort()[0][-6:]][::-1]

        return render_template('index.html', result_xg=top_colleges_xg, result_dt=top_colleges_dt)
    else:
        # Render the prediction form
        return render_template('index.html')


# Route to handle the resources page
@app.route('/resources')
def resources():
    return render_template('resources.html')

# Route to handle the tips page
@app.route('/tips_and_tricks')
def tips():
    return render_template('Tips_and_Tricks.html')

# Route to handle the links page
@app.route('/links')
def links():
    return render_template('Links.html')

# Entry point of the application
if __name__ == '__main__':
    app.run(debug=True)
