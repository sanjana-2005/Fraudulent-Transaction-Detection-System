from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Set secret key for session management
app.secret_key = 'hello'  # Replace 'your_secret_key_here' with a secure key

# Load the pre-trained model, scaler, KMeans clustering, and PCA
model_path = r"C:\flaskenv\quantum_probability_classifier_model1.pkl"
scaler_path = r"C:\flaskenv\scaler1.pkl"
kmeans_path = r"C:\flaskenv\kmeans.pkl"
pca_path = r"C:\flaskenv\pca1.pkl"

# Load models
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
kmeans = joblib.load(kmeans_path)
pca = joblib.load(pca_path)

# Sample users (In real applications, use a database)
users = {
    "admin": "password123",  # username: password
    "user": "userpassword"
}

@app.route('/')
def index():
    return render_template('login4.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    if username in users and users[username] == password:
        flash('Login successful!', 'success')
        return redirect(url_for('dashboard'))
    else:
        flash('Invalid username or password!', 'danger')
        return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    return render_template('index4.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        time = float(request.form['time'])
        amount = float(request.form['amount'])

        input_data = np.zeros(28)
        input_data[0] = time
        input_data[1] = amount

        input_data = input_data.reshape(1, -1)

        input_scaled = scaler.transform(input_data[:, :2])

        input_data[:, :2] = input_scaled

        input_pca = pca.transform(input_data)

        cluster = kmeans.predict(input_pca)

        input_with_cluster_pca = np.hstack([input_pca, cluster.reshape(-1, 1)])

        if input_with_cluster_pca.shape[1] > 3:
            input_with_cluster_pca = input_with_cluster_pca[:, :3]

        prob = model.predict_proba(input_with_cluster_pca)

        threshold = 0.1
        prediction = (prob[:, 1] > threshold).astype(int)

        fraud_probability = prob[0, 1] * 100
        non_fraud_probability = prob[0, 0] * 100

        if prediction[0] == 1:
            review = "Fraudulent Transaction"
            result = (
                f"Feedback: Based on the data provided, this transaction is likely to be fraudulent. "
                f"The model's prediction suggests a high likelihood of fraud.\n\n"
                f"Fraudulent Transaction - Probability: {fraud_probability:.2f}%\n\n"
                f"Please review the transaction carefully for any unusual behavior or inconsistencies."
            )
        else:
            review = "Non-Fraudulent Transaction"
            result = (
                f"Feedback: Based on the data provided, this transaction is likely non-fraudulent. "
                f"The model's prediction suggests a very low chance of fraud.\n\n"
                f"Non-Fraudulent Transaction - Probability: {non_fraud_probability:.2f}%\n\n"
                f"However, we recommend monitoring the transaction for any unexpected activity."
            )

        return render_template('result4.html', review=review, result=result)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
