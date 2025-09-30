from flask import Flask, request, render_template

import joblib

# Load your trained model (adjust filename if needed)
with open("models/spam_model.pkl", "rb") as f:
    model = joblib.load(f)

# If you used CountVectorizer or TfidfVectorizer, load it too
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = joblib.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        email_text = request.form["email_text"]

        # Transform input using vectorizer
        features = vectorizer.transform([email_text])

        # Predict with model
        prediction = model.predict(features)[0]

        result = "🚫 Spam" if prediction == 1 else "✅ Not Spam"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
