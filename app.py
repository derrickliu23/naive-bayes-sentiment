# app.py
# Flask backend for the Naïve Bayes sentiment demo.
# Exposes a single POST endpoint /classify that accepts JSON text
# and returns the classifier's prediction + word-level explanation.

from flask import Flask, request, jsonify, render_template
from classifier import classifier

app = Flask(__name__)


@app.route("/")
def index():
    """Serve the main demo page."""
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    """
    Accepts: { "text": "some input string" }
    Returns: {
        "label": "positive" | "negative",
        "confidence": 0.0 - 1.0,
        "probabilities": { "positive": float, "negative": float },
        "explanation": [ { "word": str, "diff": float, "scores": {...} }, ... ]
    }
    """
    data = request.get_json()

    # Basic validation
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400

    result = classifier.predict(text)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)