from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForTokenClassification, BertTokenizerFast
import torch
from nltk.tokenize import word_tokenize
import os

model = AutoModelForTokenClassification.from_pretrained(os.path.join("saved_model"))
tokenizer = BertTokenizerFast.from_pretrained(os.path.join("saved_tokenizer"))

id2label = {
    0: "O",
    1: "B-HOBBY",
    2: "I-HOBBY",
    3: "B-TOY",
    4: "I-TOY",
    5: "B-SPORT",
    6: "I-SPORT",
    7: "B-SUBJECT",
    8: "I-SUBJECT"
}

app = Flask(__name__)

def predict_interests(text):
    # Tokenize the input text
    words = word_tokenize(text.lower())
    encoded = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True)
    word_ids = encoded.word_ids(0)

    # Perform prediction
    inputs = {k: v.to("cpu") for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

    predicted_labels = [id2label[pred.item()] for pred in predictions[0]]
    interests = []
    current_interest = []
    current_type = None

    for idx, (word_id, label) in enumerate(zip(word_ids, predicted_labels)):
        if word_id is not None:
            if label.startswith("B-"):
                if current_interest:
                    interests.append({"text": " ".join(current_interest), "type": current_type})
                current_interest = [words[word_id]]
                current_type = label[2:]
            elif label.startswith("I-") and current_interest:
                current_interest.append(words[word_id])
            elif label == "O" and current_interest:
                interests.append({"text": " ".join(current_interest), "type": current_type})
                current_interest = []
                current_type = None

    if current_interest:
        interests.append({"text": " ".join(current_interest), "type": current_type})
    return interests

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    interests = predict_interests(text)
    return jsonify(interests)

if __name__ == "__main__":
    app.run(debug=True)
