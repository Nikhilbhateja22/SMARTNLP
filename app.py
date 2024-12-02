from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from pymongo import MongoClient

app = Flask(__name__)

client = MongoClient("mongodb://localhost:27017/")
db = client.smartnlp
conversations = db.conversations

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

summarizer = pipeline("summarization")
sentiment_analyzer = pipeline("sentiment-analysis")
ner_recognizer = pipeline("ner", grouped_entities=True)

@app.route("/")
def home():
    return jsonify({"message": "Welcome to SMARTNLP API!"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    response = chat_pipeline(user_input, max_length=150, num_return_sequences=1)[0]["generated_text"]
    conversations.insert_one({"user": user_input, "bot": response})
    return jsonify({"response": response})

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json
    text = data.get("text", "")
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)[0]["summary_text"]
    return jsonify({"summary": summary})

@app.route("/sentiment", methods=["POST"])
def sentiment():
    data = request.json
    text = data.get("text", "")
    sentiment = sentiment_analyzer(text)[0]
    return jsonify({"sentiment": sentiment})

@app.route("/ner", methods=["POST"])
def ner():
    data = request.json
    text = data.get("text", "")
    entities = ner_recognizer(text)
    return jsonify({"entities": entities})

@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    text = data.get("text", "")
    target_language = data.get("language", "es")
    translator = pipeline("translation_en_to_" + target_language)
    translation = translator(text)[0]["translation_text"]
    return jsonify({"translation": translation})

@app.route("/paraphrase", methods=["POST"])
def paraphrase():
    data = request.json
    text = data.get("text", "")
    paraphrase_model = pipeline("text2text-generation", model="t5-small")
    paraphrased = paraphrase_model(f"paraphrase: {text}", max_length=50, num_return_sequences=1)[0]["generated_text"]
    return jsonify({"paraphrased": paraphrased})

if __name__ == "__main__":
    app.run(debug=True)
