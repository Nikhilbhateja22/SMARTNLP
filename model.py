from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, TFAutoModelForSequenceClassification
import torch
import nltk

# Initialize necessary pipelines
summarizer = pipeline("summarization")
sentiment_analyzer = pipeline("sentiment-analysis")
ner_model = pipeline("ner")
chatbot_model_name = "gpt2" 


def load_chatbot_model():
    model = AutoModelForSequenceClassification.from_pretrained(chatbot_model_name)
    tokenizer = AutoTokenizer.from_pretrained(chatbot_model_name)
    return model, tokenizer

model, tokenizer = load_chatbot_model()


def generate_response(user_input):
    inputs = tokenizer.encode(user_input, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.92, temperature=0.85)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def summarize_text(text):
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']


def analyze_sentiment(text):
    sentiment = sentiment_analyzer(text)[0]
    return sentiment['label']  


def extract_entities(text):
    entities = ner_model(text)
    return [{"word": entity['word'], "entity": entity['entity']} for entity in entities]
