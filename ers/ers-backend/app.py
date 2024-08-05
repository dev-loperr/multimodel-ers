from flask import Flask, request, jsonify
from utils.model1_app import process_text
from utils.model2_app import process2_text
from utils.model3_app import extractor
from utils.model4_app import extract_terms
from utils.model5_app import extract_entities
from utils.model6_app import generate_text
from utils.model7_app import extract_biobert_entities
from utils.model8_app import entities_med
import numpy as np

app = Flask(__name__)
app.config['MongoDB_Connection_String'] = ''

def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

@app.route("/process_text", methods=['POST', 'GET'])  # for model1_app, i.e. chemical-disease model
def handle_text():
    text = request.get_json().get("text")
    entities = process_text(text)
    return jsonify(entities)

@app.route('/process2_text', methods=['POST'])  # for model2_app, i.e. jsylee/scibert_scivocab_uncased-finetuned-ner
def process_text_endpoint():
    data = request.get_json()
    text = data.get('text', '')
    entities = process2_text(text)
    native_entities = convert_to_native(entities)
    return jsonify(native_entities)

@app.route('/extract_keyphrases', methods=['POST']) # for model3_app, i.e. keyphrase extraction
def extract_keyphrases_endpoint():
    data = request.get_json()
    text = data.get('text', '')
    keyphrases = extractor.extract_keyphrases(text)
    return jsonify(convert_to_native(keyphrases))

@app.route('/extract_terms', methods=['POST']) # for model4_app, i.e. gliner 
def extract_terms_endpoint():
    data = request.get_json()
    text = data.get('text', '')
    entities = extract_terms(text)
    return jsonify(convert_to_native(entities))

@app.route('/extract_entities', methods=['POST']) # for model5_app, i.e. fran-martinez/scibert_scivocab_cased_ner_jnlpba (not giving the desired output)
def extract_entities_endpoint():
    data = request.get_json()
    text = data.get('text', '')
    entities = extract_entities(text)
    return jsonify(convert_to_native(entities))

@app.route('/generate_text', methods=['POST']) # for model6_app, i.e. gpt2
def generate_text_endpoint():
    data = request.get_json()
    prompt = data.get('prompt', '')
    generated_texts = generate_text(prompt)
    return jsonify(convert_to_native(generated_texts))

@app.route('/extract_biobert_entities', methods=['POST']) # for model7_app, i.e. dmis-lab/biobert-v1.1
def extract_biobert_entities_endpoint():
    data = request.get_json()
    text = data.get('text', '')
    entities = extract_biobert_entities(text)
    return jsonify(convert_to_native(entities))

@app.route('/entities_med', methods=['POST']) # for model8.py, i.e. dslim/bert-base-NER
def entities_med_endpoint():
    data = request.get_json()
    text = data.get('text', '')
    entities = entities_med(text)
    return jsonify(convert_to_native(entities))

@app.route('/', methods=['POST', 'GET'])
def index():
    return "ERS-Backend is up!"

if __name__ == "__main__":
    app.run(debug=True)
