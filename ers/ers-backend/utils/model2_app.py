from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import numpy as np

app = Flask(__name__)

# Define the model checkpoint and label mapping
model_checkpoint = "jsylee/scibert_scivocab_uncased-finetuned-ner"
label_mapping = {
    "DRUG": "Chemical Substances",
    "ADVERSE EFFECT": "Medical Conditions",
    "PER": "Person",
    "LOC": "Location"
}

def convert_to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def process2_text(text):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=5, id2label={0: 'O', 1: 'DRUG', 2: 'DRUG', 3: 'ADVERSE EFFECT', 4: 'ADVERSE EFFECT'})
    ner_pipeline = pipeline(task="ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    
    results = ner_pipeline(text)
    entities = []
    print(entities)
    for result in results:
        entity = {
            "text": result["word"],
            "label": label_mapping.get(result["entity_group"], result["entity_group"])
        }
        if "score" in result:
            entity["score"] = convert_to_native(result["score"])
        if "index" in result:
            entity["index"] = convert_to_native(result["index"])
        
        entities.append(entity)
    return entities

@app.route('/process2_text', methods=['POST'])
def process_text_endpoint():
    data = request.get_json()
    text = data.get('text', '')
    entities = process2_text(text)
    return jsonify(entities)

@app.route('/', methods=['POST', 'GET'])
def index():
    return "ERS-Backend is up!"

if __name__ == "__main__":
    app.run(debug=True)
