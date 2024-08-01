from transformers import pipeline
import numpy as np

# Load the BioBERT model for NER
ner_model = pipeline("ner", model="dmis-lab/biobert-v1.1")

def convert_to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def extract_biobert_entities(text):
    predictions = ner_model(text)
    processed_predictions = []
    for prediction in predictions:
        processed_prediction = {
            'entity': prediction['entity'],
            'score': convert_to_native(prediction['score']),
            'index': prediction['index'],
            'start': prediction['start'],
            'end': prediction['end'],
            'word': prediction['word']
        }
        processed_predictions.append(processed_prediction)
    return processed_predictions
