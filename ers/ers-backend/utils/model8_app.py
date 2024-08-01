import json
from transformers import pipeline
import numpy as np

ner_pipe = pipeline("ner", model="dslim/bert-base-NER")

def convert_to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def entities_med(text):
    try:
        predictions = ner_pipe(text)
        print(f"Predictions: {predictions}")  
        processed_predictions = []
        for prediction in predictions:
            processed_prediction = {
                'entity': prediction['entity'],
                'type': prediction['word'],
                'confidence': convert_to_native(prediction['score'])
            }
            processed_predictions.append(json.dumps(processed_prediction))
        print(f"Processed Predictions: {processed_predictions}")  # Debugging: Print the processed predictions
        return processed_predictions
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
