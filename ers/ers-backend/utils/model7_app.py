from transformers import pipeline
import numpy as np
# TODO
# Load the BioBERT model for NER
ner_model = pipeline("ner", model="dmis-lab/biobert-v1.1")

label_mapping = {
    "B-PER": "Person",
    "I-PER": "Person",
    "B-LOC": "Location",
    "I-LOC": "Location",
    "B-ORG": "Organization",
    "I-ORG": "Organization",
    "B-MISC": "Miscellaneous",
    "I-MISC": "Miscellaneous",
    "B-DISEASE": "Disease",
    "I-DISEASE": "Disease",
    "B-DRUG": "Drug",
    "I-DRUG": "Drug",
    "B-AE": "Adverse Effect",
    "I-AE": "Adverse Effect"
}

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

def extract_biobert_entities(text):
    predictions = ner_model(text)
    entity_groups = {}
    for prediction in predictions:
        label = label_mapping.get(prediction['entity'], prediction['entity'])
        entity = {
            'entity': label,
            'score': convert_to_native(prediction['score']),
            'index': prediction['index'],
            'start': prediction['start'],
            'end': prediction['end'],
            'word': prediction['word']
        }
        if label not in entity_groups:
            entity_groups[label] = []
        entity_groups[label].append(entity)
    
    # Convert the dictionary to native Python types
    native_entity_groups = convert_to_native(entity_groups)
    
    # Create the final output including entity groups
    output = {
        "entity_groups": list(native_entity_groups.keys()),
        "entities": native_entity_groups
    }
    
    return output