from transformers import pipeline
import numpy as np

model_checkpoint = "ghadeermobasher/BC5CDR-Chemical-Disease-balanced-scibert_scivocab_cased"
label_mapping = {
    "CHEMICAL": "Chemical Substances",
    "DISEASE": "Medical Conditions",
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

def process_text(text):
    ner_pipeline = pipeline(task="ner", model=model_checkpoint)
    results = ner_pipeline(text)
    entities = []
    for result in results:
        print(result)
        entity = {
            "text": "",
            "label": label_mapping.get(result["entity"], result["entity"])
        }
        if "score" in result:
            entity["score"] = convert_to_native(result["score"])
        if "index" in result:
            entity["index"] = convert_to_native(result["index"])
        
        entities.append(entity)
    return entities