from gliner import GLiNER
import numpy as np

model_checkpoint = "EmergentMethods/gliner_medium_news-v2.1"
model = GLiNER.from_pretrained(model_checkpoint)

def convert_to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def extract_terms(text, labels=["CHEMICAL", "DISEASE"]):
    entities = model.predict_entities(text, labels)
    return entities
