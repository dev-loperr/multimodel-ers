from gliner import GLiNER
import numpy as np

model_checkpoint = "EmergentMethods/gliner_medium_news-v2.1"
model = GLiNER.from_pretrained(model_checkpoint)

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

def gliner(text):
    labels = ["CHEMICAL", "DISEASE", "PERSON", "LOCATION", "DATE", "ORGANIZATION", "MISC", "DRUG", "ADVERSE EFFECT"]  
    results = model.predict_entities(text, labels)
    
    # Create a dictionary to store entities by their groups
    entity_groups = {}
    for result in results:
        label = result.get('label', 'UNKNOWN')
        entity = {
            'text': result.get('text', ''),
            'score': convert_to_native(result.get('score', 0.0)),
            'start': result.get('start', 0),
            'end': result.get('end', 0)
        }
        if label not in entity_groups:
            entity_groups[label] = []
        entity_groups[label].append(entity)
    
    native_entity_groups = convert_to_native(entity_groups)
    
    output = {
        "entity_groups": list(native_entity_groups.keys()),
        "entities": native_entity_groups
    }
    
    return output
