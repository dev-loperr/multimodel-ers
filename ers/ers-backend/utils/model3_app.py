from transformers import (
    pipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
import numpy as np

def convert_to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class KeyphraseExtractionPipeline:
    def __init__(self, model):
        self.model = AutoModelForTokenClassification.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.pipeline = pipeline(
            task="ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"
        )

    def extract_keyphrases(self, text):
        results = self.pipeline(text)
        processed_results = []
        for result in results:
            processed_results.append({
                'entity_group': result.get('entity_group', 'UNKNOWN'),
                'score': convert_to_native(result.get('score', 0.0)),
                'word': result.get('word', ''),
                'start': result.get('start', 0),
                'end': result.get('end', 0)
            })
        return processed_results
    
# Define keyphrase extraction pipeline
model_name = "ml6team/keyphrase-extraction-kbir-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)
