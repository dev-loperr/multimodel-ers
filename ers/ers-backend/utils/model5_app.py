from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

model_checkpoint = 'fran-martinez/scibert_scivocab_cased_ner_jnlpba'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

# Update tokenizer configuration
tokenizer.model_max_length = 512  
tokenizer.truncation = True       

nlp_ner = pipeline(
    task="ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

def extract_entities(text):
    lines = text.splitlines()
    entities = []
    for line in lines:
        result = nlp_ner(line)
        print(f"Result for line '{line}': {result}")  # Debug print
        for entity in result:
            entities.append(entity)
    return entities
