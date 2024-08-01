from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model_checkpoint = 'openai-community/gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

text_generator = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer
)

def generate_text(prompt, max_length=30, num_return_sequences=5):
    generated_texts = text_generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
    return generated_texts
