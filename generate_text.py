from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model")

# Generate text
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text using the model
output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=1,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
)

# Decode and print generated text
print(tokenizer.decode(output[0], skip_special_tokens=True))
