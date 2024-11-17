from multiprocessing import freeze_support
import os
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

def main():
    # Your main program logic here
    print("Program is running...")

    # Step 1: Load the Dataset
    dataset = load_dataset("text", data_files={"train": "dataset.txt"})

    # Step 2: Load Pre-trained GPT-2 and Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Set the pad_token (if not already set)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Or tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Step 3: Tokenize the Dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4)

    # Step 4: Prepare Data for Training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False  # Disable Masked Language Modeling
    )

    # Step 5: Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        evaluation_strategy="no",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
    )

    # Step 6: Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Step 7: Train the Model
    trainer.train()

    # Step 8: Save the Fine-Tuned Model
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

    print("Model fine-tuned and saved successfully!")

if __name__ == '__main__':
    freeze_support()  # Optional, mainly for frozen executables
    main()