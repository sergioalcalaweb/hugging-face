

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd

# Load the model and tokenizer
model_name = "facebook/blenderbot-3B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the dataset
df = pd.read_csv('example.csv')
dataset = Dataset.from_pandas(df)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["prompt"], text_target=examples["response"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

output = './chatbot_model'

# Training arguments
training_args = TrainingArguments(
    output_dir=output,
    eval_strategy="no", 
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./chatbot_model")
tokenizer.save_pretrained("./chatbot_model")