

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset
import pandas as pd
import torch

# Load the model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the dataset
df = pd.read_csv('example.csv')
dataset = Dataset.from_pandas(df)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["prompt"], text_target=examples["response"], 
                     padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

output = './chatbot_model'

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training arguments
training_args = TrainingArguments(
    output_dir=output,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    weight_decay=0.01,
    save_steps=200,
    save_total_limit=3,
    logging_dir="./logs",
    gradient_accumulation_steps=2,
    report_to="none",
    learning_rate=3e-5,  # 🔹 Reduce el learning rate
    warmup_steps=500,  # 🔹 Añade pasos de calentamiento
    overwrite_output_dir=True,  # 🔹 Reentrena desde cero
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,  # Agregar evaluación para medir el progreso
    tokenizer=tokenizer,  # Asegurar tokenización correcta
    data_collator=data_collator   # Puede ser mejorado con `DataCollatorForSeq2Seq` si es necesario
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./chatbot_model")
tokenizer.save_pretrained("./chatbot_model")

print(trainer.state.log_history)