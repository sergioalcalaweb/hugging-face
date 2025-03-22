import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
# from torch.utils.data import DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

dataset = load_dataset("csv", data_files="example.csv")  # O usa "csv" si es CSV
dataset = dataset["train"].train_test_split(test_size=0.2)

# print(tokenized_datasets["train"])


# train_dataloader = DataLoader(dataset["train"], batch_size=4, shuffle=True, pin_memory=False, num_workers=4)
# eval_dataloader = DataLoader(dataset["test"], batch_size=4, pin_memory=False, num_workers=4)

training_args = TrainingArguments(
    output_dir="./qa_model",
    eval_strategy="epoch",
    learning_rate=5e-5,  # Aprendizaje un poco más alto para converger más rápido
    per_device_train_batch_size=4,  # Apple Silicon es más eficiente con batches pequeños
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    optim="adamw_torch_fused",  # Optimizador más rápido en MPS
    report_to="none",  # Evita problemas con logging
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()

model.save_pretrained("./chatbot_model")
tokenizer.save_pretrained("./chatbot_model")