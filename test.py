from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Ruta donde guardaste el modelo
model_path = "./chatbot_model"

# Cargar modelo y tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Enviar modelo a GPU si está disponible
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

def generate_response(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Probar con una pregunta del dataset
test_prompt = "Do you develop mobile applications?"
response = generate_response(test_prompt)

print("User:", test_prompt)
print("Bot:", response)
