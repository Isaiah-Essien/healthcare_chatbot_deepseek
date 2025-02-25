# Fine-Tuned DeepSeek Healthcare Chatbot
![Chatbot Output](images\Screenshot 2025-02-25 114713.png)

## Project Overview
This project involves fine-tuning **DeepSeek-R1-Distill-Llama-8B** using **LoRA (Low-Rank Adaptation)** to create a specialized **healthcare chatbot**. The chatbot is designed to assist users with medical-related inquiries, providing AI-generated health insights based on trained data.

The model has been fine-tuned on **medical question-answering datasets**, making it well-suited for understanding and generating responses to healthcare-related queries.

##  Why Healthcare?
Access to reliable healthcare information is essential, especially in underserved areas where medical professionals may not always be available. AI-driven chatbots can:
- Provide **instant responses** to health-related queries.
- Assist medical professionals by **suggesting possible diagnoses**.
- Enhance **medical education** by summarizing complex conditions in layman's terms.

By fine-tuning **DeepSeek**, this project creates an **efficient, low-resource model** that can deliver **accurate** and **contextually relevant** healthcare responses.



## Links
🔗 **Dataset**: [Kaggle Healthcare QA Dataset](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)  
🔗 **Model**: [Hugging Face Model](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
🔗 **Link to Video recordig**: [Video walkthrough](https://youtu.be/ZehKvFSy8Mo)
---

## 🛠 Installation & Setup
To run this project, ensure you have the following dependencies installed:

```bash
pip install unsloth==2025.2.4
pip install unsloth_zoo==2025.2.3
pip install torch==2.5.1
pip install torchaudio==2.5.1
pip install torchvision==0.20.1
pip install vllm==0.7.2
pip install xformers==0.0.28.post3
pip install xgrammar==0.1.11
pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
pip install trl==0.8.2
```

---

##  Data Preprocessing & Exploration

### **1. Dataset Collection**
- The dataset consists of **medical question-answer pairs** from public sources such as Kaggle and Hugging Face.
- Data is structured into a **user query and AI-generated response format**.

### **2. Cleaning & Preprocessing**
- **Text Normalization:** Lowercasing, removing special characters.
- **Tokenization:** Using the **DeepSeek tokenizer**.
- **Formatting for Chat:** Applying structured templates for dialogue modeling.

---

## Fine-Tuning DeepSeek with LoRA
### **What is LoRA?**
LoRA (Low-Rank Adaptation) is a **parameter-efficient fine-tuning technique** that **reduces computational cost** while maintaining high accuracy. Instead of updating all model weights, **LoRA injects trainable layers into transformer blocks**, allowing for:
**Faster Training**  
**Lower Memory Consumption**  
**Better Adaptability**

### **Fine-Tuning Process**
- Load **DeepSeek-R1-Distill-Llama-8B** model via `FastLanguageModel.from_pretrained()`.
- Apply **chat formatting** using `get_chat_template()`.
- Use **LoRA** to inject trainable parameters.
- **Train** on a medical dataset with **4-bit quantization** for efficiency.

---

## Training Configuration
| Parameter | Value |
|-----------|--------|
| Model | `DeepSeek-R1-Distill-Llama-8B` |
| Max Seq Length | 2048 |
| Dtype | Auto |
| LoRA Enabled |  Yes |
| Quantization | 4-bit |
| Training Method | Supervised Fine-Tuning (SFT) |
| Max New Tokens | 64 |
| Temperature | 0.6 |
| Min Probability (min_p) | 0.1 |

---

## Model Evaluation
After fine-tuning, the model was evaluated using:

### **1. Perplexity (PPL)**
Perplexity measures how well the model predicts text. **Lower is better.**
```python
perplexity = torch.exp(loss)
```

### **2. BLEU Score**
BLEU evaluates text similarity between AI-generated and human responses. **Higher is better.**
```python
bleu_score = sentence_bleu(reference, generated)
```

---

##  How to Use the Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Model
finetuned_model_path = "path_to_the_model"
tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(finetuned_model_path, local_files_only=True).to("cuda")

# Chat with Model
user_input = "What are the symptoms of diabetes?"
inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Chatbot:", response)
```
- ensure you install all dependencies needed for this to work and can be found in the notebook
- Ensure you have a GPU and enough RAM
---

## Screenshots
![Training Log](images\image.png)

---

## Summary
**Fine-tuned DeepSeek for healthcare chat**  
**Efficient LoRA-based adaptation**  
**Preprocessed medical QA data**  
**Optimized for inference with 4-bit quantization**  

This chatbot is a step towards **AI-driven healthcare solutions**, making medical knowledge **accessible and conversational** for everyone. 

## Author:
- This code belongs to Isaiah Edem Essien and must not be reproduced without consent.

## License:
- MIT(Hugging Face License)

