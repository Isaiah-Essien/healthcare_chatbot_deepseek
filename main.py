'''This script exist incase someone wants to quickly try out the model on a GPU'''

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu

# Load the fine-tuned model and tokenizer from the saved directory
finetuned_model_path = "my model path"

# Load tokenizer and model from local directory
tokenizer = AutoTokenizer.from_pretrained(
    finetuned_model_path, local_files_only=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    finetuned_model_path, local_files_only=False, trust_remote_code=True).to("cuda")

# Function to calculate Perplexity


def calculate_perplexity(model, tokenizer, text, device='cuda'):
    model.eval()  # Set model to evaluation mode
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)),
                               inputs.input_ids.view(-1), ignore_index=tokenizer.pad_token_id)

    perplexity = torch.exp(loss)  # e^(cross-entropy loss)
    return perplexity.item()

# Function to calculate BLEU Score


def calculate_bleu(reference, generated):
    reference = [reference.split()]  # BLEU expects list of lists
    generated = generated.split()
    bleu_score = sentence_bleu(reference, generated)
    return bleu_score


# Set PAD token to be the same as EOS token to avoid tokenization issues
tokenizer.pad_token = tokenizer.eos_token

while True:
    # Get user input
    user_input = input("Enter your message (or type 'quit' to exit): ")

    # Break the loop if the user types 'quit'
    if user_input.lower() == "quit":
        print("Exiting...")
        break

    # Tokenize the user input
    inputs = tokenizer(user_input, return_tensors="pt",
                       padding=True, truncation=True).to("cuda")

    attention_mask = inputs["input_ids"] != tokenizer.pad_token_id

    # Generate the output
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=attention_mask,
        max_new_tokens=64,
        use_cache=True,  # use cache for faster token generation
        temperature=0.6,  # controls randomness in response
        min_p=0.1  # Sets the minimum probability threshold for token selection
    )

    # Decode the generated tokens into human-readable text
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Model Response:", text)

    # Calculate Perplexity
    perplexity = calculate_perplexity(model, tokenizer, text)
    print(f"Perplexity Score: {perplexity}")

    # Calculate BLEU Score
    bleu_score = calculate_bleu(user_input, text)
    print(f"BLEU Score: {bleu_score}")
