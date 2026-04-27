import torch
import torch.nn.functional as F
# (Assume we imported our DecoderOnlyTransformer class and Tokenizer here)

# 1. Configuration (Must match training EXACTLY)
vocab_size = 10000
d_model = 256
context_length = 128

# 2. Initialize the empty shell and load the learned brains
model = DecoderOnlyTransformer(vocab_size, d_model, num_heads=8, num_layers=4, context_length=context_length)
model.load_state_dict(torch.load("my_custom_llm_v1.pt", weights_only=True))

# CRITICAL: Put model in evaluation mode. 
# This disables things like Dropout (which we only want during training)
model.eval()

# Move to GPU if available for faster generation
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 3. The Production Generation Function
@torch.no_grad() # CRITICAL: Tells PyTorch NOT to track gradients, saving massive memory
def generate_text(model, starting_text, max_new_tokens=50):
    # Convert string prompt to tensor of integers
    input_ids = torch.tensor(encode(starting_text), dtype=torch.long).unsqueeze(0).to(device) # Shape: (1, T)
    
    for _ in range(max_new_tokens):
        # Crop the context if it gets longer than what the model can handle
        idx_cond = input_ids[:, -context_length:]
        
        # Forward pass (Get predictions for all tokens)
        logits = model(idx_cond)
        
        # We only care about the prediction for the VERY LAST token
        logits = logits[:, -1, :] # Shape: (1, vocab_size)
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample from the distribution (or just pick the max probability using torch.argmax)
        # Using multinomial allows for a bit of creativity/randomness (temperature)
        idx_next = torch.multinomial(probs, num_samples=1) # Shape: (1, 1)
        
        # Append the predicted token to the running sequence
        input_ids = torch.cat((input_ids, idx_next), dim=1) # Shape: (1, T+1)
        
    # Convert the final list of integers back to a string
    return decode(input_ids[0].tolist())

# --- RUNNING IT ---
prompt = "The AI engineer decided to"
generated_output = generate_text(model, prompt, max_new_tokens=20)
print(generated_output)