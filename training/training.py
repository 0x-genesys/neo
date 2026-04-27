import torch
import torch.nn as nn
import torch.optim as optim

# 1. Initialize the model and the optimizer
# (Assuming 'DecoderOnlyTransformer' is the class we wrote earlier)
vocab_size = 10000
model = DecoderOnlyTransformer(vocab_size=vocab_size, d_model=256, num_heads=8, num_layers=4, context_length=128)

# AdamW is the industry standard for training LLMs
optimizer = optim.AdamW(model.parameters(), lr=3e-4) 

# Standard loss function for classification/language modeling
criterion = nn.CrossEntropyLoss()

# --- THE TRAINING LOOP ---
epochs = 10
for epoch in range(epochs):
    # Imagine we have a function that gets a batch of data
    # inputs: (Batch, Time)
    # targets: (Batch, Time) - The inputs shifted one position to the right
    inputs, targets = get_batch() 
    
    # Step 1: Forward Pass
    # We feed the data in, and the model makes its predictions
    logits = model(inputs) # Shape: (Batch, Time, Vocab_Size)
    
    # PyTorch's CrossEntropy expects 2D inputs: (Batch * Time, Vocab_Size)
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    targets = targets.view(B * T)
    
    # Step 2: Calculate the Loss (How wrong is the model?)
    loss = criterion(logits, targets)
    
    # Step 3: Zero the Gradients
    # CRITICAL: PyTorch accumulates gradients by default. We must clear 
    # the old gradients from the previous step before calculating new ones.
    optimizer.zero_grad(set_to_none=True) 
    
    # Step 4: BACKPROPAGATION
    # This single line tells PyTorch to traverse the computational graph backwards
    # and calculate the gradients (dLoss/dWeight) for every parameter in the model.
    loss.backward() 
    
    # Step 5: Update the Weights
    # The optimizer looks at the gradients calculated in step 4 and adjusts 
    # the model's weights to make the loss smaller next time.
    optimizer.step() 

    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    
# --- SAVING THE MODEL ---

# 1. We extract the state_dict (the dictionary of weights)
weights_dictionary = model.state_dict()

# 2. We save it to a file on the hard drive
torch.save(weights_dictionary, "my_custom_llm_v1.pt")
print("Model successfully saved to disk!")

   