import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import torch.nn.functional as F
from model import SimCSEModel
from dataset import SimCSEDataset
import time
from datetime import timedelta


MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 32
TEMPERATURE = 0.05
CHECKPOINT_PATH = './checkpoint'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SimCSEModel(MODEL_NAME).to(device)

raw_dataset = load_dataset("daily_dialog")
dialogs = raw_dataset['train']['dialog']
sentences = []
for dialog in dialogs:
    sentences.extend(dialog)

dataset = SimCSEDataset(sentences, tokenizer, max_len=MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

os.makedirs(CHECKPOINT_PATH, exist_ok=True)

best_loss = float('inf')
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    epoch_start = time.time()

    #TODO 
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask = batch 
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    
        token_type_ids = torch.zeros_like(input_ids).to(device)
    
        input_ids = torch.cat([input_ids, input_ids], dim=0)
        attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
        token_type_ids = torch.cat([token_type_ids, token_type_ids], dim=0)
    
        embeddings = model(input_ids, attention_mask, token_type_ids)  # [2N, hidden_size]
    
        embeddings = F.normalize(embeddings, dim=1)
        
        similarity_matrix = embeddings @ embeddings.T  # [2N, 2N]
        labels = torch.arange(BATCH_SIZE).to(device)
        labels = torch.cat([labels + BATCH_SIZE, labels], dim=0)
        similarity_matrix = similarity_matrix / TEMPERATURE
    
        loss = F.cross_entropy(similarity_matrix, labels)
    
        loss.backward()
        optimizer.step()
    
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.4f}. "
          f"Time taken: {timedelta(seconds=int(epoch_time))}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, "best_model.pth"))
        print(f"Checkpoint saved at epoch {epoch + 1} with loss {best_loss:.4f}")

total_time = time.time() - start_time
print(f"Training finished. Total time: {timedelta(seconds=int(total_time))}")
