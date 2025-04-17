import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sentencepiece as spm


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even positions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd positions

        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class TransformersLanguageModel(nn.Module):
    def __init__(self, vocab_size=10000, d_model = 512, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.2, max_len=512):
        super(TransformersLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)  

        #defining one decoder layer
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        #stacking them
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, input_ids):
        #embed tokens
        embeddings = self.embedding(input_ids) * math.sqrt(self.d_model)
        #positional encodings
        embeddings = self.pos_encoder(embeddings)
        #masking tokens
        seq_len = input_ids.size(1)
        mask = self.generate_square_subsequent_mask(seq_len).to(input_ids.device)
        pad_mask = (input_ids==3)
        

        output = self.transformer_encoder(embeddings, mask=mask, src_key_padding_mask=pad_mask)
        logits = self.fc_out(output)
        return logits

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    
    def predict_next_token(self, input_ids, temperature=0.8, top_p=0.9):
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids)
            #now we grab the last token of the sequence
            logits = logits[:, -1, :]
            #this is where like temeprature would come into effect but mine stays the same because its 1
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            #sort tokens from highest to lowest probability
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            #cumulative sum of the sorted probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            #the smallest set where cumulative prob exceeds top_p
            cutoff = cumulative_probs > top_p
            if torch.any(cutoff):
                cutoff_index = torch.min(torch.where(cutoff)[1]) + 1
            else:
                cutoff_index = sorted_probs.shape[-1]
           

            # Slice to get the nucleus set
            filtered_probs = sorted_probs[:, :cutoff_index]
            filtered_indices = sorted_indices[:, :cutoff_index]

            # Re-normalize
            filtered_probs = filtered_probs / torch.sum(filtered_probs, dim=-1, keepdim=True)

            sampled_token_idx = torch.multinomial(filtered_probs, num_samples=1)
            next_token_id = filtered_indices[0, sampled_token_idx].item()
            
            return next_token_id
        

    def generate(self, tokenizer, prompt, max_length=50, temperature=0.8, device='cuda', top_p=0.9):
        self.eval()
        input_ids = tokenizer.encode(prompt, out_type=int)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        generated_ids = []
        
        for _ in range(max_length):
            next_token_id = self.predict_next_token(input_tensor, temperature=temperature, top_p=top_p)
            generated_ids.append(next_token_id)
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=device)], dim=1)
        
        return tokenizer.decode(generated_ids, out_type=str)




#to handle sequences and load the .pt files 
class TokenizedDataset(Dataset):
    
    def __init__(self, filepath):
        self.sequences = torch.load(filepath)

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        sequence = self.sequences[index]
        if len(sequence) < 2:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        #so now we split sequences into input and target
        #like[5,10,15] - [10,15,20]
        return sequence[:-1], sequence[1:]
    
def perplexity(model, test_loader, criterion, device='cuda'):
    model.eval()
    model.to(device)
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = criterion(logits, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    perplexity = math.exp(avg_loss)
    print(f"Test Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
    return avg_loss, perplexity

def bleu(model, tokenizer, testdata_path, device='cuda', max_samples=100):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import json

    model.eval()
    model.to(device)

    scores = []
    smoother = SmoothingFunction()

    with open(testdata_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break

            data = json.loads(line)
            prompt = data["prompt"]
            reference = data["completion"]

            generated = model.generate(tokenizer, prompt, device=device)
            reference_tokens = [reference.split()]
            generated_tokens = generated.split()

            score = sentence_bleu(
                reference_tokens,
                generated_tokens,
                weights=(0.5, 0.5),
                smoothing_function=smoother.method1
            )
            scores.append(score)

    avg_bleu = sum(scores) / len(scores)
    print(f"\nAverage BLEU score on {len(scores)} samples: {avg_bleu:.4f}")
    return avg_bleu


def setup_transformer_training():
    model = TransformersLanguageModel()
    criterion = nn.CrossEntropyLoss(ignore_index=3, label_smoothing = 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)  
    print("Transformers Training setup done", flush=True)
    return model, criterion, optimizer, scheduler

def collate_fn(batch):
    batch = [pair for pair in batch if pair[0].nelement() > 0]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0)

    input_batch, target_batch = zip(*batch)
    input_batch = pad_sequence(input_batch, batch_first=True, padding_value=3)
    target_batch = pad_sequence(target_batch, batch_first=True, padding_value=3)
    return input_batch, target_batch

def train_transformer(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=24, device='cuda'):
    model.to(device)

    training_loss_history = []
    validation_loss_history = []

    print("Starting the Transformers training loop", flush=True)
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # since hidden state is not used for training
            logits = model(inputs)
            #flatten for loss
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        training_loss_history.append(avg_train_loss)

        #validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                loss = criterion(logits, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        validation_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}", flush=True)
        scheduler.step(avg_val_loss)

    torch.save(model.state_dict(), "transformer_model.pt")
    print("Transformer model saved", flush=True)
    return training_loss_history, validation_loss_history

def load_tokenizer(model_path='./modelsptfiles/bpe_tokenizer.model'):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp

def generate_from_prompt(model, tokenizer, prompt, device='cuda', top_p=0.9):
    model.eval()
    model.to(device)
    response = model.generate(tokenizer, prompt, device=device, top_p=top_p)
    print(f"\nPrompt: {prompt}\nGenerated: {response}\n")

def plot_loss_curves(training_loss, validation_loss, model_name="Transformers Language Model"):
    epochs = range(1, len(training_loss) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, training_loss, label="Training Loss")
    plt.plot(epochs, validation_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.title(f"Training and Validation Loss Curves for {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig("transformerloss_curve.png")
    plt.close()


def train_main():
    print("Starting Transformers training main loop", flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = TokenizedDataset("./modelsptfiles/train.pt")
    # Split the dataset into 80% training and 20% validation.
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    model, criterion, optimizer, scheduler = setup_transformer_training()
    train_loss, val_loss = train_transformer(model, criterion, optimizer, scheduler, train_loader, val_loader, device=device,)
    plot_loss_curves(train_loss, val_loss, model_name="Transformers Language Model")

def test_main():
    print("Starting Transformer evaluation loop", flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = load_tokenizer()
    model = TransformersLanguageModel()
    model.load_state_dict(torch.load("./modelsptfiles/transformer_model.pt", map_location=device))
    model.to(device)

    # Load test dataset
    test_dataset = TokenizedDataset("./modelsptfiles/test.pt")
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

    # Compute perplexity
    criterion = nn.CrossEntropyLoss(ignore_index=3)
    print("\nPerplexity ")
    perplexity(model, test_loader, criterion, device=device)

    # Generate some outputs
    print("\nSample Generations")
    generate_from_prompt(model, tokenizer, "Which do you prefer? Dogs or cats?", device=device)
    generate_from_prompt(model, tokenizer, "Messi is the goat, wouldn't you agree?", device=device)
    generate_from_prompt(model, tokenizer, "She was a fairy and", device=device)

    # Compute BLEU score
    print("\n BLEU Score")
    bleu(model, tokenizer, "./data/test.jsonl", device=device, max_samples=100)

    

if __name__ == "__main__":
    print("Checking if we got in")
    test_main() 











