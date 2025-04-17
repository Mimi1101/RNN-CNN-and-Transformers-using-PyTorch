import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import sentencepiece as spm
import math
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction



class LSTM(nn.Module):
    """
    A LSTM language model for next token prediction.
    """
    def __init__(self, embedding_dim=256, hidden_dim=512, vocab_size=10000, num_layers=2, dropout=0.2):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        embedded = self.embedding(input_ids) 
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden
    
    def predict_next_token(self, input_ids, hidden=None, temperature=0.8, top_p =0.9):
        self.eval()
        with torch.no_grad():
            logits, hidden = self.forward(input_ids, hidden)
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
            next_token_id = filtered_indices[0, sampled_token_idx]
            
            return next_token_id.item(), hidden
        
    def generate(self, tokenizer, prompt, max_length=50,  temperature=0.8, device='cuda', top_p = 0.9):
        self.eval()
        #encoding the prompt
        input_ids = tokenizer.encode(prompt, out_type=int)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        generated_ids = []
        hidden = None

        for _ in range(max_length):
            next_token_id, hidden = self.predict_next_token(input_tensor, hidden, temperature, top_p = top_p)

            generated_ids.append(next_token_id)
            #generated token as input for the next step
            input_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

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

def setup_lstm_training():
    model = LSTM()
    criterion = nn.CrossEntropyLoss(ignore_index=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    print("LSTM Training setup done", flush=True)
    return model, criterion, optimizer

def collate_fn(batch):
    batch = [pair for pair in batch if pair[0].nelement() > 0]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0)

    input_batch, target_batch = zip(*batch)
    input_batch = pad_sequence(input_batch, batch_first=True, padding_value=3)
    target_batch = pad_sequence(target_batch, batch_first=True, padding_value=3)
    return input_batch, target_batch
   


def train_LSTM(model, criterion, optimizer, train_loader, val_loader, num_epochs=24, device='cuda'):
    model.to(device)

    training_loss_history = []
    validation_loss_history = []

    print("Starting the LSTM training loop", flush=True)
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # since hidden state is not used for training
            logits, _ = model(inputs)
            #flatten for loss
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

            loss = criterion(logits, targets)
            loss.backward()
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
                logits, _ = model(inputs)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                loss = criterion(logits, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        validation_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}", flush=True)

    torch.save(model.state_dict(), "lstm_model.pt")
    print("LSTM model saved", flush=True)
    return training_loss_history, validation_loss_history

def generate_from_prompt(model, tokenizer, prompt, device='cuda'):
    model.eval()
    model.to(device)
    response = model.generate(tokenizer, prompt, device=device)
    print(f"\nPrompt: {prompt}\nGenerated: {response}\n")

def perplexity(model, test_loader, criterion, device='cuda'):
    model.eval()
    model.to(device)
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = criterion(logits, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    perplexity = math.exp(avg_loss)
    print(f"Test Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
    return avg_loss, perplexity

def bleu(model, tokenizer, testdata, device='cuda', max_samples=100):
    model.eval()
    model.to(device)

    scores = []
    smoother = SmoothingFunction()

    with open(testdata, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break

            data = json.loads(line)
            prompt = data["prompt"]
            reference = data["completion"]

            generated = model.generate(tokenizer, prompt, device=device)
            #splitting the real and generated answer into a list of words
            reference_tokens = [reference.split()]
            generated_tokens = generated.split()
            #calculate blue
            score = sentence_bleu(
                reference_tokens,
                generated_tokens,
                weights=(0.5, 0.5),
                smoothing_function=smoother.method1
            )
            scores.append(score)

            

    avg_bleu = sum(scores) / len(scores)
    print(f"Average bleu score on {len(scores)} samples: {avg_bleu:.4f}")
    return avg_bleu

def plot_loss_curves(training_loss, validation_loss, model_name="LSTM Language Model"):
    epochs = range(1, len(training_loss) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, training_loss, label="Training Loss")
    plt.plot(epochs, validation_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.title(f"Training and Validation Loss Curves for {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig("lstmloss_curve.png")
    plt.close()

def train_main():
    print("Starting LSTM training main loop", flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = TokenizedDataset("./modelsptfiles/train.pt")
    # Split the dataset into 80% training and 20% validation.
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(validation_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    model, criterion, optimizer = setup_lstm_training()
    train_loss, val_loss = train_LSTM(model, criterion, optimizer, train_loader, val_loader, device=device)
    plot_loss_curves(train_loss, val_loss, model_name="LSTM Language Model")

def test_main():
    print("Testing LSTM model on prompt generation")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = LSTM()
    model.load_state_dict(torch.load("./modelsptfiles/lstm_model.pt", map_location=device))

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load("./modelsptfiles/bpe_tokenizer.model")

    # Generate prompts
    generate_from_prompt(model, sp, "Which do you prefer? Dogs or cats?", device=device)
    generate_from_prompt(model, sp, "Messi is the goat, wouldn't you agree?", device=device)
    generate_from_prompt(model, sp, "She was a fairy and", device=device)


    #evaluate perplexity
    test_dataset = TokenizedDataset("./modelsptfiles/test.pt")
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    criterion = nn.CrossEntropyLoss(ignore_index=3)
    print("\nEvaluating Perplexity...")
    perplexity(model, test_loader, criterion, device=device)

    #bleu
    print("\nEvaluating BLEU score...")
    bleu(model, sp, "./data/test.jsonl", device=device, max_samples=100)



if __name__ == "__main__":
    print("Checking if we got in")
    test_main() 













    

    





