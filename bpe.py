import json
import sentencepiece as spm
import torch


class bpetokenizer:


    def train_BPE():
        spm.SentencePieceTrainer.train(
        input = 'corpus.txt',       
        model_prefix='bpe_tokenizer',    
        vocab_size=10000,                 
        model_type='bpe',                 
        user_defined_symbols=['<bos>', '<eos>', '<pad>'],  
        bos_id =1, eos_id=2, pad_id=3   
    )
        
    def tokenize_data(input_path, output_path, model_path = 'bpe_tokenizer.model'):
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        tokenized_sequences = []
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    prompt = data["prompt"]
                    completion = data["completion"]

                    #combining and encoding
                    combined = f"{prompt} {completion}"
                    token_ids = sp.encode(combined, out_type=int)


                    tokenized_sequences.append(torch.tensor(token_ids, dtype=torch.long))

            torch.save(tokenized_sequences, output_path)
            print(f"Saved tokenized data!")
        except Exception as e:
            print(f"Error processing file {input_path}: {e}")


if __name__ == "__main__":
    lala = bpetokenizer
    # #training the bpe model
    lala.train_BPE()
    # # #tokenizing train and test
    lala.tokenize_data("./data/test.jsonl", "train.pt")
    lala.tokenize_data("./data/test.jsonl", "test.pt")
   






    


