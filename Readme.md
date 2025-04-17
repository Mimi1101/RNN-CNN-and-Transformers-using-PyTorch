# Language Modeling with RNN, LSTM, and Transformers using PyTorch

This repo has three language models built using PyTorch:
- Vanilla RNN
- An LSTM
- A decoder-only Transformer

The models are already trained. After training, you can test them using prompts. You can also check out the BLEU and Perplexity scores.

---

## Setup

First, install the Python packages you'll need:

```bash
pip install torch sentencepiece tqdm matplotlib nltk


## How to Run

Each model has a `test_main()` function. The models are already trained — you just run the scripts:

```bash
python lstm.py
python rnn.py
python transformers.py


Each script will:
-Load the trained model
-Generate sample outputs from prompts
-Calculate Perplexity
-Calculate BLEU Score

## Want to Change the Prompt?

You can change the prompts by editing the test_main() function in each .py file.

generate_from_prompt(model, tokenizer, "Your prompt",device=device)







