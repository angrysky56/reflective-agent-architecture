import sys

from transformers import AutoTokenizer

try:
    print(
        "Attempting to load gpt2 tokenizer with revision 6c0e6080953db56375760c0471a8c5f2929baf11..."
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2", revision="6c0e6080953db56375760c0471a8c5f2929baf11"
    )
    print("Success! Tokenizer loaded.")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Pad token: {tokenizer.pad_token}")
    print(f"EOS token: {tokenizer.eos_token}")
except Exception as e:
    print(f"FAILED to load tokenizer: {e}")
    sys.exit(1)
