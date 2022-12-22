from transformers import BertForMaskedLM, BertTokenizerFast
import numpy as np
import torch

model = BertForMaskedLM.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
text = ["hello I want to eat some [MASK] meat today. It's thanksgiving [MASK] all!", "yo yo what's up"]

# tokenize text and pass into model
tokens = tokenizer(text, return_tensors="pt", padding=True)
input_ids = tokens["input_ids"]
token_logits = model(input_ids).logits

# Find the location of [MASK] and extract its logits
mask_token_indices = torch.argwhere(input_ids == tokenizer.mask_token_id)[:, 1]
mask_replacements = []
for mask_token_index in mask_token_indices:
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_5_tokens = torch.argsort(-mask_token_logits)[:5].tolist()
    mask_replacements.append(top_5_tokens)

# Print outputs
for replacement_tokens in list(zip(*mask_replacements)):
    copy_text = str(text)
    for i, token in enumerate(replacement_tokens):
        mask_index = copy_text.index("[MASK]")
        copy_text = copy_text[:mask_index] + tokenizer.decode([token]) + copy_text[mask_index + 6 :]
    print(f">>> {copy_text}")
