# This exists as an additional debugging tool so that I can step into the pytorch internals. (for some reason, debugger for Jupyter notebooks is skipping the internals)
import torch

emb = torch.nn.Embedding(num_embeddings=5, embedding_dim=3, padding_idx=1)
emb_loaded = torch.nn.Embedding(num_embeddings=5, embedding_dim=3, padding_idx=1)

assert (emb.weight != emb_loaded.weight).any()
print(emb.weight == emb_loaded.weight)

save_torch_path = "emb.pt"
torch.save(emb.state_dict(), save_torch_path)

save_torch_path = "emb.pt"
emb_loaded.load_state_dict(torch.load(save_torch_path))

assert (emb.weight == emb_loaded.weight).all()
print(emb.weight == emb_loaded.weight)
