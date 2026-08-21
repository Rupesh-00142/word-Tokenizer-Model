import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import faiss 
import json
import os
import re
import random
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer

batch_size = 16
block_size = 64
max_iters = 1500
eval_interval = 300
learning_rate = 3e-4
repetition_penalty = 1.0

d_model = 192
n_heads = 4
n_layers = 4
dropout = 0.1

torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def clean_and_duplicate(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    lines = text.split(".")
    lines = [l.strip() for l in lines if len(l.strip())>20]
    lines = list(set(lines))
    random.shuffle(lines)
    return ".".join(lines)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# tokenizer banate hain
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# words split karega
tokenizer.pre_tokenizer = Whitespace()

# trainer define karo
trainer = BpeTrainer(
    vocab_size=5000,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"]
)

if os.path.exists("tokenizer.json"):
    os.remove("tokenizer.json")

tokenizer.train(["input.txt"], trainer)
tokenizer.save("tokenizer.json")

tokenizer = Tokenizer.from_file("tokenizer.json")

print(tokenizer.token_to_id("[UNK]"))

#stoi = {ch:i for i,ch in enumerate(chars)}
#itos = {i:ch for ch, i in stoi.items()}


def encode(text):
    return tokenizer.encode(text if text else " ").ids

def decode(tokens):
    return tokenizer.decode(tokens)

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

vocab_size = tokenizer.get_vocab_size()

responses = {
    "greeting": "Hello! How can I help you?",
    "timing_query": "College timing is 10 AM to 5 PM.",
    "fees_query": "Fees depend on the course.",
    "goodbye": "Goodbye! Have a nice day."
}

def get_intent(text):
    text = text.lower()

    if "hello" in text or "hi" in text:
        return "greeting"
    elif "timing" in text:
        return "timing_query"
    elif "fees" in text:
        return "fees_query"
    elif "bye" in text:
        return "goodbye"
    else:
        return "unknown"

def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])

    return x.to(device),y.to(device)

class causalselfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = d_model//n_heads
        self.qkv = nn.Linear(d_model, d_model*3)
        self.fc = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        qkv = qkv.reshape(B,T,n_heads, 3*self.head_dim)
        q,k,v = qkv.chunk(3,dim=-1)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        att = (q @ k.transpose(-2,-1))/math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:T,:T]==0, float('-inf'))
        att = F.softmax(att,dim=-1)
        out = att@v
        out = out.transpose(1,2).contiguous().view(B,T,C)
        return self.fc(out)
    
class feedforward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model,d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model)
        )
    def forward(self,x):
        return self.net(x)
    
class block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = causalselfAttention()
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = feedforward()
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self,x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size,d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[block()for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model,vocab_size)

    def forward(self, idx, target=None):
        B,T = idx.size()
        token_emb = self.token_emb(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(pos)

        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)


        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1,vocab_size), target.view(-1))

        return logits, loss
    def generate (self, idx, max_new_tokens, temperature= 1.1, top_k=None):

        for _ in range(max_new_tokens):
            idx_cond = idx[:,-block_size:]
            logits, _  = self(idx_cond)
            logits = logits[:,-1,:]/temperature

            for token in set(idx[0].tolist()):
                logits[0, token] /= repetition_penalty

            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            idx =torch.cat((idx, next_token), dim=1)
        return idx
    
model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

print("total parameters:", sum(p.numel() for p in model.parameters())/1e6,"Million")
print(device)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses ={}
    for split in ['train','val']:
        total_loss = 0
        for _ in range(20):
            x,y = get_batch(split)
            _,loss = model(x,y)
            total_loss +=loss.item()
        losses[split] = total_loss/20
    model.train()
    return losses

best_val_loss = float('inf')

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step{iter}| Train{losses['train']:.4f}| val{losses['val']:.4f}")
        if losses['val']<best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(),"gpt_model_1.pth")
            print("Model saved!")

    x,y = get_batch('train')
    logits, loss = model(x,y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    optimizer.step()


torch.save(model.state_dict(),"gpt_model.pth")
#model.load_state_dict(torch.load("gpt.pth"))
print("T.. completed...")

model.load_state_dict(torch.load("gpt_model_1.pth"))
model.eval()

print(loss.item())

while True:
    user_input = input("You: ")

    intent = get_intent(user_input)
    
    if user_input=="exit":
        break
    prompt=f"User:{user_input} AI:"
   # if intent in responses:
    #    print("AI:", responses[intent])

        #if intent == "goodbye":
           # break 

    #else:
    context = torch.tensor([encode(user_input)], dtype=torch.long).to(device)
    output = model.generate(context, max_new_tokens=50, temperature=0.8, top_k=40)
    new_tokens = output[0][context.shape[1]:]
    result = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
    if len(new_tokens) == 0:
       result = "I am still learning. Please try another question."

        #print("AI:", result)
        #print("Generated tokens:", new_tokens.tolist())
    print("AI:", result) 
