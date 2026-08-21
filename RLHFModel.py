import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import sentencepiece as spm


# ==========================================================
# CONFIG
# ==========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

block_size = 64
d_model = 128
n_heads = 4
n_layers = 4
learning_rate = 3e-4

# ==========================================================
# LOAD DATA
# ==========================================================
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

spm.SentencePieceTrainer.train(input = 'input.txt',model_prefix = 'gpt_tokenizer', vocab_size = 2000, model_type = 'bpe')

sp = spm.SentencePieceProcessor()
sp.load("gpt_tokenozer.model")
vocab_size = sp.get_piece_size()
def encode(text):
    return sp.encode(text)

def decode(tokens):
    return sp.decode(tokens)
#chars = sorted(list(set(text)))
#vocab_size = len(chars)

#stoi = {ch:i for i,ch in enumerate(chars)}
#itos = {i:ch for ch,i in stoi.items()}

#def encode(s):
  #  return [stoi.get(c,0) for c in s]

#def decode(l):
#    return "".join([itos[i] for i in l])

#encode_data =[]
#for line in text.split("\n"):
 #   encode_data.extend(encode(line))

data = torch.tensor(encode, dtype=torch.long)

# ==========================================================
# LoRA LAYER
# ==========================================================

class LoRALinear(nn.Module):
    def __init__(self, linear, rank=4):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        
        in_f = linear.in_features
        out_f = linear.out_features
        
        self.A = nn.Parameter(torch.randn(in_f, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, out_f) * 0.01)

    def forward(self, x):
        return self.linear(x) + x @ self.A @ self.B

# ==========================================================
# TRANSFORMER BLOCK
# ==========================================================

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out,_ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x

# ==========================================================
# GPT MODEL
# ==========================================================

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)

        # LoRA applied to output head
        self.head = LoRALinear(nn.Linear(d_model, vocab_size))

    def forward(self, idx):
        B,T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln(x)
        return self.head(x)

    def generate(self, idx, max_new=100, temperature=0.7):
        for _ in range(max_new):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits/temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs,1)
            idx = torch.cat([idx,next_token],dim=1)
            if next_token.item() == stoi["<END>"]:
                break
        return idx

# ==========================================================
# REWARD MODEL (RLHF Concept)
# ==========================================================

class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, hidden):
        return self.fc(hidden.mean(dim=1))

# ==========================================================
# TRAINING FUNCTIONS
# ==========================================================

def get_batch():
    ix = torch.randint(0, len(data)-block_size, (16,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x,y

def pretrain(model, optimizer, steps=2000):
    model.train()
    for step in range(steps):
        x,y = get_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1,vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%100==0:
            print("Pretrain step:",step,"loss:",loss.item())

def freeze_base(model):
    for name,param in model.named_parameters():
        if "A" not in name and "B" not in name:
            param.requires_grad=False

def lora_finetune(model, optimizer, steps=400):
    model.train()
    for step in range(steps):
        x,y = get_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1,vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%100==0:
            print("LoRA step:",step,"loss:",loss.item())

def evaluate_perplexity(model):
    model.eval()
    total_loss=0
    total_steps=0
    for i in range(0,len(data)-block_size,block_size):
        x=data[i:i+block_size].unsqueeze(0).to(device)
        y=data[i+1:i+block_size+1].unsqueeze(0).to(device)
        logits=model(x)
        loss=F.cross_entropy(logits.view(-1,vocab_size), y.view(-1))
        total_loss+=loss.item()
        total_steps+=1
    return math.exp(total_loss/total_steps)

# ==========================================================
# RLHF SIMULATION
# ==========================================================

def simulate_rlhf(model):
    reward_model = RewardModel().to(device)
    optimizer_rm = torch.optim.Adam(reward_model.parameters(), lr=1e-4)

    model.eval()
    prompt="User: Explain AI\nAI:"
    idx=torch.tensor([encode(prompt)],dtype=torch.long).to(device)

    ans1=decode(model.generate(idx,60)[0].tolist())
    ans2=decode(model.generate(idx,60)[0].tolist())

    label=0 if len(ans1)>len(ans2) else 1

    hidden=torch.randn(1,10,d_model).to(device)
    r1=reward_model(hidden)
    r2=reward_model(hidden)

    loss=-torch.log(torch.sigmoid(r1-r2)).mean()

    optimizer_rm.zero_grad()
    loss.backward()
    optimizer_rm.step()

    print("RLHF simulated. Reward loss:",loss.item())

#=========== Memory =====
def compress_memory(text):
    words = text.split()
    if len(words) > 40:
        summary = " ".join(words[:40])
    else:
        summary = text

    return summary


# ==========================================================
# CHAT MODE
# ==========================================================

def chat(model):
    model.eval()
    while True:
        user=input("You: ")
        if user=="exit":
            break
        prompt=f"User:{user} AI:"
        idx=torch.tensor([encode(prompt)],dtype=torch.long).to(device)
        out=decode(model.generate(idx,100)[0].tolist())
        print("AI:",out)
       # add_memory("user:"+user)
        #compressed = compress_memory(out)
        #memory.add("AI:"+ compressed)

# ==========================================================
# MAIN PIPELINE
# ==========================================================

model=GPT().to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)

print("Starting Base Training...")
pretrain(model,optimizer)

print("Freezing Base Model...")
freeze_base(model)

lora_optimizer=torch.optim.AdamW(
    filter(lambda p:p.requires_grad,model.parameters()),
    lr=1e-4
)

print("Starting LoRA Fine-Tuning...")
lora_finetune(model,lora_optimizer)

print("Evaluating...")
print("Perplexity:",evaluate_perplexity(model))

print("Simulating RLHF...")
simulate_rlhf(model)

print("Training Complete. Starting Chat...")
chat(model)