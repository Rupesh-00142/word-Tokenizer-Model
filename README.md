# Word Tokenizer Model

A lightweight **Word-Level Tokenizer** built in Python for Natural Language Processing (NLP) and custom language-model projects.

This tokenizer converts raw text into words/tokens and maps them to numerical token IDs that can be directly used as input for machine-learning and Transformer-based language models.

## 🚀 Features

* Word-level tokenization
* Text → Token conversion
* Token → ID conversion
* ID → Token decoding
* Vocabulary generation
* Special token support
* Unknown token handling
* Vocabulary size detection
* Easy integration with custom GPT/Transformer models
* Lightweight and easy to customize

## 🧠 How It Works

The tokenizer follows a simple pipeline:

```text
Raw Text
   ↓
Text Cleaning
   ↓
Word Tokenization
   ↓
Vocabulary Mapping
   ↓
Token IDs
   ↓
Transformer / Language Model
```

Example:

```text
Input:
"Hello, how are you?"

Tokens:
["Hello", "how", "are", "you"]

Token IDs:
[125, 42, 78, 91]
```

The generated token IDs can then be passed to a custom neural-network language model.


```python
from tokenizer import WordTokenizer

tokenizer = WordTokenizer()

text = "Hello, welcome to my AI project."

tokens = tokenizer.encode(text)

print(tokens)
```

To decode token IDs:

```python
text = tokenizer.decode(tokens)

print(text)
```

## 📚 Vocabulary

The tokenizer maintains a vocabulary that maps words to unique numerical IDs.

Example:

```text
<UNK>    → 0
<PAD>    → 1
<START>  → 2
<END>    → 3
hello    → 4
world    → 5
```

The vocabulary can be saved and loaded for use during model training and inference.

## 🤖 Use With Transformer Models

This tokenizer is designed to work with custom Transformer/GPT-style architectures.

```text
Dataset
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Embedding Layer
   ↓
Transformer
   ↓
Language Model Head
   ↓
Generated Text
```

It can be used as the preprocessing layer for a custom AI assistant or language model.

## 🎯 Applications

* Custom GPT models
* Transformer models
* Chatbots
* AI assistants
* NLP projects
* Text classification
* Language-model training
* Educational ML projects
* Experimental LLM architectures

## 🛠️ Technologies

* Python
* Natural Language Processing
* Tokenization
* Machine Learning
* Deep Learning
* Transformer Architecture
* PyTorch

## 📌 Future Improvements

* [ ] Subword/BPE tokenization
* [ ] SentencePiece support
* [ ] Better punctuation handling
* [ ] Unicode normalization
* [ ] Faster vocabulary processing
* [ ] Padding and attention-mask generation
* [ ] Hugging Face tokenizer compatibility
* [ ] Integration with custom GPT architecture

## 📄 License

This project is open-source and available under the MIT License.

## 👨‍💻 Author

**Rupesh Tandan**

Built as part of an experimental custom AI/NLP project.

---

⭐ If you find this project useful, consider giving the repository a star!
