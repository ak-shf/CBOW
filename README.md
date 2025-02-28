# CBOW Model Implementation in Python

This repository contains a Python implementation of the **Continuous Bag-of-Words (CBOW) model** for generating word embeddings using text extracted from a PDF document. The implementation is done **from scratch** using NumPy for numerical computations and PyMuPDF for text extraction.

## 📌 Overview

The **CBOW model** is a word embedding technique that predicts a **target word** based on its surrounding **context words**. This project follows these steps:

1. **Text Extraction:** Reads text from a PDF file (`corpus.pdf`) using PyMuPDF.
2. **Preprocessing:** Tokenizes the text, removes punctuation and stopwords, and converts words to lowercase.
3. **CBOW Pair Generation:** Forms (context, target) word pairs using a specified window size.
4. **One-Hot Encoding:** Converts words into numerical vectors using one-hot encoding.
5. **Training Data Preparation:** Sums the one-hot vectors of context words to form input vectors.
6. **Neural Network Training:** Implements a **two-layer neural network** to train word embeddings using gradient descent.
7. **Prediction:** Predicts a target word when given a set of context words.

---

## 🛠️ Requirements

- Python 3.x
- [PyMuPDF](https://pypi.org/project/PyMuPDF/) (for PDF text extraction)
- [NumPy](https://numpy.org/) (for numerical operations)

### 📥 Installation

To install the required libraries, run:

```bash
pip install PyMuPDF numpy
