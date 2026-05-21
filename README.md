# 🎬 Netflix Movie Genre Classification using LSTM

A deep learning project implemented in a Jupyter Notebook environment that builds a Recurrent Neural Network (RNN) using Long Short-Term Memory (LSTM) networks to automatically classify Netflix movie titles into their respective genres based purely on their textual summaries/descriptions.

---

## 📌 Project Overview
With thousands of titles available on streaming platforms, automated content tagging and metadata enrichment are critical. This project treats movie descriptions as sequence data, preprocesses the text data, handles tokenization, and feeds them into a sequence-optimized deep learning model to accurately predict major film classifications.

---

## 🏗️ Model Architecture

The deep learning network is constructed using the Keras Functional API, ensuring sequential data flow:

1. **Input Layer**: Accepts padded token sequences of length `max_len = 30`.
2. **Embedding Layer**: Transforms integer-encoded words into dense vectors of fixed size (`embedding_dim = 50`).
3. **Stacked LSTM Layers**: Two recurrent layers capture long-range contextual dependencies within descriptions.
4. **Regularization (Dropout)**: Built-in dropout fractions (0.2) prevent overfitting during training transitions.
5. **Output Dense Layer**: Uses a Softmax classifier to yield categorical probabilities across your targeted genres.

### 🧠 Core Mathematical Activations Inside the LSTM Cell
To cleanly gate memory cells and avoid common vanishing/exploding gradient flaws over sequences, the network explicitly relies on two fundamental activations:

* **`tanh` (Hyperbolic Tangent Function)**: Applied to regulate hidden state values and cell state updates. It maps data between $[-1, 1]$, centering the sequence representations around zero to maintain stable gradient transitions.
* **`sigmoid` Function**: Utilized in the internal gating mechanisms (Forget Gate, Input Gate, and Output Gate). It strictly scales numbers between $[0, 1]$, acting as a binary filter to decide exactly what percentage of contextual information to forget or pass forward.

---

## 🛠️ Step-by-Step Project Pipeline

### Step 1: Library Setup
Installs and imports core numerical computation, model design (`TensorFlow`/`Keras`), and metrics tracking infrastructure.

### Step 2: Data Engineering & Cleaning
* Loads `netflix_titles.csv`.
* Strips individual movie categories out of the raw `listed_in` string.
* Filters the dataset to isolate high-density primary target genres: *Dramas, Comedies, Action & Adventure, Horror Movies,* and *Documentaries*.

### Step 3: Text Vectorization & Categorical Preprocessing
* Formulates a `Tokenizer` limited to the top 10,000 architectural keywords.
* Standardizes feature arrays using post-padding and truncating mechanics to keep text sequences at exactly 30 indices.
* Standardizes string labels using `LabelEncoder` combined with an ultimate transformation to `to_categorical` matrices.

### Step 4: Model Design & Training Execution
Defines the functional computational graph, binds the optimization compiler using `adam` adjustments, and performs a 15-epoch cross-validated training execution loop.

### Step 5: Robust Metric Diagnostics
Resolves dimensional mismatch execution errors when matching multi-class probabilities to categorical tracking arrays. Features an explicit configuration utilizing `np.argmax(..., axis=1)` to provide safe `classification_report` data maps and plots a descriptive visual performance matrix using `seaborn`.

### Step 6: Live Interface Inferencing
Exposes a clean custom function `predict_movie_genre(description)` to seamlessly transform string input descriptions and output predictions instantly with calculated percentage confidence scores.

---

## 📊 Requirements & Dependencies
To run this project flawlessly inside a local Jupyter environment or on Google Colab, prepare your space using the following libraries:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
