# SMS_spam_detector_ML_MiniProject

# SMS Spam Filter — Naive Bayes

A binary text classifier that detects spam vs ham (legitimate) SMS messages, built as part of a hands-on ML portfolio covering classic algorithms end-to-end.

## Problem statement

Given an SMS message, predict whether it is `spam` or `ham`. This is a supervised binary classification problem on unstructured text data.

## Dataset

[SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (UCI) — 5,572 labeled SMS messages (~13% spam, ~87% ham).

## Concepts used

### Naive Bayes

Naive Bayes is a probabilistic classifier built on Bayes' theorem:

```
P(spam | message) = P(message | spam) * P(spam) / P(message)
```

Rather than computing `P(message | spam)` directly (which is hard — a message is a sequence of many words), Naive Bayes makes an independence assumption: it treats every word's presence as conditionally independent of every other word, given the class. This lets it estimate:

```
P(message | spam) ≈ P(word1 | spam) * P(word2 | spam) * ... * P(wordN | spam)
```

Each `P(word | spam)` is just: *how often does this word appear in spam messages, relative to all words in spam messages, in the training data?*

**Why "naive"?** Because words are obviously not independent in real language (grammar, context, word order all create dependencies). The assumption is wrong in a literal sense, but works well in practice for tasks like spam detection where certain vocabulary ("free", "winner", "click") is strongly associated with one class, regardless of grammatical correctness.

**Laplace smoothing**: if a word never appeared in the spam class during training, `P(word | spam) = 0`, which would zero out the entire product for any message containing that word. Smoothing adds a small constant (default 1) to every word count so no probability is ever exactly zero.

### Text vectorization: CountVectorizer vs TF-IDF

Models can't consume raw text — every message must become a fixed-length numeric vector first.

**CountVectorizer (Bag of Words)**
- Builds a vocabulary of every unique word seen during `fit()`.
- Each message becomes a vector of raw word counts against that vocabulary.
- Simple, and *frequency itself* is treated as a useful signal — a word appearing 3 times counts more than appearing once.

**TF-IDF (Term Frequency – Inverse Document Frequency)**
- Also builds a vocabulary, but instead of raw counts, it weights each word by:
  - **TF**: how often the word appears in *this* message
  - **IDF**: how rare the word is *across all messages* — common words (the, is, you) get down-weighted; rare, distinctive words get boosted
- Designed to surface words that are distinctive to a specific document, while suppressing generic, everywhere-words.

**Why CountVectorizer outperformed TF-IDF here**: spam detection relies on certain words (`free`, `win`, `call now`, `claim`) appearing *frequently and repeatedly across many spam messages* — that repetition across the spam class is itself the signal. TF-IDF's IDF term penalizes exactly those words for being common, since "common across many documents" is usually treated as "less informative" in TF-IDF's worldview. So TF-IDF made the model more conservative: it only flagged messages with rarer, more unusual words as spam, which improved precision but hurt recall. This is a useful, general lesson — TF-IDF is not a strictly "better" choice than raw counts; it depends on whether the signal lives in word rarity or word frequency.

**Important caveat shared by both vectorizers**: the vocabulary is fixed at `fit()` time (on the training set only). Any word seen in production/test data that wasn't in the training vocabulary is silently ignored — it contributes nothing to the prediction. This is a real limitation of Bag-of-Words style methods, and one reason production NLP systems eventually move toward subword tokenization or embeddings, which can represent unseen words via shared sub-word units or learned semantic similarity.

## Results

| Vectorizer | Accuracy | Precision | Recall |
|---|---|---|---|
| CountVectorizer | 0.9839 | 0.9853 | 0.8933 |
| TF-IDF | 0.9623 | 1.0000 | 0.7200 |

### Confusion matrix (CountVectorizer)

|  | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | 963 | 2 |
| **Actual Spam** | 16 | 134 |

### Metric tradeoff: precision vs recall

- **Precision** — of everything predicted spam, what fraction was actually spam? High precision = few false alarms (real messages wrongly blocked).
- **Recall** — of everything actually spam, what fraction did we catch? High recall = few spam messages slip through.

For a spam filter, a **false positive (blocking a real message)** is generally more costly than a **false negative (a spam message reaching the inbox)** — a missed OTP, job offer, or bank alert is worse than seeing one extra spam message. This pushes the design choice toward optimizing for **precision over recall**, which is the reasoning behind preferring the higher-precision configuration in production, even at some cost to recall.

## Tech stack

- Python, pandas, scikit-learn
- `MultinomialNB`, `CountVectorizer`, `TfidfVectorizer`
- (Planned) FastAPI endpoint for serving predictions

## Project structure

```
sms_spam_detection/
├── spam.csv
├── train.py
├── spam_model.pkl
├── vectorizer.pkl
└── README.md
```

## How to run

```bash
pip install pandas scikit-learn
python train.py
```

