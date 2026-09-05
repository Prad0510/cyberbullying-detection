# Cyberbullying Detection Using PCA-Extracted GloVe Features and RoBERTa

## 📌 Project Overview

This project focuses on detecting and classifying cyberbullying in social-media text.

The proposed approach uses:

* **GloVe** for extracting word-level semantic features
* **PCA** for dimensionality reduction of GloVe features
* **RoBERTa** for transformer-based contextual text representation and classification

The final system is intended to classify tweets into different types of cyberbullying.

---

# 📂 Project Structure

```text
cyberbullying-detection/
│
├── data/
│   ├── raw/
│   │   └── cbtweets/
│   │       └── CBTWweets.csv
│   │
│   └── processed/
│       └── cyberbullying_cleaned.csv
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   └── 02_preprocessing.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   └── preprocessing/
│       ├── __init__.py
│       └── clean_text.py
│
├── experiments/
│
├── models/
│
├── reports/
│   └── dataset_analysis.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ✅ Work Completed So Far

## 1. Dataset Selection

The current dataset being used is the **CBTweets dataset**.

Original file:

```text
CBTweets.csv
```

The dataset contains:

* **47,692 rows**
* **2 columns**
* `tweet_text`
* `cyberbullying_type`

There are six classes:

```text
age
religion
ethnicity
gender
not_cyberbullying
other_cyberbullying
```

The original dataset was found to have no missing values.

---

## 2. Initial Dataset Analysis

The dataset was inspected for:

* Number of rows and columns
* Missing values
* Duplicate rows
* Class distribution
* Tweet length
* Very long tweets
* Social-media-specific patterns

The original dataset contained:

```text
Rows: 47,692
Columns: 2
Missing values: 0
Exact duplicate rows: 36
```

There were also **23 tweets longer than 500 characters**.

These long tweets were **not automatically deleted** because long text can still contain useful contextual information for cyberbullying detection.

Instead, maximum sequence length should be handled later during model tokenization.

---

# 🧹 3. Text Preprocessing

A preprocessing function was implemented in:

```text
src/preprocessing/clean_text.py
```

The function performs the following operations:

### HTML decoding

Example:

```text
&amp; → &
```

### Unicode normalization

Unicode representation is normalized using `NFKC`.

### Retweet removal

A leading:

```text
RT
```

is removed.

### URL replacement

URLs are replaced with:

```text
<URL>
```

Example:

```text
https://example.com/test
```

becomes:

```text
<URL>
```

### User mention replacement

Twitter-style mentions are replaced with:

```text
<USER>
```

Example:

```text
@username
```

becomes:

```text
<USER>
```

### Punctuation preservation

Punctuation such as:

```text
!!!
?
#
```

is preserved because it may contain useful information for cyberbullying detection.

### Hashtag preservation

Hashtags are preserved.

Example:

```text
#bullying
```

remains:

```text
#bullying
```

### Emoji preservation

Emojis are preserved because they may contain contextual or emotional information.

### Offensive-word preservation

Offensive words are **not removed**.

This is intentional because offensive language can be an important signal for cyberbullying classification.

---

# ⚠️ 4. Duplicate and Label-Conflict Analysis

After preprocessing, duplicate analysis was performed.

There were:

```text
Exact duplicate rows: 36
Duplicate cleaned texts: 2,184
```

However, not all duplicate cleaned texts represent actual duplicate tweets.

For example, different URLs:

```text
#BlameOneNotAll http://t.co/ABC
#BlameOneNotAll http://t.co/XYZ
```

become:

```text
#BlameOneNotAll <URL>
```

Therefore, duplicate cleaned texts were not blindly deleted.

---

## Label conflicts

An important dataset-quality issue was discovered.

The same original tweet text sometimes appeared with different labels.

Results:

```text
Original tweets with conflicting labels: 1,639
Rows involved: 3,278
```

Example:

```text
#MKR I really hope they get out-sassed
```

appeared with different labels in the original dataset.

This means the conflicts are present in the original dataset and are not caused by our preprocessing.

These conflicting examples were removed rather than arbitrarily choosing one of the labels.

---

# 📊 5. Final Cleaned Dataset

After removing conflicting-label examples and remaining duplicate `(tweet_text, label)` pairs:

```text
Final dataset size: 44,378 rows
```

Final class distribution:

| Class               |       Rows | Percentage |
| ------------------- | ---------: | ---------: |
| age                 |      7,992 |     18.01% |
| religion            |      7,991 |     18.01% |
| ethnicity           |      7,952 |     17.92% |
| gender              |      7,772 |     17.51% |
| not_cyberbullying   |      6,428 |     14.48% |
| other_cyberbullying |      6,243 |     14.07% |
| **Total**           | **44,378** |   **100%** |

The final dataset remains reasonably balanced across the six classes.

---

# 🚧 Current Status

### Completed

* [x] Dataset selected
* [x] Dataset downloaded
* [x] Dataset structure inspected
* [x] Missing-value check
* [x] Duplicate analysis
* [x] Class-distribution analysis
* [x] Long-text analysis
* [x] Text preprocessing function
* [x] URL normalization
* [x] Mention normalization
* [x] Retweet removal
* [x] HTML decoding
* [x] Unicode normalization
* [x] Duplicate/label-conflict analysis
* [x] Conflicting-label examples removed
* [x] Final cleaned dataset created

---

# 🚀 Next Steps

The following work has **not been completed yet** and can be continued by the team.

## Step 1 — Final Dataset Validation

Before modeling:

* Check remaining duplicate texts
* Check empty records
* Verify class distribution
* Verify that conflicting labels no longer exist

---

## Step 2 — Train / Validation / Test Split

Create a **stratified** split so that all six classes are represented proportionally.

For example:

```text
Training:   70%
Validation: 15%
Testing:    15%
```

The split should be performed carefully to avoid duplicate-text leakage between training and testing.

---

## Step 3 — Label Encoding

Convert:

```text
age
religion
ethnicity
gender
not_cyberbullying
other_cyberbullying
```

into numerical labels for model training.

The mapping should be saved and reused consistently across training, validation, and testing.

---

# 🧠 Step 4 — GloVe Feature Extraction

Use pretrained **GloVe embeddings** to represent the cleaned tweets numerically.

The team needs to decide:

* Which GloVe version to use
* How to represent a complete tweet from individual word embeddings
* How to handle words not present in GloVe

Possible tweet-level representations include:

* Mean word embedding
* Weighted/TF-IDF weighted embedding
* Other aggregation methods

---

# 📉 Step 5 — PCA

Apply PCA to the GloVe-based feature vectors.

Important:

> PCA must be fitted on the training data only.

Then apply the learned PCA transformation to validation and test data.

The number of principal components should be selected based on explained variance or experimental comparison.

---

# 🤖 Step 6 — RoBERTa Model

Use RoBERTa for contextual representation of the cleaned tweets.

Tasks include:

* Tokenization
* Maximum sequence length selection
* Fine-tuning
* Validation
* Hyperparameter tuning
* Multiclass classification

Long tweets should be handled during tokenization/model preparation rather than deleted during preprocessing.

---

# 🔗 Step 7 — GloVe/PCA + RoBERTa Integration

The exact architecture for combining the two approaches still needs to be finalized.

Possible approach:

```text
Cleaned Tweet
     │
     ├──────────────► GloVe
     │                   │
     │                   ▼
     │                  PCA
     │                   │
     │                   ▼
     │            Reduced GloVe Features
     │
     └──────────────► RoBERTa
                         │
                         ▼
                  RoBERTa Features
                         │
                         ▼
                  Feature Fusion
                         │
                         ▼
                   Classifier
                         │
                         ▼
                  Cyberbullying Class
```

The team should agree on the exact fusion strategy before implementation.

**Important:** PCA-reduced GloVe vectors should not simply be inserted as RoBERTa token embeddings. They should be treated as a separate feature representation and fused at an appropriate later stage unless the architecture is deliberately designed otherwise.

---

# 📏 Step 8 — Model Evaluation

Evaluate the final model using metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Macro F1
* Confusion matrix

Because this is a six-class classification problem, **macro F1-score** should be reported in addition to accuracy.

---

# 🔬 Step 9 — Comparison and Analysis

Compare:

1. GloVe-based model
2. GloVe + PCA model
3. RoBERTa model
4. Combined GloVe/PCA + RoBERTa model

This will help demonstrate whether PCA-based GloVe features actually provide additional value over the transformer model alone.

---

# 📌 Important Notes for Team Members

### Do not modify the raw dataset

The original dataset should remain unchanged in:

```text
data/raw/
```

### Do not perform aggressive text cleaning

Do not automatically remove:

* Offensive words
* Hashtags
* Emojis
* Punctuation
* Contextual words

These can be useful features for cyberbullying detection.

### Do not fit preprocessing components on the test set

For example, PCA must be fitted using training data only.

### Avoid data leakage

Do not allow identical or conflicting examples to appear across training and testing sets.

---

# 👥 Where to Continue

If you are joining the project now, start from:

```text
data/processed/cyberbullying_cleaned.csv
```

The preprocessing stage has already been completed.

The next immediate task is:

```text
Final Dataset Validation
        ↓
Stratified Train/Validation/Test Split
        ↓
        ├── GloVe → PCA
        │
        └── RoBERTa
                ↓
          Feature Fusion
                ↓
           Classification
                ↓
            Evaluation
```

The notebooks documenting the completed preprocessing are:

```text
notebooks/01_dataset_analysis.ipynb
notebooks/02_preprocessing.ipynb
```

The preprocessing implementation is:

```text
src/preprocessing/clean_text.py
```
