# ML_PROJCT_01
# Sentiment Analysis on Amazon Product Reviews

## Overview
This project implements a sentiment analysis model to classify Amazon product reviews as positive or negative based on their textual content. The dataset consists of reviews labeled with binary sentiment values, where `1` indicates a positive sentiment and `0` indicates a negative sentiment.

## Table of Contents
- [Dataset Overview](#dataset-overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Selection](#model-selection)
- [Model Training](#model-training)
- [Evaluation Metrics](#evaluation-metrics)
- [Comparative Analysis](#comparative-analysis)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)

## Dataset Overview
- **Source**: The dataset can be accessed from [this link](https://raw.githubusercontent.com/rashakil-ds/Public-Datasets/refs/heads/main/amazon.csv).
- **Columns**:
  - `reviewText`: The textual content of the review.
  - `Positive`: The sentiment label (1 for positive, 0 for negative).

### Sample Data
| reviewText                                           | Positive |
|-----------------------------------------------------|----------|
| This is one of the best apps according to a bunch...| 1        |
| This is a pretty good version of the game for...    | 1        |
| This is a silly game and can be frustrating...       | 1        |
| This is a terrific game on any pad. Hrs of fun...   | 1        |

## Data Preprocessing
1. **Handle Missing Values**: Remove any rows with missing values in `reviewText` or `sentiment`.
2. **Text Preprocessing**:
   - Convert text to lowercase.
   - Remove stop words, punctuation, and special characters.
   - Tokenize and lemmatize the text data.
3. **Split Dataset**: The dataset is split into training (80%) and testing (20%) sets.

### Code Snippet
```python
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load dataset
url = 'https://raw.githubusercontent.com/rashakil-ds/Public-Datasets/refs/heads/main/amazon.csv'
df = pd.read_csv(url)

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^a-z\s]', '', text.lower())
    tokens = word_tokenize(text)
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply preprocessing
df['cleanedText'] = df['reviewText'].apply(preprocess_text)
```

## Model Selection
We selected several machine learning models for sentiment classification:
- **Statistical Models**:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - Naïve Bayes
  - Gradient Boosting (e.g., XGBoost)
  
- **Neural Models**:
  - LSTM (Long Short-Term Memory)
  - GRUs (Gated Recurrent Units)

### Vectorization
Text data is converted to TF-IDF features to prepare for model training.

### Code Snippet
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
```

## Model Training
Each selected model is trained on the training dataset, and performance is evaluated using metrics such as accuracy, precision, recall, and F1 score.

### Example Model Training
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Train Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)

# Predictions and Evaluation
lr_predictions = lr_model.predict(X_test_tfidf)
print(classification_report(y_test, lr_predictions))
print("Accuracy:", accuracy_score(y_test, lr_predictions))
```

## Evaluation Metrics
The performance of each model is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

### Example Output
```plaintext
Logistic Regression Results:
              precision    recall  f1-score   support
           0       0.84      0.65      0.73       958
           1       0.90      0.96      0.93      3042
    accuracy                           0.89      4000
```

## Comparative Analysis
A comparative analysis of all models is conducted to identify strengths and weaknesses based on evaluation
