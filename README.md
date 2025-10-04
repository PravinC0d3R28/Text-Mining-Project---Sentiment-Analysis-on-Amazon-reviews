# Sentiment Analysis of Amazon Musical Instrument Reviews

This project performs sentiment analysis on customer reviews of musical instruments from Amazon. The primary goal is to classify reviews as positive, negative, or neutral and to extract actionable insights from the textual data. By understanding customer feedback, businesses can identify product strengths, weaknesses, and overall market sentiment.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Exploratory Data Analysis (EDA)](#eda)
5. [Model Building & Evaluation](#model-building-evaluation)
6. [Results & Key Findings](#results-key-findings)
7. [How to Run](#how-to-run)
8. [Dependencies](#dependencies)

---

## Project Overview
The project follows a comprehensive NLP workflow:

- **Data Cleaning & Preprocessing:** Prepare raw text data by handling missing values, removing noise, and performing normalization.
- **Exploratory Data Analysis:** Visualize the data to uncover patterns such as word frequencies and sentiment distribution over time.
- **Feature Engineering:** Transform cleaned text into numerical features using TF-IDF vectorization for machine learning.
- **Model Training:** Build and train multiple classification models to predict sentiment.
- **Performance Evaluation:** Assess model performance using metrics like accuracy, precision, recall, and F1-score to select the best model.

---

## Dataset
The dataset used is `Musical_instruments_reviews.csv`, containing 10,261 customer reviews.

**Key Columns:**
- `reviewText`: Full text of the customer review.
- `summary`: Short summary of the review.
- `overall`: Numerical rating given by the user (1–5).
- `helpful`: Tuple indicating helpfulness (e.g., [2, 3] means 2 out of 3 people found it helpful).
- `reviewTime`: Date of the review.

---

## Methodology

### 1. Data Preprocessing
The raw review text is processed as follows:
- **Concatenation:** Combine `reviewText` and `summary` for comprehensive analysis.
- **Sentiment Labeling:** Create a sentiment column based on `overall` score:
  - Positive: Rating > 3
  - Negative: Rating < 3
  - Neutral: Rating = 3
- **Text Cleaning:** Apply custom function to:
  - Convert text to lowercase.
  - Remove punctuation, URLs, and numbers.
  - Apply custom stopword removal while retaining important negative words (e.g., "not", "no").

### 2. Feature Extraction
- **TF-IDF (Term Frequency-Inverse Document Frequency):** Converts cleaned text into numerical features for classification, capturing the importance of words in the corpus.

### 3. Handling Class Imbalance
- **SMOTE (Synthetic Minority Over-sampling Technique):** The dataset has more positive reviews. SMOTE generates synthetic samples for minority classes (negative and neutral) to balance the training set.

---

## Exploratory Data Analysis (EDA)

### Sentiment Distribution
The dataset shows a significant class imbalance with positive reviews dominating.

![Sentiment Distribution](path/to/your/sentiment_distribution_pie_chart.png)

### Word Clouds
Visualizing the most frequent words for each sentiment:

**Positive Sentiment Word Cloud**  
![Positive Word Cloud](path/to/your/positive_wordcloud.png)

**Negative Sentiment Word Cloud**  
![Negative Word Cloud](path/to/your/negative_wordcloud.png)

### N-gram Analysis
Bigram and trigram analysis identifies common multi-word phrases associated with each sentiment.

![Top Negative Bigrams](path/to/your/negative_bigrams_chart.png)

---

## Model Building & Evaluation
Several machine learning classifiers were trained on an 80%-20% train-test split:

**Models Implemented:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Bernoulli Naive Bayes
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)

Performance metrics include Accuracy, Precision, Recall, and F1-Score.

---

## Results & Key Findings

### Model Performance Metrics

| Model                   | Accuracy | F1-Score (Positive) | F1-Score (Negative) | F1-Score (Neutral) |
|-------------------------|---------|-------------------|-------------------|------------------|
| Logistic Regression     | 0.95    | 0.97              | 0.87              | 0.82             |
| Support Vector Machine  | 0.95    | 0.97              | 0.86              | 0.82             |
| Random Forest Classifier| 0.94    | 0.96              | 0.84              | 0.79             |
| K-Nearest Neighbors     | 0.94    | 0.96              | 0.88              | 0.75             |
| Decision Tree Classifier| 0.88    | 0.92              | 0.74              | 0.67             |
| Bernoulli Naive Bayes   | 0.85    | 0.91              | 0.69              | 0.63             |

### Confusion Matrix
Logistic Regression shows high accuracy in classifying sentiments:

![Confusion Matrix](path/to/your/logistic_regression_confusion_matrix.png)

### Key Findings
- **Class Imbalance is Critical:** SMOTE improved recall and F1-scores for minority classes (Negative and Neutral).  
- **Custom Stopwords Matter:** Retaining negative indicators like "not" is essential.  
- **N-grams Provide Context:** Multi-word phrases like "customer service" and "not good" improve sentiment detection.  
- **High-Performance Models:** Logistic Regression and SVC achieved 95% accuracy, effectively distinguishing sentiments.

---

## How to Run
```bash
# Clone the repository
git clone https://your-repository-url.git
cd your-repository-directory

# Install dependencies
pip install -r requirements.txt

# Ensure dataset is in root directory
# Launch Jupyter Notebook
```
jupyter notebook
Open Sentiment_Analysis_Amazon_reviews.ipynb and run all cells.

Dependencies
```
A requirements.txt file should include:

pandas
numpy
nltk
scikit-learn
matplotlib
seaborn
textblob
plotly
cufflinks
imbalanced-learn
wordcloud
```
