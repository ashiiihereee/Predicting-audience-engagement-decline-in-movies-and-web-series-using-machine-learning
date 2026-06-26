# 🎬 Predicting Audience Engagement Decline in Movies & Web Series
### Machine Learning Project | NMIMS, Chandigarh Campus

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?style=flat&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?style=flat&logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)
![Accuracy](https://img.shields.io/badge/Accuracy-86.89%25-brightgreen?style=flat)
![AUC](https://img.shields.io/badge/AUC-0.94-blue?style=flat)

---

## 📌 Project Overview

This project builds a **machine learning pipeline to predict audience engagement decline** in movies and web series using real-world streaming and film datasets. Five classification models were trained, compared, and evaluated — with **Random Forest achieving the best performance at 86.89% accuracy and AUC of 0.94**.

Built as part of the **Machine Learning** course at **NMIMS, Chandigarh Campus** under the guidance of **Ms. Harsimran Kaur**.

---

## ✨ Key Highlights

| Feature | Details |
|---|---|
| 📊 Datasets Merged | 4 Kaggle datasets — Netflix, IMDB, TMDB Movies & Credits |
| 🧹 Preprocessing | JSON parsing, null handling, binary classification, normalization |
| 🧠 Models Trained | Logistic Regression, KNN, Random Forest, SVM, Gradient Boosting |
| 🏆 Best Model | Random Forest — 86.89% Accuracy, AUC = 0.94 |
| 📈 Custom Feature | Engagement Score built from vote_count, vote_average & popularity |
| 💾 Model Saved | Exported as `movie_success_model.pkl` using Pickle |

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Libraries:** Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn
- **Environment:** Jupyter Notebook
- **Models:** Logistic Regression · KNN · Random Forest · SVM · Gradient Boosting

---

## 📂 Project Structure

```
Audience-Engagement-Prediction-ML/
│
├── MLPROJECTCODE_Ashna_D026_.ipynb    # Main Jupyter Notebook (full pipeline)
├── MLProjectReport_Ashna.pdf          # Full project report
├── movie_success_model.pkl            # Saved Random Forest model
├── requirements.txt                   # Python dependencies
│
├── datasets/
│   ├── netflix_titles_nov_2019.csv    # Netflix titles dataset
│   ├── imdb_top_1000.csv              # IMDB Top 1000 movies
│   ├── tmdb_5000_movies.csv           # TMDB 5000 movies metadata
│   └── tmdb_5000_credits.csv          # TMDB cast & crew data
│
└── README.md
```

---

## 📦 Datasets Used

| Dataset | Source | Records | Key Features |
|---|---|---|---|
| Netflix Titles (Nov 2019) | Kaggle | 5,837 | title, genre, duration, rating, type |
| IMDB Top 1000 | Kaggle | 1,000 | IMDB rating, gross, runtime, genre |
| TMDB 5000 Movies | Kaggle | 4,803 | budget, revenue, popularity, vote stats |
| TMDB 5000 Credits | Kaggle | 4,803 | cast, crew, director |

> **Final merged dataset:** 4,803 rows × 24 columns

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/ashiiihereee/Audience-Engagement-Prediction-ML.git
cd Audience-Engagement-Prediction-ML
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Open the notebook
```bash
jupyter notebook MLPROJECTCODE_Ashna_D026_.ipynb
```

### requirements.txt contents
```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

---

## 🔄 Methodology

### 1. Data Collection & Merging
- Four publicly available Kaggle datasets loaded and merged
- Final dataframe: 4,803 rows × 24 columns after cleaning

### 2. Data Preprocessing
- Handled missing values across all columns
- Removed duplicate entries (0 duplicates found)
- Dropped irrelevant columns: `homepage`, `tagline`, `original_title`
- Parsed JSON fields (genres, keywords, cast, crew) using Python's `ast` library
- Filled missing `runtime` values with column mean
- Binary classification target:
```python
df['success'] = df['revenue'].apply(lambda x: 1 if x > df['revenue'].median() else 0)
```

### 3. Feature Engineering
- Extracted top 3 cast members and director from JSON crew data
- Created a `combined_features` column: genres + keywords + cast + crew
- Applied `CountVectorizer` with max 5,000 text features
- Built a custom **Engagement Score**:
```python
df['engagement_score'] = (
    df['vote_count'] * 0.5 +
    df['vote_average'] * 0.3 +
    df['popularity'] * 0.2
)
```
- Classified into `High` or `Low` engagement levels based on mean score

### 4. Model Training — 80:20 Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 📊 Results & Model Comparison

| Model | Accuracy |
|---|---|
| Logistic Regression | 85.54% |
| K-Nearest Neighbors | 84.60% |
| **Random Forest** ⭐ | **86.89%** |
| Support Vector Machine | 80.96% |
| Gradient Boosting | 86.78% |

### Random Forest — Classification Report
```
              precision    recall  f1-score   support
           0       0.89      0.85      0.87       491
           1       0.85      0.89      0.87       470
    accuracy                           0.87       961
```

### ROC Curve — AUC = 0.94
The Random Forest model achieved an AUC of **0.94**, indicating excellent discriminative ability between high and low engagement content.

---

## 🔍 Key Findings

- **Best Model:** Random Forest — 86.89% accuracy, AUC = 0.94
- **Feature Importance:** `vote_count` was the strongest predictor of engagement (importance score ~0.58), followed by `popularity`
- **Ensemble models** (Random Forest, Gradient Boosting) consistently outperformed simpler models due to their ability to handle complex non-linear patterns
- **Top 10 High Engagement Movies identified:**

```
1. Inception          5. Deadpool
2. The Dark Knight    6. Interstellar
3. Avatar             7. Django Unchained
4. The Avengers       8. Guardians of the Galaxy
                      9. Mad Max: Fury Road
                     10. The Hunger Games
```

---

## 📸 Visualizations Included

- ✅ Model Comparison Bar Chart (LR vs KNN vs RF vs SVM vs GB)
- ✅ Confusion Matrix — Random Forest
- ✅ ROC Curve (AUC = 0.94)
- ✅ Feature Importance Plot (Numeric Features)
- ✅ Movie Ratings Distribution Histogram
- ✅ Budget vs Revenue Scatter Plot
- ✅ Engagement Level Distribution (High vs Low)
- ✅ Top 10 High Engagement Movies (Horizontal Bar Chart)

---

## 🚀 Future Scope

- Incorporate real-time user interaction data from streaming platforms
- Apply deep learning techniques (LSTM, Transformers) for improved accuracy
- Expand dataset to include more diverse and global content
- Build a real-time prediction dashboard
- Integrate sentiment analysis from social media and reviews

---

## 👩‍💻 Author

**Ashna Bansal**
SAP ID: 70572400052 | Roll No: D026 | Batch: A
NMIMS, Chandigarh Campus — School of Technology Management and Engineering

---

## 🙏 Acknowledgements

- **Ms. Harsimran Kaur** — Machine Learning Subject Teacher, NMIMS Chandigarh
- [Kaggle](https://www.kaggle.com) — Datasets
- [Scikit-learn](https://scikit-learn.org/) — ML Library
- [TMDB](https://www.themoviedb.org/) — Movie Metadata
- Research references from Google Scholar

