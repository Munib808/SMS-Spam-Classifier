# SMS-Spam-Classifier

# Email/SMS Spam Classifier

An end-to-end Machine Learning project that classifies SMS or Email messages into **Spam** or **Ham** (Not Spam) using Natural Language Processing (NLP) and a Multinomial Naive Bayes model.

## Project Overview
This project involves building a text classification model trained on a dataset of 5,572 messages. The workflow includes rigorous data cleaning, Exploratory Data Analysis (EDA), and text preprocessing (stemming, stopword removal) to achieve high precision and accuracy.

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn
- **Web Framework:** Streamlit (for the UI)
- **Deployment:** Pickle (for model serialization)

## Key Features
- **Data Cleaning:** Handled duplicates and missing values.
- **EDA:** Analyzed message length, word count, and sentence count to find patterns between spam and ham.
- **NLP Pipeline:** - Tokenization
  - Removing special characters & punctuation
  - Removing stopwords
  - Stemming using PorterStemmer
- **Model:** Multinomial Naive Bayes (chosen for its high precision in text classification).
- **Vectorization:** TF-IDF Vectorizer with `max_features=3000`.

## 🖥️ How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`

## 📈 Model Performance
| Algorithm | Accuracy | Precision |
| :--- | :--- | :--- |
| **Multinomial Naive Bayes** | ~97% | 1.00 |

*Note: In spam detection, Precision is prioritized to ensure important messages are not incorrectly flagged as spam.*
