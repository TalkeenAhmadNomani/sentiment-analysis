# Twitter Hate and Offensive Content Detection

## Motivation
Twitter empowers free speech, but this freedom is sometimes misused to spread hate or offensive content. This project aims to detect such tweets using supervised learning techniques and modern NLP preprocessing, fostering healthier online conversations.

---

## Dataset
- **Total Samples:** 24,783 tweets  
- **Columns:** `count`, `hate_speech`, `offensive_language`, `neither`, `class`, `tweet`  
- **Class Distribution:**
  - Offensive Language (1): 77.43%
  - Neither (2): 16.80%
  - Hate Speech (0): 5.77%

---

## Preprocessing Steps
1. Lowercasing all text.  
2. Removal of URLs, mentions, hashtags, numbers, and punctuation.  
3. Stopword removal using **NLTK**.  
4. Stemming using **PorterStemmer**.  
5. Tokenization and padding (for deep learning models).  
6. Vectorization using **CountVectorizer** (for ML models).  

---

## Handling Imbalance
- Used **RandomOverSampler** from `imblearn` to balance class distribution in the training set.
---
<img width="1087" height="712" alt="Screenshot 2025-08-17 133605" src="https://github.com/user-attachments/assets/b1abfec4-4948-4313-a68b-5ab7f2afdba3" />



<img width="425" height="307" alt="Screenshot 2025-08-17 133619" src="https://github.com/user-attachments/assets/af7892d8-5b16-422e-b291-16774097dbcc" />

## Models Trained

### Machine Learning Models
| Model               | Accuracy | Precision | Recall  | F1 Score |
|--------------------|---------|----------|--------|----------|
| Voting Ensemble     | 0.8687  | 0.8952   | 0.8687 | 0.8792   |
| XGBoost             | 0.8590  | 0.9028   | 0.8590 | 0.8740   |
| Logistic Regression | 0.8550  | 0.8915   | 0.8550 | 0.8689   |
| SVM                 | 0.8465  | 0.8854   | 0.8465 | 0.8622   |
| Naive Bayes         | 0.8396  | 0.8827   | 0.8396 | 0.8570   |


<img width="1417" height="721" alt="Screenshot 2025-08-17 133639" src="https://github.com/user-attachments/assets/9601c1d6-60bf-4e98-9245-0dc552485a1e" />


### Deep Learning Models
| Model | Accuracy | Precision | Recall  | F1 Score |
|-------|---------|----------|--------|----------|
| CNN   | 0.8545  | 0.8582   | 0.8545 | 0.8560   |
| GRU   | 0.8570  | 0.8543   | 0.8570 | 0.8550   |
| LSTM  | 0.8564  | 0.8530   | 0.8564 | 0.8545   |



<img width="1408" height="700" alt="Screenshot 2025-08-17 133704" src="https://github.com/user-attachments/assets/c7cb3799-4fde-40f4-8774-e27fb9fd38c5" />

---

## Final Model
- **Voting Ensemble** (Logistic Regression + Naive Bayes + SVM) chosen for deployment.  
- Trained using **scikit-learn**.  
- Vectorization with **CountVectorizer**.  


---

## Example Predictions

| Tweet                                         | Prediction             |
|-----------------------------------------------|----------------------|
| "I hate this community, they are all terrible people" | Hate Speech (Class 0) |
| "This is just a bunch of nonsense words"      | Offensive Language (Class 1) |
| "You are an amazing person, keep shining!"    | Neither (Class 2)     |

---

