# Credit Card Fraud Detection

## Project Overview
This project explores the application of various Machine Learning and Deep Learning algorithms to detect fraudulent credit card transactions. Dealing with financial fraud requires high accuracy and the ability to distinguish between legitimate and fraudulent activities within highly imbalanced datasets.

We implemented and compared four different models—Logistic Regression, Random Forest, XGBoost, and a Convolutional Neural Network (CNN)—to identify the most effective solution for minimizing financial losses while maintaining a smooth user experience.

## Dataset
The dataset consists of transactions made by credit cards in September 2013 by European cardholders.

- Source: Kaggle Credit Card Fraud Detection
- Transactions: 284,807 total transactions.
- Imbalance: Only 492 transactions are fraudulent (0.172%), making this a highly unbalanced classification problem.
- Features:
  - Time: Seconds elapsed between this transaction and the first transaction in the dataset.
  - Amount: The transaction amount.
  - V1 - V28: Principal components obtained with PCA (features are anonymized for privacy).
  - Class: Target variable (1 = Fraud, 0 = Legitimate).

## Technologies & Libraries Used
The project is implemented in Python using the following libraries:

- Data Manipulation: pandas, numpy
- Visualization: seaborn, matplotlib
- Machine Learning: scikit-learn, xgboost
- Deep Learning: tensorflow, keras
- Imbalance Handling: imbalanced-learn (SMOTE)

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Visualized the class distribution to highlight the severe imbalance.
- Analyzed the distribution of Time and Amount features.
- Generated correlation matrices to identify relationships between features and the target variable.

### 2. Data Preprocessing
- Scaling: Applied StandardScaler and RobustScaler to the Amount and Time columns to normalize their range.
- Splitting: Divided the dataset into training (80%) and testing (20%) sets.
- Resampling: Applied SMOTE (Synthetic Minority Over-sampling Technique) to the training data to balance the classes, ensuring models don't just predict the majority class.

### 3. Model Training
We trained four distinct models to compare their performance:

1. Logistic Regression: A baseline model for binary classification.
2. Random Forest: An ensemble method using multiple decision trees.
3. XGBoost: A high-performance gradient boosting framework.
4. Convolutional Neural Network (CNN): A deep learning model designed with 1D convolutional layers to capture patterns in the tabular data.

## Results & Evaluation
Models were evaluated on the untouched Test Set using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

| Model | ROC AUC | Precision | Recall | F1-Score | Accuracy |
|-------|---------|-----------|--------|----------|----------|
| XGBoost | 0.9995 | 0.81 | 0.89 | 0.85 | 0.9862 |
| Random Forest | 0.9996 | 0.94 | 0.84 | 0.89 | 0.9527 |
| CNN | 0.9964 | 0.30 | 0.83 | 0.44 | 0.9607 |
| Logistic Regression | 0.9992 | 0.83 | 0.65 | 0.73 | 0.9581 |

(Note: Metrics are based on the test set evaluation.)

### Key Findings
- XGBoost emerged as the best overall model, offering the highest ROC-AUC and excellent Recall (detecting 89% of fraud cases).
- Random Forest achieved the highest Precision, making it the best choice if minimizing false alarms is the priority.
- CNN showed promise with high recall but suffered from low precision in this specific configuration.

## Usage
1. Clone the repository:
   git clone https://github.com/yourusername/CreditCardFraud.git

2. Install dependencies:
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow imbalanced-learn

3. Run the Notebook:
   Open CreditCardFraud.ipynb in Jupyter Notebook or Google Colab and execute the cells sequentially.

## Authors
- Namrata Mali
- Divakar Reddy Ravi
- Khushiben Nareshbhai Patel
- Tung Tran
- City University of Seattle - MS Computer Science
