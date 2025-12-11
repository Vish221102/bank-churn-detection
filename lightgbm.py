import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('bank-full.csv', sep=';')

# Encode categorical variables
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Split features and labels
X = df.drop('y', axis=1)
y = df['y']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Train base model: LightGBM
lgbm = LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42)
lgbm.fit(X_train, y_train)

# Use LightGBM to generate prediction probabilities
lgbm_train_prob = lgbm.predict_proba(X_train)[:, 1].reshape(-1, 1)
lgbm_test_prob = lgbm.predict_proba(X_test)[:, 1].reshape(-1, 1)

# Use these probabilities as input to logistic regression
meta_model = LogisticRegression()
meta_model.fit(lgbm_train_prob, y_train)

# Predict using the stacked model
final_probs = meta_model.predict_proba(lgbm_test_prob)[:, 1]
final_preds = (final_probs >= 0.6).astype(int)

# Evaluation
print("Classification Report:\n", classification_report(y_test, final_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, final_preds))
print("ROC AUC Score:", roc_auc_score(y_test, final_probs))

