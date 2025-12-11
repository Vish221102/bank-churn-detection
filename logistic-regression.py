# INSTALL REQUIRED PACKAGES (run once)
# !pip install optuna scikit-learn pandas matplotlib

import optuna
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, classification_report, 
                            confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

# 1. Load and Prepare Data
data = pd.read_csv('bank-full.csv', sep=';')
X = pd.get_dummies(data.drop('y', axis=1))  # Auto-handle categoricals
y = data['y'].map({'no':0, 'yes':1})  # Binary encoding

# 2. Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 3. Optuna Optimization
def objective(trial):
    params = {
        'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0, 1),
        'penalty': 'elasticnet',
        'solver': 'saga',
        'class_weight': 'balanced',
        'max_iter': 1000
    }
    
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:,1]
    return roc_auc_score(y_test, y_proba)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, show_progress_bar=True)

# 4. Train Final Model
best_params = study.best_params
best_params.update({'penalty':'elasticnet', 'solver':'saga', 'class_weight':'balanced'})

final_model = LogisticRegression(**best_params)
final_model.fit(X_train, y_train)

# 5. Evaluation
y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:,1]

print("\nBest Parameters:", study.best_params)
print("Test AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

# Feature Importance (absolute coefficient values)
importance = pd.Series(np.abs(final_model.coef_[0]), 
                      index=pd.get_dummies(data.drop('y', axis=1)).columns)
importance.nlargest(10).plot(kind='barh', ax=ax1, title='Top 10 Feature Importances')

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax2, 
                                       display_labels=['No Churn', 'Churn'])
ax2.set_title('Confusion Matrix')

plt.tight_layout()
plt.show()