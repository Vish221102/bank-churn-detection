import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd

# Load data (replace with your data loading)
data = pd.read_csv('bank-full.csv', sep=';')

# Preprocessing
# Convert categorical variables (example for 'job' column)
data = pd.get_dummies(data, columns=['job', 'marital', 'education', 'default', 
                                   'housing', 'loan', 'contact', 'month', 'poutcome'])

# Convert target to binary
data['y'] = data['y'].map({'no':0, 'yes':1}).astype(int)

# Split data
X = data.drop('y', axis=1)
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Calculate scale_pos_weight for imbalance correction
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

# Build XGBoost model
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,  # Critical for imbalanced data
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0, # L2 regularization
    random_state=42,
    eval_metric='auc',
    early_stopping_rounds=10,
    tree_method='hist'  # Faster than 'exact'
)

# Train with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]  # Probabilities for AUC

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score: {roc_auc:.4f}")

# Feature Importance
plt.figure(figsize=(10,6))
xgb.plot_importance(model, max_num_features=15)
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()