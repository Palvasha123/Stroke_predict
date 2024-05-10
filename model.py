import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# Load dataset
dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Preprocessing
dataset = dataset.drop(['id'], axis=1)
dataset["gender"] = dataset["gender"].replace("Other", "Female")
print(dataset.head())
print("Number of columns in preprocessed dataset:", dataset.shape[1])
dataset[["hypertension", "heart_disease", "stroke"]].astype(str)
dataset.info()
dataset = pd.get_dummies(dataset, drop_first=True)
print(dataset.head())
print("Number of columns in preprocessed dataset:", dataset.shape[1])
dataset.info()
# Resampling
oversample = RandomOverSampler(sampling_strategy="minority")
x = dataset.drop(['stroke'], axis=1)
y = dataset["stroke"]
x_over, y_over = oversample.fit_resample(x, y)

# Feature scaling
scaler = StandardScaler()
x_over_scaled = scaler.fit_transform(x_over)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
x_over_scaled = imputer.fit_transform(x_over_scaled)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_over_scaled, y_over, test_size=0.2, random_state=42)
num_columns = x_train.shape[1]
print("Number of columns in X_train:", num_columns)

# Define KNeighborsClassifier
# knn_classifier = KNeighborsClassifier()

# Train the KNeighborsClassifier
# knn_classifier.fit(x_train, y_train)

# Make predictions on test set
# y_pred = knn_classifier.predict(x_test)
xgb_classifier = XGBClassifier()

# Train the XGBClassifier
xgb_classifier.fit(x_train, y_train)

# Make predictions on test set
y_pred = xgb_classifier.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("AUC-ROC:", auc_roc)
print("Confusion Matrix:")
print(cm)

pickle.dump(xgb_classifier,open("model.pkl","wb"))
