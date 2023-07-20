import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
from imblearn.over_sampling import SMOTE
import pickle

# Data Exploration
dataset = pd.read_csv("WISDM_at_v2.0_transformed.csv")

# Splitting Dataset into x and y
x = dataset.drop(["Activity", "user", "id"], axis=1)
y = dataset["Activity"]

# Data Preprocessing
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Split the input features to x, and output prediction to y
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

# Use SMOTE for oversampling the imbalanced data
smote = SMOTE(random_state= 27)
x_train, y_train = smote.fit_resample(x_train, y_train)

# Decision Tree Model
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(x_train, y_train)

pickle.dump(tree_clf, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))