import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
from imblearn.over_sampling import SMOTE


# Data Exploration
dataset = pd.read_csv("WISDM_at_v2.0_transformed.csv")
"""
pd.options.display.max_columns = None
print("Head of Dataset\n", dataset.head(10))
print("Info of Dataset\n", dataset.info())
print("Description of Dataset\n", dataset.describe())
print(pd.crosstab(index=dataset["Activity"], columns="count"))
plt.figure(figsize=(10,5))
ax = sns.countplot(x="Activity", data=dataset)
plt.xticks(x=dataset['Activity'],  rotation='vertical')
plt.show(
)
"""

# Data Preprocessing
# Splitting Dataset into x and y
x = dataset.drop("Activity", axis=1)
y = dataset["Activity"]

"""
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
"""

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
y_tree = tree_clf.predict(x_train)

# Accuracy Before Performing Cross Validation
train_acc = accuracy_score(y_tree, y_train)
print("\nTraining Accuracy before CV: {:.4f}".format(train_acc))

# Confusion Matrix Before Performing Cross Validation
cm = confusion_matrix(y_train, y_tree)
print("\nConfusion Matrix before CV:\n", cm)

# Performing Cross Validation
k_scores = cross_val_score(tree_clf, x_train, y_train, cv=5, scoring='accuracy')
print("\nTraining Accuracy after CV:\n", k_scores)

# Confusion Matrix
y_tree_cv = cross_val_predict(tree_clf, x_train, y_train, cv=5)
cm = confusion_matrix(y_train, y_tree_cv)
print("\nConfusion Matrix after CV:\n", cm)
classes = unique_labels(y_train, y_tree_cv)

# Create figure and axis
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

# Add title, labels, and tick marks
ax.set(title='Decision Tree',
       xlabel='Predicted Value',
       ylabel='Actual Value')
tick_marks = np.arange(len(classes))
ax.set(xticks=tick_marks, xticklabels=classes,
       yticks=tick_marks, yticklabels=classes)

# Rotate the tick labels for better readability
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

# Show the plot
plt.tight_layout()
plt.show()

# Per Class Accuracy
matrix = confusion_matrix(y_train, y_tree_cv)
print(matrix.diagonal()/matrix.sum(axis=1))

# Measure the Accuracy, Precision, Recall and f1 score
print('\nAccuracy  = {:.4f}'.format(accuracy_score(y_train, y_tree_cv)))
print('Precision = {:.4f}'.format(precision_score(y_train, y_tree_cv, average='macro')))
print('Recall    = {:.4f}'.format(recall_score(y_train, y_tree_cv, average='macro')))
print('f1 score  = {:.4f}'.format(f1_score(y_train, y_tree_cv, average='macro')))


# # Fine-Tune the Model Using Grid Search
# param_grid = {
#     'max_depth': [2, 3, 5, 10, 20],
#     'min_samples_leaf': [5, 10, 20, 50, 100],
#     'criterion': ["gini", "entropy"]
# }
# grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
# print ('Performing grid search...')
# grid_search.fit(x_train, y_train)
# print('done')

# # View the accuracy score
# print('Best score for training data:', grid_search.best_score_, "\n")

# # View the best parameters for the model found using grid search
# print('Best Parameter:', grid_search.best_params_)
# #'criterion': 'gini', 'max_depth': 20, 'min_samples_leaf': 5

# final_model = grid_search.best_estimator_
# print(final_model)

# # Predicting the test set
# y_test = final_model.predict(x_test)
# print('Test accuracy = {:.4f}'.format(accuracy_score(y_test, y_test)))
# print('Test precision = {:.4f}'.format(precision_score(y_test, y_test, average='macro')))
# print('Test recall = {:.4f}'.format(recall_score (y_test, y_test, average='macro')))
# print('Test f1 score = {:.4f}'.format(f1_score(y_test, y_test, average='macro')))
