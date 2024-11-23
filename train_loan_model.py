"""import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
file_path = "C:\\Users\\Yash\\OneDrive\\Desktop\\Django\\LoanApproval\\loan_approval_dataset.csv"  # Adjust path
loan_data = pd.read_csv(file_path)

# Strip whitespace from column names
loan_data.columns = loan_data.columns.str.strip()

# Handle missing values
# Fill missing numeric values with the median
loan_data.fillna(loan_data.median(numeric_only=True), inplace=True)

# Fill missing categorical values with mode (more domain-relevant than 'Unknown')
for column in loan_data.select_dtypes(include='object').columns:
    loan_data[column].fillna(loan_data[column].mode()[0], inplace=True)

# Encode categorical columns
label_encoders = {}
categorical_columns = ['education', 'self_employed', 'loan_status']
for column in categorical_columns:
    le = LabelEncoder()
    loan_data[column] = le.fit_transform(loan_data[column].astype(str))
    label_encoders[column] = le

# Define features and target
X = loan_data[['no_of_dependents', 'education', 'self_employed', 'income_annum',
               'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
               'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']]
y = loan_data['loan_status']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a pipeline for scaling and model training
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),  # Scale features
    ('model', LogisticRegression(max_iter=2000, class_weight='balanced'))  # Logistic Regression Model
])

# Hyperparameter tuning
param_grid = {
    'model__C': [0.01, 0.1, 1, 10],  # Regularization strength
    'model__solver': ['liblinear', 'lbfgs'],  # Solvers
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model
best_pipeline = grid_search.best_estimator_

# Cross-validation to evaluate model performance
cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")

# Train the best model
best_pipeline.fit(X_train, y_train)

# Evaluate on test set
y_pred = best_pipeline.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate model accuracy
accuracy = best_pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

# Save the pipeline and label encoders
joblib.dump(best_pipeline, 'loan_approval_pipeline.pkl')
joblib.dump(label_encoders, 'loan_label_encoders.pkl')
# Load the saved pipeline
"""
# Updating the 'train_loan_model.py' file with the suggested changes while retaining joblib for saving models and scaler.


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
file_path = "C:\\Users\\Yash\\OneDrive\\Desktop\\Django\\LoanApproval\\loan_approval_dataset.csv"  # Adjust path
loan_data = pd.read_csv(file_path)

# Strip whitespace from column names
loan_data.columns = loan_data.columns.str.strip()

# Handle missing values
# Fill missing numeric values with the median
loan_data.fillna(loan_data.median(numeric_only=True), inplace=True)

# Fill missing categorical values with mode (more domain-relevant than 'Unknown')
for column in loan_data.select_dtypes(include='object').columns:
    loan_data[column].fillna(loan_data[column].mode()[0], inplace=True)

# Add calculated Assets column
loan_data['Assets'] = (loan_data['residential_assets_value'] +
                       loan_data['commercial_assets_value'] +
                       loan_data['luxury_assets_value'])

# Encode categorical columns
label_encoders = {}
categorical_columns = ['education', 'self_employed', 'loan_status']
for column in categorical_columns:
    le = LabelEncoder()
    loan_data[column] = le.fit_transform(loan_data[column].astype(str))
    label_encoders[column] = le

# Define features and target
X = loan_data[['no_of_dependents', 'education', 'self_employed', 'income_annum',
               'loan_amount', 'loan_term', 'cibil_score', 'Assets']]
y = loan_data['loan_status']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a pipeline for scaling and model training
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Use StandardScaler
    ('model', LogisticRegression(max_iter=2000, class_weight='balanced'))  # Logistic Regression Model
])

# Hyperparameter tuning
param_grid = {
    'model__C': [0.01, 0.1, 1, 10],  # Regularization strength
    'model__solver': ['liblinear', 'lbfgs'],  # Solvers
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model
best_pipeline = grid_search.best_estimator_

# Cross-validation to evaluate model performance
cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")

# Train the best model
best_pipeline.fit(X_train, y_train)

# Evaluate on test set
y_pred = best_pipeline.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate model accuracy
accuracy = best_pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

# Save the pipeline and label encoders using joblib
joblib.dump(best_pipeline, 'loan_approval_pipeline.pkl')
joblib.dump(label_encoders, 'loan_label_encoders.pkl')