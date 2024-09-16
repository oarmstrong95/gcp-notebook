import pandas as pd
import logging
import time
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt
from google.cloud import bigquery
from google.cloud import aiplatform
from diagnostics.functions import create_oneway_plot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define your project and location
PROJECT_ID = 'heart-attack-ml-435812'
LOCATION = 'europe-west2'

# Initialize a BigQuery client
client = bigquery.Client(project=PROJECT_ID, location=LOCATION)

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Query the table
query = f"""
    SELECT *
    FROM `{PROJECT_ID}.data.heart-disease`
"""

# Execute the query
query_job = client.query(query)

# Convert the query result to a DataFrame
data = query_job.to_dataframe()
logger.info('BQ table read successfully')

# Split the data into features and target
features = data.drop('target', axis=1)
target = data['target']

# Log the sizes of the datasets
logger.info(f"Features shape: {features.shape}")
logger.info(f"Target shape: {target.shape}")

# Split the data into training, validation, and test sets
train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)
train_subset_features, val_features, train_subset_target, val_target = train_test_split(train_features, train_target, test_size=0.2, random_state=42)

# Log the sizes of the datasets
logger.info(f"Train set size: {len(train_features)} rows")
logger.info(f"Validation set size: {len(val_features)} rows")
logger.info(f"Test set size: {len(test_features)} rows")

# Check the splits
assert len(train_features) == len(train_subset_features) + len(val_features)

# Define the hyperparameter search space
param_dist = {
    'n_estimators': randint(50, 150),  # Fewer trees to prevent overfitting
    'max_depth': randint(2, 6),        # Shallower trees to reduce complexity
    'learning_rate': uniform(0.05, 0.15),  # Moderate learning rates
}

# Define the model
model = xgb.XGBClassifier()

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=5, scoring='recall', verbose=2, n_jobs=-1, random_state=42, refit=True)

# Train the model with hyperparameter tuning
start_time = time.time()
random_search.fit(train_subset_features, train_subset_target)
end_time = time.time()
fitting_time = end_time - start_time
logger.info(f'Fitting Time with Hyperparameter Tuning: {fitting_time} seconds')

# Get the best model
best_model = random_search.best_estimator_
logger.info(f'Best Parameters: {random_search.best_params_}')

# Evaluate the best model on the validation set
val_predictions = best_model.predict(val_features)

# Generate confusion matrix
confusion_matrix = metrics.confusion_matrix(val_target, val_predictions)
tn, fp, fn, tp = confusion_matrix.ravel()

# Calculate total number of predictions
total_predictions = tn + fp + fn + tp
logger.info(f'Total Predictions: {total_predictions}')

# Log the confusion matrix
logger.info(f'True Negatives: {tn}')
logger.info(f'False Positives: {fp}')
logger.info(f'False Negatives: {fn}')
logger.info(f'True Positives: {tp}')

# Calculate false positive rate
false_positive_rate = fp / (tn + fp)
logger.info(f'False Positive Rate: {false_positive_rate}')

# Calculate false negative rate
false_negative_rate = fn / (fn + tp)
logger.info(f'False Negative Rate: {false_negative_rate}')

# Calculate ROC score
roc_score = metrics.roc_auc_score(val_target, val_predictions)
logger.info(f'ROC Score: {roc_score}')

# Calculate F1 score
f1_score = metrics.f1_score(val_target, val_predictions)
logger.info(f'F1 Score: {f1_score}')

# Combine val_features and val_target into a single DataFrame
val_data = pd.concat([val_features, val_target], axis=1)

# Create oneway plots for each feature using validation data
for feature in val_data.columns[:-1]:
    create_oneway_plot(val_data, feature, 'target')