import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
import logging
import time

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Read the CSV file
data = pd.read_csv('data.csv')
logger.info('CSV file read successfully')

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
logger.info(f"Train set size: {len(train_features)}")
logger.info(f"Validation set size: {len(val_features)}")
logger.info(f"Test set size: {len(test_features)}")

# Define the model architecture
model = xgb.XGBClassifier()

# Train the model
start_time = time.time()
model.fit(train_subset_features, train_subset_target)
end_time = time.time()
fitting_time = end_time - start_time
logger.info(f'Fitting Time: {fitting_time} seconds')

# # Evaluate the model on the validation set
# val_predictions = model.predict(val_features)

# # Calculate validation accuracy using sklearn metrics
# val_accuracy = metrics.accuracy_score(val_target, val_predictions)
# logger.info(f'Validation Accuracy: {val_accuracy}')

# # Generate confusion matrix
# confusion_matrix = metrics.confusion_matrix(val_target, val_predictions)
# logger.info(f'Confusion Matrix:\n{confusion_matrix}')

# # Calculate false positive rate
# false_positive_rate = confusion_matrix[0, 1] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
# logger.info(f'False Positive Rate: {false_positive_rate}')

# # Calculate false negative rate
# false_negative_rate = confusion_matrix[1, 0] / (confusion_matrix[1, 0] + confusion_matrix[1, 1])
# logger.info(f'False Negative Rate: {false_negative_rate}')

# # Create oneway plots for each feature
# for feature in features.columns:
#     create_oneway_plot(data, feature, 'target')

# # Evaluate the model on the test set
# test_predictions = model.predict(test_features)
# test_accuracy = (test_predictions == test_target).mean()
# print(f'Test Accuracy: {test_accuracy}')
