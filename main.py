# /**********************************************************************************************************
# Step 1: Import necessary libraries
# /**********************************************************************************************************
import pandas as pd
import logging
import time
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
from datetime import datetime, timedelta

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
    FROM `{PROJECT_ID}.data.training_data`
"""

# Execute the query
query_job = client.query(query)

# Convert the query result to a DataFrame
data = query_job.to_dataframe()
logger.info('BQ table read successfully')

# Function to remove the letter 'c' and convert to integer
def convert_date(value):
    return int(value.replace('c-', ''))

# Relabel the column 'dep_delayed_15min' to 'target'
data = data.rename(columns={'dep_delayed_15min': 'target'})

# Change the values of True to 1 and False to 0 in the 'target' column
data['target'] = data['target'].map({True: 1, False: 0})
logger.info('Relablelled target successfully')

# /**********************************************************************************************************
# Step 2: Preprocess the data
# /**********************************************************************************************************
# Apply the function to the 'month' column
data['Month'] = data['Month'].apply(convert_date)
data['DayofMonth'] = data['DayofMonth'].apply(convert_date)
data['DayOfWeek'] = data['DayOfWeek'].apply(convert_date)

# Convert DepTime to a time type
data['DepTime'] = pd.to_datetime(data['DepTime'], format='%H%M', errors='coerce').dt.time
data['DepTime'] = pd.to_datetime(data['DepTime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else None))

# Convert the 'UniqueCarrier' column to a categorical type
data['UniqueCarrier'] = data['UniqueCarrier'].astype('category')

# Convert the 'Origin' and 'Dest' columns to a categorical type
data['Origin'] = data['Origin'].astype('category')
data['Dest'] = data['Dest'].astype('category')

def create_continuous_date_index(df, start_year=2000):
    # Sort the dataframe
    df = df.sort_values(['Month', 'DayofMonth', 'DepTime'])
    
    # Create initial date column
    df['Date'] = pd.to_datetime(df.apply(lambda row: f'{start_year}-{int(row["Month"]):02d}-{int(row["DayofMonth"]):02d}', axis=1), errors='coerce')
    
    # Drop rows where date creation failed (e.g., invalid day for the month)
    df = df.dropna(subset=['Date'])
    
    # Initialize year and create continuous date
    current_year = start_year
    continuous_date = []
    prev_date = None
    
    for _, row in df.iterrows():
        current_date = row['Date']
        
        if prev_date is not None:
            # Check if we've wrapped around to a new year
            if current_date < prev_date:
                current_year += 1
            # Check for large gaps
            elif (current_date - prev_date).days > 180:  # Assume gaps larger than 6 months indicate a new year
                current_year += 1
        
        new_date = current_date.replace(year=current_year)
        continuous_date.append(new_date)
        prev_date = current_date
    
    df['ContinuousDate'] = continuous_date
    
    # Recreate DayOfWeek based on ContinuousDate
    df['RecreatedDayOfWeek'] = df['ContinuousDate'].dt.dayofweek + 1  # Monday = 1, Sunday = 7

    # Drop the 'DayOfWeek' column
    df = df.drop('DayOfWeek', axis=1)
    df = df.drop('Date', axis=1)
    
    # Sort the data
    df['ContinuousDate'] = pd.to_datetime(df['ContinuousDate'])
    
    return df

# Apply the updated function
data = create_continuous_date_index(data)

# Check for any NaN, None or missing values in all columns
missing_values = data.isnull().sum()

# Drop the rows with missing values
data = data.dropna()

# /**********************************************************************************************************
# Step 2: Time based features
# /**********************************************************************************************************
# Create a new column 'TimeOfDay' that indicates whether it's night time, afternoon, lunch, or morning
data['TimeOfDay'] = pd.cut(data['DepTime'].dt.hour,
                           bins=[0, 6, 12, 18, 24],
                           labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                           include_lowest=True)

# Create a new column 'HourOfDay' that indicates the hour of the day
data['HourOfDay'] = data['DepTime'].dt.hour

# Create a new column 'PeakHour' that indicates whether it's peak hour or not
data['PeakHour'] = data['DepTime'].dt.hour.isin(range(7, 10)) | data['DepTime'].dt.hour.isin(range(16, 19))

# Create a new column 'HasHolidays' that indicates whether that month has holidays
data['HasHolidays'] = data['Month'].isin([1, 7, 12])

# # Create a new column 'DistanceCategory' that categorizes the distance into short, midrange, long, or extra long
# data['DistanceCategory'] = pd.cut(data['Distance'],
#                                  bins=[0, 30, 500, 1500, 5000],
#                                  labels=['Short', 'Midrange', 'Long', 'Extra Long'],
#                                  include_lowest=True)

# /**********************************************************************************************************
# Step 3: Time series features
# /**********************************************************************************************************
def create_rolling_features(df, group_columns, target_column='target', window=7):
    # Ensure the dataframe is sorted properly
    df = df.sort_values(group_columns + ['ContinuousDate', 'DepTime'])
    
    # List of features to create rolling statistics for
    features = ['Distance', 'RecreatedDayOfWeek']
    
    for feature in features:
        # Rolling mean
        df[f'{feature}_rolling_mean'] = df.groupby(group_columns)[feature].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
        
        # Rolling standard deviation
        df[f'{feature}_rolling_std'] = df.groupby(group_columns)[feature].transform(
            lambda x: x.rolling(window=window, min_periods=1).std().shift(1)
        )
    
    # Create rolling delay statistics
    df['is_delayed'] = df[target_column] == 1
    
    # Rolling delay rate
    df['rolling_delay_rate'] = df.groupby(group_columns)['is_delayed'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
    )
    
    # Rolling delay count
    df['rolling_delay_count'] = df.groupby(group_columns)['is_delayed'].transform(
        lambda x: x.rolling(window=window, min_periods=1).sum().shift(1)
    )
    
    # Create time-based features
    df['DayOfYear'] = df['ContinuousDate'].dt.dayofyear
    df['WeekOfYear'] = df['ContinuousDate'].dt.isocalendar().week
    
    # Rolling statistics for these time-based features
    for time_feature in ['DayOfYear', 'WeekOfYear']:
        df[f'{time_feature}_rolling_delay_rate'] = df.groupby(time_feature)['is_delayed'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
    
    # Clean up intermediate columns
    df = df.drop(['is_delayed'], axis=1)
    
    return df

# Define the grouping columns
group_columns = ['UniqueCarrier', 'Origin']

# Apply the function with a window of 7 past observations
data = create_rolling_features(data, group_columns, window=7)

# /**********************************************************************************************************
# Step 3: Delayed features
# /**********************************************************************************************************

# Sort the DataFrame by UniqueCarrier, Origin, ContinuousDate, and DepTime to ensure historical order
data = data.sort_values(by=['UniqueCarrier', 'Origin', 'ContinuousDate', 'DepTime'])

# Check if the previous flight for the 'UniqueCarrier' on the same 'ContinuousDate' was delayed
data['PreviousFlightDelayed'] = data.groupby(['UniqueCarrier', 'ContinuousDate'], observed=True)['target'].shift(1)

# Convert the NaN values to 0
data['PreviousFlightDelayed'] = data['PreviousFlightDelayed'].fillna(0)

# Count the number of flights delayed by 'Origin', 'UniqueCarrier', 'ContinuousDate' before the DepTime for that row
data['CountDelayedBefore'] = data.groupby(['Origin', 'UniqueCarrier', 'ContinuousDate'], observed=True)['target'].cumsum().subtract(data['target'])

# Add a randomly generated feature
data['RandomFeature'] = np.random.rand(len(data))

logger.info('Features created successfully')

# /**********************************************************************************************************
# Step 3: Drop cols
# /**********************************************************************************************************

# Drop the 'DepTime' column
data = data.drop(['DepTime', 'ContinuousDate'], axis=1)

logger.info('Features dropped successfully')

# /**********************************************************************************************************
# Step 4: Split the data into features and target
# /**********************************************************************************************************
# Split the data into features and target
features = data.drop('target', axis=1)
target = data['target']

# Log the sizes of the datasets
logger.info(f"Features shape: {features.shape}")
logger.info(f"Target shape: {target.shape}")

# Split the data into training, validation, and test sets
train_features, val_features, train_target, val_target = train_test_split(features, target, test_size=0.2, random_state=42)

# Log the sizes of the datasets
logger.info(f"Train set size: {len(train_features)} rows")
logger.info(f"Validation set size: {len(val_features)} rows")
                                         
# Define the hyperparameter search space
param_grid = {
    'eta': [0.01, 0.1],
    'max_depth': [3, 5],
    'min_child_weight': [1, 3],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0],
    'gamma': [0, 1],
    'alpha': [0, 1],
    'lambda': [0, 1]
}

# Define the model
xgb_model = xgb.XGBClassifier(enable_categorical=True)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=1, n_jobs=-1, refit=True)

# Train the model with hyperparameter tuning
start_time = time.time()
grid_search.fit(train_features, train_target)
end_time = time.time()
fitting_time = end_time - start_time
logger.info(f'Fitting Time with Hyperparameter Tuning: {fitting_time} seconds')

# Get the best model
best_model = grid_search.best_estimator_
logger.info(f'Best Parameters: {grid_search.best_params_}')

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

# Create a bar chart of the number of flights by carrier and the proportion of delayed flights

# Calculate the number of flights by carrier
carrier_counts = data['UniqueCarrier'].value_counts()

# Calculate the proportion of delayed flights by carrier
delayed_proportion = data.groupby('UniqueCarrier')['target'].mean()

fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot the number of flights by carrier on the left y-axis
carrier_counts.plot(kind='bar', ax=ax1, color='b', alpha=0.6, position=0)
ax1.set_xlabel('Carrier')
ax1.set_ylabel('Number of Flights', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_xticklabels(carrier_counts.index, rotation=45)

# Create a second y-axis for the proportion of delayed flights
ax2 = ax1.twinx()
delayed_proportion.plot(kind='line', ax=ax2, color='r', marker='o', linewidth=2)
ax2.set_ylabel('Proportion of Delayed Flights', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('Number of Flights and Proportion of Delayed Flights by Carrier')
fig.tight_layout()
plt.show()