# /**********************************************************************************************************
# Step 1: Import necessary libraries
# /**********************************************************************************************************
import pandas as pd
import logging
import time
from catboost import CatBoostClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
from datetime import datetime, timedelta
from sklearn.utils.class_weight import compute_class_weight

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

# Create a new column 'HourOfDay' that indicates the hour of the day
data['HourOfDay'] = data['DepTime'].dt.hour

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

# Calculate class weights
classes = np.array([0, 1])
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=data['target'])
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
logger.info('Class Weights:', class_weight_dict)

# /**********************************************************************************************************
# Step 3: Define function that creates features
# /**********************************************************************************************************
def engineer_features(df):
    # Make a copy of the dataframe to avoid modifying the original
    data = df.copy()
    
    # Ensure the data is sorted by ContinuousDate
    data = data.sort_values('ContinuousDate')

    # Helper function to calculate lagged aggregations
    def lagged_agg(group, agg_func):
        return group.shift().expanding().agg(agg_func)

    # 1. Temporal Features
    data['DayOfYear'] = data['ContinuousDate'].dt.dayofyear
    data['WeekOfYear'] = data['ContinuousDate'].dt.isocalendar().week
    data['IsWeekend'] = data['ContinuousDate'].dt.dayofweek.isin([5, 6]).astype(int)
    data['Month'] = data['ContinuousDate'].dt.month
    data['Season'] = pd.cut(data['Month'], bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'], include_lowest=True)
    data['QuarterOfYear'] = data['ContinuousDate'].dt.quarter

    # 2. Carrier-related Features
    data['CarrierDelayRate'] = data.groupby('UniqueCarrier')['target'].transform(lambda x: lagged_agg(x, 'mean'))
    data['CarrierOriginDelayRate'] = data.groupby(['UniqueCarrier', 'Origin'])['target'].transform(lambda x: lagged_agg(x, 'mean'))
    data['CarrierRouteDelayRate'] = data.groupby(['UniqueCarrier', 'Origin', 'Dest'])['target'].transform(lambda x: lagged_agg(x, 'mean'))
    data['CarrierRouteTimeDelayRate'] = data.groupby(['UniqueCarrier', 'Origin', 'Dest', 'HourOfDay'])['target'].transform(lambda x: lagged_agg(x, 'mean'))

    # 3. Airport-related Features
    data['OriginAirportDelayRate'] = data.groupby('Origin')['target'].transform(lambda x: lagged_agg(x, 'mean'))
    data['DestAirportDelayRate'] = data.groupby('Dest')['target'].transform(lambda x: lagged_agg(x, 'mean'))
    data['OriginAirportBusyness'] = data.groupby('Origin')['target'].transform(lambda x: lagged_agg(x, 'count'))
    data['DestAirportBusyness'] = data.groupby('Dest')['target'].transform(lambda x: lagged_agg(x, 'count'))

    # 4. Route-specific Features
    data['RoutePopularity'] = data.groupby(['Origin', 'Dest'])['target'].transform(lambda x: lagged_agg(x, 'count'))
    data['RouteDelayRate'] = data.groupby(['Origin', 'Dest'])['target'].transform(lambda x: lagged_agg(x, 'mean'))

    # 5. Distance-related Features
    data['DistanceBins'] = pd.cut(data['Distance'], bins=[0, 500, 1000, np.inf], labels=['Short', 'Medium', 'Long'])

    # 6. Time-related Features
    data['IsMorningFlight'] = ((data['HourOfDay'] >= 6) & (data['HourOfDay'] < 10)).astype(int)
    data['IsEveningFlight'] = ((data['HourOfDay'] >= 18) & (data['HourOfDay'] < 22)).astype(int)
    data['IsNightFlight'] = ((data['HourOfDay'] >= 22) | (data['HourOfDay'] < 6)).astype(int)

    # 7. Historical Delay Features
    for window in [7, 30]:
        # Calculate rolling average for each Origin-Dest pair
        data[f'MovingAverageDelay{window}Day'] = data.groupby(['Origin', 'Dest'])['target'].transform(
            lambda x: x.shift().rolling(window=window, min_periods=1).mean()
        )

    # Exponential Moving Average
    data['ExponentialMovingAverageDelay'] = data.groupby(['Origin', 'Dest'])['target'].transform(
        lambda x: x.shift().ewm(span=7, adjust=False).mean()
    )

    # Calculate delay trend
    data['DelayTrend'] = data.groupby(['Origin', 'Dest'])['target'].transform(
        lambda x: x.shift().diff().rolling(window=7, min_periods=1).mean()
    )

    # 8. Combination Features
    data['CarrierDayOfWeek'] = data['UniqueCarrier'] + '_' + data['ContinuousDate'].dt.dayofweek.astype(str)
    data['OriginDayOfWeek'] = data['Origin'] + '_' + data['ContinuousDate'].dt.dayofweek.astype(str)
    data['DestDayOfWeek'] = data['Dest'] + '_' + data['ContinuousDate'].dt.dayofweek.astype(str)
    data['CarrierMonth'] = data['UniqueCarrier'] + '_' + data['Month'].astype(str)
    data['OriginMonth'] = data['Origin'] + '_' + data['Month'].astype(str)
    data['DestMonth'] = data['Dest'] + '_' + data['Month'].astype(str)

    # 9. Congestion Features
    data['FlightsDepartingSameHour'] = data.groupby(['Origin', 'ContinuousDate', 'HourOfDay'])['target'].transform('count') - 1
    data['FlightsArrivingSameHour'] = data.groupby(['Dest', 'ContinuousDate', 'HourOfDay'])['target'].transform('count') - 1
    data['TotalFlightsForDay'] = data.groupby(data['ContinuousDate'].dt.date)['target'].transform('count') - 1

    # 10. Lagged Features
    data['PreviousFlightDelay'] = data.groupby(['UniqueCarrier', 'ContinuousDate'])['target'].shift()
    
    for shift in [1, 7, 30]:
        data[f'PreviousDelayRate_{shift}'] = data.groupby(['Origin', 'Dest'])['target'].shift(shift)

    # 11. Competitive Features
    data['CarriersOnRoute'] = data.groupby(['Origin', 'Dest', data['ContinuousDate'].dt.to_period('D')])['UniqueCarrier'].transform('nunique')
    data['CarrierMarketShareOnRoute'] = (data.groupby(['Origin', 'Dest', 'UniqueCarrier', data['ContinuousDate'].dt.to_period('D')])['target'].transform('count') - 1) / \
                                        (data.groupby(['Origin', 'Dest', data['ContinuousDate'].dt.to_period('D')])['target'].transform('count') - 1)

    # 12. Derived from Existing Features
    data['DistanceTimesTimeOfDay'] = data['Distance'] * data['HourOfDay']
    data['CarrierDistance'] = data['UniqueCarrier'] + '_' + pd.cut(data['Distance'], bins=5).astype(str)
    data['DistanceSquared'] = data['Distance'] ** 2
    data['HourSquared'] = data['HourOfDay'] ** 2

    # 13. First and last flights of the day
    data['IsFirstFlight'] = data.groupby(['UniqueCarrier', 'Origin', data['ContinuousDate'].dt.date])['HourOfDay'].transform('min') == data['HourOfDay']
    data['IsLastFlight'] = data.groupby(['UniqueCarrier', 'Origin', data['ContinuousDate'].dt.date])['HourOfDay'].transform('max') == data['HourOfDay']

    # 14. Delay rates by morning, evening and night
    for time_of_day in ['IsMorningFlight', 'IsEveningFlight', 'IsNightFlight']:
        data[f'{time_of_day[2:-6]}DelayRate'] = data.groupby(['UniqueCarrier', time_of_day])['target'].transform(
            lambda x: lagged_agg(x, 'mean')
        )

    # 15. Non-linear impacts of delays
    data['DelayRateSquared'] = data['RouteDelayRate'] ** 2
    data['DelayRateCubed'] = data['RouteDelayRate'] ** 3

    # 16. Small airlines impact (using lagged calculation to prevent leakage)
    def lagged_size(group):
        return group.expanding().count().shift(1)

    airline_size = data.groupby('UniqueCarrier').size().reset_index(name='size')
    airline_size['lagged_size'] = airline_size.groupby('UniqueCarrier')['size'].transform(lagged_size)
    
    # Merge the airline size back to the main dataframe
    data = data.merge(airline_size[['UniqueCarrier', 'lagged_size']], on='UniqueCarrier', how='left')
    
    # Now we can calculate IsSmallAirline
    small_airline_threshold = data['lagged_size'].quantile(0.25)
    data['IsSmallAirline'] = (data['lagged_size'] < small_airline_threshold).astype(int)
    
    data['SmallAirlineDelayImpact'] = data['IsSmallAirline'] * data.groupby('UniqueCarrier')['target'].transform(
        lambda x: lagged_agg(x, 'mean')
    )

    # Don't forget to drop the 'lagged_size' column if you don't need it in the final dataset
    data = data.drop('lagged_size', axis=1)
    
    # 17. Smaller airports (using lagged calculation to prevent leakage)

    # For Origin airports
    origin_airport_size = data.groupby('Origin').size().reset_index(name='size')
    origin_airport_size['lagged_size_origin'] = origin_airport_size.groupby('Origin')['size'].transform(lagged_size)
    data = data.merge(origin_airport_size[['Origin', 'lagged_size_origin']], on='Origin', how='left')
    
    origin_threshold = data['lagged_size_origin'].quantile(0.25)
    data['IsSmallOriginAirport'] = (data['lagged_size_origin'] < origin_threshold).astype(int)
    
    # For Destination airports
    dest_airport_size = data.groupby('Dest').size().reset_index(name='size')
    dest_airport_size['lagged_size_dest'] = dest_airport_size.groupby('Dest')['size'].transform(lagged_size)
    data = data.merge(dest_airport_size[['Dest', 'lagged_size_dest']], on='Dest', how='left')
    
    dest_threshold = data['lagged_size_dest'].quantile(0.25)
    data['IsSmallDestAirport'] = (data['lagged_size_dest'] < dest_threshold).astype(int)
    
    data['SmallAirportRoute'] = data['IsSmallOriginAirport'] * data['IsSmallDestAirport']

    # 18. Hub airport feature (using lagged calculation to prevent leakage)
    hub_threshold = data['lagged_size_origin'].quantile(0.9)
    data['IsHubAirport'] = (data['lagged_size_origin'] > hub_threshold).astype(int)
    
    # 19. Interaction between carrier and airport size
    data['CarrierSmallAirportInteraction'] = data['IsSmallAirline'] * (data['IsSmallOriginAirport'] | data['IsSmallDestAirport'])
    
    # 20. Busiest hours for each airport
    data['OriginAirportHourlyTraffic'] = data.groupby(['Origin', 'HourOfDay'])['target'].transform('count')
    data['DestAirportHourlyTraffic'] = data.groupby(['Dest', 'HourOfDay'])['target'].transform('count')
    
    # Clean up intermediate columns
    data = data.drop(['lagged_size_origin', 'lagged_size_dest', 'DepTime'], axis=1)
    
    # Identify numeric columns (both integer and float)
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns

    # Replace NULL values with 0 for all numeric columns
    data[numeric_columns] = data[numeric_columns].fillna(0)

    # Log the number of NaNs replaced in numeric columns
    nan_counts_numeric = data[numeric_columns].isna().sum()
    
    if nan_counts_numeric.sum() > 0:
        logger.info("NaN values replaced with 0 in the following numeric columns:")
        logger.info(nan_counts_numeric[nan_counts_numeric > 0])
    else:
        logger.info("No NaN values found in numeric columns after feature engineering.")

    # Check for remaining NaNs in non-numeric columns
    non_numeric_columns = data.columns.difference(numeric_columns)
    remaining_nans = data[non_numeric_columns].isna().sum()
    if remaining_nans.sum() > 0:
        logger.info("\nRemaining NaN values in non-numeric columns:")
        logger.info(remaining_nans[remaining_nans > 0])
    else:
        logger.info("\nNo remaining NaN values in non-numeric columns.")

    return data

# Apply function
engineered_df = engineer_features(data)

# Add a randomly generated feature
engineered_df['RandomFeature'] = np.random.rand(len(engineered_df))

# Log all features
logger.info("Features created. List of all features:")
logger.info(engineered_df.columns.tolist())

# /**********************************************************************************************************
# Step 4: Data transformation
# /**********************************************************************************************************
# Convertcolumns to a categorical type
engineered_df['UniqueCarrier'] = engineered_df['UniqueCarrier'].astype('category')
engineered_df['Origin'] = engineered_df['Origin'].astype('category')
engineered_df['Dest'] = engineered_df['Dest'].astype('category')
engineered_df['Season'] = engineered_df['Season'].astype('category')
engineered_df['DistanceBins'] = engineered_df['DistanceBins'].astype('category')
engineered_df['CarrierDayOfWeek'] = engineered_df['CarrierDayOfWeek'].astype('category')
engineered_df['OriginDayOfWeek'] = engineered_df['OriginDayOfWeek'].astype('category')
engineered_df['DestDayOfWeek'] = engineered_df['DestDayOfWeek'].astype('category')
engineered_df['CarrierMonth'] = engineered_df['CarrierMonth'].astype('category')
engineered_df['OriginMonth'] = engineered_df['OriginMonth'].astype('category')
engineered_df['DestMonth'] = engineered_df['DestMonth'].astype('category')
engineered_df['CarrierDistance'] = engineered_df['CarrierDistance'].astype('category')

# Identify categorical features
categorical_features = [
    'UniqueCarrier', 'Origin', 'Dest', 'Season', 'DistanceBins', 
    'CarrierDayOfWeek', 'OriginDayOfWeek', 'DestDayOfWeek', 
    'CarrierMonth', 'OriginMonth', 'DestMonth', 'CarrierDistance'
]

# /**********************************************************************************************************
# Step 5: Split the data into features and target
# /**********************************************************************************************************
# Sort the data by ContinuousDate
engineered_df = engineered_df.sort_values('ContinuousDate')

# Determine the split date (e.g., use the 80th percentile date as the split point)
split_date = engineered_df['ContinuousDate'].quantile(0.8)

# Split the data into training and validation sets
train_data = engineered_df[engineered_df['ContinuousDate'] <= split_date]
val_data = engineered_df[engineered_df['ContinuousDate'] > split_date]

# Split the features and target for both training and validation sets
train_features = train_data.drop(['target', 'ContinuousDate'], axis=1)
train_target = train_data['target']

val_features = val_data.drop(['target', 'ContinuousDate'], axis=1)
val_target = val_data['target']

# Log the sizes of the datasets
logger.info(f"Total dataset size: {len(data)} rows")
logger.info(f"Train set size: {len(train_features)} rows")
logger.info(f"Validation set size: {len(val_features)} rows")
logger.info(f"Split date: {split_date}")

# Log the date ranges
logger.info(f"Training data date range: {train_data['ContinuousDate'].min()} to {train_data['ContinuousDate'].max()}")
logger.info(f"Validation data date range: {val_data['ContinuousDate'].min()} to {val_data['ContinuousDate'].max()}")

# Log the shapes of the datasets
logger.info(f"Train features shape: {train_features.shape}")
logger.info(f"Train target shape: {train_target.shape}")
logger.info(f"Validation features shape: {val_features.shape}")
logger.info(f"Validation target shape: {val_target.shape}")
                                         
# Define the hyperparameter search space
param_grid = {
    'learning_rate': [0.01, 0.1],
    'depth': [4, 6, 10]
}

# Define the model with categorical features
model = CatBoostClassifier(cat_features=categorical_features)
logger.info(f'Model created with categorical features: {model}')

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=2, verbose=1, n_jobs=-1, refit=True)

# Train the model with hyperparameter tuning
start_time = time.time()
grid_search.fit(
    train_features, train_target,
    eval_set=(val_features, val_target),  # Pass the evaluation set
    early_stopping_rounds=50,  # Stop if no improvement after 50 rounds
    verbose=10  # Print updates every 10 rounds
)
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

# Get feature importance
feature_importance = best_model.feature_importances_

# Create a DataFrame with feature names and importance scores
feature_importance_df = pd.DataFrame({'Feature': train_features.columns, 'Importance': feature_importance})

# Sort the DataFrame by importance scores in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance graph
plt.figure(figsize=(12, 6))
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()
