from google.cloud.aiplatform import CustomTrainingJob
from google.cloud import aiplatform

# Define your project and location
PROJECT_ID = 'heart-attack-ml-435812'
LOCATION = 'europe-west2'

# Initialize AI Platform
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Define your training script
file_path = 'main.py'

# Create a custom training job
job = CustomTrainingJob(
    display_name='test-training-job',
    script_path=file_path,
    container_uri='us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.1-3:latest',
    requirements=['pandas', 'scikit-learn', 'xgboost']
)

# Run the job
job.run(
    replica_count=1,
    machine_type='n1-standard-4',
    args=['--data-path', 'gs://heart-attack-ml/heart-disease.csv']
)
