# backend/tasks.py

from celery import Celery
from backend.utils import process_file
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Celery with the correct app name and broker/backend URLs
app = Celery(
    'backend.tasks',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)

# Optionally, load additional configurations from a config module
app.config_from_object('backend.celeryconfig')


@app.task
def process_file_task(file_path, meta_model, provider_weights, confidence_threshold):
    """
    Celery task to process a file with the specified meta_model, provider_weights, and confidence_threshold.

    Args:
        file_path (str): Path to the uploaded requirements file.
        meta_model (str): The meta-model to use for classification.
        provider_weights (dict): Weights assigned to each provider.
        confidence_threshold (float): Confidence threshold for voting algorithm.

    Returns:
        dict: Contains 'results' and 'output_file' path.
    """
    return process_file(file_path, meta_model, provider_weights, confidence_threshold)
