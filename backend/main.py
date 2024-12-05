# backend/main.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from backend.tasks import process_file_task
import os
from dotenv import load_dotenv
import shutil
import uuid
import json

# Load environment variables
load_dotenv()

app = FastAPI()

# Define the directory to save uploaded files
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploaded_files")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    meta_model: str = Form(...),
    provider_weights: str = Form(...),
    confidence_threshold: float = Form(...)
):
    """
    Endpoint to upload a requirements file along with the selected meta_model, provider_weights, and confidence_threshold.

    Args:
        file (UploadFile): The uploaded requirements file.
        meta_model (str): The selected meta-model for classification.
        provider_weights (str): JSON string of provider weights.
        confidence_threshold (float): Confidence threshold for voting algorithm.

    Returns:
        dict: Contains the Celery task ID.
    """
    # Validate meta_model
    valid_meta_models = ['openai', 'anthropic', 'groq', 'gemma2']
    if meta_model not in valid_meta_models:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid meta_model. Choose from {valid_meta_models}."}
        )

    # Validate confidence_threshold
    if not (0.5 <= confidence_threshold <= 1.0):
        return JSONResponse(
            status_code=400,
            content={"error": "Confidence threshold must be between 0.5 and 1.0."}
        )

    # Parse provider_weights from JSON string to dict
    try:
        provider_weights_dict = json.loads(provider_weights)
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid provider_weights format. Must be a valid JSON string."}
        )

    # Check if all four providers are present
    required_providers = ['openai', 'anthropic', 'groq', 'gemma2']
    missing_providers = [p for p in required_providers if p not in provider_weights_dict]
    if missing_providers:
        return JSONResponse(
            status_code=400,
            content={"error": f"Missing weights for providers: {', '.join(missing_providers)}."}
        )

    # Ensure all weights are non-negative numbers
    for provider, weight in provider_weights_dict.items():
        if not isinstance(weight, (int, float)) or weight < 0:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid weight for provider '{provider}'. Must be a non-negative number."}
            )

    # Check if the file is a text file
    if not file.filename.endswith(".txt"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only text files are supported. Please upload a .txt file."}
        )

    # Save the uploaded file with a unique name to prevent overwriting
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_location = os.path.join(UPLOAD_DIR, unique_filename)

    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Start the Celery task with all parameters
    task = process_file_task.delay(file_location, meta_model, provider_weights_dict, confidence_threshold)
    return {"task_id": task.id}


@app.get("/results/{task_id}")
def get_results(task_id: str):
    """
    Endpoint to retrieve the status and results of a classification task.

    Args:
        task_id (str): The Celery task ID.

    Returns:
        dict: Contains the task status and data if completed.
    """
    # Import Celery app dynamically to prevent circular imports
    from backend.tasks import app as celery_app
    task_result = celery_app.AsyncResult(task_id)
    if task_result.state == 'SUCCESS':
        return {"status": "completed", "results": task_result.result.get('results', [])}
    elif task_result.state == 'PENDING':
        return {"status": "pending"}
    elif task_result.state == 'FAILURE':
        return {"status": "error", "message": str(task_result.result)}
    else:
        return {"status": task_result.state}


@app.get("/download/{task_id}")
def download_file(task_id: str):
    """
    Endpoint to download the classified results as a CSV file.

    Args:
        task_id (str): The Celery task ID.

    Returns:
        FileResponse or JSONResponse: Returns the CSV file or an error message.
    """
    # Import Celery app dynamically to prevent circular imports
    from backend.tasks import app as celery_app
    task_result = celery_app.AsyncResult(task_id)
    if task_result.state == 'SUCCESS':
        output_file_path = task_result.result.get('output_file', '')
        if os.path.exists(output_file_path):
            return FileResponse(output_file_path, filename=os.path.basename(output_file_path), media_type='text/csv')
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Output file not found."}
            )
    elif task_result.state == 'FAILURE':
        return JSONResponse(
            status_code=500,
            content={"error": "Task failed."}
        )
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "Task not completed yet."}
        )
