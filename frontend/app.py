# frontend/app.py

import streamlit as st
import requests
import time
import os
import pandas as pd
import json

BACKEND_URL = "http://localhost:8000"

st.title("Requirement Classification App")

# Step 1: Select Meta-Model
st.header("Select Meta-Model")
meta_model_options = ["OpenAI", "Anthropic", "Groq", "Gemma2"]
meta_model_mapping = {
    'OpenAI': 'openai',
    'Anthropic': 'anthropic',
    'Groq': 'groq',
    'Gemma2': 'gemma2'
}
meta_model = st.selectbox("Choose the Meta-Model to use for classification:", meta_model_options)
selected_meta_model = meta_model_mapping[meta_model]

# Step 2: Set Provider Weights
st.header("Set Provider Weights")
st.write("Assign weights to each provider. Higher weights give more influence to the provider's classification.")

# Collect weights for all four providers
provider_weights = {}
for provider in meta_model_options:
    provider_key = meta_model_mapping[provider]
    weight = st.number_input(
        f"Weight for {provider}",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        key=provider_key
    )
    provider_weights[provider_key] = weight

# Step 3: Set Confidence Threshold
st.header("Set Confidence Threshold")
st.write("Set the confidence threshold for the voting algorithm (between 0.5 and 1.0).")
confidence_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.5,
    max_value=1.0,
    value=0.6,
    step=0.05
)

# Step 4: Upload Requirements File
st.header("Upload Requirements File")
uploaded_file = st.file_uploader("Upload your requirements file (.txt only)", type=["txt"])

if uploaded_file is not None:
    if st.button("Process File"):
        with st.spinner("Processing..."):
            try:
                # Prepare the files and data for the POST request
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                data = {
                    "meta_model": selected_meta_model,
                    "provider_weights": json.dumps(provider_weights),
                    "confidence_threshold": str(confidence_threshold)
                }

                # Send a POST request to the backend
                response = requests.post(
                    f"{BACKEND_URL}/upload/",
                    files=files,
                    data=data
                )
                if response.status_code == 200:
                    task_id = response.json().get('task_id')
                    st.success(f"Task submitted successfully! Task ID: {task_id}")
                    status = 'pending'
                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    while status == 'pending':
                        time.sleep(2)
                        result_response = requests.get(
                            f"{BACKEND_URL}/results/{task_id}"
                        )
                        if result_response.status_code == 200:
                            result = result_response.json()
                            status = result.get('status')
                            if status == 'pending':
                                progress_bar.progress(30)
                                progress_text.text("Status: Processing...")
                            elif status == 'completed':
                                progress_bar.progress(100)
                                progress_text.text("Status: Completed!")
                            elif status == 'error':
                                progress_bar.progress(100)
                                progress_text.text("Status: Error occurred during processing.")
                                st.error("An error occurred during processing.")
                                break
                        else:
                            st.error("Error fetching results: " + result_response.text)
                            break

                    if status == 'completed':
                        st.success("Processing completed!")
                        data = result.get('results', [])
                        if data:
                            df = pd.DataFrame(data)
                            st.write("### Classified Requirements")
                            st.dataframe(df)
                        else:
                            st.info("No classification results found.")

                        # Fetch the CSV file from the backend
                        download_response = requests.get(
                            f"{BACKEND_URL}/download/{task_id}"
                        )
                        if download_response.status_code == 200:
                            st.download_button(
                                label="Download Classified Requirements",
                                data=download_response.content,
                                file_name="classified_requirements.csv",
                                mime='text/csv'
                            )
                        else:
                            st.error("Error downloading classified file: " + download_response.text)
                    elif status == 'error':
                        st.error("An error occurred during processing.")
                    else:
                        st.error("Unknown status: " + status)
                else:
                    st.error("Failed to upload file: " + response.text)

            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {e}") 
