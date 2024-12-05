# backend/utils.py

import os
import re
import json
import time
import logging
from groq import Groq
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from openai import OpenAI
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from owlready2 import *

# Placeholder imports for Groq and Gemma2
# Replace these with actual client libraries

# ============================
# Configuration and Initialization
# ============================

# Load environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY must be set in the environment.")
openai_client = OpenAI(api_key=openai_api_key)

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY must be set in the environment.")
anthropic_client = Anthropic(api_key=anthropic_api_key)

# Assuming Groq and Gemma2 use the same API key
groq_api_key = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=groq_api_key)


# Configure logging
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, 'logs', 'backend.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Ensure logs directory exists

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,  # Set to INFO to capture more details
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Define weights for providers
weights = {
    'openai': 1.0,
    'anthropic': 0.9,
    'groq': 0.8,
    'gemma2': 0.85,
}

# Temperature settings for diversity in responses
temperatures = [0.0]  # Use deterministic setting to minimize API calls

# Label mapping dictionary to standardize labels
label_mapping = {
    "functional": "functional",
    "nonfunctional": "nonfunctional",
    "primary": "primary",
    "secondary": "secondary",
    "usertask": "UserTask",
    "systemtask": "SystemTask",
    "undetermined": "undetermined",
    "UserTask": "UserTask",
    "SystemTask": "SystemTask",
    "Primary": "Primary",
    "Secondary": "Secondary",
    "Functional": "functional",
    "Nonfunctional": "nonfunctional",
}


# ============================
# Load Ontology
# ============================

# Define the ontologies directory path
ontologies_dir = os.path.join(script_dir, "ontologies")
os.makedirs(ontologies_dir, exist_ok=True)  # Ensure ontologies directory exists

# Define the full path to the ontology file
ontology_path = os.path.join(ontologies_dir, "NFRsOntology.owl")

# Load the ontology
if not os.path.exists(ontology_path):
    raise FileNotFoundError(f"Ontology file '{ontology_path}' not found. Please ensure it exists.")

onto = get_ontology(ontology_path).load()

# Map quality attribute names to ontology classes
quality_attribute_classes = {
    'availability': onto.Availability,
    'usability': onto.Usability,
    'security': onto.Security,
    'performance': onto.Performance,
    'maintainability': onto.Maintainability,
    'reliability': onto.Reliability,
    'scalability': onto.Scalability,
    'portability': onto.Portability,
    'interoperability': onto.Interoperability
}

# List of NFRs for prompts
nfr_list = list(quality_attribute_classes.keys())

# Definitions of quality attributes for prompts
quality_attribute_definitions = {
    'availability': 'The degree to which a system is operational and accessible when required for use.',
    'usability': 'The ease with which users can learn and use a system to achieve their goals effectively.',
    'security': 'The protection of information and systems from unauthorized access, use, disclosure, disruption, modification, or destruction.',
    'performance': 'The ability of the system to perform its required functions within acceptable response times and resource utilization levels.',
    'maintainability': 'The ease with which a system can be modified to correct faults, improve performance, or adapt to a changed environment.',
    'reliability': 'The ability of the system to perform its required functions under stated conditions for a specified period of time.',
    'scalability': 'The capability of a system to handle increased load without impacting performance or compromising functionality.',
    'portability': 'The ease with which a system or component can be transferred from one environment to another.',
    'interoperability': 'The ability of a system to work with or use the parts or equipment of another system.'
}

# Definitions for functional requirement types
functional_type_definitions = {
    'primary': 'Requirements that define core functionalities of the system, directly contributing to its main goals or providing direct value to the users.',
    'secondary': 'Requirements that operationalize Non-Functional Requirements (NFRs) or quality attributes such as Security, Performance, etc. They are functionalities implemented as solutions to meet NFRs.'
}

# Definitions for task types
task_type_definitions = {
    'UserTask': 'Tasks that require user initiation or direct user interaction.',
    'SystemTask': 'Tasks that the system performs automatically without user involvement.'
}

# ============================
# Retry Decorator for Robustness
# ============================

def retry_on_failure(max_retries=3, delay=2, backoff=2):
    """
    Decorator to retry a function upon failure.

    Args:
        max_retries (int): Maximum number of retries.
        delay (int): Initial delay between retries in seconds.
        backoff (int): Multiplier applied to delay after each retry.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            num_items = len(args[0]) if args else 1
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in {func.__name__}: {e}. Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    retries += 1
                    current_delay *= backoff
            logger.error(f"Max retries exceeded for {func.__name__}")
            # Return appropriate default based on function's expected return type
            if func.__name__.startswith("llm_classify"):
                if "functional" in func.__name__:
                    return [('undetermined', 'N/A')] * num_items
                elif "task" in func.__name__:
                    return ['undetermined'] * num_items
                else:
                    return ['undetermined'] * num_items
            else:
                return ['undetermined'] * num_items
        return wrapper
    return decorator

# ============================
# Core Functions
# ============================

def create_batch_prompt(requirements):
    """
    Create a prompt that asks the LLM to classify multiple requirements.

    Args:
        requirements (list of str): The list of requirements to classify.

    Returns:
        str: The prompt to send to the LLM.
    """
    prompt = """
You are an expert in software engineering requirements classification. Classify each of the following requirements as 'functional' or 'nonfunctional'.

Definitions:
- **Functional Requirements**: Statements of the functionalities that the system must perform. They define what the system does or must not do.
- **Nonfunctional Requirements**: Constraints or qualities that the system must have, such as performance, security, usability, etc.

Examples:
1. "The system shall allow users to register and log in using their email and password." -> functional
2. "The application shall process user data within 2 seconds to ensure a responsive user experience." -> nonfunctional
3. "The software shall comply with GDPR regulations to protect user data." -> nonfunctional
4. "The system shall provide an API to allow third-party applications to integrate with the service." -> functional
5. "The application shall be available 99.9% of the time to meet service level agreements." -> nonfunctional

Now, classify the following requirements:

"""
    for idx, req in enumerate(requirements, 1):
        prompt += f"{idx}. \"{req}\"\n"

    prompt += "\nProvide your classifications in the following format:\n"
    prompt += "1. functional\n2. nonfunctional\n..."

    return prompt


def parse_batch_output(output, num_requirements):
    """
    Parse the LLM output to extract classifications.

    Args:
        output (str): The raw output from the LLM.
        num_requirements (int): The number of requirements.

    Returns:
        list of str: The list of classifications.
    """
    classifications = ['undetermined'] * num_requirements
    lines = output.strip().splitlines()
    
    for line in lines:
        # Extract the requirement number at the beginning of the line
        num_match = re.match(r'^(\d+)\.', line.strip())
        if not num_match:
            continue  # Skip lines that do not start with a number and a dot
        
        idx = int(num_match.group(1)) - 1  # Zero-based indexing
        
        # Search for 'functional' or 'nonfunctional' anywhere in the line
        classification_match = re.search(r'\b(functional|nonfunctional)\b', line.strip(), re.IGNORECASE)
        if classification_match:
            classification = classification_match.group(1).lower()
            # Map the classification using label_mapping
            classifications[idx] = label_mapping.get(classification, 'undetermined')
        else:
            # If classification keywords are not found, log a warning
            logger.warning(f"No valid classification found in line: '{line.strip()}'")
    
    return classifications


def create_qa_classification_prompt(requirements):
    """
    Create a prompt that asks the LLM to classify each nonfunctional requirement into a quality attribute.

    Args:
        requirements (list of str): The list of nonfunctional requirements to classify.

    Returns:
        str: The prompt to send to the LLM.
    """
    quality_attributes_formatted = '\n'.join([f"- {qa.capitalize()}: {quality_attribute_definitions[qa]}" for qa in quality_attribute_classes.keys()])

    prompt = f"""
You are an expert in software engineering requirements classification. For each of the following nonfunctional requirements, determine which quality attribute it relates to. Choose the most appropriate one from the following list and their definitions:

{quality_attributes_formatted}

Requirements:
"""
    for idx, req in enumerate(requirements, 1):
        prompt += f"{idx}. \"{req}\"\n"

    prompt += "\nProvide your answers in the following format:\n"
    prompt += "1. [Quality Attribute]\n2. [Quality Attribute]\n..."

    return prompt

def parse_qa_classification_output(output, num_requirements):
    """
    Parse the LLM output to extract quality attribute classifications.

    Args:
        output (str): The raw output from the LLM.
        num_requirements (int): The number of requirements.

    Returns:
        list of str: The list of quality attribute classifications.
    """
    classifications = ['undetermined'] * num_requirements
    lines = output.strip().splitlines()
    for line in lines:
        match = re.match(r'(\d+)\.\s*([a-zA-Z]+)', line.strip(), re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1
            classification = match.group(2).lower()
            if 0 <= idx < num_requirements and classification in [qa.lower() for qa in quality_attribute_classes.keys()]:
                classifications[idx] = classification.capitalize()
            else:
                classifications[idx] = 'Undetermined'
    return classifications

def create_functional_classification_prompt(requirements):
    """
    Create a prompt that asks the LLM to classify functional requirements into 'primary' or 'secondary',
    and specify which NFR(s) they operationalize if they are 'secondary'.

    Args:
        requirements (list of str): The list of functional requirements to classify.

    Returns:
        str: The prompt to send to the LLM.
    """
    quality_attributes_formatted = ', '.join([qa.capitalize() for qa in nfr_list])

    prompt = f"""
You are an expert in software engineering requirements classification. Classify each of the following functional requirements as 'primary' or 'secondary' functional requirements based on the following definitions:

- **Primary Functional Requirements**: Core functionalities that define the main actions and services the system must perform to meet the user's needs and achieve its primary objectives.

- **Secondary Functional Requirements**: Functionalities that are implemented to fulfill Non-Functional Requirements (NFRs) or quality attributes such as {quality_attributes_formatted}. These requirements operationalize NFRs by providing specific functionalities that address quality concerns.

For each requirement:

1. Determine if it is 'primary' or 'secondary' based on the definitions above.
2. If it is 'secondary', specify which NFR(s) it operationalizes (choose from the list provided).

Examples:

1. "The system shall allow users to create and edit their profiles." -> primary
2. "The application shall encrypt all user data at rest and in transit." -> secondary (operationalizes 'Security')
3. "The system shall provide real-time analytics dashboards for administrators." -> primary
4. "The application shall maintain 99.9% uptime to ensure availability." -> secondary (operationalizes 'Availability')
5. "The system shall automatically scale resources to handle increased load." -> secondary (operationalizes 'Scalability')
6. "Users shall be able to reset their passwords via email verification." -> primary
7. "The system shall log all user activities for auditing purposes." -> secondary (operationalizes 'Maintainability', 'Security')

Now, classify the following functional requirements:

"""
    for idx, req in enumerate(requirements, 1):
        prompt += f"{idx}. \"{req}\"\n"

    prompt += "\nProvide your answers in the following format:\n"
    prompt += "1. primary\n2. secondary (operationalizes 'NFR')\n3. primary\n..."

    return prompt


def parse_functional_classification_output(output, num_requirements):
    """
    Parse the LLM output to extract functional classifications and NFR(s) operationalized.

    Args:
        output (str): The raw output from the LLM.
        num_requirements (int): The number of requirements.

    Returns:
        list of tuples: Each tuple contains ('classification', 'operationalizes_nfr').
    """
    results = [('undetermined', 'N/A')] * num_requirements
    lines = output.strip().splitlines()
    for line in lines:
        match = re.match(
            r'(\d+)\.\s*(primary|secondary)(?:\s*\(operationalizes\s*\'?([a-zA-Z,\s]+)\'?\))?',
            line.strip(),
            re.IGNORECASE
        )
        if match:
            idx = int(match.group(1)) - 1
            classification = match.group(2).lower()
            operationalizes_nfr = match.group(3) if match.group(3) else 'N/A'
            if 0 <= idx < num_requirements:
                classification = label_mapping.get(classification, 'undetermined')
                if operationalizes_nfr != 'N/A':
                    # Normalize and validate NFRs
                    nfrs = [nfr.strip().lower() for nfr in operationalizes_nfr.split(',')]
                    valid_nfrs = [nfr.capitalize() for nfr in nfrs if nfr in nfr_list]
                    if valid_nfrs:
                        operationalizes_nfr = ', '.join(valid_nfrs)
                    else:
                        operationalizes_nfr = 'Undetermined'
                else:
                    operationalizes_nfr = 'N/A'
                results[idx] = (classification, operationalizes_nfr)
    return results


def create_task_classification_prompt(requirements):
    """
    Create a prompt that asks the LLM to classify functional requirements into 'UserTask' or 'SystemTask'.
    """
    prompt = f"""
You are an expert in software engineering requirements classification. Classify each of the following functional requirements as **'UserTask'** or **'SystemTask'** based on the following definitions:

- **UserTask**: {task_type_definitions['UserTask']}

- **SystemTask**: {task_type_definitions['SystemTask']}

**Instructions:**
- Provide only the classification (**'UserTask'** or **'SystemTask'**) for each requirement.
- Do **not** include any explanations, comments, or additional text.
- Ensure that each classification is on a new line corresponding to the requirement number.

**Examples:**
1. "The system shall allow users to upload files." -> UserTask
2. "The system shall automatically backup data every hour." -> SystemTask
3. "Users shall be able to search for products." -> UserTask
4. "The system shall send email notifications when an event occurs." -> SystemTask
5. "The application shall provide a dashboard for users to monitor their activities." -> UserTask
6. "The system shall process transactions in real-time." -> SystemTask

**Now, classify the following functional requirements:**

"""
    for idx, req in enumerate(requirements, 1):
        prompt += f"{idx}. \"{req}\"\n"

    prompt += "\nProvide your classifications in the following format:\n"
    prompt += "1. UserTask\n2. SystemTask\n..."

    return prompt


def parse_task_classification_output(output, num_requirements):
    """
    Parse the LLM output to extract task classifications.

    Args:
        output (str): The raw output from the LLM.
        num_requirements (int): The number of requirements.

    Returns:
        list of str: The list of task classifications.
    """
    classifications = ['undetermined'] * num_requirements
    lines = output.strip().splitlines()
    for line in lines:
        # Remove punctuation and make lowercase
        clean_line = re.sub(r'[^\w\s]', '', line.strip()).lower()
        match = re.match(r'(\d+)\.\s*(usertask|systemtask)', clean_line)
        if match:
            idx = int(match.group(1)) - 1
            classification = match.group(2)
            if classification == 'usertask':
                classifications[idx] = 'UserTask'
            elif classification == 'systemtask':
                classifications[idx] = 'SystemTask'
    return classifications


def parse_meta_model_output_classification(output, expected_labels):
    """
    Parse the meta-model's output to extract the classification.

    Args:
        output (str): The raw output from the meta-model.
        expected_labels (list): List of valid classification labels.

    Returns:
        str: The classification result.
    """
    # Convert expected labels to lower case for case-insensitive matching
    expected_labels_lower = [label.lower() for label in expected_labels]
    
    # Create a regex pattern that matches any of the expected labels as whole words
    pattern = r'\b(' + '|'.join(map(re.escape, expected_labels_lower)) + r')\b'
    
    # Search for the pattern in the output
    match = re.search(pattern, output, re.IGNORECASE)
    
    if match:
        classification = match.group(1).capitalize()
        # Ensure the classification matches one of the expected labels (case-insensitive)
        for label in expected_labels:
            if classification.lower() == label.lower():
                return label
    else:
        # If no match is found, log a warning and return 'Undetermined'
        logger.warning(f"Meta-model returned invalid classification: '{output}'")
        return 'Undetermined'


def parse_meta_model_output_task_classification(output, expected_labels):
    """
    Parse the meta-model's output to extract the classification.

    Args:
        output (str): The raw output from the meta-model.
        expected_labels (list): List of valid classification labels.

    Returns:
        str: The classification result.
    """
    classification = output.strip()
    # Use regex to find 'UserTask' or 'SystemTask' irrespective of spaces or punctuation
    match = re.search(r'\b(User\s*Task|System\s*Task)\b', classification, re.IGNORECASE)
    if match:
        # Normalize the classification by removing spaces and ensuring correct casing
        classification = match.group(1).replace(' ', '').lower()
        if classification == 'usertask':
            return 'UserTask'
        elif classification == 'systemtask':
            return 'SystemTask'
    logger.warning(f"Meta-model returned invalid classification: '{classification}'")
    return 'Undetermined'


    
def parse_meta_model_output_functional_classification(output, expected_labels):
    """
    Parse the meta-model's output to extract the classification and operationalizes_nfr.

    Args:
        output (str): The raw output from the meta-model.
        expected_labels (list): List of valid classification labels.

    Returns:
        tuple: ('classification', 'operationalizes_nfr')
    """
    # Expected format: [Classification] (operationalizes 'NFR1, NFR2, ...')
    match = re.match(r'(Primary|Secondary)(?:\s*\(operationalizes\s*\'?([a-zA-Z,\s]+)\'?\))?', output.strip(), re.IGNORECASE)
    if match:
        classification = match.group(1).capitalize()
        operationalizes_nfr = match.group(2) if match.group(2) else 'N/A'
        if classification not in expected_labels:
            classification = 'Undetermined'
        if operationalizes_nfr != 'N/A':
            # Validate NFRs
            nfrs = [nfr.strip().lower() for nfr in operationalizes_nfr.split(',')]
            valid_nfrs = [nfr.capitalize() for nfr in nfrs if nfr in nfr_list]
            if valid_nfrs:
                operationalizes_nfr = ', '.join(valid_nfrs)
            else:
                operationalizes_nfr = 'Undetermined'
        else:
            operationalizes_nfr = 'N/A'
        return (classification, operationalizes_nfr)
    else:
        logger.warning(f"Meta-model returned invalid functional classification: '{output}'")
        return ('Undetermined', 'N/A')
# ============================
# Batch Classification Functions for Functional Requirements
# ============================


@retry_on_failure(max_retries=3, delay=2, backoff=2)
def llm_classify_openai_batch(requirements, temperature):
    """
    Classify requirements as functional or nonfunctional using OpenAI's API.

    Args:
        requirements (list of str): The list of requirements to classify.
        temperature (float): The temperature setting for randomness.

    Returns:
        list of str: List of classifications corresponding to each requirement.
    """
    try:
        prompt = create_batch_prompt(requirements)
        logger.info(f"OpenAI: Sending batch prompt with temperature {temperature}")
        response = openai_client.chat.completions.create(
            model= "gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=100 * len(requirements),  # Adjust based on expected output
            n=1,
            temperature=temperature
        )
        output = response.choices[0].message.content.strip()
        logger.info(f"OpenAI: Received batch classifications:\n{output}")

        # Parse the output to extract classifications
        classifications = parse_batch_output(output, len(requirements))
        return classifications
    except Exception as e:
        logger.error(f"OpenAI Batch Classification Error: {e}")
        return ['undetermined'] * len(requirements)

@retry_on_failure(max_retries=3, delay=2, backoff=2)
def llm_classify_anthropic_batch(requirements, temperature):
    """
    Classify requirements as functional or nonfunctional using Anthropic's API.

    Args:
        requirements (list of str): The list of requirements to classify.
        temperature (float): The temperature setting for randomness.

    Returns:
        list of str: List of classifications corresponding to each requirement.
    """
    try:
        prompt = create_batch_prompt(requirements)
        anthropic_prompt = HUMAN_PROMPT + prompt + AI_PROMPT
        logger.info(f"Anthropic: Sending batch prompt with temperature {temperature}")
        response = anthropic_client.completions.create(
            prompt=anthropic_prompt,
            model="claude-2",
            max_tokens_to_sample=100 * len(requirements),
            temperature=temperature,
        )
        output = response.completion.strip()
        logger.info(f"Anthropic: Received batch classifications:\n{output}")

        classifications = parse_batch_output(output, len(requirements))
        return classifications
    except Exception as e:
        logger.error(f"Anthropic Batch Classification Error: {e}")
        return ['undetermined'] * len(requirements)

@retry_on_failure(max_retries=3, delay=2, backoff=2)
def llm_classify_groq_batch(requirements, temperature):
    """
    Classify requirements as functional or nonfunctional using Groq's API.

    Args:
        requirements (list of str): The list of requirements to classify.
        temperature (float): The temperature setting for randomness.

    Returns:
        list of str: List of classifications corresponding to each requirement.
    """
    try:
        prompt = create_batch_prompt(requirements)
        logger.info(f"Groq: Sending batch prompt with temperature {temperature}")
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",  # Use the specified model name
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=100 * len(requirements),
            n=1,
            temperature=temperature
        )
        output = response.choices[0].message.content.strip()
        logger.info(f"Groq: Received batch classifications:\n{output}")

        classifications = parse_batch_output(output, len(requirements))
        return classifications
    except Exception as e:
        logger.error(f"Groq Batch Classification Error: {e}")
        return ['undetermined'] * len(requirements)

@retry_on_failure(max_retries=3, delay=2, backoff=2)
def llm_classify_gemma2_batch(requirements, temperature):
    """
    Classify requirements as functional or nonfunctional using Gemma2's API.

    Args:
        requirements (list of str): The list of requirements to classify.
        temperature (float): The temperature setting for randomness.

    Returns:
        list of str: List of classifications corresponding to each requirement.
    """
    try:
        prompt = create_batch_prompt(requirements)
        logger.info(f"Gemma2: Sending batch prompt with temperature {temperature}")
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",  # Use the specified model name
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=100 * len(requirements),
            n=1,
            temperature=temperature
        )
        output = response.choices[0].message.content.strip()
        logger.info(f"Gemma2: Received batch classifications:\n{output}")

        classifications = parse_batch_output(output, len(requirements))
        return classifications
    except Exception as e:
        logger.error(f"Gemma2 Batch Classification Error: {e}")
        return ['undetermined'] * len(requirements)

@retry_on_failure(max_retries=3, delay=2, backoff=2)
def llm_classify_openai_functional_batch(requirements, temperature):
    """
    Classify functional requirements into primary or secondary using OpenAI's API.

    Args:
        requirements (list of str): The list of functional requirements to classify.
        temperature (float): The temperature setting for randomness.

    Returns:
        list of tuples: Each tuple contains ('classification', 'operationalizes_nfr').
    """
    try:
        prompt = create_functional_classification_prompt(requirements)
        logger.info(f"OpenAI Functional: Sending batch prompt with temperature {temperature}")
        response = openai_client.chat.completions.create(
            model= "gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150 * len(requirements),  # Adjust based on expected output length
            n=1,
            temperature=temperature
        )
        output = response.choices[0].message.content.strip()
        logger.info(f"OpenAI Functional: Received batch classifications:\n{output}")

        # Parse the output to extract classifications
        results = parse_functional_classification_output(output, len(requirements))
        return results
    except Exception as e:
        logger.error(f"OpenAI Functional: Error during batch classification: {e}")
        return [('undetermined', 'N/A')] * len(requirements)

@retry_on_failure(max_retries=3, delay=2, backoff=2)
def llm_classify_anthropic_functional_batch(requirements, temperature):
    """
    Classify functional requirements into primary or secondary using Anthropic's API.

    Args:
        requirements (list of str): The list of functional requirements to classify.
        temperature (float): The temperature setting for randomness.

    Returns:
        list of tuples: Each tuple contains ('classification', 'operationalizes_nfr').
    """
    try:
        prompt = create_functional_classification_prompt(requirements)
        anthropic_prompt = HUMAN_PROMPT + prompt + AI_PROMPT
        logger.info(f"Anthropic Functional: Sending batch prompt with temperature {temperature}")
        response = anthropic_client.completions.create(
            prompt=anthropic_prompt,
            model="claude-2",
            max_tokens_to_sample=150 * len(requirements),
            temperature=temperature,
        )
        output = response.completion.strip()
        logger.info(f"Anthropic Functional: Received batch classifications:\n{output}")

        # Parse the output to extract classifications
        results = parse_functional_classification_output(output, len(requirements))
        return results
    except Exception as e:
        logger.error(f"Anthropic Functional: Error during batch classification: {e}")
        return [('undetermined', 'N/A')] * len(requirements)

@retry_on_failure(max_retries=3, delay=2, backoff=2)
def llm_classify_groq_functional_batch(requirements, temperature):
    """
    Classify functional requirements into primary or secondary using Groq's API.

    Args:
        requirements (list of str): The list of functional requirements to classify.
        temperature (float): The temperature setting for randomness.

    Returns:
        list of tuples: Each tuple contains ('classification', 'operationalizes_nfr').
    """
    try:
        prompt = create_functional_classification_prompt(requirements)
        logger.info(f"Groq Functional: Sending batch prompt with temperature {temperature}")
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",  # Use the specified model name
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150 * len(requirements),
            n=1,
            temperature=temperature
        )
        output = response.choices[0].message.content.strip()
        logger.info(f"Groq Functional: Received batch classifications:\n{output}")

        # Parse the output to extract classifications
        results = parse_functional_classification_output(output, len(requirements))
        return results
    except Exception as e:
        logger.error(f"Groq Functional: Error during batch classification: {e}")
        return [('undetermined', 'N/A')] * len(requirements)

@retry_on_failure(max_retries=3, delay=2, backoff=2)
def llm_classify_gemma2_functional_batch(requirements, temperature):
    """
    Classify functional requirements into primary or secondary using Gemma2's API.

    Args:
        requirements (list of str): The list of functional requirements to classify.
        temperature (float): The temperature setting for randomness.

    Returns:
        list of tuples: Each tuple contains ('classification', 'operationalizes_nfr').
    """
    try:
        prompt = create_functional_classification_prompt(requirements)
        logger.info(f"Gemma2 Functional: Sending batch prompt with temperature {temperature}")
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",  # Use the specified model name
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150 * len(requirements),
            n=1,
            temperature=temperature
        )
        output = response.choices[0].message.content.strip()
        logger.info(f"Gemma2 Functional: Received batch classifications:\n{output}")

        # Parse the output to extract classifications
        results = parse_functional_classification_output(output, len(requirements))
        return results
    except Exception as e:
        logger.error(f"Gemma2 Functional: Error during batch classification: {e}")
        return [('undetermined', 'N/A')] * len(requirements)

# ============================
# Batch Classification Functions for Task Classification
# ============================

@retry_on_failure(max_retries=3, delay=2, backoff=2)
def llm_classify_openai_task_batch(requirements, temperature):
    """
    Classify functional requirements into UserTask or SystemTask using OpenAI's API.

    Args:
        requirements (list of str): The list of functional requirements to classify.
        temperature (float): The temperature setting for randomness.

    Returns:
        list of str: List of task classifications corresponding to each requirement.
    """
    try:
        prompt = create_task_classification_prompt(requirements)
        logger.info(f"OpenAI Task: Sending batch prompt with temperature {temperature}")
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=5 * len(requirements),
            n=1,
            temperature=temperature
        )
        output = response.choices[0].message.content.strip()
        logger.info(f"OpenAI Task: Received batch classifications:\n{output}")

        classifications = parse_task_classification_output(output, len(requirements))
        return classifications
    except Exception as e:
        logger.error(f"OpenAI Task: Error during batch classification: {e}")
        return ['undetermined'] * len(requirements)

@retry_on_failure(max_retries=3, delay=2, backoff=2)
def llm_classify_anthropic_task_batch(requirements, temperature):
    """
    Classify functional requirements into UserTask or SystemTask using Anthropic's API.

    Args:
        requirements (list of str): The list of functional requirements to classify.
        temperature (float): The temperature setting for randomness.

    Returns:
        list of str: List of task classifications corresponding to each requirement.
    """
    try:
        prompt = create_task_classification_prompt(requirements)
        anthropic_prompt = HUMAN_PROMPT + prompt + AI_PROMPT
        logger.info(f"Anthropic Task: Sending batch prompt with temperature {temperature}")
        response = anthropic_client.completions.create(
            prompt=anthropic_prompt,
            model="claude-2",
            max_tokens_to_sample=5 * len(requirements),
            temperature=temperature,
        )
        output = response.completion.strip()
        logger.info(f"Anthropic Task: Received batch classifications:\n{output}")

        classifications = parse_task_classification_output(output, len(requirements))
        return classifications
    except Exception as e:
        logger.error(f"Anthropic Task: Error during batch classification: {e}")
        return ['undetermined'] * len(requirements)

@retry_on_failure(max_retries=3, delay=2, backoff=2)
def llm_classify_groq_task_batch(requirements, temperature):
    """
    Classify functional requirements into UserTask or SystemTask using Groq's API.

    Args:
        requirements (list of str): The list of functional requirements to classify.
        temperature (float): The temperature setting for randomness.

    Returns:
        list of str: List of task classifications corresponding to each requirement.
    """
    try:
        prompt = create_task_classification_prompt(requirements)
        logger.info(f"Groq Task: Sending batch prompt with temperature {temperature}")
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",  # Use the specified model name
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=5 * len(requirements),
            n=1,
            temperature=temperature
        )
        output = response.choices[0].message.content.strip()
        logger.info(f"Groq Task: Received batch classifications:\n{output}")

        classifications = parse_task_classification_output(output, len(requirements))
        return classifications
    except Exception as e:
        logger.error(f"Groq Task: Error during batch classification: {e}")
        return ['undetermined'] * len(requirements)

@retry_on_failure(max_retries=3, delay=2, backoff=2)
def llm_classify_gemma2_task_batch(requirements, temperature):
    """
    Classify functional requirements into UserTask or SystemTask using Gemma2's API.

    Args:
        requirements (list of str): The list of functional requirements to classify.
        temperature (float): The temperature setting for randomness.

    Returns:
        list of str: List of task classifications corresponding to each requirement.
    """
    try:
        prompt = create_task_classification_prompt(requirements)
        logger.info(f"Gemma2 Task: Sending batch prompt with temperature {temperature}")
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",  # Use the specified model name
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=5 * len(requirements),
            n=1,
            temperature=temperature
        )
        output = response.choices[0].message.content.strip()
        logger.info(f"Gemma2 Task: Received batch classifications:\n{output}")

        classifications = parse_task_classification_output(output, len(requirements))
        return classifications
    except Exception as e:
        logger.error(f"Gemma2 Task: Error during batch classification: {e}")
        return ['undetermined'] * len(requirements)

# ============================
# Batch Classification Functions for Quality Attribute Classification
# ============================

@retry_on_failure(max_retries=3, delay=2, backoff=2)
def llm_classify_openai_qa_classification_batch(requirements, temperature):
    """
    Classify the quality attributes of a list of nonfunctional requirements using OpenAI's API.

    Args:
        requirements (list of str): The nonfunctional requirement texts.
        temperature (float): The temperature setting for randomness.

    Returns:
        list of str: The most likely quality attribute for each requirement.
    """
    try:
        prompt = create_qa_classification_prompt(requirements)
        logger.info(f"OpenAI QA: Sending batch prompt with temperature {temperature}")
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=10 * len(requirements),
            n=1,
            temperature=temperature
        )
        output = response.choices[0].message.content.strip()
        logger.info(f"OpenAI QA: Received batch classifications:\n{output}")

        classifications = parse_qa_classification_output(output, len(requirements))
        return classifications
    except Exception as e:
        logger.error(f"OpenAI QA: Error during batch classification: {e}")
        return ['Undetermined'] * len(requirements)

@retry_on_failure(max_retries=3, delay=2, backoff=2)
def llm_classify_anthropic_qa_classification_batch(requirements, temperature):
    """
    Classify the quality attributes of a list of nonfunctional requirements using Anthropic's API.

    Args:
        requirements (list of str): The nonfunctional requirement texts.
        temperature (float): The temperature setting for randomness.

    Returns:
        list of str: The most likely quality attribute for each requirement.
    """
    try:
        prompt = create_qa_classification_prompt(requirements)
        anthropic_prompt = HUMAN_PROMPT + prompt + AI_PROMPT
        logger.info(f"Anthropic QA: Sending batch prompt with temperature {temperature}")
        response = anthropic_client.completions.create(
            prompt=anthropic_prompt,
            model="claude-2",
            max_tokens_to_sample=10 * len(requirements),
            temperature=temperature,
        )
        output = response.completion.strip()
        logger.info(f"Anthropic QA: Received batch classifications:\n{output}")

        classifications = parse_qa_classification_output(output, len(requirements))
        return classifications
    except Exception as e:
        logger.error(f"Anthropic QA: Error during batch classification: {e}")
        return ['Undetermined'] * len(requirements)

@retry_on_failure(max_retries=3, delay=2, backoff=2)
def llm_classify_groq_qa_classification_batch(requirements, temperature):
    """
    Classify the quality attributes of a list of nonfunctional requirements using Groq's API.

    Args:
        requirements (list of str): The nonfunctional requirement texts.
        temperature (float): The temperature setting for randomness.

    Returns:
        list of str: The most likely quality attribute for each requirement.
    """
    try:
        prompt = create_qa_classification_prompt(requirements)
        logger.info(f"Groq QA: Sending batch prompt with temperature {temperature}")
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",  # Use the specified model name
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=10 * len(requirements),
            n=1,
            temperature=temperature
        )
        output = response.choices[0].message.content.strip()
        logger.info(f"Groq QA: Received batch classifications:\n{output}")

        classifications = parse_qa_classification_output(output, len(requirements))
        return classifications
    except Exception as e:
        logger.error(f"Groq QA: Error during batch classification: {e}")
        return ['Undetermined'] * len(requirements)

@retry_on_failure(max_retries=3, delay=2, backoff=2)
def llm_classify_gemma2_qa_classification_batch(requirements, temperature):
    """
    Classify the quality attributes of a list of nonfunctional requirements using Gemma2's API.

    Args:
        requirements (list of str): The nonfunctional requirement texts.
        temperature (float): The temperature setting for randomness.

    Returns:
        list of str: The most likely quality attribute for each requirement.
    """
    try:
        prompt = create_qa_classification_prompt(requirements)
        logger.info(f"Gemma2 QA: Sending batch prompt with temperature {temperature}")
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",  # Use the specified model name
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=10 * len(requirements),
            n=1,
            temperature=temperature
        )
        output = response.choices[0].message.content.strip()
        logger.info(f"Gemma2 QA: Received batch classifications:\n{output}")

        classifications = parse_qa_classification_output(output, len(requirements))
        return classifications
    except Exception as e:
        logger.error(f"Gemma2 QA: Error during batch classification: {e}")
        return ['Undetermined'] * len(requirements)

# ============================
# Meta-Model Functions for Stacking
# ============================

def create_meta_model_prompt_qa_classification(requirement, base_model_responses, provider_weights):
    """
    Create a prompt for the meta-model to classify a nonfunctional requirement into a quality attribute,
    given the responses from the base models and their weights.

    Args:
        requirement (str): The nonfunctional requirement text.
        base_model_responses (list of tuples): Each tuple contains ('provider', 'classification').
        provider_weights (dict): Weights assigned to each provider.

    Returns:
        str: The prompt to send to the meta-model.
    """
    prompt = f"""
You are an expert in software engineering requirements classification.

For the following nonfunctional requirement:

"{requirement}"

We have the following classifications provided by different models, along with their weights:

"""
    for provider, classification in base_model_responses:
        weight = provider_weights.get(provider, 1.0)
        prompt += f"- {provider.capitalize()} (Weight: {weight}): {classification}\n"

    prompt += """
Your task is to determine which quality attribute the requirement most likely relates to.
Choose the most appropriate one from the following list and their definitions:

"""
    for qa in quality_attribute_classes.keys():
        definition = quality_attribute_definitions[qa]
        prompt += f"- {qa.capitalize()}: {definition}\n"

    prompt += """
Consider the weights of each provider when making your decision to ensure that higher-weighted providers have more influence.

Provide your answer in the following format:

[Quality Attribute]

Examples:

- Security
- Performance
"""

    return prompt


def llm_classify_meta_model_qa_classification(requirement, base_model_responses, provider_weights, meta_model):
    """
    Use the specified meta-model to classify a requirement into a quality attribute,
    given the responses from the base models and their weights.

    Args:
        requirement (str): The nonfunctional requirement text.
        base_model_responses (list of tuples): Each tuple contains ('provider', 'classification').
        provider_weights (dict): Weights assigned to each provider.
        meta_model (str): The meta-model to use ('openai', 'anthropic', 'groq', 'gemma2').

    Returns:
        str: The classification result.
    """
    try:
        # Create the meta-model prompt incorporating provider weights
        meta_prompt = create_meta_model_prompt_qa_classification(requirement, base_model_responses, provider_weights)
        
        # Determine which meta-model to use and make the appropriate API call
        if meta_model == 'openai':
            logger.info(f"Meta-Model QA Classification: Using OpenAI as meta-model.")
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": meta_prompt}
                ],
                max_tokens=100,
                n=1,
                temperature=0.0
            )
            output = response.choices[0].message.content.strip()
            logger.info(f"OpenAI Meta-Model Response:\n{output}")

        elif meta_model == 'anthropic':
            logger.info(f"Meta-Model QA Classification: Using Anthropic as meta-model.")
            anthropic_prompt = HUMAN_PROMPT + meta_prompt + AI_PROMPT
            response = anthropic_client.completions.create(
                prompt=anthropic_prompt,
                model="claude-2",
                max_tokens_to_sample=10,
                temperature=0.0,
            )
            output = response.completion.strip()
            logger.info(f"Anthropic Meta-Model Response:\n{output}")

        elif meta_model == 'groq':
            logger.info(f"Meta-Model QA Classification: Using Groq as meta-model.")
            response = groq_client.chat.completions.create(
                model="llama3-8b-8192",  # Specify the Groq model name
                messages=[
                    {"role": "user", "content": meta_prompt}
                ],
                max_tokens=100,
                n=1,
                temperature=0.0
            )
            output = response.choices[0].message.content.strip()
            logger.info(f"Groq Meta-Model Response:\n{output}")

        elif meta_model == 'gemma2':
            logger.info(f"Meta-Model QA Classification: Using Gemma2 as meta-model.")
            response = groq_client.chat.completions.create(
                model="gemma2-9b-it",  # Specify the Gemma2 model name
                messages=[
                    {"role": "user", "content": meta_prompt}
                ],
                max_tokens=100,
                n=1,
                temperature=0.0
            )
            output = response.choices[0].message.content.strip()
            logger.info(f"Gemma2 Meta-Model Response:\n{output}")

        else:
            logger.error(f"Invalid meta-model specified: {meta_model}")
            return 'Undetermined'

        # Parse the output to get the classification
        classification = parse_meta_model_output_classification(
            output, 
            list(map(str.capitalize, quality_attribute_classes.keys()))
        )
        logger.info(f"Final Meta-Model Classification: {classification}")
        return classification

    except Exception as e:
        logger.error(f"Meta-model QA Classification Error: {e}")
        return 'Undetermined'

def create_meta_model_prompt_functional_classification(requirement, base_model_responses, provider_weights):
    """
    Create a prompt for the meta-model to classify a functional requirement into primary or secondary,
    and identify operationalized NFRs, given the responses from the base models and their weights.

    Args:
        requirement (str): The functional requirement text.
        base_model_responses (list of tuples): Each tuple contains ('provider', ('classification', 'operationalizes_nfr')).
        provider_weights (dict): Weights assigned to each provider.

    Returns:
        str: The prompt to send to the meta-model.
    """
    prompt = f"""
You are an expert in software engineering requirements classification.

For the following functional requirement:

"{requirement}"

We have the following classifications provided by different models, along with their weights:

"""
    for provider, (classification, operationalizes_nfr) in base_model_responses:
        weight = provider_weights.get(provider, 1.0)
        prompt += f"- {provider.capitalize()} (Weight: {weight}): {classification.capitalize()}"
        if classification.lower() == 'secondary' and operationalizes_nfr != 'N/A':
            prompt += f" (operationalizes '{operationalizes_nfr}')"
        prompt += "\n"

    prompt += f"""
Your task is to classify the requirement as 'Primary' or 'Secondary' functional requirement based on the following definitions:

- **Primary Functional Requirements**: {functional_type_definitions['primary']}

- **Secondary Functional Requirements**: {functional_type_definitions['secondary']}

If it is 'Secondary', specify which NFR(s) it operationalizes (choose from the following list):

"""
    for qa in quality_attribute_classes.keys():
        prompt += f"- {qa.capitalize()}\n"

    prompt += """
Consider the weights of each provider when making your decision to ensure that higher-weighted providers have more influence.

Provide your answer in the following format:

[Classification] (operationalizes 'NFR1, NFR2, ...')

Examples:
- Primary
- Secondary (operationalizes 'Security, Performance')
"""

    return prompt



@retry_on_failure(max_retries=3, delay=2, backoff=2)
def llm_classify_meta_model_functional_classification(requirement, base_model_responses, provider_weights, meta_model):
    """
    Use the specified meta-model to classify a functional requirement into primary or secondary,
    and identify operationalized NFRs, given the prompt.

    Args:
        requirement (str): The functional requirement text.
        base_model_responses (list of tuples): Each tuple contains ('provider', ('classification', 'operationalizes_nfr')).
        provider_weights (dict): Weights assigned to each provider.
        meta_model (str): The meta-model to use.

    Returns:
        tuple: ('classification', 'operationalizes_nfr')
    """
    try:
        meta_prompt = create_meta_model_prompt_functional_classification(requirement, base_model_responses, provider_weights)
        if meta_model == 'openai':
            response = openai_client.chat.completions.create(
                model= "gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": meta_prompt}
                ],
                max_tokens=50,
                n=1,
                temperature=0.0
            )
            output = response.choices[0].message.content.strip()
        elif meta_model == 'anthropic':
            anthropic_prompt = HUMAN_PROMPT + meta_prompt + AI_PROMPT
            response = anthropic_client.completion(
                prompt=anthropic_prompt,
                model="claude-2",
                max_tokens_to_sample=50,
                temperature=0.0,
            )
            output = response.completion.strip()
        elif meta_model == 'groq':
            logger.info(f"Meta-Model QA Classification: Using Groq as meta-model.")
            response = groq_client.chat.completions.create(
                model="llama3-8b-8192",  # Specify the Groq model name
                messages=[
                    {"role": "user", "content": meta_prompt}
                ],
                max_tokens=100,
                n=1,
                temperature=0.0
            )
            output = response.choices[0].message.content.strip()
            logger.info(f"Groq Meta-Model Response:\n{output}")

        elif meta_model == 'gemma2':
            logger.info(f"Meta-Model QA Classification: Using Gemma2 as meta-model.")
            response = groq_client.chat.completions.create(
                model="gemma2-9b-it",  # Specify the Gemma2 model name
                messages=[
                    {"role": "user", "content": meta_prompt}
                ],
                max_tokens=100,
                n=1,
                temperature=0.0
            )
            output = response.choices[0].message.content.strip()
            logger.info(f"Gemma2 Meta-Model Response:\n{output}")

        else:
            logger.error(f"Invalid meta-model specified: {meta_model}")
            return ('undetermined', 'N/A')

        # Parse the output to get the classification and operationalizes_nfr
        result = parse_meta_model_output_functional_classification(output, ['Primary', 'Secondary'])
        return result

    except Exception as e:
        logger.error(f"Meta-model Functional Classification Error: {e}")
        return ('Undetermined', 'N/A')

def create_meta_model_prompt_task_classification(requirement, base_model_responses, provider_weights):
    """
    Create a prompt for the meta-model to classify a functional requirement into UserTask or SystemTask,
    given the responses from the base models and their weights.

    Args:
        requirement (str): The functional requirement text.
        base_model_responses (list of tuples): Each tuple contains ('provider', 'classification').
        provider_weights (dict): Weights assigned to each provider.

    Returns:
        str: The prompt to send to the meta-model.
    """
    prompt = f"""
You are an expert in software engineering requirements classification.

For the following functional requirement:

"{requirement}"

We have the following classifications provided by different models, along with their weights:

"""
    for provider, classification in base_model_responses:
        weight = provider_weights.get(provider, 1.0)
        prompt += f"- {provider.capitalize()} (Weight: {weight}): {classification}\n"

    prompt += f"""
**Instructions:**
- Classify the requirement as **'UserTask'** or **'SystemTask'** based on the following definitions:
  
  - **UserTask**: {task_type_definitions['UserTask']}

  - **SystemTask**: {task_type_definitions['SystemTask']}

- Provide only the classification (**'UserTask'** or **'SystemTask'**) without any additional explanations or text.

**Examples:**
- UserTask
- SystemTask

**Now, provide your classification:**
"""

    return prompt

def llm_classify_meta_model_task_classification(requirement, base_model_responses, provider_weights, meta_model):
    """
    Use the specified meta-model to classify a functional requirement into UserTask or SystemTask,
    given the responses from the base models and their weights.

    Args:
        requirement (str): The functional requirement text.
        base_model_responses (list of tuples): Each tuple contains ('provider', 'classification').
        provider_weights (dict): Weights assigned to each provider.
        meta_model (str): The meta-model to use ('openai', 'anthropic', 'groq', 'gemma2').

    Returns:
        str: The classification result.
    """
    try:
        meta_prompt = create_meta_model_prompt_task_classification(requirement, base_model_responses, provider_weights)
        if meta_model == 'openai':
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": meta_prompt}
                ],
                max_tokens=5,  # Minimal tokens needed
                n=1,
                temperature=0.0  # Deterministic output
            )
            output = response.choices[0].message.content.strip()
        elif meta_model == 'anthropic':
            anthropic_prompt = HUMAN_PROMPT + meta_prompt + AI_PROMPT
            response = anthropic_client.completions.create(
                prompt=anthropic_prompt,
                model="claude-2",
                max_tokens_to_sample=5,
                temperature=0.0,
            )
            output = response.completion.strip()
        elif meta_model == 'groq':
            response = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "user", "content": meta_prompt}
                ],
                max_tokens=5,
                n=1,
                temperature=0.0
            )
            output = response.choices[0].message.content.strip()
        elif meta_model == 'gemma2':
            response = groq_client.chat.completions.create(  # Using groq_client as per user instruction
                model="gemma2-9b-it",
                messages=[
                    {"role": "user", "content": meta_prompt}
                ],
                max_tokens=5,
                n=1,
                temperature=0.0
            )
            output = response.choices[0].message.content.strip()
        else:
            logger.error(f"Invalid meta-model specified: {meta_model}")
            return 'Undetermined'

        # Parse the output to get the classification
        classification = parse_meta_model_output_task_classification(output, ['UserTask', 'SystemTask'])
        return classification

    except Exception as e:
        logger.error(f"Meta-model Task Classification Error: {e}")
        return 'Undetermined'


# ============================
# Voting Algorithms for Initial Classification
# ============================
def voting_algorithm(responses, provider_weights, confidence_threshold):
    """
    Aggregate classification responses using weighted voting.

    Args:
        responses (list of tuples): Each tuple contains ('provider', 'classification').
        provider_weights (dict): Weights assigned to each provider.
        confidence_threshold (float): Confidence threshold for determining final classification.

    Returns:
        str: 'functional', 'nonfunctional', or 'undetermined'.
    """
    total_weights = defaultdict(float)

    for provider, classification in responses:
        provider_weight = provider_weights.get(provider, 1.0)

        if classification in ['functional', 'nonfunctional']:
            total_weights[classification] += provider_weight
            logger.info(f"Voting Algorithm: Provider='{provider}', Classification='{classification}', Weight={provider_weight}")
        else:
            logger.warning(f"Voting Algorithm: Invalid classification '{classification}' from provider '{provider}'.")

    logger.info(f"Voting Algorithm: Total Weights = {dict(total_weights)}")

    if not total_weights:
        return 'undetermined'

    # Sort classifications by total weight descending
    sorted_classifications = sorted(total_weights.items(), key=lambda x: x[1], reverse=True)
    top_class, top_weight = sorted_classifications[0]
    total_weight = sum(total_weights.values())

    # Calculate confidence
    confidence = top_weight / total_weight

    if confidence >= confidence_threshold:
        logger.info(f"Voting Algorithm: Final Classification Decision = '{top_class}' with confidence {confidence:.2f}")
        return top_class
    else:
        # To minimize 'undetermined', lower the threshold dynamically or choose the top class anyway
        # Here, we choose to return the top class regardless, unless confidence is extremely low
        if confidence >= 0.3:  # Set a lower bound to still make a decision
            logger.info(f"Voting Algorithm: Confidence below threshold. Choosing top class '{top_class}' with confidence {confidence:.2f}")
            return top_class
        else:
            logger.info("Voting Algorithm: Confidence threshold not met, returning 'undetermined'")
            return 'undetermined'



def voting_algorithm_batch(responses_list, provider_weights, confidence_threshold):
    """
    Aggregate classification responses using weighted voting for batches.

    Args:
        responses_list (list of list of tuples): Each element corresponds to a requirement and contains a list of tuples ('provider', 'classification').
        provider_weights (dict): Weights assigned to each provider.
        confidence_threshold (float): Confidence threshold for determining final classification.

    Returns:
        list of str: Final classifications for each requirement.
    """
    final_classifications = []
    for responses in responses_list:
        final_classification = voting_algorithm(responses, provider_weights, confidence_threshold)
        final_classifications.append(final_classification)
    return final_classifications


# ============================
# Classification Pipeline
# ============================

def classify_quality_attributes_batch(requirements, meta_model, provider_weights):
    """
    Classify the quality attributes of a list of nonfunctional requirements using stacking.

    Args:
        requirements (list of str): The nonfunctional requirement texts.
        meta_model (str): The meta-model to use ('openai', 'anthropic', 'groq', 'gemma2').
        provider_weights (dict): Weights assigned to each provider.

    Returns:
        list of str: The most likely quality attribute for each requirement.
    """
    batch_size = 40  # Adjust as needed
    qa_results = []

    for i in range(0, len(requirements), batch_size):
        batch_requirements = requirements[i:i+batch_size]
        responses_list = [[] for _ in batch_requirements]

        base_models = ['openai', 'anthropic', 'groq', 'gemma2']  # Included 'groq' and 'gemma2'
        if meta_model in base_models:
            base_models.remove(meta_model)
        else:
            logger.warning(f"Meta-model '{meta_model}' not in base models list. Proceeding without removal.")

        for temperature in temperatures:
            # Collect responses from base models concurrently
            with ThreadPoolExecutor(max_workers=4) as executor:  # Increased max_workers to accommodate more providers
                future_to_provider = {}
                for model in base_models:
                    if model == 'openai':
                        future_to_provider[executor.submit(
                            llm_classify_openai_qa_classification_batch, 
                            batch_requirements, 
                            temperature
                        )] = 'openai'
                    elif model == 'anthropic':
                        future_to_provider[executor.submit(
                            llm_classify_anthropic_qa_classification_batch, 
                            batch_requirements, 
                            temperature
                        )] = 'anthropic'
                    elif model == 'groq':
                        future_to_provider[executor.submit(
                            llm_classify_groq_qa_classification_batch, 
                            batch_requirements, 
                            temperature
                        )] = 'groq'
                    elif model == 'gemma2':
                        future_to_provider[executor.submit(
                            llm_classify_gemma2_qa_classification_batch, 
                            batch_requirements, 
                            temperature
                        )] = 'gemma2'
                    else:
                        logger.warning(f"Unknown base model '{model}' encountered. Skipping.")

                for future in as_completed(future_to_provider):
                    provider = future_to_provider[future]
                    try:
                        classifications = future.result()
                        for idx, classification in enumerate(classifications):
                            responses_list[idx].append((provider, classification))
                    except Exception as e:
                        logger.error(f"Error from {provider}: {e}")
                        for idx in range(len(batch_requirements)):
                            responses_list[idx].append((provider, 'Undetermined'))

        # Now, for each requirement, create a prompt for the meta-model
        meta_model_classifications = []

        for idx, responses in enumerate(responses_list):
            # Create a prompt for the meta-model
            meta_prompt = create_meta_model_prompt_qa_classification(batch_requirements[idx], responses, provider_weights)
            # Send the prompt to the meta-model and get the final classification
            meta_classification = llm_classify_meta_model_qa_classification(
                requirement=batch_requirements[idx],
                base_model_responses=responses,
                provider_weights=provider_weights,
                meta_model=meta_model
            )
            meta_model_classifications.append(meta_classification)

        qa_results.extend(meta_model_classifications)

    return qa_results 


def classify_functional_requirements_batch(requirements, meta_model, provider_weights):
    """
    Classify functional requirements into primary or secondary and identify operationalized NFRs.

    Args:
        requirements (list of str): The functional requirement texts.
        meta_model (str): The meta-model to use for stacking ('openai', 'anthropic', 'groq', 'gemma2').
        provider_weights (dict): Weights assigned to each provider.

    Returns:
        list of tuples: Each tuple contains ('classification', 'operationalizes_nfr').
    """
    batch_size = 40  # Adjust based on API limitations and token usage
    functional_classifications = []

    for i in range(0, len(requirements), batch_size):
        batch_requirements = requirements[i:i+batch_size]
        responses_list = [[] for _ in batch_requirements]

        base_models = ['openai', 'anthropic', 'groq', 'gemma2']  # Included 'groq' and 'gemma2'
        if meta_model in base_models:
            base_models.remove(meta_model)
        else:
            logger.warning(f"Meta-model '{meta_model}' not in base models list. Proceeding without removal.")

        for temperature in temperatures:
            # Collect responses from base models concurrently
            with ThreadPoolExecutor(max_workers=4) as executor:  # Increased max_workers to accommodate more providers
                future_to_provider = {}
                for model in base_models:
                    if model == 'openai':
                        future_to_provider[executor.submit(
                            llm_classify_openai_functional_batch, batch_requirements, temperature
                        )] = 'openai'
                    elif model == 'anthropic':
                        future_to_provider[executor.submit(
                            llm_classify_anthropic_functional_batch, batch_requirements, temperature
                        )] = 'anthropic'
                    elif model == 'groq':
                        future_to_provider[executor.submit(
                            llm_classify_groq_functional_batch, batch_requirements, temperature
                        )] = 'groq'
                    elif model == 'gemma2':
                        future_to_provider[executor.submit(
                            llm_classify_gemma2_functional_batch, batch_requirements, temperature
                        )] = 'gemma2'
                    else:
                        logger.warning(f"Unknown base model '{model}' encountered. Skipping.")

                for future in as_completed(future_to_provider):
                    provider = future_to_provider[future]
                    try:
                        classifications = future.result()
                        for idx, classification in enumerate(classifications):
                            responses_list[idx].append((provider, classification))
                    except Exception as e:
                        logger.error(f"Error from {provider}: {e}")
                        for idx in range(len(batch_requirements)):
                            responses_list[idx].append((provider, ('Undetermined', 'N/A')))

        # Now, for each requirement, create a prompt for the meta-model
        meta_model_classifications = []

        for idx, responses in enumerate(responses_list):
            # Create a prompt for the meta-model
            meta_prompt = create_meta_model_prompt_functional_classification(batch_requirements[idx], responses, provider_weights)
            # Send the prompt to the meta-model and get the final classification
            meta_classification = llm_classify_meta_model_functional_classification(
                requirement=batch_requirements[idx],
                base_model_responses=responses,
                provider_weights=provider_weights,
                meta_model=meta_model
            )
            meta_model_classifications.append(meta_classification)

        functional_classifications.extend(meta_model_classifications)

    return functional_classifications


def classify_task_requirements_batch(requirements, meta_model, provider_weights):
    """
    Classify functional requirements into UserTask or SystemTask using stacking.

    Args:
        requirements (list of str): The functional requirement texts.
        meta_model (str): The meta-model to use for stacking ('openai', 'anthropic', 'groq', 'gemma2').
        provider_weights (dict): Weights assigned to each provider.

    Returns:
        list of str: The task classifications for each requirement.
    """
    batch_size = 40  # Adjust based on API limitations and token usage
    task_classifications = []

    for i in range(0, len(requirements), batch_size):
        batch_requirements = requirements[i:i+batch_size]
        responses_list = [[] for _ in batch_requirements]

        # Include 'groq' and 'gemma2' in the base models
        base_models = ['openai', 'anthropic', 'groq', 'gemma2']  # Included 'groq' and 'gemma2'
        if meta_model in base_models:
            base_models.remove(meta_model)
        else:
            logger.warning(f"Meta-model '{meta_model}' not in base models list. Proceeding without removal.")

        for temperature in temperatures:
            # Collect responses from base models concurrently
            with ThreadPoolExecutor(max_workers=4) as executor:  # Increased max_workers to accommodate more providers
                future_to_provider = {}
                for model in base_models:
                    if model == 'openai':
                        future_to_provider[executor.submit(
                            llm_classify_openai_task_batch, batch_requirements, temperature
                        )] = 'openai'
                    elif model == 'anthropic':
                        future_to_provider[executor.submit(
                            llm_classify_anthropic_task_batch, batch_requirements, temperature
                        )] = 'anthropic'
                    elif model == 'groq':
                        future_to_provider[executor.submit(
                            llm_classify_groq_task_batch, batch_requirements, temperature
                        )] = 'groq'
                    elif model == 'gemma2':
                        future_to_provider[executor.submit(
                            llm_classify_gemma2_task_batch, batch_requirements, temperature
                        )] = 'gemma2'
                    else:
                        logger.warning(f"Unknown base model '{model}' encountered. Skipping.")

                for future in as_completed(future_to_provider):
                    provider = future_to_provider[future]
                    try:
                        classifications = future.result()
                        for idx, classification in enumerate(classifications):
                            responses_list[idx].append((provider, classification))
                    except Exception as e:
                        logger.error(f"Error from {provider}: {e}")
                        for idx in range(len(batch_requirements)):
                            responses_list[idx].append((provider, 'Undetermined'))

        # Now, for each requirement, create a prompt for the meta-model
        meta_model_classifications = []

        for idx, responses in enumerate(responses_list):
            # Create a prompt for the meta-model
            meta_prompt = create_meta_model_prompt_task_classification(batch_requirements[idx], responses, provider_weights)
            # Send the prompt to the meta-model and get the final classification
            meta_classification = llm_classify_meta_model_task_classification(
               requirement=batch_requirements[idx],
                base_model_responses=responses,
                provider_weights=provider_weights,
                meta_model=meta_model
            )
            meta_model_classifications.append(meta_classification)

        task_classifications.extend(meta_model_classifications)

    return task_classifications 


# ============================
# Requirement Classification Pipeline
# ============================

def classify_requirements_batch(requirements, meta_model, provider_weights, confidence_threshold):
    """
    Classify a list of requirements in batch.

    Args:
        requirements (list of str): The list of requirement texts to classify.
        meta_model (str): The meta-model to use ('openai', 'anthropic', 'groq', 'gemma2').
        provider_weights (dict): Weights assigned to each provider.
        confidence_threshold (float): Confidence threshold for voting algorithm.

    Returns:
        list of dict: Contains 'requirement', 'classification', 'functional_type', 'task_type', 'quality_attributes', and 'operationalizes_nfr' for each requirement.
    """
    batch_size = 40  # Adjust based on API limitations and token usage
    results = []

    for i in range(0, len(requirements), batch_size):
        batch_requirements = requirements[i:i+batch_size]
        responses_list = [[] for _ in batch_requirements]

        # Step 1: Classify as functional or nonfunctional using the voting algorithm
        base_models_initial = ['openai', 'anthropic', 'groq', 'gemma2']  # Included 'groq' and 'gemma2'
        for temperature in temperatures:
            # Collect responses from all LLMs concurrently
            with ThreadPoolExecutor(max_workers=4) as executor:  # Increased max_workers to accommodate more providers
                future_to_provider = {}
                for model in base_models_initial:
                    if model == 'openai':
                        future_to_provider[executor.submit(llm_classify_openai_batch, batch_requirements, temperature)] = 'openai'
                    elif model == 'anthropic':
                        future_to_provider[executor.submit(llm_classify_anthropic_batch, batch_requirements, temperature)] = 'anthropic'
                    elif model == 'groq':
                        future_to_provider[executor.submit(llm_classify_groq_batch, batch_requirements, temperature)] = 'groq'
                    elif model == 'gemma2':
                        future_to_provider[executor.submit(llm_classify_gemma2_batch, batch_requirements, temperature)] = 'gemma2'
                    else:
                        logger.warning(f"Unknown base model '{model}' encountered. Skipping.")

                for future in as_completed(future_to_provider):
                    provider = future_to_provider[future]
                    try:
                        classifications = future.result()
                        for idx, classification in enumerate(classifications):
                            responses_list[idx].append((provider, classification))
                    except Exception as e:
                        logger.error(f"Error from {provider}: {e}")
                        for idx in range(len(batch_requirements)):
                            responses_list[idx].append((provider, 'undetermined'))

        # Aggregate responses for the batch using weighted voting with confidence_threshold
        batch_final_classifications = voting_algorithm_batch(
            responses_list, 
            provider_weights, 
            confidence_threshold
        )

        # Prepare lists for further classification
        nonfunctional_requirements = []
        nonfunctional_indices = []
        functional_requirements = []
        functional_indices = []

        for idx, classification in enumerate(batch_final_classifications):
            if classification == 'nonfunctional':
                nonfunctional_requirements.append(batch_requirements[idx])
                nonfunctional_indices.append(idx)
            elif classification == 'functional':
                functional_requirements.append(batch_requirements[idx])
                functional_indices.append(idx)
            else:
                logger.info(f"Requirement '{batch_requirements[idx]}' classified as 'undetermined'.")

        # Step 2: Handle quality attribute classification for nonfunctional requirements using stacking
        qa_results = ['N/A'] * len(batch_requirements)
        if nonfunctional_requirements:
            qa_classifications = classify_quality_attributes_batch(
                nonfunctional_requirements, 
                meta_model, 
                provider_weights
            )
            for idx, qa in zip(nonfunctional_indices, qa_classifications):
                qa_results[idx] = qa.capitalize() if qa != 'Undetermined' else 'Undetermined'

        # Step 3: Handle functional requirement classification into primary/secondary and operationalizes NFR using stacking
        functional_type_results = ['N/A'] * len(batch_requirements)
        operationalization_results = ['N/A'] * len(batch_requirements)
        if functional_requirements:
            func_classifications = classify_functional_requirements_batch(
                functional_requirements, 
                meta_model, 
                provider_weights
            )
            for idx, (func_type, op_nfr) in zip(functional_indices, func_classifications):
                functional_type_results[idx] = func_type.capitalize() if func_type != 'Undetermined' else 'Undetermined'
                operationalization_results[idx] = op_nfr if op_nfr != 'N/A' else 'N/A'

        # Step 4: Handle task classification for functional requirements using stacking
        task_results = ['N/A'] * len(batch_requirements)
        if functional_requirements:
            task_classifications = classify_task_requirements_batch(
                functional_requirements, 
                meta_model, 
                provider_weights
            )
            for idx, task_type in zip(functional_indices, task_classifications):
                task_results[idx] = task_type.capitalize() if task_type != 'Undetermined' else 'Undetermined'

        # Compile the results
        for idx, req in enumerate(batch_requirements):
            classification = batch_final_classifications[idx]
            result = {
                'requirement': req,
                'classification': classification,
                'functional_type': functional_type_results[idx],
                'task_type': task_results[idx],
                'quality_attributes': qa_results[idx],
                'operationalizes_nfr': operationalization_results[idx]
            }
            results.append(result)

    return results 


# ============================
# Utility Functions
# ============================

def process_file(file_path, meta_model, provider_weights, confidence_threshold):
    """
    Process a file containing requirements, classify each requirement, and save the results.

    Args:
        file_path (str): Path to the input file.
        meta_model (str): The meta-model to use ('openai', 'anthropic', 'groq', 'gemma2').
        provider_weights (dict): Weights assigned to each provider.
        confidence_threshold (float): Confidence threshold for voting algorithm.

    Returns:
        dict: Contains 'results' and 'output_file' path.
    """
    logger.info(f"Processing file: {file_path} with meta-model: {meta_model}, provider_weights: {provider_weights}, confidence_threshold: {confidence_threshold}")
    requirements = extract_requirements(file_path)
    final_results = classify_requirements_batch(
        requirements, 
        meta_model, 
        provider_weights, 
        confidence_threshold
    )
    output_file_path = save_results(final_results, file_path)
    logger.info(f"Finished processing file. Results saved to {output_file_path}")
    return {'results': final_results, 'output_file': output_file_path}


def extract_requirements(file_path):
    """
    Extract non-empty lines from the input file as requirements.

    Args:
        file_path (str): Path to the input file.

    Returns:
        list: List of requirement strings.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except UnicodeDecodeError:
        # Try reading the file with a different encoding
        with open(file_path, 'r', encoding='latin1') as file:
            lines = file.readlines()

    requirements = [line.strip() for line in lines if line.strip()]
    logger.info(f"Extracted {len(requirements)} requirements from '{file_path}'")
    return requirements

def save_results(results, original_file_path):
    """
    Save the classification results to a CSV file.

    Args:
        results (list of dict): List containing requirements and their classifications.
        original_file_path (str): Path to the original input file.

    Returns:
        str: Path to the saved CSV file.
    """
    output_file = original_file_path + "_classified.csv"
    try:
        with open(output_file, "w", encoding='utf-8') as f:
            f.write("Requirement,Classification,FunctionalType,TaskType,QualityAttributes,OperationalizesNFR\n")
            for item in results:
                req = item['requirement'].replace('"', '""')  # Escape double quotes
                cls = item['classification']
                func_type = item.get('functional_type', 'N/A')
                task_type = item.get('task_type', 'N/A')
                qas = item.get('quality_attributes', 'N/A')
                op_nfr = item.get('operationalizes_nfr', 'N/A')
                f.write(f'"{req}","{cls}","{func_type}","{task_type}","{qas}","{op_nfr}"\n')  # Enclose in quotes to handle any commas
        logger.info(f"Results successfully saved to '{output_file}'")
    except Exception as e:
        logger.error(f"Error saving results to '{output_file}': {e}")
    return output_file

# ============================
# Main Execution Block (For Testing)
# ============================


if __name__ == "__main__":
    # Example usage: Process a sample file
    sample_file = os.path.join(script_dir, "uploaded_files", "Reqs.txt")
    if not os.path.exists(sample_file):
        logger.error(f"Sample file '{sample_file}' does not exist.")
        print(f"Sample file '{sample_file}' does not exist.")
    else:
        # Specify the meta-model to use (e.g., 'openai', 'anthropic', 'groq', 'gemma2')
        # Define provider weights and confidence threshold
        provider_weights = {
            'openai': 1.0,
            'anthropic': 0.9,
            'groq': 0.8,
            'gemma2': 0.85,
        }
        confidence_threshold = 0.6  # Value between 0.5 and 1.0
        result = process_file(sample_file, meta_model='openai', provider_weights=provider_weights, confidence_threshold=confidence_threshold)
        print(f"Processing completed. Results saved to: {result['output_file']}")
