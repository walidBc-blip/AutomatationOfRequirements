# ontology.py

from owlready2 import *
import os
import re
import spacy
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import pandas as pd

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")  # Ensure this model is installed

# Create a new ontology for Requirements
onto = get_ontology("http://example.org/RequirementsOntology.owl")

# ============================
# Ontology Definitions
# ============================

with onto:
    # General Requirement Class
    class Requirement(Thing):
        """A general class for requirements."""
        pass

    # Functional Requirement Class
    class FunctionalRequirement(Requirement):
        """Requirements specifying functions the system must perform."""
        pass

    # Nonfunctional Requirement Class
    class NonFunctionalRequirement(Requirement):
        """Requirements specifying qualities the system must have."""
        pass

    # Quality Attribute Class
    class QualityAttribute(Thing):
        """A class representing quality attributes."""
        pass

    # Operationalization Class
    class Operationalization(Thing):
        """Defines specific operationalization measures for NFRs."""
        pass

    # Task Class
    class Task(Thing):
        """A task that is performed in the system."""
        pass

    class UserTask(Task):
        """Tasks that require user initiation or direct user interaction."""
        pass

    class SystemTask(Task):
        """Tasks that the system performs automatically without user involvement."""
        pass

    # Primary and Secondary Functional Requirements
    class PrimaryFunctionalRequirement(FunctionalRequirement):
        """FRs directly contributing to the main goal of the system."""
        pass

    class SecondaryFunctionalRequirement(FunctionalRequirement):
        """FRs that are auxiliary to the main goal, such as logging or compliance."""
        pass

    # Properties
    class isRealizedBy(ObjectProperty):
        """Links FunctionalRequirements to Tasks that realize them."""
        domain = [FunctionalRequirement]
        range = [Task]

    class isDecomposedInto(ObjectProperty):
        """Defines decomposition of Requirements into sub-requirements."""
        domain = [Requirement]
        range = [Requirement]
        transitive = True  # Enables cascading decomposition relations

    class hasQualityAttribute(ObjectProperty):
        """Associates NFRs with Quality Attributes."""
        domain = [NonFunctionalRequirement]
        range = [QualityAttribute]

    class isAssociatedWith(ObjectProperty):
        """Associates Requirements with other entities."""
        domain = [Requirement]
        range = [Thing]

    class operationalizes(ObjectProperty):
        """Indicates that a FunctionalRequirement operationalizes a QualityAttribute."""
        domain = [FunctionalRequirement]
        range = [QualityAttribute]

    class hasOperationalization(ObjectProperty):
        """Defines how a QualityAttribute is operationalized in the system."""
        domain = [QualityAttribute]
        range = [Operationalization]

    class hasKeyword(DataProperty):
        """Associates keywords with ontology classes or instances."""
        domain = [Thing]
        range = [str]

    class hasDescription(DataProperty):
        """Provides a detailed description of the class or instance."""
        domain = [Thing]
        range = [str]
        functional = [True]  # Ensures only one description per instance

    # ============================
    # Quality Attributes and Tactics
    # ============================

    # Example: Security Quality Attribute
    class Security(QualityAttribute):
        """Requirement to protect system data and ensure secure access."""
        pass

    # Define Tactics for Security
    class SecurityTactic(Operationalization):
        """A tactic used to achieve security requirements."""
        pass

    class Authentication(SecurityTactic):
        """Verifying the identity of users or systems."""
        pass

    class Authorization(SecurityTactic):
        """Determining access rights for authenticated users."""
        pass

    class Encryption(SecurityTactic):
        """Protecting data by encoding it."""
        pass

    class AuditAndMonitoring(SecurityTactic):
        """Tracking system activities to detect security breaches."""
        pass

    class InputValidation(SecurityTactic):
        """Ensuring that inputs meet expected criteria to prevent attacks."""
        pass

    # Create instances of tactics
    authentication_tactic = Authentication("AuthenticationMechanism")
    authorization_tactic = Authorization("AccessControl")
    encryption_tactic = Encryption("DataEncryption")
    audit_tactic = AuditAndMonitoring("AuditLogging")
    input_validation_tactic = InputValidation("InputValidationMechanism")

    # Add keywords to tactics
    authentication_tactic.hasKeyword = ["login", "user verification", "credentials", "multi-factor authentication", "password reset"]
    authorization_tactic.hasKeyword = ["permissions", "access control", "roles", "privileges"]
    encryption_tactic.hasKeyword = ["encryption", "decryption", "cipher", "SSL/TLS", "HTTPS"]
    audit_tactic.hasKeyword = ["logging", "monitoring", "audit trails", "security logs", "intrusion detection"]
    input_validation_tactic.hasKeyword = ["sanitization", "input checking", "validation", "whitelisting", "blacklisting"]

    # Link Security to Tactics
    Security.hasOperationalization = [
        authentication_tactic,
        authorization_tactic,
        encryption_tactic,
        audit_tactic,
        input_validation_tactic
    ]

    # Repeat similar definitions for other Quality Attributes (e.g., Usability, Performance, etc.)
    # For brevity, only Security is fully defined here. You can expand similarly for others.

    # ============================
    # Mapping Quality Attributes
    # ============================

    quality_attribute_classes = {
        'security': Security,
        # Add other Quality Attributes like 'usability', 'performance', etc., similarly
    }

    # ============================
    # Functional Requirements and Tasks
    # ============================

    # Example Functional Requirements
    fr1 = FunctionalRequirement("FR1")
    fr1.hasDescription = ["The system shall allow users to reset their password using a recovery email."]
    fr1.is_a.append(SecondaryFunctionalRequirement)
    password_reset_task = UserTask("PasswordResetTask")
    fr1.isRealizedBy.append(password_reset_task)

    fr2 = FunctionalRequirement("FR2")
    fr2.hasDescription = ["The application shall provide real-time notifications for new messages and alerts."]
    fr2.is_a.append(PrimaryFunctionalRequirement)
    realtime_notification_task = SystemTask("RealTimeNotificationTask")
    fr2.isRealizedBy.append(realtime_notification_task)

    # Determine if FRs operationalize any NFRs (e.g., Security)
    def operationalize_nfr(fr_instance):
        fr_text = fr_instance.hasDescription[0].lower()  # Assuming hasDescription is a list
        for qa_name, qa_class in quality_attribute_classes.items():
            for tactic in qa_class.hasOperationalization:
                if hasattr(tactic, 'hasKeyword'):
                    for keyword in tactic.hasKeyword:
                        if keyword.lower() in fr_text:
                            fr_instance.operationalizes.append(qa_class)
                            print(f"Functional Requirement '{fr_instance.hasDescription[0]}' operationalizes '{qa_name.capitalize()}' via '{tactic.name}'.")
                            break  # Assuming one tactic match is sufficient

    operationalize_nfr(fr1)
    operationalize_nfr(fr2)

# ============================
# Finalizing the Ontology
# ============================

# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the ontologies directory path
ontologies_dir = os.path.join(script_dir, "ontologies")

# Ensure the ontologies directory exists
os.makedirs(ontologies_dir, exist_ok=True)

# Define the full path to save the ontology
ontology_save_path = os.path.join(ontologies_dir, "RequirementsOntology.owl")

# Save the ontology to the ontologies directory
onto.save(file=ontology_save_path, format="rdfxml")

print("Requirements Ontology initialized and saved.")

# ============================
# Enhancements: Classification and Learning
# ============================

# Function to extract requirements from a file
def extract_requirements(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin1') as file:
            lines = file.readlines()

    requirements = [line.strip() for line in lines if line.strip()]
    print(f"Extracted {len(requirements)} requirements from '{file_path}'")
    return requirements

# Function to tokenize and lemmatize text using spaCy
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Prepare data for machine learning
def prepare_data(requirements):
    texts = [preprocess_text(req) for req in requirements]
    return texts

# Function to get labels from the ontology based on keywords
def get_labels_from_ontology(requirement_text):
    labels = set()
    req_text_lower = requirement_text.lower()
    for qa_name, qa_class in quality_attribute_classes.items():
        for tactic in qa_class.hasOperationalization:
            if hasattr(tactic, 'hasKeyword'):
                for keyword in tactic.hasKeyword:
                    if keyword.lower() in req_text_lower:
                        labels.add(qa_class.name)
                        break  # Avoid multiple additions from the same quality attribute
    # Additionally, determine if the requirement is Primary or Secondary and its Task Type
    # This requires semantic understanding; for simplicity, we'll use keyword matching
    # Define keywords for Primary/Secondary and Task Types
    primary_keywords = ["directly contribute", "main goal", "core functionality", "primary function"]
    secondary_keywords = ["auxiliary", "supporting function", "log", "compliance", "maintenance"]

    task_keywords_user = ["user", "users", "customer", "client", "operator"]
    task_keywords_system = ["system", "automatically", "background", "process"]

    # Check for Primary or Secondary
    for keyword in primary_keywords:
        if keyword in req_text_lower:
            labels.add("PrimaryFunctionalRequirement")
            break
    for keyword in secondary_keywords:
        if keyword in req_text_lower:
            labels.add("SecondaryFunctionalRequirement")
            break

    # Check for Task Type
    for keyword in task_keywords_user:
        if keyword in req_text_lower:
            labels.add("UserTask")
            break
    for keyword in task_keywords_system:
        if keyword in req_text_lower:
            labels.add("SystemTask")
            break

    return list(labels)

# Load the model and MultiLabelBinarizer
def load_model_and_mlb():
    model_path = os.path.join(script_dir, 'uploaded_files', 'model.pkl')
    mlb_path = os.path.join(script_dir, 'uploaded_files', 'mlb.pkl')

    if not os.path.exists(model_path) or not os.path.exists(mlb_path):
        print("Model and MultiLabelBinarizer not found. Please train the model first.")
        return None, None

    model = joblib.load(model_path)
    mlb = joblib.load(mlb_path)
    print("Model and MultiLabelBinarizer loaded successfully.")
    return model, mlb

# Function to classify requirements
def classify_requirements(requirements, pipeline, mlb):
    texts = prepare_data(requirements)
    # Predict using the model
    y_pred = pipeline.predict(texts)
    # Convert predictions back to labels
    predicted_labels = mlb.inverse_transform(y_pred)

    # Combine with ontology-based labels
    final_results = []
    for idx, req in enumerate(requirements):
        ontology_labels = get_labels_from_ontology(req)
        ml_labels = list(predicted_labels[idx])

        # Combine labels
        combined_labels = set(ontology_labels + ml_labels)
        if not combined_labels:
            combined_labels = ["Undetermined"]

        result = {
            'requirement': req,
            'classification': list(combined_labels)
        }
        final_results.append(result)

    return final_results

# Function to save results
def save_results(results, output_file):
    try:
        with open(output_file, "w", encoding='utf-8') as f:
            f.write("Requirement,Classification\n")
            for item in results:
                req = item['requirement'].replace('"', '""')
                cls = ', '.join(item['classification'])
                f.write(f'"{req}","{cls}"\n')
        print(f"Results successfully saved to '{output_file}'")
    except Exception as e:
        print(f"Error saving results to '{output_file}': {e}")

# ============================
# Main Execution Flow
# ============================

if __name__ == "__main__":
    # ============================
    # Model Training (Only once)
    # ============================
    # Uncomment the following lines if you need to train the model

    """
    # Sample labeled data for training (requirement text and their labels)
    # In practice, you should load this from a dataset
    training_requirements = [
        "The system shall encrypt all user data.",
        "Users must authenticate using multi-factor authentication.",
        "Implement caching to improve performance.",
        "The application should handle exceptions gracefully.",
        "Load balancing should distribute traffic evenly.",
        "System must be fault-tolerant and recover from errors.",
        "The application shall provide real-time notifications for new messages and alerts.",
        "The system shall allow users to reset their password using a recovery email."
    ]

    training_labels = [
        ["Security"],
        ["Security"],
        ["Performance"],
        ["Reliability"],
        ["Performance"],
        ["Reliability"],
        ["Usability", "SystemTask"],
        ["Security", "UserTask"]
    ]

    # Prepare training data
    training_texts = prepare_data(training_requirements)

    # Train the model
    pipeline, mlb = train_model(training_texts, training_labels)
    """

    # ============================
    # Classification of New Requirements
    # ============================

    # Load the trained model and MultiLabelBinarizer
    model, mlb = load_model_and_mlb()
    if not model or not mlb:
        exit()

    # Define the path to the new requirements file
    requirements_file = os.path.join(script_dir, 'uploaded_files', 'new_requirements.txt')

    # Load requirements
    if not os.path.exists(requirements_file):
        print(f"Sample file '{requirements_file}' does not exist.")
        exit()

    requirements = extract_requirements(requirements_file)

    # Classify requirements
    results = classify_requirements(requirements, model, mlb)

    # Define the output file path
    output_file = os.path.join(script_dir, 'uploaded_files', 'classified_requirements.csv')

    # Save the classification results
    save_results(results, output_file)

    print("Requirements classified successfully.")


