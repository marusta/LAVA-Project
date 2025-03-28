import os
import torch
from simpletransformers.ner import NERModel
from config import get_default_model_args

def create_ner_model(model_type, model_name, labels, output_dir):
    """Initializes or loads a NER model."""
    
    model_args = get_default_model_args(output_dir, labels)

    if os.path.exists(model_name):
        print(f"Loading fine-tuned NER model from {model_name}...")
        model = NERModel(model_type, model_name, labels=labels, args=model_args, use_cuda=torch.cuda.is_available())
    
        # Check if the classifier layer needs to be replaced
        model_config = model.model.config
        if len(labels) != model_config.num_labels:
            print(f"Label mismatch detected: Model has {model_config.num_labels} labels, but dataset has {len(labels)} labels.")
            print("Reinitializing classifier layer to match new dataset labels.")

            # Reinitialize classifier layer
            model.model.classifier = torch.nn.Linear(model_config.hidden_size, len(labels))
            model.model.config.num_labels = len(labels)
            
    else:
        print(f"Initializing new NER model: {model_name}...")
        model = NERModel(model_type, model_name, labels=labels, args=model_args, use_cuda=torch.cuda.is_available())

    return model

def load_ner_model(model_dir):
    """Loads a trained NER model."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found.")
    
    return NERModel("bert", model_dir)
