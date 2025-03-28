from simpletransformers.ner import NERArgs

MODEL_REGISTRY = {
    "bert": "google-bert/bert-base-uncased",
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "bluebert": "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16",
    "clinical-bert": "emilyalsentzer/Bio_ClinicalBERT",
    "biomed_roberta": "allenai/biomed_roberta_base",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "lava": "./models/lava"  # Your fine-tuned model stored locally
}

def get_model_name(short_name):
    """Returns the full model name for a given short model name."""
    if short_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{short_name}' not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[short_name]

def get_default_model_args(output_dir, labels):
    model_args = NERArgs()
    model_args.labels_list = labels
    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.sliding_window = True
    model_args.num_train_epochs = 3 #10
    model_args.train_batch_size = 4 #16
    model_args.fp16 = False #True
    model_args.output_dir = output_dir
    model_args.best_model_dir = f"{output_dir}/best_model/"
    model_args.evaluate_during_training = True
    model_args.show_running_loss = True
    model_args.use_early_stopping = True
    # model_args.wandb_project = "huggingface"
    # model_args.use_multiprocessing = False  
    # model_args.use_cuda = False  
    return model_args
