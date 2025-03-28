import os
import pandas as pd

# Dataset paths (assumes datasets are stored in a "datasets" folder)
DATASET_PATHS = { 'ner' : {
                            "bc5cdr": {'train' : "datasets/preprocessed_NER/bc5cdr/train.csv", 'test' : "datasets/preprocessed_NER/bc5cdr/test.csv"},
                            "biored": {'train' : "datasets/preprocessed_NER/biored/train.csv", 'test' : "datasets/preprocessed_NER/biored/test.csv"},
                            "chemprot": {'train' : "datasets/preprocessed_NER/chemprot/train.csv", 'test' : "datasets/preprocessed_NER/chemprot/test.csv"},
                            "ncbi-disease": {'train' : "datasets/preprocessed_NER/ncbi-disease/train.csv", 'test' : "datasets/preprocessed_NER/ncbi-disease/test.csv"}
                        },
                  're' : {
                            "biored": {'train' : "datasets/preprocessed_RE/biored/train.csv", 'test' : "datasets/preprocessed_RE/biored/test.csv"},
                        }
        }

def load_dataset(dataset_name, task  ='ner'):
    """Loads a single dataset as a DataFrame."""
    if task not in DATASET_PATHS.keys():
        raise ValueError(f"Task '{task}' not found. Available: {list(DATASET_PATHS.keys())}")
    if dataset_name not in DATASET_PATHS[task].keys():
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(DATASET_PATHS[task].keys())}")
    
    train_path = DATASET_PATHS[task][dataset_name]['train']
    test_path = DATASET_PATHS[task][dataset_name]['test']
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Dataset file '{train_path}' not found. Make sure datasets are placed in the 'datasets' folder.")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset file '{test_path}' not found. Make sure datasets are placed in the 'datasets' folder.")
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    return df_train.astype(str), df_test.astype(str)

def load_multiple_datasets(dataset_names, task = 'ner'):
    """Loads and combines multiple datasets for training and testing."""
    
    train_dfs = []
    test_dfs = []
    
    counter = 0
    for dataset_name in dataset_names:
        df_train = load_dataset(dataset_name, 'train')
        df_test = load_dataset(dataset_name, 'test')
        
        df_train['sentence_id'] = df_train['sentence_id'].apply(lambda x: (chr(ord('a') + counter) + str(x)))
        df_test['sentence_id'] = df_test['sentence_id'].apply(lambda x: (chr(ord('a') + counter) + str(x)))

        train_dfs.append(df_train)
        test_dfs.append(df_test)
        counter += 1
    
    train_df = pd.concat(train_dfs) if train_dfs else None
    test_df = pd.concat(test_dfs) if test_dfs else None
    
    print(f"Loaded {len(dataset_names)} dataset(s): {dataset_names}")
    print(f"Training samples: {len(train_df)} | Test samples: {len(test_df)}")
    
    return train_df.astype(str), test_df.astype(str)