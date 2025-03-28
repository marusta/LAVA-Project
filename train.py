from ner_model import create_ner_model
from re_model import create_re_model

def train_ner(model_type, model_name, df_train, df_dev, output_dir):
    """Trains the NER model."""
    
    labels = list(df_train["labels"].unique())
    model = create_ner_model(model_type, model_name, labels, output_dir)

    model.train_model(df_train, eval_data=df_dev)
    print(f"NER training complete. Model saved to {output_dir}")

def train_re(model_type, model_name, df_train, df_dev, output_dir):
    """Trains the Relation Extraction model."""
    
    labels = list(df_train["relation"].unique())
    model = create_re_model(model_type, model_name, labels, output_dir)

    model.train_model(df_train, eval_data=df_dev)
    print(f"RE training complete. Model saved to {output_dir}")
