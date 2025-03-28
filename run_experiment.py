from data_loader import load_dataset
from train import train_ner, train_re
from evaluate import evaluate_ner, evaluate_re
from pipeline import full_pipeline

NER_MODEL_DIR = "ner_model"
RE_MODEL_DIR = "re_model"

df_train, df_test = load_dataset('bc5cdr', 'ner')
# print(df_train.head())

print("\n=== Training Models ===")
train_ner("bert","bert-base-uncased", df_train, df_test, NER_MODEL_DIR)
# train_re("bert", "bert-base-uncased", df_train, df_test, RE_MODEL_DIR)

# print("\n=== Evaluating Models ===")
# evaluate_ner(NER_MODEL_DIR, df)
# evaluate_re(RE_MODEL_DIR, df)

# print("\n=== Running Full Pipeline ===")
# full_pipeline(NER_MODEL_DIR, RE_MODEL_DIR, df)
