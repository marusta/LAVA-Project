from ner_model import load_ner_model
from re_model import load_re_model

def extract_entities(model_dir, df_text):
    """Extracts named entities from text."""
    model = load_ner_model(model_dir)
    predictions, raw_outputs = model.predict(df_text["text"].tolist())
    return predictions

def extract_relations(model_dir, entity_pairs):
    """Extracts relations between named entities."""
    model = load_re_model(model_dir)
    predictions, raw_outputs = model.predict(entity_pairs)
    return predictions

def full_pipeline(ner_model_dir, re_model_dir, df_text):
    """Runs the full pipeline: NER → RE"""
    
    print("\n=== Extracting Entities ===")
    entities = extract_entities(ner_model_dir, df_text)

    entity_pairs = [(e1, e2) for ent_list in entities for e1 in ent_list for e2 in ent_list if e1 != e2]

    print("\n=== Extracting Relations ===")
    relations = extract_relations(re_model_dir, entity_pairs)

    print("\n=== Final Knowledge Extraction Output ===")
    for (e1, e2), relation in zip(entity_pairs, relations):
        print(f"{e1} -[{relation}]-> {e2}")
