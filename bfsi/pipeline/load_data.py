import os
import pandas as pd
from config import DATA_DIR

def load_all_data():
    """Load all source files from local data/ directory."""
    # Paths
    policy_path = os.path.join(DATA_DIR, "policy_corpus.txt")
    alerts_path = os.path.join(DATA_DIR, "alerts.csv")
    typology_path = os.path.join(DATA_DIR, "typology_thresholds.csv")
    sample_alerts_path = os.path.join(DATA_DIR, "sample_alerts.json")

    # Load policy text
    with open(policy_path, "r", encoding="utf-8") as f:
        policy_text = f.read()

    
    alerts_df = pd.read_csv(alerts_path)
    typology_df = pd.read_csv(typology_path)

    # JSON is optional for SAR testing, but we keep it
    # We'll return the path or load it – but main.py will load it directly if needed
    return alerts_df, typology_df, policy_text