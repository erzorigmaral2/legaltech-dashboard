import pandas as pd

def load_data():
    df = pd.read_csv("data/mongolia_legal_survey_synthetic_2000.csv")
    return df
