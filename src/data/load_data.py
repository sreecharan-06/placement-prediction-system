import os
import pandas as pd

def load_data():
    # Make path robust by locating it relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_path = os.path.join(base_dir, "data", "placement_data.csv")
    df = pd.read_csv(csv_path)
    return df

def get_summary(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "target": "PlacementStatus"
    }

if __name__ == "__main__":
    df = load_data()
    print(get_summary(df))
