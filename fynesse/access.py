import pandas as pd

def attendance_acs(url: str) -> pd.DataFrame:
    """
    Load and clean school attendance data from a CSV URL.
    """
    df = pd.read_csv(url, skiprows=1)
    df.columns = df.columns.str.strip()
    return df.copy()


def population_acs(url: str) -> pd.DataFrame:
    """
    Load and clean population CSV data.
    """
    df = pd.read_csv(url)
    # Convert relevant numeric columns
    df["Age"] = pd.to_numeric(df["Age"], errors='coerce')
    df["Total"] = pd.to_numeric(df["Total"], errors='coerce').fillna(0)
    return df.copy()