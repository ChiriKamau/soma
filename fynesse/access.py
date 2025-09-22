import pandas as pd

def attendance_acs(url: str) -> pd.DataFrame:
    """
    Load and clean school attendance data from a CSV URL.
    """
    df = pd.read_csv(url, skiprows=1)
    df.columns = df.columns.str.strip()
    return df.copy()
