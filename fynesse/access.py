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

def education_acs(url: str) -> pd.DataFrame:
    """
    Load and clean education level CSV data.
    """
    df = pd.read_csv(url)
    df[df.columns[0]] = df[df.columns[0]].str.upper()  # uppercase counties
    return df.copy()

def schools_acs(url: str) -> pd.DataFrame:
    """
    Load number of primary schools CSV data.
    """
    df = pd.read_csv(url)
    df[df.columns[0]] = df[df.columns[0]].str.upper()  # uppercase counties
    return df.copy()

def secondary_acs(url_high_schools: str, url_secondary: str) -> pd.DataFrame:
    """
    Load high school and secondary school CSV data.
    """
    df_high = pd.read_csv(url_high_schools)
    df_secondary = pd.read_csv(url_secondary)
    
    # Uppercase county names in secondary school data
    df_secondary["County"] = df_secondary["County"].str.upper()
    
    return df_high.copy(), df_secondary.copy()