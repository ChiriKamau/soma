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


def univ_tvet_acs(url_univ: str, url_tvet: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both universities and TVET institutions CSVs.
    Returns two DataFrames: (universities, tvets)
    """
    df_univ = pd.read_csv(url_univ)
    df_univ['County'] = df_univ['County'].str.upper().str.strip()
    
    df_tvet = pd.read_csv(url_tvet)
    df_tvet['County'] = df_tvet['County'].str.upper().str.strip()
    
    return df_univ, df_tvet


def correlation_acs(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """
    Prepare column lists and labels for correlation.
    """
    edu_cols = ["Pre-Primary_People", "Primary_People", "Secondary_People",
                "Technical_and_Vocational_Training_TVET_People", "University_People"]
    school_cols = ["Primary_Schools", "Primary_Schools", "Secondary_Schools",
                   "TVET_Schools", "University_Schools"]
    edu_labels = ["Pre-Primary", "Primary", "Secondary", "TVET", "University"]
    return edu_cols, school_cols, edu_labels