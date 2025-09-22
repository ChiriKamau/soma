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


def level_school_df_merge(county_edu, county_schools):
    """
    Merge education population and schools data to a single dataframe
    with standardized column names for correlation.
    """
    df = pd.merge(county_edu, county_schools, left_on=county_edu.columns[0], right_on=county_schools.columns[0], how='inner')

    # Rename education columns
    df = df.rename(columns={
        "Pre-Primary": "Pre-Primary_People",
        "Primary": "Primary_People",
        "Secondary": "Secondary_People",
        "Technical and Vocational Training (TVET)": "Technical_and_Vocational_Training_TVET_People",
        "University": "University_People",
        "Public_Primary_Schools": "Primary_Schools",
        "Private_Primary_Schools": "Primary_Schools",  # if you want to sum later
        "Secondary_Schools": "Secondary_Schools",
        "TVET_Schools": "TVET_Schools",
        "University_Schools": "University_Schools"
    })

    return df
