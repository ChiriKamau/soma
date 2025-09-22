import pandas as pd

# Global counties list
counties_list = [
    "BARINGO","BOMET","BUNGOMA","BUSIA","ELGEYO-MARAKWET","EMBU","GARISSA",
    "HOMA BAY","ISIOLO","KAJIADO","KAKAMEGA","KERICHO","KIAMBU","KILIFI",
    "KIRINYAGA","KISII","KISUMU","KITUI","KWALE","LAIKIPIA","LAMU","MACHAKOS",
    "MAKUENI","MANDERA","MARSABIT","MERU","MIGORI","MOMBASA","MURANGA",
    "NAIROBI","NAKURU","NANDI","NAROK","NYAMIRA","NYANDARUA","NYERI",
    "SAMBURU","SIAYA","TAITA-TAVETA","TANA RIVER","THARAKA-NITHI","TRANS NZOIA",
    "TURKANA","UASIN GISHU","VIHIGA","WAJIR","WEST POKOT"
]

def attendance_ass(df: pd.DataFrame, counties: list = counties_list) -> pd.DataFrame:
    """
    Filter for counties, clean numeric values, and sort alphabetically.
    """
    area_col, school_col = df.columns[0], df.columns[2]
    df_county = df[df[area_col].str.upper().isin([c.upper() for c in counties])].drop_duplicates(subset=[area_col]).copy()
    df_county[school_col] = pd.to_numeric(
        df_county[school_col].astype(str).str.replace('[,""]','', regex=True),
        errors='coerce'
    )
    return df_county.sort_values(by=area_col).reset_index(drop=True), area_col, school_col

def population_ass(df: pd.DataFrame, min_age: int = 4, max_age: int = 20) -> pd.DataFrame:
    """
    Filter population DataFrame by age range, sum by county, and match with the 47 counties.
    
    Returns:
        county_pop_df (pd.DataFrame): Processed population by county, sorted alphabetically.
    """
    # Filter by age range
    df_age = df[(df["Age"] >= min_age) & (df["Age"] <= max_age)]

    # Sum population by county
    county_pop = df_age.groupby("ewcounty")["Total"].sum().reset_index()
    county_pop["ewcounty"] = county_pop["ewcounty"].str.upper()

    # Filter for the 47 counties
    county_pop = county_pop[county_pop["ewcounty"].isin(counties_list)]
    
    # Sort alphabetically
    county_pop = county_pop.sort_values(by="ewcounty").reset_index(drop=True)
    return county_pop

def education_ass(df: pd.DataFrame, counties: list) -> pd.DataFrame:
    """
    Filter education data for the main counties and clean numeric columns.
    """
    edu_columns = [
        "Pre-Primary",
        "Primary",
        "Secondary",
        "Technical and Vocational Training (TVET)",
        "University"
    ]
    # Filter main counties
    df_county = df[df[df.columns[0]].isin([c.upper() for c in counties])].copy()
    
    # Clean numeric columns
    for col in edu_columns:
        df_county[col] = pd.to_numeric(
            df_county[col].astype(str).str.replace('[,"]', '', regex=True),
            errors='coerce'
        )
    
    # Sort alphabetically
    df_county = df_county.sort_values(by=df.columns[0]).reset_index(drop=True)
    return df_county
