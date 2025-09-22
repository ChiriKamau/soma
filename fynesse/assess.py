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
