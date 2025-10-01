import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
    

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

def education_ass(df: pd.DataFrame, counties: list = counties_list):
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


def schools_ass(df: pd.DataFrame, counties: list = counties_list) -> pd.DataFrame:
    """
    Filter primary schools data for the main counties and sort.
    """
    df_county = df[df[df.columns[0]].isin([c.upper() for c in counties])].copy()
    
    # Sort by the order in counties_list
    df_county[df.columns[0]] = pd.Categorical(df_county[df.columns[0]], categories=counties, ordered=True)
    df_county = df_county.sort_values(df.columns[0]).reset_index(drop=True)
    
    return df_county

def secondary_ass(df_high, df_secondary, counties: list = counties_list):
    """
    Process high school categories and merge with private secondary school counts.
    Returns a combined dataframe.
    """
    # County KNEC codes
    county_map = {
        "01": "TAITA-TAVETA", "02": "KWALE", "03": "MOMBASA", "04": "KILIFI", "05": "LAMU",
        "06": "TANA RIVER", "07": "NYANDARUA", "08": "NYERI", "09": "KIRINYAGA", "10": "MURANGA",
        "11": "KIAMBU", "12": "MACHAKOS", "13": "KITUI", "14": "EMBU", "15": "MERU",
        "16": "THARAKA-NITHI", "17": "ISIOLO", "18": "MAKUENI", "19": "THARAKA", "20": "NAIROBI",
        "21": "TURKANA", "22": "WEST POKOT", "23": "SAMBURU", "24": "TRANS NZOIA", "25": "UASIN GISHU",
        "26": "ELGEYO-MARAKWET", "27": "NANDI", "28": "BARINGO", "29": "LAIKIPIA", "30": "NAROK",
        "31": "KAJIADO", "32": "KERICHO", "33": "BOMET", "34": "KAKAMEGA", "35": "VIHIGA",
        "36": "BUNGOMA", "37": "BUSIA", "38": "SIAYA", "39": "KISUMU", "40": "HOMA BAY",
        "41": "MIGORI", "42": "KISII", "43": "NYAMIRA", "44": "GARISSA", "45": "WAJIR",
        "46": "MANDERA", "47": "MARSABIT"
    }
    
    # Map high school counties
    df_high["School_Code"] = df_high["School_Code"].astype(str).str.zfill(8)
    df_high["County_Code"] = df_high["School_Code"].str[:2]
    df_high["County"] = df_high["County_Code"].map(county_map)
    
    # Pivot high school categories
    pivot = df_high.pivot_table(
        index="County",
        columns="Category",
        values="School_Name",
        aggfunc="count",
        fill_value=0
    )
    pivot["Total"] = pivot.sum(axis=1)
    
    # Clean secondary school private counts
    df_secondary["Private_Secondary_Schools"] = pd.to_numeric(
        df_secondary["Private_Secondary_Schools"], errors='coerce'
    ).fillna(0)
    
    df_secondary_subset = df_secondary[["County", "Private_Secondary_Schools"]]
    
    # Merge pivot with secondary schools
    pivot_reset = pivot.copy()
    pivot_reset.index.name = None
    pivot_reset['County'] = pivot_reset.index
    pivot_reset = pivot_reset.reset_index(drop=True)
    
    combined = pd.merge(pivot_reset, df_secondary_subset, on="County", how="left")
    combined["Private_Secondary_Schools"] = combined["Private_Secondary_Schools"].fillna(0)
    combined["Total"] = combined["Total"] + combined["Private_Secondary_Schools"]
    
    # Filter only main counties
    combined = combined[combined["County"].isin([c.upper() for c in counties])]
    
    # Sort by counties_list
    combined["County"] = pd.Categorical(combined["County"], categories=counties, ordered=True)
    combined = combined.sort_values("County").reset_index(drop=True)
    
    return combined

def univ_tvet_ass(df_univ: pd.DataFrame, df_tvet: pd.DataFrame, counties: list = counties_list) -> tuple[pd.Series, pd.Series]:
    """
    Count universities and TVET institutions per county.
    Returns two Series with counts.
    """
    # Universities
    df_univ = df_univ[df_univ['County'].isin([c.upper() for c in counties])]
    univ_counts = df_univ['County'].value_counts()
    univ_counts = univ_counts.reindex([c.upper() for c in counties], fill_value=0)
    
    # TVET institutions
    df_tvet = df_tvet[df_tvet['County'].isin([c.upper() for c in counties])]
    tvet_counts = df_tvet['County'].value_counts()
    tvet_counts = tvet_counts.reindex([c.upper() for c in counties], fill_value=0)
    
    return univ_counts, tvet_counts


def combine_correlation_data(county_edu, county_schools, county_secondary, univ_counts, tvet_counts):
    """Combine all datasets for correlation analysis with proper alignment."""
    # Start with education data as base
    df = county_edu.copy()
    county_col = df.columns[0]
    
    # Get county names from base dataframe
    base_counties = df[county_col].values
    
    # Add primary schools (align by county name)
    primary_data = county_schools.copy()
    primary_col = primary_data.columns[0]
    
    # Merge primary schools data
    df = df.merge(primary_data[[primary_col, 'Public_Primary_Schools', 'Private_Primary_Schools']], 
                  left_on=county_col, right_on=primary_col, how='left')
    df['Primary_Schools'] = df['Public_Primary_Schools'].fillna(0) + df['Private_Primary_Schools'].fillna(0)
    df.drop([primary_col, 'Public_Primary_Schools', 'Private_Primary_Schools'], axis=1, inplace=True)
    
    # Add secondary schools (align by county name)
    secondary_data = county_secondary[['County', 'Total']].copy()
    df = df.merge(secondary_data, left_on=county_col, right_on='County', how='left')
    df['Secondary_Schools'] = df['Total'].fillna(0)
    df.drop(['County', 'Total'], axis=1, inplace=True)
    
    # Add TVET and University data (reindex to match base counties)
    tvet_aligned = tvet_counts.reindex(base_counties, fill_value=0)
    univ_aligned = univ_counts.reindex(base_counties, fill_value=0)
    
    df['TVET_Schools'] = tvet_aligned.values
    df['University_Schools'] = univ_aligned.values
    
    # Rename education columns
    df.rename(columns={'Primary': 'Primary_People', 'Secondary': 'Secondary_People', 
                       'Technical and Vocational Training (TVET)': 'Technical_and_Vocational_Training_TVET_People',
                       'University': 'University_People'}, inplace=True)
    
    return df

# Add this to your "Assess" file:

def prepare_regression_data(level_school_df, county_secondary):
    """Prepare data for regression analysis."""
    secondary_features = county_secondary[["County", "National", "Extra County", "county sch", "Sub County", "Private_Secondary_Schools"]].copy()
    regression_df = level_school_df.merge(secondary_features, left_on=level_school_df.columns[0], right_on="County", how="left")
    X = regression_df[["National", "Extra County", "county sch", "Sub County", "Private_Secondary_Schools"]].fillna(0)
    y = regression_df["University_People"]
    return X, y

def run_tvet_regression(level_school_df, county_secondary):
    """Run regression analysis for TVET institutions."""

    
    secondary_features = county_secondary[["County", "National", "Extra County", "county sch", "Sub County", "Private_Secondary_Schools"]].copy()
    regression_df = level_school_df.merge(secondary_features, left_on=level_school_df.columns[0], right_on="County", how="left")
    X = regression_df[["National", "Extra County", "county sch", "Sub County", "Private_Secondary_Schools"]].fillna(0)
    y = regression_df["TVET_Schools"]
    
    X_scaled = StandardScaler().fit_transform(X)
    model = LinearRegression().fit(X_scaled, y)
    return model, X, X_scaled, y

def regression2_ass(level_school_df):
    """Prepare data for education level regression analysis."""
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
    X = pd.DataFrame({
        "Primary_Schools": level_school_df["Primary_Schools"],
        "Secondary_Schools": level_school_df["Secondary_Schools"],
        "TVET_Schools": level_school_df["TVET_Schools"],
        "University_Schools": level_school_df["University_Schools"]
    })
    
    y = (level_school_df["Primary_People"] * 1 +
         level_school_df["Secondary_People"] * 2 +
         level_school_df["Technical_and_Vocational_Training_TVET_People"] * 3 +
         level_school_df["University_People"] * 4)
    
    X_scaled = StandardScaler().fit_transform(X)
    return X, X_scaled, y


def probability_ass(level_school_df, county_secondary):
    """Prepare data for university probability prediction model."""
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
    # Merge with secondary school data to get school categories
    secondary_features = county_secondary[["County", "National", "Extra County", "county sch", "Sub County", "Private_Secondary_Schools"]].copy()
    prob_df = level_school_df.merge(secondary_features, left_on=level_school_df.columns[0], right_on="County", how="left")
    
    # Calculate total population for ratios
    total_population = (prob_df["Primary_People"] + prob_df["Secondary_People"] + 
                       prob_df["Technical_and_Vocational_Training_TVET_People"] + prob_df["University_People"])
    
    # Create target: University probability (university people / total population)
    y = prob_df["University_People"] / total_population
    
    # Create features
    features_df = pd.DataFrame({
        # School infrastructure per capita (per 1000 people)
        "Primary_Schools_per_capita": (prob_df["Primary_Schools"] / total_population) * 1000,
        "Secondary_Schools_per_capita": (prob_df["Secondary_Schools"] / total_population) * 1000,
        "University_Schools_per_capita": (prob_df["University_Schools"] / total_population) * 1000,
        
        # Secondary school quality indicators (weighted by prestige)
        "National_Schools_per_capita": (prob_df["National"] / total_population) * 1000,
        "Extra_County_per_capita": (prob_df["Extra County"] / total_population) * 1000,
        "County_Schools_per_capita": (prob_df["county sch"] / total_population) * 1000,
        "Sub_County_per_capita": (prob_df["Sub County"] / total_population) * 1000,
        "Private_Secondary_per_capita": (prob_df["Private_Secondary_Schools"] / total_population) * 1000,
        
        # Education pathway indicators
        "Secondary_Education_Rate": prob_df["Secondary_People"] / total_population,
        "Higher_Ed_Infrastructure": (prob_df["TVET_Schools"] + prob_df["University_Schools"]) / total_population * 1000
    })
    
    # Fill NaN values with 0
    features_df.fillna(0, inplace=True)
    
    # Scale features
    X_scaled = StandardScaler().fit_transform(features_df)
    
    return features_df, X_scaled, y, prob_df