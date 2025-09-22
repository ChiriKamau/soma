import matplotlib.pyplot as plt
import pandas as pd

def attendance_add(df_county, area_col: str, school_col: str, figsize=(15, 8)):
    """
    Plot school attendance by county.
    """
    plt.figure(figsize=figsize)
    plt.bar(df_county[area_col], df_county[school_col], color='skyblue')
    plt.xticks(rotation=90)
    plt.ylabel('People in School')
    plt.title('School Attendance by County')
    plt.tight_layout()
    plt.show()

def population_add(df_county, county_col: str = "ewcounty", total_col: str = "Total", figsize=(15,8)):
    """
    Plot population by county.
    """
    plt.figure(figsize=figsize)
    plt.bar(df_county[county_col], df_county[total_col], color='lightgreen')
    plt.xticks(rotation=90)
    plt.ylabel(f'Population Ages')
    plt.title(f'Population by County')
    plt.tight_layout()
    plt.show() 

def population_vs_attendance(county_pop, county_att, pop_col="Total", att_col=None, figsize=(18,6)):
    """
    Plot population (under 20) vs school attendance side by side for all counties.
    
    Parameters:
        county_pop (DataFrame): population data from population_ass()
        county_att (DataFrame): attendance data from attendance_ass()
        pop_col (str): column name for population numbers in county_pop
        att_col (str): column name for attendance numbers in county_att
        figsize (tuple): figure size
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if att_col is None:
        att_col = county_att.columns[2]  # default: 3rd column from attendance_ass

    x = np.arange(len(county_pop))  # positions
    width = 0.4

    plt.figure(figsize=figsize)
    plt.bar(x - width/2, county_pop[pop_col], width=width, color='lightgreen', label='Population Under 20')
    plt.bar(x + width/2, county_att[att_col], width=width, color='skyblue', label='People in School')

    plt.xticks(x, county_pop[county_pop.columns[0]], rotation=90)
    plt.ylabel("Number of People")
    plt.title("Population Under 20 vs School Attendance by County")
    plt.legend()
    plt.tight_layout()
    plt.show()

def education_add(df_county, county_col=None, figsize=(18, 8)):
    """
    Plot stacked bar chart of education levels by county.
    """
    if county_col is None:
        county_col = df_county.columns[0]
    
    edu_columns = [
        "Pre-Primary",
        "Primary",
        "Secondary",
        "Technical and Vocational Training (TVET)",
        "University"
    ]
    
    x = df_county[county_col]
    
    plt.figure(figsize=figsize)
    bottom = 0
    for col in edu_columns:
        plt.bar(x, df_county[col], bottom=bottom, label=col)
        bottom += df_county[col]
    
    plt.xticks(rotation=90)
    plt.ylabel("Number of People")
    plt.title("Education Levels by County (2019 Census)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def schools_add(df_county, county_col=None, public_col="Public_Primary_Schools", private_col="Private_Primary_Schools", figsize=(18,8)):
    """
    Plot stacked bar chart of public vs private primary schools by county.
    """
    if county_col is None:
        county_col = df_county.columns[0]
    
    x = df_county[county_col]
    
    plt.figure(figsize=figsize)
    plt.bar(x, df_county[public_col], label="Public", color="skyblue")
    plt.bar(x, df_county[private_col], bottom=df_county[public_col], label="Private", color="salmon")
    
    plt.xticks(rotation=90)
    plt.ylabel("Number of Schools")
    plt.title("Number of Primary Schools by County")
    plt.legend()
    plt.tight_layout()
    plt.show()


def secondary_add(df_county, county_col="County", figsize=(18,8)):
    """
    Plot stacked bar chart of high school categories + private secondary schools by county.
    """
    categories = [col for col in df_county.columns if col not in [county_col, "Total"]]
    bottom = pd.Series([0]*df_county.shape[0])
    
    plt.figure(figsize=figsize)
    
    for cat in categories:
        plt.bar(df_county[county_col], df_county[cat], bottom=bottom, label=cat)
        bottom += df_county[cat]
    
    plt.xticks(rotation=90)
    plt.ylabel("Number of Schools")
    plt.title("Schools per County (High School Categories + Private Secondary Schools)")
    plt.legend(title="Category")
    plt.tight_layout()
    plt.show()
