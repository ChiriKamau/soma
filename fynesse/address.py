import matplotlib.pyplot as plt

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
