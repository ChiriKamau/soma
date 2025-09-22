import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd   


def attendance_add(df, area_col, school_col, figsize=(15,8)):
    plt.figure(figsize=figsize)
    plt.bar(df[area_col], df[school_col], color='skyblue')
    plt.xticks(rotation=90); plt.ylabel('People in School')
    plt.title('School Attendance by County'); plt.tight_layout(); plt.show()

def population_add(df, county_col="ewcounty", total_col="Total", figsize=(15,8)):
    plt.figure(figsize=figsize)
    plt.bar(df[county_col], df[total_col], color='lightgreen')
    plt.xticks(rotation=90); plt.ylabel('Population Ages')
    plt.title('Population by County'); plt.tight_layout(); plt.show()

def population_vs_attendance(pop_df, att_df, pop_col="Total", att_col=None, figsize=(18,6)):
    if att_col is None: att_col = att_df.columns[2]
    x = np.arange(len(pop_df)); width = 0.4
    plt.figure(figsize=figsize)
    plt.bar(x-width/2, pop_df[pop_col], width, color='lightgreen', label='Population Under 20')
    plt.bar(x+width/2, att_df[att_col], width, color='skyblue', label='People in School')
    plt.xticks(x, pop_df[pop_df.columns[0]], rotation=90)
    plt.ylabel("Number of People"); plt.title("Population vs School Attendance")
    plt.legend(); plt.tight_layout(); plt.show()

def education_add(df, county_col=None, figsize=(18,8)):
    if county_col is None: county_col = df.columns[0]
    cols = ["Pre-Primary","Primary","Secondary","Technical and Vocational Training (TVET)","University"]
    x = df[county_col]; bottom = np.zeros(len(df))
    plt.figure(figsize=figsize)
    for c in cols: plt.bar(x, df[c], bottom=bottom, label=c); bottom += df[c]
    plt.xticks(rotation=90); plt.ylabel("Number of People"); plt.title("Education Levels by County")
    plt.legend(); plt.tight_layout(); plt.show()

def schools_add(df, county_col=None, public_col="Public_Primary_Schools", private_col="Private_Primary_Schools", figsize=(18,8)):
    if county_col is None: county_col = df.columns[0]
    x = df[county_col]
    plt.figure(figsize=figsize)
    plt.bar(x, df[public_col], label="Public", color="skyblue")
    plt.bar(x, df[private_col], bottom=df[public_col], label="Private", color="salmon")
    plt.xticks(rotation=90); plt.ylabel("Number of Schools"); plt.title("Number of Primary Schools")
    plt.legend(); plt.tight_layout(); plt.show()

def secondary_add(df, county_col="County", figsize=(18,8)):
    categories = [c for c in df.columns if c not in [county_col, "Total"]]
    bottom = np.zeros(len(df))
    plt.figure(figsize=figsize)
    for cat in categories: plt.bar(df[county_col], df[cat], bottom=bottom, label=cat); bottom += df[cat]
    plt.xticks(rotation=90); plt.ylabel("Number of Schools"); plt.title("Schools per County")
    plt.legend(title="Category"); plt.tight_layout(); plt.show()

def univ_tvet_add(univ_counts, tvet_counts, figsize=(15,6)):
    x = np.arange(len(univ_counts)); width=0.4
    plt.figure(figsize=figsize)
    plt.bar(x-width/2, univ_counts.values, width, label='Universities', color='skyblue')
    plt.bar(x+width/2, tvet_counts.values, width, label='TVET Institutions', color='lightgreen')
    plt.xticks(x, univ_counts.index, rotation=90)
    plt.ylabel("Number of Institutions"); plt.title("Universities and TVET Institutions per County")
    plt.legend(); plt.tight_layout(); plt.show()


def correlation_add(corr_matrix: pd.DataFrame, figsize=(8,6)):
    """
    Plots a heatmap of the correlation matrix.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xlabel("Schools per Level")
    plt.ylabel("Population per Education Level")
    plt.title("Correlation between Education Level Population and Number of Schools per Level (Full Matrix)")
    plt.tight_layout()
    plt.show()