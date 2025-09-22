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


# Add this to your "Address" file (visualization functions):

def correlation_add(level_school_df, figsize=(10,6)):
    """Plot correlation heatmap between education populations and school counts."""

    
    edu_cols = ["Primary_People", "Secondary_People", "Technical_and_Vocational_Training_TVET_People", "University_People"]
    school_cols = ["Primary_Schools", "Secondary_Schools", "TVET_Schools", "University_Schools"]
    
    corr_matrix = pd.DataFrame([[level_school_df[edu].corr(level_school_df[school]) for school in school_cols] for edu in edu_cols],
                              index=["Primary", "Secondary", "TVET", "University"], columns=school_cols)
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.xlabel("Schools per Level"); plt.ylabel("Population per Education Level")
    plt.title("Correlation between Education Level Population and Number of Schools per Level")
    plt.tight_layout(); plt.show()

def regression_add(model, X, X_scaled, y, figsize=(8,5)):
    """Plot regression model results - coefficients and predicted vs actual."""
    coeff_df = pd.DataFrame({
        "School_Category": X.columns,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)
    
    plt.figure(figsize=figsize)
    plt.bar(coeff_df["School_Category"], coeff_df["Coefficient"], color="skyblue")
    plt.title("Impact of Secondary School Category on University Population")
    plt.ylabel("Coefficient (Effect on University Population)")
    plt.xlabel("Secondary School Category")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Predicted vs Actual plot
    y_pred = model.predict(X_scaled)
    
    plt.figure(figsize=figsize)
    plt.scatter(y, y_pred, color="salmon")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel("Actual University Population")
    plt.ylabel("Predicted University Population")
    plt.title("Predicted vs Actual University Population by County")
    plt.tight_layout()
    plt.show()

def tvet_regression_add(model, X, X_scaled, y, figsize=(8,5)):
    """Plot TVET regression results."""
    import matplotlib.pyplot as plt
    
    coeff_df = pd.DataFrame({"School_Category": X.columns, "Coefficient": model.coef_}).sort_values(by="Coefficient", ascending=False)
    
    print("Coefficients for TVET model:")
    print(coeff_df)
    print(f"\nR^2 score: {model.score(X_scaled, y)}")
    
    plt.figure(figsize=figsize)
    plt.bar(coeff_df["School_Category"], coeff_df["Coefficient"], color="lightgreen")
    plt.title("Impact of Secondary School Category on TVET Institutions")
    plt.ylabel("Coefficient (Effect on TVET Institutions)")
    plt.xlabel("Secondary School Category")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def regression2_add(model, X, X_scaled, y, figsize=(8,5)):
    """Plot education level regression results."""
    import matplotlib.pyplot as plt
    
    coeff_df = pd.DataFrame({
        "School_Category": X.columns,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)
    
    print("Coefficients (effect of school category on education level):")
    print(coeff_df)
    print(f"\nR^2 score: {model.score(X_scaled, y)}")
    
    plt.figure(figsize=figsize)
    plt.bar(coeff_df["School_Category"], coeff_df["Coefficient"], color="mediumpurple")
    plt.ylabel("Coefficient (Effect on Education Level)")
    plt.xlabel("School Category")
    plt.title("Impact of Number of Schools per Category on County Education Level")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def cluster_add(result_df, figsize=(12,8)):
    """Visualize clustering results."""
    import matplotlib.pyplot as plt
    county_col = result_df.columns[0]
    
    # Print cluster summary
    print("CLUSTER SUMMARY")
    print("=" * 30)
    for cluster in sorted(result_df['Cluster'].unique()):
        counties = result_df[result_df['Cluster'] == cluster][county_col].tolist()
        avg_unis = result_df[result_df['Cluster'] == cluster]['University_Schools'].mean()
        print(f"Cluster {cluster} ({len(counties)} counties): {', '.join(counties[:5])}")
        print(f"  Avg Universities: {avg_unis:.1f}\n")
    
    # Visualizations
    colors = ['red', 'blue', 'green', 'orange']
    plt.figure(figsize=figsize)
    
    plt.subplot(2,2,1)
    for cluster in sorted(result_df['Cluster'].unique()):
        data = result_df[result_df['Cluster'] == cluster]
        plt.scatter(data['University_Schools'], data['University_People'], 
                   c=colors[cluster], label=f'Cluster {cluster}', alpha=0.7)
    plt.xlabel('University Schools'); plt.ylabel('University People')
    plt.title('Universities vs University Population'); plt.legend()
    
    plt.subplot(2,2,2)
    for cluster in sorted(result_df['Cluster'].unique()):
        data = result_df[result_df['Cluster'] == cluster]
        plt.scatter(data['Primary_Schools'], data['Secondary_Schools'], 
                   c=colors[cluster], label=f'Cluster {cluster}', alpha=0.7)
    plt.xlabel('Primary Schools'); plt.ylabel('Secondary Schools')
    plt.title('Primary vs Secondary Schools'); plt.legend()
    
    plt.subplot(2,2,3)
    cluster_means = result_df.groupby('Cluster')[["Primary_People", "Secondary_People", "University_People"]].mean()
    cluster_means.plot(kind='bar', ax=plt.gca())
    plt.title('Education Levels by Cluster'); plt.ylabel('Population'); plt.xticks(rotation=0)
    
    plt.subplot(2,2,4)
    cluster_counts = result_df['Cluster'].value_counts().sort_index()
    plt.pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index], autopct='%1.1f%%')
    plt.title('County Distribution')
    
    plt.tight_layout(); plt.show()