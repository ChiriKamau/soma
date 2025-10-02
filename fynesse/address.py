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


from sklearn.utils import resample

def regression_add(model, X, X_scaled, y, figsize=(8,5), n_bootstrap=1000): 
    """Plot regression model results - coefficients (with bootstrap error bars) and predicted vs actual (y scaled by 10,000)."""
    
    # Bootstrap to estimate standard errors
    coefs = []
    for _ in range(n_bootstrap):
        X_res, y_res = resample(X_scaled, y)
        model_res = model.__class__().fit(X_res, y_res)
        coefs.append(model_res.coef_)
    
    coefs = np.array(coefs)
    coef_mean = np.mean(coefs, axis=0) / 10000
    coef_std = np.std(coefs, axis=0) / 10000  # error bars
    
    coeff_df = pd.DataFrame({
        "School_Category": X.columns,
        "Coefficient": coef_mean,
        "Error": coef_std
    }).sort_values(by="Coefficient", ascending=False)
    
    # Coefficients bar plot with error bars
    plt.figure(figsize=figsize)
    plt.bar(coeff_df["School_Category"], coeff_df["Coefficient"], 
            yerr=coeff_df["Error"], color="skyblue", capsize=5)
    plt.title("How Different Types of Secondary Schools Affect University Enrollment")
    plt.ylabel("Coefficient (Effect on University Enrollment / 10,000)")
    plt.xlabel("Secondary School Category")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Predicted vs Actual plot
    y_pred = model.predict(X_scaled) / 10000
    y_scaled = y / 10000
    
    plt.figure(figsize=figsize)
    plt.scatter(y_scaled, y_pred, color="salmon", alpha=0.7, edgecolor="k")
    plt.plot([y_scaled.min(), y_scaled.max()], [y_scaled.min(), y_scaled.max()], 'k--', lw=2)
    plt.xlabel("Actual University Population (/10,000)")
    plt.ylabel("Predicted University Population (/10,000)")
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

def probability_add(model, features_df, X_scaled, y, prob_df, figsize=(15,10)):
    """Visualize university probability prediction results."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    county_col = prob_df.columns[0]
    
    # Model performance
    y_pred = model.predict(X_scaled)
    r2 = model.score(X_scaled, y)
    
    print("UNIVERSITY PROBABILITY PREDICTION MODEL")
    print("=" * 50)
    print(f"R² Score: {r2:.3f}")
    print(f"Mean University Probability: {y.mean():.3f} ({y.mean()*100:.1f}%)")
    print(f"Prediction Range: {y_pred.min():.3f} - {y_pred.max():.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        "Feature": features_df.columns,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", key=abs, ascending=False)
    
    print("\nTOP PREDICTIVE FACTORS:")
    print("-" * 30)
    for i, row in feature_importance.head(5).iterrows():
        direction = "increases" if row["Coefficient"] > 0 else "decreases"
        print(f"{row['Feature']}: {direction} probability by {abs(row['Coefficient']):.4f}")
    
    # Visualizations
    plt.figure(figsize=figsize)
    
    # 1. Predicted vs Actual
    plt.subplot(2, 3, 1)
    plt.scatter(y, y_pred, alpha=0.7, color='blue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual University Probability')
    plt.ylabel('Predicted University Probability')
    plt.title(f'Predicted vs Actual (R² = {r2:.3f})')
    plt.grid(True, alpha=0.3)
    
    # 2. Feature importance
    plt.subplot(2, 3, 2)
    top_features = feature_importance.head(8)
    colors = ['green' if x > 0 else 'red' for x in top_features["Coefficient"]]
    plt.barh(range(len(top_features)), top_features["Coefficient"], color=colors)
    plt.yticks(range(len(top_features)), top_features["Feature"])
    plt.xlabel('Coefficient (Impact on University Probability)')
    plt.title('Top Predictive Factors')
    plt.grid(True, alpha=0.3)
    
    # 3. Secondary school quality impact
    plt.subplot(2, 3, 3)
    secondary_features = ["National_Schools_per_capita", "Extra_County_per_capita", "County_Schools_per_capita", 
                         "Sub_County_per_capita", "Private_Secondary_per_capita"]
    secondary_coeffs = [feature_importance[feature_importance["Feature"] == f]["Coefficient"].iloc[0] 
                       if f in feature_importance["Feature"].values else 0 for f in secondary_features]
    colors = ['darkblue', 'blue', 'lightblue', 'orange', 'red']
    plt.bar(range(len(secondary_features)), secondary_coeffs, color=colors)
    plt.xticks(range(len(secondary_features)), ['National', 'Extra County', 'County', 'Sub County', 'Private'], rotation=45)
    plt.ylabel('Impact on University Probability')
    plt.title('Secondary School Type Impact')
    plt.grid(True, alpha=0.3)
    
    # 4. County predictions ranking
    plt.subplot(2, 3, 4)
    county_results = pd.DataFrame({
        'County': prob_df[county_col],
        'Actual': y,
        'Predicted': y_pred
    }).sort_values('Predicted', ascending=False)
    
    top_counties = county_results.head(10)
    x_pos = range(len(top_counties))
    plt.bar(x_pos, top_counties['Predicted'], alpha=0.7, color='skyblue', label='Predicted')
    plt.bar(x_pos, top_counties['Actual'], alpha=0.7, color='orange', label='Actual')
    plt.xticks(x_pos, top_counties['County'], rotation=90)
    plt.ylabel('University Probability')
    plt.title('Top 10 Counties - University Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Infrastructure vs Probability
    plt.subplot(2, 3, 5)
    total_schools = prob_df["Primary_Schools"] + prob_df["Secondary_Schools"] + prob_df["University_Schools"]
    plt.scatter(total_schools, y, alpha=0.7, color='green')
    plt.xlabel('Total Schools in County')
    plt.ylabel('University Probability')
    plt.title('School Infrastructure vs University Probability')
    plt.grid(True, alpha=0.3)
    
    # 6. Residuals plot
    plt.subplot(2, 3, 6)
    residuals = y - y_pred
    plt.scatter(y_pred, residuals, alpha=0.7, color='purple')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted University Probability')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Prediction Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # County-specific insights
    print(f"\nHIGHEST PREDICTED UNIVERSITY PROBABILITY:")
    print("-" * 40)
    for i, row in county_results.head(5).iterrows():
        print(f"{row['County']}: {row['Predicted']:.3f} ({row['Predicted']*100:.1f}%)")
    
    print(f"\nLOWEST PREDICTED UNIVERSITY PROBABILITY:")
    print("-" * 40)
    for i, row in county_results.tail(5).iterrows():
        print(f"{row['County']}: {row['Predicted']:.3f} ({row['Predicted']*100:.1f}%)")

# Add to address.py

def regression2_pop_normalized_add(model, X, X_scaled, y, figsize=(8,5)):
    """Plot normalized education level regression results (per capita features, average education level target)."""
    import matplotlib.pyplot as plt
    
    # Clean column names for display (remove '_per_1000')
    display_cols = [col.replace('_per_1000', '') for col in X.columns]
    
    coeff_df = pd.DataFrame({
        "School_Category": display_cols,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)
    
    print("Normalized Coefficients (effect of schools per 1000 population on average education level):")
    print(coeff_df)
    print(f"\nR^2 score: {model.score(X_scaled, y)}")
    
    plt.figure(figsize=figsize)
    plt.bar(coeff_df["School_Category"], coeff_df["Coefficient"], color="mediumpurple")
    plt.ylabel("Coefficient (Effect on Average Education Level)")
    plt.xlabel("School Category (per 1000 population)")
    plt.title("Impact of Schools per Capita on Average County Education Level")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()