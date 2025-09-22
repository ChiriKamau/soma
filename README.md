# Kenya Education Analysis

A data analysis project examining the relationship between education levels, school infrastructure, and population demographics across Kenya's 47 counties using the Fynesse framework.

## Project Overview

This project analyzes educational data from Kenya to understand:
- Distribution of education levels across counties
- Correlation between school availability and population education levels
- Impact of different school types on educational outcomes
- Predictive relationships between secondary school categories and higher education

## Data Sources

The analysis uses several datasets:
- **Population by Education Level**: Distribution of population aged 3+ by highest education level (2019 Census)
- **Primary Schools**: Number of public and private primary schools per county
- **Secondary Schools**: Breakdown by school categories (National, Extra County, County, Sub County, Private)
- **TVET Institutions**: Technical and Vocational Education Training centers
- **Universities**: Higher education institutions by county
- **School Attendance**: Population currently attending school by county

## Framework Structure

The project follows the Fynesse framework with three main components:

```
├── access.py      # Data loading and cleaning functions
├── assess.py      # Data processing and preparation functions  
├── address.py     # Visualization and analysis functions
```

### Access Functions
- `education_acs()`: Load education level data
- `schools_acs()`: Load primary school data
- `secondary_acs()`: Load secondary school data
- `univ_tvet_acs()`: Load university and TVET data
- `population_acs()`: Load population data
- `attendance_acs()`: Load school attendance data

### Assess Functions
- `education_ass()`: Process education level data for 47 counties
- `schools_ass()`: Process primary school counts
- `secondary_ass()`: Combine secondary school categories and private schools
- `univ_tvet_ass()`: Count universities and TVET institutions per county
- `combine_correlation_data()`: Merge all datasets for correlation analysis
- `regression2_ass()`: Prepare data for regression modeling

### Address Functions
- `education_add()`: Visualize education levels by county
- `schools_add()`: Plot primary school distributions
- `secondary_add()`: Display secondary school categories
- `univ_tvet_add()`: Show university and TVET distributions
- `correlation_add()`: Generate correlation heatmaps
- `regression2_add()`: Plot regression results

## Key Analysis Components

### 1. Correlation Analysis
Examines relationships between education level populations and school availability:
```python
level_school_df = combine_correlation_data(county_edu, county_schools, county_secondary, univ_counts, tvet_counts)
correlation_add(level_school_df)
```

### 2. Regression Modeling
Predicts education outcomes based on school infrastructure:
```python
X, X_scaled, y = regression2_ass(level_school_df)
edu_model = LinearRegression().fit(X_scaled, y)
regression2_add(edu_model, X, X_scaled, y)
```

## Quick Start

### Prerequisites
- Python 3.7+
- Required packages: pandas, numpy, matplotlib, seaborn, scikit-learn

### Installation
```bash
# Clone the repository
git clone [your-repo-url]
cd kenya-education-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Usage Example
```python
import pandas as pd
from access import education_acs, schools_acs
from assess import education_ass, schools_ass
from address import education_add, schools_add

# Load and process data
df_edu = education_acs("education_data_url")
county_edu = education_ass(df_edu)

# Visualize results
education_add(county_edu)
```

## Data Processing Pipeline

1. **Data Access**: Load raw datasets from CSV URLs
2. **Data Assessment**: Clean, filter, and standardize data for 47 counties
3. **Data Integration**: Combine datasets with proper county alignment
4. **Analysis**: Generate correlations, regression models, and visualizations

## Key Findings

The analysis reveals patterns in:
- Education level distribution across Kenyan counties
- Correlation between school infrastructure and education outcomes
- Predictive factors for higher education enrollment
- Regional disparities in educational resources

## Counties Analyzed

All 47 Kenyan counties including: Baringo, Bomet, Bungoma, Busia, Elgeyo-Marakwet, Embu, Garissa, Homa Bay, Isiolo, Kajiado, Kakamega, Kericho, Kiambu, Kilifi, Kirinyaga, Kisii, Kisumu, Kitui, Kwale, Laikipia, Lamu, Machakos, Makueni, Mandera, Marsabit, Meru, Migori, Mombasa, Muranga, Nairobi, Nakuru, Nandi, Narok, Nyamira, Nyandarua, Nyeri, Samburu, Siaya, Taita-Taveta, Tana River, Tharaka-Nithi, Trans Nzoia, Turkana, Uasin Gishu, Vihiga, Wajir, West Pokot.

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request
