import requests
import getdata
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import geopandas as gpd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# --- SETUP: Download and Import Data Helper ---
# This is required to load the 'votes' GeoDataFrame
url = 'https://raw.githubusercontent.com/linogaliana/python-datascientist/master/content/modelisation/get_data.py'
r = requests.get(url, allow_redirects=True)
open('getdata.py', 'wb').write(r.content)

# Load Data
votes = getdata.create_votes_dataframes()

# ==============================================================================
# Q1: Descriptive Analysis (Shape, Info, Describe)
# ==============================================================================
print("=" * 60)
print("Q1: Descriptive Analysis")
print("=" * 60)

# Size of the dataframe (shape)
print("Shape (Rows, Columns):", votes.shape)

# Column names and their types (info)
print("\nColumn Info and Types:")
votes.info()

# Statistics of each numerical column (describe)
print("\nStatistics of Numerical Columns:")
print(votes.describe())

# ==============================================================================
# Q2 & Q3: Data Transformation and Reduction
# ==============================================================================

# Q2: Different values of 'winner'
print("\nQ2: Unique 'winner' values:")
print(votes['winner'].unique())

# Recode 'winner' to numeric 'winner2' (0, 1, 2, ...)
votes['winner2'] = votes['winner'].astype('category').cat.codes

# Q3: Create reduced DataFrame df3, set index, rename column
columns_to_include = [
    "GEOID", "winner2", "votes_gop", "Unemployment_rate_2021",
    "Median_Household_Income_2021",
    "Percent of adults with less than a high school diploma, 2018-22",
    "Percent of adults with a bachelor's degree or higher, 2018-22"
]

# Create df3, set 'GEOID' as index
df3 = votes[columns_to_include].set_index("GEOID")

# Rename 'winner2' to 'winner'
df3.rename(columns={'winner2': 'winner'}, inplace=True)

print("\nQ3: Reduced DataFrame (df3) Head:")
print(df3.head(3))

# ==============================================================================
# Q4: Frequency Analysis and Plot (winner)
# ==============================================================================
print("\n" + "=" * 60)
print("Q4: Frequency Analysis (winner)")
print("=" * 60)

# Create the frequency table
frequency_table = df3['winner'].value_counts()
print("\nFrequency Table for 'winner':")
print(frequency_table)

# Plot the frequency table as a horizontal bar plot
plt.figure(figsize=(8, 5))
frequency_table.plot(
    kind='barh',
    title='Q4: Frequency of Winners (Recoded)',
    xlabel='Count (Frequency)',
    ylabel='Winner Code',
    color='skyblue'
)
plt.tight_layout()
plt.show()

# ==============================================================================
# Q5: Categorical Transformation (Median_Household_Income_2021)
# ==============================================================================
print("\n" + "=" * 60)
print("Q5: Income Categorization and Analysis")
print("=" * 60)

# Transform the variable into 5 categories using pd.cut
df3['Median_Household_Income_2021_cat'] = pd.cut(
    df3['Median_Household_Income_2021'],
    bins=5
)

# Create the frequency table
frequency_table_income = df3['Median_Household_Income_2021_cat'].value_counts().sort_index()
print("\nFrequency Table for 'Median_Household_Income_2021_cat':")
print(frequency_table_income)

# Create the associated bar plot
plt.figure(figsize=(10, 6))
frequency_table_income.plot(
    kind='bar',
    title='Q5: Frequency of Median Household Income Categories',
    xlabel='Median Household Income Category (USD)',
    ylabel='Count (Frequency)',
    color='coral',
    rot=45
)
plt.tight_layout()
plt.show()

# ==============================================================================
# Q6: Descriptive Statistics of df3
# ==============================================================================
print("\n" + "=" * 60)
print("Q6: Descriptive Statistics (All df3 Columns)")
print("=" * 60)
print(df3.describe(include='all'))

# ==============================================================================
# Q7: Histogram for votes_gop
# ==============================================================================
print("\n" + "=" * 60)
print("Q7: Histogram for votes_gop")
print("=" * 60)

plt.figure(figsize=(8, 5))
df3['votes_gop'].hist(
    bins=20,
    color='darkblue',
    edgecolor='black'
)
plt.title('Q7: Distribution of votes_gop')
plt.xlabel('Percentage of Votes for GOP')
plt.ylabel('Frequency (Count)')
plt.grid(axis='y', alpha=0.5)
plt.show()

# ==============================================================================
# Q8: Correlation Matrix, Heatmap, and Scatter Matrix
# ==============================================================================
print("\n" + "=" * 60)
print("Q8: Correlation Analysis")
print("=" * 60)

# Remove the categorical column (Correlation requires numerical data)
df_corr = df3.drop(columns=['Median_Household_Income_2021_cat'])

# Calculate the correlation matrix
correlation_matrix = df_corr.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap='vlag',
    linewidths=.5,
    cbar_kws={'label': 'Correlation Coefficient'}
)
plt.title('Q8: Correlation Heatmap of Numerical Variables')
plt.show()

# Scatter Matrix
pd.plotting.scatter_matrix(
    df_corr,
    figsize=(15, 15),
    diagonal='hist',
    alpha=0.6,
    marker='o'
)
plt.suptitle('Q8: Scatter Matrix of Variables', y=1.02, fontsize=16)
plt.show()

# ==============================================================================
# Q10 & Q11: Standardization
# ==============================================================================

# Q10: Standardize variables to create df_scaled
df_scale_data = df3.drop(columns=['Median_Household_Income_2021_cat', 'winner'])
scaler = StandardScaler()
scaled_array = scaler.fit_transform(df_scale_data)
df_scaled = pd.DataFrame(
    scaled_array,
    columns=df_scale_data.columns,
    index=df_scale_data.index
)

# Q11: Verification of Standardization
income_col = 'Median_Household_Income_2021'
income_stats = df_scaled[income_col].agg(['mean', 'var'])

print("\n" + "=" * 60)
print("Q11: Verification of Standardized Income Variable")
print("=" * 60)
print(f"Mean (μ): {income_stats['mean']:.15f} (Should be ~0)")
print(f"Variance (σ²): {income_stats['var']:.15f} (Should be 1.0)")

# Histogram comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
n_bins = 50

# Before Standardization
df3[income_col].hist(ax=axes[0], bins=n_bins, color='blue', edgecolor='black', alpha=0.7)
axes[0].set_title('Q11: Before Standardization (Original Scale)')
axes[0].set_xlabel('Median Household Income (USD)')

# After Standardization
df_scaled[income_col].hist(ax=axes[1], bins=n_bins, color='red', edgecolor='black', alpha=0.7)
axes[1].set_title('Q11: After Standardization (Z-Scores)')
axes[1].set_xlabel('Median Household Income (Z-Score)')

plt.suptitle('Distribution of Median Household Income Before and After Standardization', fontsize=16)
plt.show()

# ==============================================================================
# Q12: Scaler on First 1000 Rows
# ==============================================================================
print("\n" + "=" * 60)
print("Q12: Scaler Fitted on First 1000 Rows")
print("=" * 60)

cols_to_fit = df3.drop(columns=['winner', 'Median_Household_Income_2021_cat']).columns
df_train_1000 = df3.loc[:, cols_to_fit].iloc[0:1000, :]

scaler_1000 = StandardScaler()
scaler_1000.fit(df_train_1000)

print(f"Mean of columns (scaler_1000.mean_): {scaler_1000.mean_}")
print(f"Standard deviation (scaler_1000.scale_): {scaler_1000.scale_}")

# ==============================================================================
# Q13: Boxplot Analysis
# ==============================================================================
print("\n" + "=" * 60)
print("Q13: Boxplot Outlier Analysis")
print("=" * 60)

df_plot = df_scaled.reset_index().drop(columns=['GEOID'])
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_plot, orient="h", palette="Set2")
plt.title('Q13: Boxplots of Standardized Variables (Outlier Detection)')
plt.xlabel('Z-Score')
plt.show()
print("\nAnalysis: Outliers (individual dots) are visible in most variables, particularly 'Unemployment_rate_2021' and the educational attainment percentages, indicating counties with extreme values.")

# ==============================================================================
# Q14: Outlier Identification (3-Sigma Rule)
# ==============================================================================
print("\n" + "=" * 60)
print("Q14: Outlier Identification (3-Sigma Rule)")
print("=" * 60)

# Identify rows with ANY value outside +/- 3 standard deviations
outlier_flags = (df_scaled.abs() > 3)
rows_to_remove = outlier_flags.any(axis=1)
total_lines_removed = rows_to_remove.sum()

print(f"Total number of lines (rows) in df_scaled: {len(df_scaled)}")
print(f"Number of lines to remove (at least one outlier > +/- 3*std): {total_lines_removed}")
print(f"Percentage of lines to remove: {(total_lines_removed / len(df_scaled) * 100):.2f}%")

# ==============================================================================
# Q15: Isolation Forest
# ==============================================================================
print("\n" + "=" * 60)
print("Q15: Isolation Forest Outlier Detection")
print("=" * 60)

model = IsolationForest(random_state=42, n_estimators=100, contamination='auto')
df_scaled['IF_OUTLIER'] = model.fit_predict(df_scaled)
if_outlier_count = (df_scaled['IF_OUTLIER'] == -1).sum()

print(f"Total outliers identified by Isolation Forest: {if_outlier_count}")
print(f"Comparison: Isolation Forest detects multivariate outliers, resulting in {if_outlier_count - total_lines_removed} more/fewer outliers than the simple 3-Sigma rule.")

# ==============================================================================
# Q16: Scatter Plot with Outlier Coloring
# ==============================================================================
print("\n" + "=" * 60)
print("Q16: Scatter Plot with Outlier Coloring")
print("=" * 60)

plt.figure(figsize=(9, 7))
sns.scatterplot(
    data=df_scaled,
    x='votes_gop',
    y='Unemployment_rate_2021',
    hue='IF_OUTLIER',
    palette={1: 'green', -1: 'red'},
    style='IF_OUTLIER',
    markers={1: 'o', -1: 'X'},
    s=70,
    alpha=0.6
)
plt.title('Q16: Votes GOP vs. Unemployment Rate (Outliers in Red)')
plt.xlabel('Votes GOP (Z-Score)')
plt.ylabel('Unemployment Rate (Z-Score)')
plt.legend(title='Outlier Status', labels=['Inlier', 'Outlier'], loc='upper right')
plt.show()
print("\nInterpretation: Red markers (outliers) often lie far from the main cluster of green points, representing counties with unusual combinations of GOP votes and unemployment rates.")

# ==============================================================================
# Q17: Pairplot
# ==============================================================================
print("\n" + "=" * 60)
print("Q17: Pairplot")
print("=" * 60)

# Create the pairplot
df_plot_final = df_scaled.drop(columns=['IF_OUTLIER'])
sns.pairplot(df_plot_final)
plt.suptitle('Q17: Pairplot of All Standardized Variables', y=1.02, fontsize=16)
plt.show()
print("\nInterpretation: The pairplot visually confirms strong relationships (e.g., negative correlation between the two educational attainment variables) and highlights distributions and potential multivariate outliers.")
