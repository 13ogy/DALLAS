import requests
import getdata
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly
import plotly.graph_objects as go
import geopandas as gpd
import numpy as np
from sklearn.preprocessing import StandardScaler # Needed for Q10

# Download the getdata.py script
url = 'https://raw.githubusercontent.com/linogaliana/python-datascientist/master/content/modelisation/get_data.py'
r = requests.get(url, allow_redirects=True)
open('getdata.py', 'wb').write(r.content)

# Load Data (REQUIRED for subsequent steps)
votes = getdata.create_votes_dataframes()

# Q1-Q3: Create the Reduced DataFrame (df3) (REQUIRED for Q10)
columns_to_include = [
    "GEOID",
    "winner",
    "votes_gop",
    "Unemployment_rate_2021",
    "Median_Household_Income_2021",
    "Percent of adults with less than a high school diploma, 2018-22",
    "Percent of adults with a bachelor's degree or higher, 2018-22"
]
df3 = votes[columns_to_include].set_index("GEOID")
df3['winner2'] = df3['winner'].astype('category').cat.codes
df3 = df3.drop(columns=['winner'])
df3.rename(columns={'winner2': 'winner'}, inplace=True)

# Q5: Add the categorical income column (REQUIRED for Q10 to drop it later)
df3['Median_Household_Income_2021_cat'] = pd.cut(
    df3['Median_Household_Income_2021'],
    bins=5
)

# -------------------- Q10: Standardize Variables --------------------

# 1. Select the numerical variables (excluding the categorical column)
columns_to_scale = df3.drop(columns=['Median_Household_Income_2021_cat']).columns
df_scale_data = df3.drop(columns=['Median_Household_Income_2021_cat'])

# 2. Initialize the StandardScaler
scaler = StandardScaler()

# 3. Fit and Transform the data
# This returns a NumPy array
scaled_array = scaler.fit_transform(df_scale_data)

# 4. Convert the NumPy array back to a pandas DataFrame
df_scaled = pd.DataFrame(
    scaled_array,
    columns=columns_to_scale,
    index=df_scale_data.index  # Keep the GEOID index
)

# Display the first few rows of the standardized DataFrame
print("------- Q10: Head of Standardized DataFrame (df_scaled) --------")
print(df_scaled.head())
print("\n------- Q10: Descriptive Statistics of Standardized Data --------")
print(df_scaled.describe())

# Plot Histograms Before and After Standardization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
income_col = 'Median_Household_Income_2021'
n_bins = 50

# 1. Histogram BEFORE Standardization (df3)
df3[income_col].hist(
    ax=axes[0],
    bins=n_bins,
    color='blue',
    edgecolor='black',
    alpha=0.7
)
axes[0].set_title('Before Standardization (Original Scale)')
axes[0].set_xlabel('Median Household Income (USD)')
axes[0].set_ylabel('Frequency')

# 2. Histogram AFTER Standardization (df_scaled)
df_scaled[income_col].hist(
    ax=axes[1],
    bins=n_bins,
    color='red',
    edgecolor='black',
    alpha=0.7
)
axes[1].set_title('After Standardization (Z-Scores)')
axes[1].set_xlabel('Median Household Income (Z-Score)')

plt.suptitle('Distribution of Median Household Income Before and After Standardization', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# -------------------- End Q10 --------------------


# The rest of the code is commented out as it is not needed for Q10.
# # Q1: Initial Description
# print("------- SHAPE --------")
# print(votes.shape)
# print("------- INFO --------")
# print(votes.info())
# print("------- DESCRIBE --------")
# print(votes.describe())
# print(df3.head(3))

# # Q4: Frequency Table and Horizontal Bar Plot for 'winner'
# frequency_table = df3['winner'].value_counts()
# print("------- Q4: Frequency Table for 'winner' --------")
# print(frequency_table)
# plt.figure(figsize=(8, 5))
# frequency_table.plot(
#     kind='barh',
#     title='Q4: Frequency of Winners (Recoded)',
#     xlabel='Count (Frequency)',
#     ylabel='Winner Code',
#     color='skyblue'
# )
# plt.tight_layout()
# plt.show()

# # Q5: Frequency Table and Bar Plot for Income Category
# frequency_table_income = df3['Median_Household_Income_2021_cat'].value_counts().sort_index()
# print("------- Q5: Frequency Table for Income Category --------")
# print(frequency_table_income)
# plt.figure(figsize=(10, 6))
# frequency_table_income.plot(
#     kind='bar',
#     title='Q5: Frequency of Median Household Income Categories',
#     xlabel='Median Household Income Category (USD)',
#     ylabel='Count (Frequency)',
#     color='coral',
#     rot=45
# )
# plt.tight_layout()
# plt.show()

# # Q6: Descriptive Statistics of all df3 variables
# print("------- Q6: Descriptive Statistics (ALL df3 Columns) --------")
# print(df3.describe(include='all'))

# # Q7: Histogram for 'votes_gop'
# plt.figure(figsize=(8, 5))
# df3['votes_gop'].hist(
#     bins=20,
#     color='darkblue',
#     edgecolor='black'
# )
# plt.title('Q7: Distribution of votes_gop')
# plt.xlabel('Percentage of Votes for GOP')
# plt.ylabel('Frequency (Count)')
# plt.grid(axis='y', alpha=0.5)
# plt.show()

# # Q8: Correlation Matrix, Heatmap, and Scatter Matrix
# df_corr = df3.drop(columns=['Median_Household_Income_2021_cat'])
# correlation_matrix = df_corr.corr()
# print("------- Q8: Correlation Matrix --------")
# print(correlation_matrix)
# plt.figure(figsize=(10, 8))
# sns.heatmap(
#     correlation_matrix,
#     annot=True,
#     fmt=".2f",
#     cmap='vlag',
#     linewidths=.5,
#     cbar_kws={'label': 'Correlation Coefficient'}
# )
# plt.title('Q8: Correlation Heatmap of df3 Numerical Variables')
# plt.show()
# pd.plotting.scatter_matrix(
#     df_corr,
#     figsize=(15, 15),
#     diagonal='hist',
#     alpha=0.6,
#     marker='o'
# )
# plt.suptitle('Q8: Scatter Matrix of df3 Variables', y=1.02, fontsize=16)
# plt.show()

# # Final Geopandas/Plotly Visualization
# color_dict = {'republican': '#FF0000', 'democrats': '#0000FF'}
# fig, ax = plt.subplots(figsize = (12,12))
# grouped = votes.groupby('winner')
# for key, group in grouped:
#     group.plot(ax=ax, label=key, color=color_dict[key])
# plt.title('Choropleth Map of Election Winners (by Area)')
# plt.axis('off')
# plt.show()
# centroids = votes.copy()
# if centroids.crs and centroids.crs.to_string() != 'EPSG:4326':
#     centroids = centroids.to_crs(epsg=4326)
# centroids.geometry = centroids.centroid
# color_dict = {"republican": '#FF0000', 'democrats': '#0000FF'}
# centroids["winner"] =  np.where(centroids['votes_gop'] > centroids['votes_dem'], 'republican', 'democrats')
# centroids['lon'] = centroids['geometry'].x
# centroids['lat'] = centroids['geometry'].y
# df = pd.DataFrame(centroids[["county_name",'lon','lat','winner', 'CENSUS_2020_POP',"state_name"]])
# df['color'] = df['winner'].replace(color_dict)
# df['size'] = df['CENSUS_2020_POP']/6000
# df['text'] = df['CENSUS_2020_POP'].astype(int).apply(lambda x: '<br>Population: {:,} people'.format(x))
# df['hover'] = df['county_name'].astype(str) +  df['state_name'].apply(lambda x: ' ({}) '.format(x)) + df['text']
# fig_plotly = go.Figure(
#     data=go.Scattergeo(
#         locationmode = 'USA-states',
#         lon=df["lon"], lat=df["lat"],
#         text = df["hover"],
#         mode = 'markers',
#         marker_color = df["color"],
#         marker_size = df['size'],
#         hoverinfo="text"
#     )
# )
# fig_plotly.update_traces(
#     marker = {'opacity': 0.5, 'line_color': 'rgb(40,40,40)', 'line_width': 0.5, 'sizemode': 'area'}
# )
# fig_plotly.update_layout(
#     title_text = "Reproduction of the \"Acres don't vote, people do\" map <br>(Click legend to toggle traces)",
#     showlegend = True,
#     geo = {"scope": 'usa', "landcolor": 'rgb(217, 217, 217)'}
# )
# fig_plotly.show()