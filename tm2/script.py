import requests
import getdata
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly
import plotly.graph_objects as go
import geopandas as gpd
import numpy as np

url = 'https://raw.githubusercontent.com/linogaliana/python-datascientist/master/content/modelisation/get_data.py'
r = requests.get(url, allow_redirects=True)
open('getdata.py', 'wb').write(r.content)

'''
votes = getdata.create_votes_dataframes()

print("------- SHAPE --------")
print(votes.shape)
print("------- INFO --------")
print(votes.info())
print("------- DESCRIBE --------")
print(votes.describe())

df2 = votes[['winner']].copy()
df2['winner2'] = df2['winner'].astype('category').cat.codes
print(df2.head())

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
print(df3.head(3))

# --- Step 1: Create the frequency table ---
frequency_table = df3['winner'].value_counts()

print("Frequency Table for 'winner':")
print(frequency_table)
print("\n")

frequency_table.plot(
    kind='barh',
    title='Frequency of Winners in df2',
    xlabel='Count (Frequency)',
    ylabel='Winner Code',
    figsize=(8, 5),
    color='skyblue'
)

plt.tight_layout()
plt.show()


# --- 1. Transform the continuous variable into a categorical one with 5 bins ---
# The resulting categories will be intervals (e.g., (min_val, val1], (val1, val2], etc.)
df3['Median_Household_Income_2021_cat'] = pd.cut(
    df3['Median_Household_Income_2021'],
    bins=5
)

# --- 2. Create the frequency table ---
frequency_table_income = df3['Median_Household_Income_2021_cat'].value_counts().sort_index()

print("Frequency Table for 'Median_Household_Income_2021_cat':")
print(frequency_table_income)
print("\n")

# --- 3. Create the associated bar plot ---
# Use sort_index() to plot the bars in the correct order of income intervals
# Use kind='bar' for a standard vertical bar plot
frequency_table_income.plot(
    kind='bar',
    title='Frequency of Median Household Income Categories',
    xlabel='Median Household Income Category (USD)',
    ylabel='Count (Frequency)',
    figsize=(10, 6),
    color='coral',
    rot=45  # Rotate x-axis labels for readability
)

plt.tight_layout()
plt.show()


print("\nDescriptive Statistics for ALL Columns (Numerical and Categorical):")
print(df3.describe(include='all'))


# 1. Select the 'votes_gop' column and call the .hist() method
df3['votes_gop'].hist(
    bins=20,  # Optional: specifies the number of bins (groups) for the data
    figsize=(8, 5),
    color='darkblue',
    edgecolor='black'  # Adds borders to the bars for better visualization
)

# 2. Add labels and title for clarity
plt.title('Distribution of votes_gop')
plt.xlabel('Percentage of Votes for GOP')
plt.ylabel('Frequency (Count)')
plt.grid(axis='y', alpha=0.5) # Optional: adds a subtle grid line

# 3. Display the plot
plt.show()
'''

'''
df3['Median_Household_Income_2021_cat'] = pd.cut(
    df3['Median_Household_Income_2021'],
    bins=5
)
df_corr = df3.drop(columns=['Median_Household_Income_2021_cat'])

# 2. Calculate the correlation matrix
correlation_matrix = df_corr.corr()

print("Correlation Matrix:")
print(correlation_matrix)

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,          # Show the correlation values on the plot
    fmt=".2f",           # Format the values to two decimal places
    cmap='vlag',         # Choose a divergent color map (good for correlation)
    linewidths=.5,       # Adds lines between cells
    cbar_kws={'label': 'Correlation Coefficient'}
)
plt.title('Correlation Heatmap of df2 Numerical Variables')
plt.show()

# Create the scatter matrix
pd.plotting.scatter_matrix(
    df_corr,
    figsize=(15, 15),
    diagonal='hist', # Show a histogram on the diagonal
    alpha=0.6,       # Transparency of the points
    marker='o'
)
plt.suptitle('Scatter Matrix of df2 Variables', y=1.02, fontsize=16)
plt.show()

'''
color_dict = {'republican': '#FF0000', 'democrats': '#0000FF'}

fig, ax = plt.subplots(figsize = (12,12))
grouped = votes.groupby('winner')
for key, group in grouped:
    group.plot(ax=ax, label=key, color=color_dict[key])
plt.axis('off')

centroids = votes.copy()
centroids.geometry = centroids.centroid
centroids['size'] = centroids['CENSUS_2020_POP'] / 10000  # to get reasonable plotable number

color_dict = {"republican": '#FF0000', 'democrats': '#0000FF'}
centroids["winner"] =  np.where(centroids['votes_gop'] > centroids['votes_dem'], 'republican', 'democrats')


centroids['lon'] = centroids['geometry'].x
centroids['lat'] = centroids['geometry'].y
centroids = pd.DataFrame(centroids[["county_name",'lon','lat','winner', 'CENSUS_2020_POP',"state_name"]])
groups = centroids.groupby('winner')

df = centroids.copy()

df['color'] = df['winner'].replace(color_dict)
df['size'] = df['CENSUS_2020_POP']/6000
df['text'] = df['CENSUS_2020_POP'].astype(int).apply(lambda x: '<br>Population: {:,} people'.format(x))
df['hover'] = df['county_name'].astype(str) +  df['state_name'].apply(lambda x: ' ({}) '.format(x)) + df['text']

fig_plotly = go.Figure(
    data=go.Scattergeo(
        locationmode = 'USA-states',
        lon=df["lon"], lat=df["lat"],
        text = df["hover"],
        mode = 'markers',
        marker_color = df["color"],
        marker_size = df['size'],
        hoverinfo="text"
    )
)

fig_plotly.update_traces(
    marker = {'opacity': 0.5, 'line_color': 'rgb(40,40,40)', 'line_width': 0.5, 'sizemode': 'area'}
)

fig_plotly.update_layout(
    title_text = "Reproduction of the \"Acres don't vote, people do\" map <br>(Click legend to toggle traces)",
    showlegend = True,
    geo = {"scope": 'usa', "landcolor": 'rgb(217, 217, 217)'}
)
