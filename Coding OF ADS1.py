import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# Load dataset
data_path = 'democracy-index-eiu.csv'
df = pd.read_csv(data_path)

# Data Cleaning
df_cleaned = df.dropna(subset=['Democracy score'])

# Statistical Analysis
def calculate_statistics(data):
    """
    Calculate and display skewness and kurtosis for the Democracy score column.
    
    Parameters:
        data (DataFrame): Cleaned dataset with Democracy score column.
    
    Returns:
        dict: A dictionary containing skewness and kurtosis of Democracy score.
    """
    democracy_skewness = skew(data['Democracy score'])
    democracy_kurtosis = kurtosis(data['Democracy score'])
    print("Democracy Score Statistical Analysis")
    print(f"Skewness: {democracy_skewness:.2f}")
    print(f"Kurtosis: {democracy_kurtosis:.2f}")
    return {'Skewness': democracy_skewness, 'Kurtosis': democracy_kurtosis}

# Plot Distribution of Democracy Scores
def plot_distribution(data):
    """
    Plot the histogram of Democracy scores to show distribution.
    
    Parameters:
        data (DataFrame): Cleaned dataset with Democracy score column.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data['Democracy score'], bins=30, edgecolor='black', color='skyblue')
    plt.title('Distribution of Democracy Scores (2006-2023)', fontsize=14, fontweight='bold')
    plt.xlabel('Democracy Score', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Plot Trend of Average Democracy Score Over Time
def plot_yearly_trend(data):
    """
    Plot the yearly trend of average Democracy scores over time.
    
    Parameters:
        data (DataFrame): Cleaned dataset with Year and Democracy score columns.
    """
    yearly_trend = data.groupby('Year')['Democracy score'].mean()
    plt.figure(figsize=(12, 6))
    plt.plot(yearly_trend.index, yearly_trend.values, marker='o', linestyle='-', color='b')
    plt.title('Average Democracy Score Over Time (2006-2023)', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontweight='bold')
    plt.ylabel('Average Democracy Score', fontweight='bold')
    plt.grid(True)
    plt.show()

# Plot Heatmap for High Variance Countries
def plot_heatmap_high_variance(data):
    """
    Plot a heatmap showing Democracy scores over time for high variance countries.
    
    Parameters:
        data (DataFrame): Cleaned dataset with Entity, Year, and Democracy score columns.
    """
    # Ensure we only calculate the standard deviation on the numeric 'Democracy score' column
    high_variance_countries = (
        data.groupby('Entity')['Democracy score']
        .std()
        .nlargest(10)
        .index
    )
    
    subset_data = data[data['Entity'].isin(high_variance_countries)]
    heatmap_data = subset_data.pivot(index='Entity', columns='Year', values='Democracy score')
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'label': 'Democracy Score'})
    plt.title('Democracy Score Heatmap for High Variance Countries (2006-2023)', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontweight='bold')
    plt.ylabel('Country', fontweight='bold')
    plt.show()

# Plot Average Democracy Score by Region
def plot_region_avg_score(data, latest_year):
    """
    Plot the average democracy score by region for the latest available year.
    
    Parameters:
        data (DataFrame): Cleaned dataset with Region, Year, and Democracy score columns.
        latest_year (int): The latest year to filter the data by.
    """
    # Map countries to regions (as per example region mapping)
    region_mapping = {
        'United States': 'North America', 'Canada': 'North America',
        'Brazil': 'South America', 'Germany': 'Europe', 'United Kingdom': 'Europe',
        'India': 'Asia', 'China': 'Asia', 'South Africa': 'Africa', 'Nigeria': 'Africa',
    }
    data['Region'] = data['Entity'].map(region_mapping)
    recent_data_with_regions = data[(data['Year'] == latest_year) & data['Region'].notnull()]
    region_avg_scores = recent_data_with_regions.groupby('Region')['Democracy score'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    region_avg_scores.plot(kind='bar', color='teal', edgecolor='black')
    plt.title(f'**Average Democracy Score by Region in {latest_year}**', fontsize=14, fontweight='bold')
    plt.xlabel('Region', fontweight='bold')
    plt.ylabel('Average Democracy Score', fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for index, value in enumerate(region_avg_scores):
        plt.text(index, value + 0.1, f"{value:.2f}", ha='center', fontweight='bold')
    plt.show()

# Calculate Correlation for Two Specific Countries
def calculate_correlation_between_countries(data, country1, country2):
    """
    Calculate and return the correlation of Democracy scores between two specified countries.
    
    Parameters:
        data (DataFrame): Cleaned dataset with 'Entity', 'Year', and 'Democracy score' columns.
        country1 (str): Name of the first country.
        country2 (str): Name of the second country.
    
    Returns:
        float: Correlation coefficient between the democracy scores of the two countries.
    """
    # Filter data for the selected countries
    country_data = data[data['Entity'].isin([country1, country2])]
    
    # Pivot data to have years as index and countries as columns
    country_pivot = country_data.pivot(index='Year', columns='Entity', values='Democracy score')
    
    # Calculate correlation between the two countries' democracy scores
    correlation = country_pivot[country1].corr(country_pivot[country2])
    print(f"Correlation between {country1} and {country2}: {correlation:.2f}")
    return correlation

# Running the functions to perform analysis and visualize
latest_year = df_cleaned['Year'].max()
stats = calculate_statistics(df_cleaned)
plot_distribution(df_cleaned)
plot_yearly_trend(df_cleaned)
plot_heatmap_high_variance(df_cleaned)
plot_region_avg_score(df_cleaned, latest_year)

# Example correlation between two specific countries
calculate_correlation_between_countries(df_cleaned, 'United States', 'Canada')