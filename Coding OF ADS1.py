import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew


def load_dataset(file_path):
    """
    Load the dataset from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file containing the dataset.

    Returns:
    pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    return pd.read_csv(file_path)


def statistical_analysis(df):
    """
    Perform statistical analysis on the dataset.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.
    """
    print("Basic Statistical Description:\n", df.describe())
    print("\nCorrelation Matrix:\n", df.corr(numeric_only=True))
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        print(f"\nKurtosis of {column}: {kurtosis(df[column])}")
        print(f"Skewness of {column}: {skew(df[column])}")


def line_chart(df):
    """
    Create a line chart showing 5G Network Coverage over the years for each country.

    Parameters:
    df (pd.DataFrame): The dataset containing columns 'Year', '5G Network Coverage (%)', and 'Country'.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Year', y='5G Network Coverage (%)', hue='Country', marker='o', linewidth=2.5)
    plt.title('5G Network Coverage over the Years for Each Country', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=14, fontweight='bold')
    plt.ylabel('5G Network Coverage (%)', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.legend(title='Country', fontsize=12, title_fontsize=14)
    plt.tight_layout()

    # Adding annotations for the last data point of each line
    for country in df['Country'].unique():
        latest_year = df['Year'].max()
        latest_value = df[(df['Country'] == country) & (df['Year'] == latest_year)]['5G Network Coverage (%)'].values[0]
        plt.text(latest_year + 0.1, latest_value, f"{latest_value:.1f}%", fontsize=10, color='black', fontweight='bold')

    plt.show()


def bar_chart(df):
    """
    Create a bar chart showing the average number of startups by tech sector and compare between countries.

    Parameters:
    df (pd.DataFrame): The dataset containing columns 'Tech Sector', 'Number of Startups', and 'Country'.
    
    This chart shows which tech sectors have the highest average number of startups, providing insight into the growth and focus areas of the technology industry for each country.
    """
    """
    Create a bar chart showing the average number of startups by tech sector.

    Parameters:
    df (pd.DataFrame): The dataset containing columns 'Tech Sector' and 'Number of Startups'.
    
    This chart shows which tech sectors have the highest average number of startups, providing insight into the growth and focus areas of the technology industry.
    """
    """
    Create a bar chart showing the average number of startups by tech sector.

    Parameters:
    df (pd.DataFrame): The dataset containing columns 'Tech Sector' and 'Number of Startups'.
    """
    plt.figure(figsize=(14, 6))
    avg_startups = df.groupby(['Tech Sector', 'Country'])['Number of Startups'].mean().reset_index()
    sns.barplot(data=avg_startups, x='Tech Sector', y='Number of Startups', hue='Country', palette='viridis', dodge=True)
    plt.title('Average Number of Startups by Tech Sector', fontsize=16, fontweight='bold')
    plt.xlabel('Tech Sector', fontsize=14, fontweight='bold')
    plt.ylabel('Average Number of Startups', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.tight_layout()

    # Adding annotations on each bar to display values
    for index, row in avg_startups.iterrows():
        tech_sector = row['Tech Sector']
        country = row['Country']
        num_startups = row['Number of Startups']
        plt.text(
            x=index,
            y=num_startups + 5,
            s=f'{num_startups:.1f}',
            ha='center', fontsize=10, color='black', fontweight='bold'
        )

    plt.show()


def heatmap(df):
    """
    Create a heatmap showing the correlation between key metrics.

    Parameters:
    df (pd.DataFrame): The dataset containing numerical columns for correlation analysis.
    
    This heatmap helps in identifying the strength and direction of relationships between different numerical metrics, useful for understanding dependencies in the data.
    """
    """
    Create a heatmap showing the correlation between key metrics.

    Parameters:
    df (pd.DataFrame): The dataset containing numerical columns for correlation analysis.
    """
    plt.figure(figsize=(14, 10))
    correlation_matrix = df.corr(numeric_only=True)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=1, linecolor='white', annot_kws={"size": 10, "weight": "bold"})
    plt.title('Correlation Heatmap of Key Metrics', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold', rotation=45)
    plt.yticks(fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def box_plot(df):
    """
    Create a box plot showing the distribution of venture capital funding by country.

    Parameters:
    df (pd.DataFrame): The dataset containing columns 'Country' and 'Venture Capital Funding (in USD)'.
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Country', y='Venture Capital Funding (in USD)', palette='Set2')
    plt.title('Distribution of Venture Capital Funding by Country', fontsize=16, fontweight='bold')
    plt.xlabel('Country', fontsize=14, fontweight='bold')
    plt.ylabel('Venture Capital Funding (in USD)', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.tight_layout()

    # Adding annotations to highlight key insights
    for i, box in enumerate(df['Country'].unique()):
        median = df[df['Country'] == box]['Venture Capital Funding (in USD)'].median()
        plt.annotate(
            f'Median: ${median:,.0f}',
            xy=(i, median),
            xytext=(i, median + 1e9),
            ha='center', fontsize=10, color='blue', fontweight='bold',
            arrowprops=dict(facecolor='blue', arrowstyle='->')
        )

    plt.show()


if __name__ == "__main__":
    # Load the dataset
    dataset_path = 'Big_Japan_vs_China_Technology.csv'
    df = load_dataset(dataset_path)

    # Perform statistical analysis
    statistical_analysis(df)

    # Create visualizations
    line_chart(df)
    bar_chart(df)
    heatmap(df)
    box_plot(df)
