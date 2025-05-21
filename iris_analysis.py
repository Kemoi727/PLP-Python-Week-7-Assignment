
# Iris Dataset Analysis and Visualization

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Describe the dataset
print("\nStatistical Summary:")
print(df.describe())

# Group by target and calculate mean
print("\nMean values grouped by target:")
print(df.groupby('target').mean())

# Add species name for visualization
df['species'] = df['target'].apply(lambda x: iris.target_names[x])

# Visualizations

# Line chart
df.groupby('target')[df.select_dtypes(include='number').columns].mean().T.plot(title='Mean Features per Target', marker='o')

plt.xlabel('Features')
plt.ylabel('Mean Value')
plt.grid(True)
plt.tight_layout()
plt.savefig('line_chart.png')
plt.close()

# Bar chart
df.groupby('species')['petal length (cm)'].mean().plot(kind='bar', title='Avg Petal Length per Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.savefig('bar_chart.png')
plt.close()

# Histogram
df['sepal length (cm)'].plot(kind='hist', bins=20, title='Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.tight_layout()
plt.savefig('histogram.png')
plt.close()

# Scatter plot
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length')
plt.tight_layout()
plt.savefig('scatter_plot.png')
plt.close()

# code to manage potential issues such as file not found, data loading errors, etc. using try-except blocks.
# Error handling
import os
import sys
import traceback
# Function to handle errors
def handle_error(e):
    print("An error occurred:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print("Traceback:")
    traceback.print_exc()
    sys.exit(1)


# Main function to run the analysis
def main():
    try:
        # Load dataset
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target

        # Display first few rows
        print("First 5 rows of the dataset:")
        print(df.head())

        # Check for missing values
        print("\nMissing values:")
        print(df.isnull().sum())

        # Describe the dataset
        print("\nStatistical Summary:")
        print(df.describe())

        # Group by target and calculate mean
        print("\nMean values grouped by target:")
        print(df.groupby('target').mean())

    except Exception as e:
        handle_error(e)
# Run the main function
if __name__ == "__main__":
    main()
# This code performs an analysis of the Iris dataset, including loading the data, checking for missing values,
# and generating various visualizations. It also includes error handling to manage potential issues.    
