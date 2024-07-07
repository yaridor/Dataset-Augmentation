# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Set plotting styles
sns.set(style="whitegrid")


def display_basic_statistics(df):
    # Display basic information about the dataset
    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nSummary statistics for numeric columns:")
    print(df.describe())

    print("\nDataset info:")
    print(df.info())

    print("\nMissing values in each column:")
    print(df.isnull().sum())


def plot_distributions(df):
    df = df.select_dtypes(include=[np.number])
    num_rows = int(np.ceil(len(df.columns) ** 0.5))
    num_cols = int(np.ceil(len(df.columns) / num_rows))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 6 * num_rows))
    for ax, column in zip(axes.flat, df.columns):
        if np.unique(df[column]).shape[0] < 5:
            sns.histplot(df[column], kde=False, bins=30, ax=ax)
        else:
            sns.histplot(df[column], kde=True, bins=30, ax=ax)
        ax.set_title(f'Distribution of {column}')
    plt.tight_layout()
    plt.show()


def plot_scatter_matrix(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    pd.plotting.scatter_matrix(df[num_cols], figsize=(15, 10), diagonal='kde')
    plt.suptitle('Scatter Matrix for Numeric Features')
    plt.tight_layout()
    plt.show()


def plot_boxplots(df):
    df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df)
    plt.title(f'Boxplot')
    plt.tight_layout()
    plt.show()


def plot_count_plots(df):
    for column in df.columns:
        if isinstance(df[column].dtype, pd.CategoricalDtype) or df[column].dtype == object:
            plt.figure(figsize=(10, 6))
            sns.countplot(y=df[column], order=df[column].value_counts().index)
            plt.title(f'Count plot of {column}')
            plt.tight_layout()
            plt.show()


def detect_outliers(df, threshold=3):
    outliers = {}
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            outliers[column] = np.where(z_scores > threshold)[0]
    return outliers


def exploratory_data_analysis(data):
    display_basic_statistics(data)
    plot_distributions(data)

    print("Pair plot for numeric features:")
    sns.pairplot(data.select_dtypes(include=[np.number]))
    plt.title('Pair Plot for Numeric Features')
    plt.tight_layout()
    plt.show()

    # Correlation matrix
    print("Correlation matrix:")
    corr_matrix = data.select_dtypes(include=[np.number]).corr()
    print(corr_matrix)

    print("Plotting correlation heatmap:")
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.show()

    print("Plotting scatter matrix for numeric features:")
    plot_scatter_matrix(data)

    print("Plotting boxplots for numeric features:")
    plot_boxplots(data)

    print("Plotting count plots for categorical features:")
    plot_count_plots(data)

    print("Detecting outliers using Z-score method:")
    outliers = detect_outliers(data)
    print(outliers)

    # Summary
    print("\nSummary of Exploratory Data Analysis:")
    print(f"Dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")
    print(f"Numeric columns: {data.select_dtypes(include=[np.number]).columns.tolist()}")
    print(f"Categorical columns: {data.select_dtypes(include=['object']).columns.tolist()}")


def main():
    # data_path = "Output Data/caesarian_original.csv"
    data_path = "Output Data/caesarian.csv"
    data = pd.read_csv(data_path)

    exploratory_data_analysis(data)


if __name__ == '__main__':
    main()
