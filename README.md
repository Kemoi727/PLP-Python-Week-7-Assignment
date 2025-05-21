# Data Analysis and Visualization Project

## Overview

This project demonstrates the process of loading, exploring, analyzing, and visualizing a dataset using Python. The chosen dataset is the classic Iris dataset, which contains measurements of different iris flower species.

The project is structured into three main tasks:

1. **Load and Explore the Dataset**
2. **Basic Data Analysis**
3. **Data Visualization**

---

## Task 1: Load and Explore the Dataset

- Loaded the Iris dataset using `pandas`.
- Displayed the first few rows to inspect the data structure.
- Checked for data types and missing values.
- Cleaned the dataset by confirming no missing values are present.

## Task 2: Basic Data Analysis

- Computed descriptive statistics (mean, median, standard deviation) for numerical columns using `.describe()`.
- Grouped data by species and calculated the mean of numerical attributes for each species.
- Identified patterns such as differences in measurements between species.

## Task 3: Data Visualization

Created four types of visualizations using `matplotlib` and `seaborn`:

- **Line Chart:** Trends of sepal length over the dataset index.
- **Bar Chart:** Average petal length by iris species.
- **Histogram:** Distribution of sepal width.
- **Scatter Plot:** Relationship between sepal length and petal length, colored by species.

All plots include appropriate titles, axis labels, and legends for clarity.

---

## How to Run

1. Ensure you have Python 3.x installed.
2. Install the required libraries if not already installed:

```bash
pip install pandas matplotlib seaborn scikit-learn
