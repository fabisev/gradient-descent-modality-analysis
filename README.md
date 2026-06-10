# Gradient Descent Modality Analysis

This repository contains the analysis code for the exploratory quantitative comparisons used in my CSE3000 research project on teaching gradient descent through different representations.

The analysis compares two instructional conditions:

- classic textbook-style condition
- multiple-represenations condition

The repository includes anonymised score data and Python scripts for calculating:

- descriptive statistics
- Mann--Whitney U tests
- Cliff's delta effect sizes

## Repository structure

```text
analysis/
    mann_whitney_analysis.py

data/
    anonymised_scores.csv

results/
    statistical_results.csv