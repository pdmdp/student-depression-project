# Student Depression Classification – Machine Learning Project

This repository contains a machine learning project developed as part of the *Machine Learning Module 1 (2024/2025)* for the **Bachelor in Artificial Intelligence** at the **University of Pavia (UNIPV)**. The objective is to build a binary classification model capable of predicting whether a student is experiencing depression, based on various lifestyle, demographic, academic, and psychological indicators.

---

## 📘 Project Overview

- **Problem Statement**: Classify students as either *depressed (1)* or *not depressed (0)* based on personal, academic, lifestyle, and mental health-related features.
- **Dataset**: [Student Depression Dataset – Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)
- **Target Variable**: `Depression` (binary: `0` = No, `1` = Yes)
- **Approach**: 
  - Simulate real-world data imperfections by introducing controlled missing values
  - Preprocess data, handle missingness, and apply various classification models
  - Evaluate performance using appropriate metrics

---

## 📁 Files Included

- `MLProjectDiPilato.ipynb`: Main notebook containing data exploration, preprocessing, model training, and evaluation.
- `Init.py`: Handles Python package installation and environment setup.
- `StudentDepressionDataset.csv`: dataset

---

## 🧠 Dataset Description

The dataset contains responses from students across a range of factors:
- **Demographics**: Age, gender, etc.
- **Academic**: Study hours, academic performance
- **Lifestyle**: Sleep patterns, physical activity, screen time
- **Family History**: Mental illness in family
- **Psychological**: Stress levels, suicidal thoughts, self-harm, anxiety, etc.


```bash
git clone https://github.com/yourusername/student-depression-ml.git
cd student-depression-ml
