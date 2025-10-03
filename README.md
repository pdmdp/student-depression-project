# Student Depression Prediction Using Machine Learning

A binary classification project predicting student depression based on lifestyle, demographic, academic, and psychological indicators. Developed as part of the Machine Learning Module 1 (2024/2025) for the Bachelor in Artificial Intelligence at the University of Pavia (UNIPV).

## 🎯 Project Overview

Depression among students is a growing concern in academic environments. This project employs machine learning techniques to identify patterns that contribute to student depression, enabling early detection and intervention.

**Objective:** Build a robust binary classification model capable of predicting whether a student is experiencing depression (1) or not (0).

## 📊 Dataset

**Source:** [Student Depression Dataset – Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)

The dataset contains responses from students across multiple dimensions:

### Features
- **Demographics:** Age, Gender, City, Degree
- **Academic:** CGPA, Academic Pressure, Study Satisfaction, Work/Study Hours
- **Lifestyle:** Sleep Duration, Dietary Habits, Physical Activity
- **Psychological:** Suicidal Thoughts, Family History of Mental Illness, Financial Stress

### Target Variable
- **Depression:** Binary classification (0 = No, 1 = Yes)

## 🔧 Methodology

### 1. Data Preprocessing

**Feature Engineering:**
- Renamed columns for better readability
- Removed irrelevant features: `id`, `Profession`, `Work Pressure`, `Job Satisfaction`
- Consolidated high-cardinality categorical variables:
  - **Degree:** Grouped into Pre-University, Undergraduate, Postgraduate, PhD
  - **City:** Categorized into Major Metropolitan, Industrial/Commercial, Suburban, Other

**Missing Data Handling:**
- Intentionally introduced 10% missing values to simulate real-world scenarios
- Treated infrequent 'Others' categories as missing data
- Dropped rows with missing values in `Financial Stress` and `Sleep Duration` (minimal impact)
- Applied imputation strategies for remaining missing data

### 2. Feature Transformation Pipeline

Created specialized pipelines for different feature types:

| Feature Type | Transformation |
|--------------|----------------|
| Numerical | Standard Scaling |
| Binary Categorical | One-Hot Encoding |
| Ordinal (Sleep, Diet) | Ordinal Encoding + Scaling |
| Nominal Categorical | Imputation + One-Hot Encoding |

### 3. Model Selection & Optimization

**Approach:** Nested Cross-Validation with Randomized Search

**Explored Components:**
- **Sampling Techniques:** None, SMOTE, RandomOverSampler
- **Dimensionality Reduction:** None, PCA, LDA, Sequential Feature Selection
- **Classifiers:** 
  - Perceptron
  - Logistic Regression (L1/L2 regularization)
  - Random Forest
  - XGBoost

**Hyperparameter Search:**
- Initial broad search across all configurations (~5x combinations)
- Refined search around best-performing model
- 5-fold cross-validation for model evaluation
- F1 score as primary evaluation metric

### 4. Final Model

**Selected Architecture:**
```
Pipeline:
├── Feature Transformation (ColumnTransformer)
└── Logistic Regression (L1 penalty, saga solver)
    └── Optimal C parameter: ~5-15 (determined via RandomizedSearchCV)
```

## 📈 Results

### Model Performance

| Metric | Test Set Score |
|--------|----------------|
| **F1 Score** | 0.87+ |
| **Accuracy** | 0.87+ |
| **Precision** | 0.87+ |
| **Recall** | 0.87+ |

### Key Insights

**Learning Curve Analysis:**
- Model shows good convergence with increasing training data
- Minimal gap between training and validation scores indicates low overfitting
- Performance stabilizes around 60-70% of training data

**Validation Curve Analysis:**
- Optimal regularization strength (C) identified in range [5, 15]
- Model robust to hyperparameter variations within this range

## 🚀 Installation & Usage

### Prerequisites
```bash
Python 3.8+
```

### Required Libraries
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn missingno matplotlib mlxtend
```

### Running the Project

1. **Clone the repository:**
```bash
git clone https://github.com/pdmdp/student-depression-project.git
cd student-depression-project
```

2. **Install dependencies:**
```bash
python Init.py  # Handles package installation
```

3. **Run the notebook:**
```bash
jupyter notebook MLProjectDiPilato.ipynb
```

### Dataset Setup
Place `StudentDepressionDataset.csv` in the `/work/` directory or update the file path in the notebook.

## 📁 Project Structure

```
student-depression-project/
│
├── MLProjectDiPilato.ipynb    # Main analysis notebook
├── Init.py                     # Dependency installer
├── StudentDepressionDataset.csv # Dataset (place in /work/)
└── README.md                   # Project documentation
```

## 🔍 Key Features

✅ **Real-world data simulation** with controlled missing values  
✅ **Comprehensive preprocessing pipeline** handling diverse feature types  
✅ **Systematic model selection** via nested cross-validation  
✅ **Class imbalance handling** with SMOTE and class weighting  
✅ **Hyperparameter optimization** with RandomizedSearchCV  
✅ **Robust evaluation** using multiple metrics and visualization

## 🎓 Academic Context

This project demonstrates proficiency in:
- End-to-end machine learning pipeline development
- Advanced preprocessing techniques
- Model selection and hyperparameter tuning
- Cross-validation strategies
- Performance evaluation and interpretation

## ⚠️ Ethical Considerations

**Important:** This model is developed for educational purposes and demonstrates machine learning concepts. Mental health prediction requires:
- Clinical validation before real-world deployment
- Consideration of ethical implications and biases
- Integration with professional mental health services
- Privacy protection and informed consent
- Awareness of model limitations

**This model should NOT be used as a standalone diagnostic tool.**

## 📝 Future Improvements

- [ ] Explore ensemble methods (stacking, blending)
- [ ] Implement SHAP values for model interpretability
- [ ] Collect more diverse data to improve generalization
- [ ] Develop web application for model deployment
- [ ] Add confidence intervals for predictions
- [ ] Investigate deep learning approaches

## 👨‍💻 Author

**Filippo Di Pilato**  
Bachelor in Artificial Intelligence  
University of Pavia (UNIPV)  
Academic Year 2024/2025

## 📄 License

This project is developed for academic purposes as part of the Machine Learning Module 1 course.

## 🙏 Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)
- University of Pavia, Department of Computer Engineering
- Course instructors and teaching assistants

---

⭐ If you found this project helpful, please consider giving it a star!
