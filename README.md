# Student depression classification

This project examines how preprocessing and input questions affect classification of a survey dataset's binary depression label. It compares logistic regression with a majority-class baseline and checks what changes when the suicidal-thought item is removed.

The reviewed L1 logistic model reaches **F1 of 0.8821** on the historical test split. Removing that item lowers F1 to **0.8382**, showing that a substantial part of the observed performance depends on which questions are available.

[Reviewed notebook](Student_Depression_Review.ipynb) · [Original coursework](MLProjectDiPilato.ipynb) · [Results](results/review.json)

## Approach

The original coursework explores Perceptron, logistic regression, random forest and XGBoost, including resampling and dimensionality reduction. The focused review studies the final L1 logistic model with a more controlled preprocessing pipeline:

- Imputation, encoding and scaling are fitted inside cross-validation.
- Sleep-duration categories follow their natural order.
- Regularization is selected separately for the full and reduced feature sets using five-fold stratified cross-validation.
- A majority-class baseline makes class imbalance visible, and each confusion matrix comes from the model it describes.

## Results

![F1 and balanced accuracy for the baseline and the two logistic models](images/review-comparison.png)

| Model | F1, positive class | Accuracy | Balanced accuracy | ROC AUC |
| :--- | ---: | ---: | ---: | ---: |
| Majority-class baseline | 0.7386 | 0.5855 | 0.5000 | 0.5000 |
| L1 logistic regression | **0.8821** | **0.8594** | **0.8514** | **0.9290** |
| L1 logistic, without suicidal-thought item | 0.8382 | 0.8061 | 0.7955 | 0.8796 |

The majority baseline already has a positive-class F1 of 0.7386, despite balanced accuracy of 0.5. That is why F1 alone would give an incomplete picture. Both logistic models improve on the baseline, while the reduced feature set has lower performance across all four metrics.

![Confusion matrices for the reviewed logistic models](images/review-confusion-matrices.png)

The original notebook reported F1 0.8791 and accuracy 0.8560. The review uses the same historical split, so its comparison with the original is retrospective rather than an independent new test.

## Contribution

**Matteo Pietro Di Pilato** · Machine Learning, Module 1, BSc in Artificial Intelligence, 2024/25. The work covers preprocessing, classifier comparisons, model selection and the later feature-sensitivity review.

## Limits and next steps

The target is the dataset's `Depression` label. Its assessment method and timing are not established here, nor is the timing of the suicidal-thought item relative to that label. This is a classification study, with no clinical or prospective validation; it is not a diagnostic tool or a model of future onset.

A stronger evaluation would use an independently collected sample with documented label definitions and measurement times. The reduced-feature comparison helps identify a dependency, but does not establish that the remaining inputs are sufficient for clinical use.

## Data and reproduction

The [Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) contains 27,901 rows and 18 columns. Historical cleaning retains 27,880 rows: 22,304 training and 5,576 test examples, using seed 30 and stratification. The review preserves the original missing-data exercise, masking the same randomly chosen 10% of rows in four fields.

```bash
git clone https://github.com/pdmdp/student-depression-project.git
cd student-depression-project
python -m pip install -r requirements.txt
python -m jupyter lab Student_Depression_Review.ipynb
```

Run the notebook from the repository root. It verifies the input checksum before fitting. `requirements.txt` pins the reviewed environment; the original exploration has separate dependencies in `requirements-original.txt`. Run `python scripts/plot_results.py` to redraw the figures from saved metrics.

Code: [MIT](LICENSE). The source dataset retains its own terms.
