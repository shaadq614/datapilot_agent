# ML Agent Report

- **Task:** classification
- **Target:** churned
- **Best model:** random_forest
- **Test metrics:** {'accuracy': 0.8167, 'f1_macro': 0.8124}
- **Baseline CV score:** 0.353

## Warnings
- Dropped columns (IDs/text/too empty): ['customer_id', 'monthly_charges']

## Model comparison (cross-validation)
- baseline: 0.353
- logistic_regression: 0.818
- random_forest: 0.827
- gradient_boosting: 0.793