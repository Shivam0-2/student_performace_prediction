# Student Performance Prediction

This is a complete beginner-friendly Data Science mini project in Python. It follows a typical college Data Science syllabus and uses only simple libraries and concepts.

## Project Objective

The objective of this project is to predict a student's final exam score using basic student information such as study hours, attendance, previous score, sleep hours, internet access, and extracurricular activity.

## Libraries Used

| Library | Use in Project |
| --- | --- |
| pandas | Loading CSV data, cleaning data, handling missing values, and preprocessing. |
| numpy | Mathematical operations such as calculating RMSE. |
| matplotlib | Creating and saving basic graphs. |
| seaborn | Creating attractive statistical visualizations such as heatmaps and bar charts. |
| scikit-learn | Splitting data, building regression models, and evaluating model performance. |

## Machine Learning Algorithms Used

### Linear Regression

Linear Regression was selected because it is one of the simplest regression algorithms. It is useful when the target value has a linear relationship with input features.

### Decision Tree Regressor

Decision Tree Regressor was selected because it is easy to understand and can handle non-linear relationships in the dataset.

### KNN Regressor

KNN Regressor was selected because it is a simple algorithm that predicts values based on similar nearby records.

## Project Structure

```text
student-performance-prediction/
│
├── data/
│   └── student_performance.csv
│
├── outputs/
│   └── Generated graph images appear here after running main.py
│
├── main.py
├── requirements.txt
├── DATASET_EXPLANATION.md
├── VIVA_QUESTIONS.md
└── README.md
```

## Steps Performed

1. Data loading
2. Data cleaning
3. Handling missing values
4. Basic preprocessing
5. Exploratory Data Analysis
6. Correlation analysis
7. Data visualization
8. Train-test split
9. Model training
10. Model performance comparison
11. Conclusion and insights

## Visualizations Included

The project generates the following graphs:

| Graph | File |
| --- | --- |
| Histogram of final scores | `outputs/final_score_histogram.png` |
| Scatter plot of study hours vs final score | `outputs/study_hours_vs_final_score.png` |
| Bar chart of parental education vs average final score | `outputs/parental_education_bar_chart.png` |
| Correlation heatmap | `outputs/correlation_heatmap.png` |
| RMSE model comparison bar chart | `outputs/model_rmse_comparison.png` |
| R2 Score model comparison bar chart | `outputs/model_r2_comparison.png` |

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

## Sample Output Explanation

After running `main.py`, the program displays:

- First 5 rows of the dataset
- Dataset information
- Statistical summary
- Missing values before and after cleaning
- Correlation matrix
- Preprocessed data
- Number of training and testing rows
- RMSE and R2 Score for each model
- A conclusion with important insights

Example model comparison output format:

```text
Model Performance Comparison:
                    Model      RMSE  R2 Score
0       Linear Regression  ...
1  Decision Tree Regressor ...
2           KNN Regressor ...
```

The exact values may change if the dataset or random state is changed.

## Conclusion

The project shows that student performance is usually related to study hours, attendance percentage, and previous academic score. Regression models can be used to estimate final scores from these features. RMSE and R2 Score help compare the models and select the best one.

## Notes for College Submission

This project avoids advanced tools such as deep learning, TensorFlow, PyTorch, Flask, FastAPI, APIs, and deployment frameworks. It is intentionally written with simple comments and readable variable names so that a 2nd-year engineering student can understand and explain it.
