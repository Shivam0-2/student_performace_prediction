"""
Student Performance Prediction

This is a beginner-friendly Data Science mini project.
It uses simple syllabus-based concepts:
data loading, cleaning, preprocessing, EDA, visualization, train-test split,
and regression models.
"""

# Standard Python module used only for creating the output folder
import os

# Data handling libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning tools from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------------
# 1. Load the CSV dataset
# ---------------------------------------------------------

data = pd.read_csv("data/student_performance.csv")

print("\nSTUDENT PERFORMANCE PREDICTION PROJECT")
print("=" * 50)
print("\nFirst 5 rows of the dataset:")
print(data.head())

print("\nDataset information:")
print(data.info())

print("\nBasic statistical summary:")
print(data.describe())


# ---------------------------------------------------------
# 2. Check and handle missing values
# ---------------------------------------------------------

print("\nMissing values before cleaning:")
print(data.isnull().sum())

# Fill missing numeric values with the mean of that column.
numeric_columns = ["study_hours", "attendance_percentage", "previous_score", "sleep_hours", "final_score"]

for column in numeric_columns:
    data[column] = data[column].fillna(data[column].mean())

# Fill missing categorical values with the mode, which means most repeated value.
categorical_columns = ["gender", "parental_education", "internet_access", "extracurricular"]

for column in categorical_columns:
    data[column] = data[column].fillna(data[column].mode()[0])

print("\nMissing values after cleaning:")
print(data.isnull().sum())


# ---------------------------------------------------------
# 3. Exploratory Data Analysis and visualizations
# ---------------------------------------------------------

# Create a folder where all graph images will be saved.
os.makedirs("outputs", exist_ok=True)

# Set a simple and clean style for graphs.
sns.set(style="whitegrid")

# Histogram: distribution of final scores.
plt.figure(figsize=(8, 5))
sns.histplot(data["final_score"], bins=10, kde=True, color="skyblue")
plt.title("Distribution of Final Scores")
plt.xlabel("Final Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("outputs/final_score_histogram.png")
plt.close()

# Scatter plot: study hours vs final score.
plt.figure(figsize=(8, 5))
sns.scatterplot(x="study_hours", y="final_score", hue="gender", data=data)
plt.title("Study Hours vs Final Score")
plt.xlabel("Study Hours per Day")
plt.ylabel("Final Score")
plt.tight_layout()
plt.savefig("outputs/study_hours_vs_final_score.png")
plt.close()

# Bar chart: average final score by parental education.
average_score_by_education = data.groupby("parental_education", as_index=False)["final_score"].mean()

plt.figure(figsize=(8, 5))
sns.barplot(x="parental_education", y="final_score", data=average_score_by_education, color="lightgreen")
plt.title("Average Final Score by Parental Education")
plt.xlabel("Parental Education")
plt.ylabel("Average Final Score")
plt.tight_layout()
plt.savefig("outputs/parental_education_bar_chart.png")
plt.close()

# Correlation heatmap for numeric columns.
correlation_data = data[numeric_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_data, annot=True, cmap="YlGnBu", linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

print("\nCorrelation matrix:")
print(correlation_data)


# ---------------------------------------------------------
# 4. Basic preprocessing for machine learning
# ---------------------------------------------------------

# Student ID is only an identifier, so it is not useful for prediction.
data_for_model = data.drop("student_id", axis=1)

# Convert categorical columns into numeric columns using one-hot encoding.
# Example: gender becomes gender_Male.
data_for_model = pd.get_dummies(data_for_model, drop_first=True)

# Convert True/False dummy columns into 1/0 values for easier understanding.
data_for_model = data_for_model.astype(float)

print("\nData after preprocessing:")
print(data_for_model.head())


# ---------------------------------------------------------
# 5. Select input features and target variable
# ---------------------------------------------------------

# X contains all columns except the value we want to predict.
X = data_for_model.drop("final_score", axis=1)

# y contains the value we want to predict.
y = data_for_model["final_score"]


# ---------------------------------------------------------
# 6. Split data into training and testing sets
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


# ---------------------------------------------------------
# 7. Train simple regression models
# ---------------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42, max_depth=4),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5)
}

results = []

for model_name, model in models.items():
    # Train the model using training data.
    model.fit(X_train, y_train)

    # Predict final scores for testing data.
    y_pred = model.predict(X_test)

    # Calculate model performance.
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": model_name,
        "RMSE": rmse,
        "R2 Score": r2
    })


# ---------------------------------------------------------
# 8. Compare model performance
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

print("\nModel Performance Comparison:")
print(results_df)

print("\nAccuracy note:")
print("Accuracy is mainly used for classification problems.")
print("This project predicts marks, so RMSE and R2 Score are more suitable.")

# Bar chart: RMSE comparison.
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="RMSE", data=results_df, color="salmon")
plt.title("Model Comparison Based on RMSE")
plt.xlabel("Machine Learning Model")
plt.ylabel("RMSE")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/model_rmse_comparison.png")
plt.close()

# Bar chart: R2 score comparison.
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="R2 Score", data=results_df, color="cornflowerblue")
plt.title("Model Comparison Based on R2 Score")
plt.xlabel("Machine Learning Model")
plt.ylabel("R2 Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/model_r2_comparison.png")
plt.close()


# ---------------------------------------------------------
# 9. Show conclusion and basic insights
# ---------------------------------------------------------

best_model_row = results_df.sort_values(by="RMSE").iloc[0]

print("\nConclusion:")
print("1. Students with more study hours generally scored higher marks.")
print("2. Attendance and previous score have a strong relationship with final score.")
print("3. Missing values were handled using mean and mode methods.")
print("4. Categorical values were converted into numeric form using one-hot encoding.")
print(f"5. Based on RMSE, the best model is: {best_model_row['Model']}.")
print("\nAll graph images are saved inside the outputs folder.")
