# %%
# 1. IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

import kagglehub
from kagglehub import KaggleDatasetAdapter


# %%
# 2. LOAD DATASET

file_path = "online_shoppers_intention.csv"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "imakash3011/online-shoppers-purchasing-intention-dataset",
    file_path
)

df.head()


# %%
# 3. INSPECT & CLEAN DATASET

print("Dataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

print("\nSummary statistics:")
print(df.describe())

print("\nRevenue counts:")
print(df["Revenue"].value_counts())

print("\nRevenue percentages:")
print(df["Revenue"].value_counts(normalize=True) * 100)

df["Revenue_numeric"] = df["Revenue"].astype(int)

df[["Revenue", "Revenue_numeric"]].head()

# %%
# 4. DATA PATTERNS & VISUALIZATION
# %%
# Figure 1: Purchase rate by visitor type

visitor_summary = df.groupby("VisitorType").agg(
    Purchase_Rate=("Revenue_numeric", "mean"),
    Session_Count=("Revenue_numeric", "count")
)

visitor_summary["Purchase_Rate"] = (
    visitor_summary["Purchase_Rate"] * 100
)

print(visitor_summary)

visitor_summary["Purchase_Rate"].plot(kind="bar")

plt.title("Purchase Rate by Visitor Type")
plt.xlabel("Visitor Type")
plt.ylabel("Purchase Rate (%)")
plt.xticks(rotation=0)

plt.savefig(
    "figures/figure1_visitor_type.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Figure 2: Purchase rate by traffic type

traffic_summary = df.groupby("TrafficType").agg(
    Purchase_Rate=("Revenue_numeric", "mean"),
    Session_Count=("Revenue_numeric", "count")
)

traffic_summary["Purchase_Rate"] = (
    traffic_summary["Purchase_Rate"] * 100
)

print(traffic_summary)

traffic_summary["Purchase_Rate"].plot(kind="bar")

plt.title("Purchase Rate by Traffic Type")
plt.xlabel("Traffic Type")
plt.ylabel("Purchase Rate (%)")
plt.xticks(rotation=0)

plt.savefig(
    "figures/figure2_traffic_type.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Figure 3: Product-page visits by purchase outcome

sns.boxplot(
    data=df,
    x="Revenue",
    y="ProductRelated"
)

plt.title("Product-Page Visits by Purchase Outcome")
plt.xlabel("Purchase Made")
plt.ylabel("Number of Product-Page Visits")

plt.savefig(
    "figures/figure3_product_page_visits.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


#group averages 
product_visits_summary = (
    df.groupby("Revenue")["ProductRelated"].mean()
)

print(product_visits_summary)

# Figure 4: Product browsing duration by purchase outcome

sns.boxplot(
    data=df,
    x="Revenue",
    y="ProductRelated_Duration"
)

plt.title("Product Browsing Duration by Purchase Outcome")
plt.xlabel("Purchase Made")
plt.ylabel("Product Browsing Duration (Seconds)")

plt.savefig(
    "figures/figure4_product_duration.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


#group averages 
product_duration_summary = (
    df.groupby("Revenue")["ProductRelated_Duration"].mean()
)

print(product_duration_summary)



#%%
# 5. LOGISTIC REGRESSION
predictors = [
    "ProductRelated",
    "ProductRelated_Duration",
    "Administrative",
    "Informational",
    "VisitorType",
    "TrafficType",
    "Month",
    "Weekend"
]

X = df[predictors]
y = df["Revenue_numeric"]

print(X.head())
print(y.head())

#categorical predictors into dummy variables
X = pd.get_dummies(
    X,
    columns=["VisitorType", "TrafficType", "Month"],
    drop_first=True
)

#predictors to numeric values
X = X.astype(float)

print(X.head())
print("Number of model features:", X.shape[1])

# Split the dataset into 80% training and 20% testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))

print("\nTraining purchase rate:")
print(y_train.mean())

print("\nTesting purchase rate:")
print(y_test.mean())

# Standardize numerical predictors

numeric_features = [
    "ProductRelated",
    "ProductRelated_Duration",
    "Administrative",
    "Informational"
]

scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

log_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

log_model.fit(X_train, y_train)
print("Logistic regression model fitted.")

#%% 
# 6. MODEL EVALUATION
y_pred = log_model.predict(X_test)
y_prop = log_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
average_precision = average_precision_score(y_test, y_prop)

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("Average Precision Score:", round(average_precision, 4))

#no-purchase baseline
baseline_accuracy = (y_test == 0).mean()
print("Baseline Accuracy (No Purchase):", round(baseline_accuracy, 4))

# Figure 5: Confusion matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Outcome")
plt.ylabel("Actual Outcome")

plt.savefig(
    "figures/figure5_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# %%
# 5-FOLD CROSS-VALIDATION
X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), numeric_features)
    ],
    remainder="passthrough"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

param_grid = {
    "model__class_weight": [
        None,
        "balanced"
    ]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
)

grid_search.fit(X_train_cv, y_train_cv)
print("Best Parameters:", grid_search.best_params_)
print("Best F1 Score:", round(grid_search.best_score_, 4))

# %%
# FINAL MODEL EVALUATION
final_model = grid_search.best_estimator_
y_pred_final = final_model.predict(X_test_cv)
y_prob_final = final_model.predict_proba(X_test_cv)[:, 1]

accuracy_final = accuracy_score(y_test_cv, y_pred_final)
precision_final = precision_score(y_test_cv, y_pred_final)
recall_final = recall_score(y_test_cv, y_pred_final)
f1_final = f1_score(y_test_cv, y_pred_final)
average_precision_final = average_precision_score(y_test_cv, y_prob_final)

print("Final Accuracy:", round(accuracy_final, 4))
print("Final Precision:", round(precision_final, 4))
print("Final Recall:", round(recall_final, 4))
print("Final F1 Score:", round(f1_final, 4))
print("Final Average Precision Score:", round(average_precision_final, 4))

# baseline
baseline_accuracy_final = (y_test_cv == 0).mean()
print("Final Baseline Accuracy (No Purchase):", round(baseline_accuracy_final, 4))

# UPDATED: Figure 5: Final model confusion matrix

cm_final = confusion_matrix(
    y_test_cv,
    y_pred_final
)

print("Final Confusion Matrix:")
print(cm_final)

sns.heatmap(
    cm_final,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Balanced Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Outcome")
plt.ylabel("Actual Outcome")

plt.savefig(
    "figures/figure5_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Figure 6: Final model precision-recall curve

curve_precision, curve_recall, curve_thresholds = precision_recall_curve(
    y_test_cv,
    y_prob_final
)

print("Number of thresholds:", len(curve_thresholds))

plt.figure(figsize=(8, 6))

plt.plot(
    curve_recall,
    curve_precision
)

plt.title("Balanced Logistic Regression Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")

plt.savefig(
    "figures/figure6_precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()