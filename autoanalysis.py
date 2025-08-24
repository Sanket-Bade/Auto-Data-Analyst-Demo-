import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Advanced Auto ML", layout="wide")
st.title("Advanced Auto ML Data Analyst - Robust & Notebook Style")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Data Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    # ---------------- 1️⃣ Data Cleaning ----------------
    with st.expander("1️⃣ Data Cleaning", expanded=True):
        st.write("**Numeric Columns:**", list(numeric_cols))
        st.write("**Categorical Columns:**", list(categorical_cols))

        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
            # Outlier clipping
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr
            df[col].clip(lower=lower_limit, upper=upper_limit, inplace=True)

        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)

        st.success("Missing values handled and outliers clipped.")

    # ---------------- 2️⃣ Problem Type Detection ----------------
    with st.expander("2️⃣ Problem Detection", expanded=False):
        target_col = st.selectbox("Select Target Variable", df.columns)
        if target_col in numeric_cols:
            problem_type = "regression"
        else:
            problem_type = "classification"
        st.write(f"Detected Problem Type: **{problem_type.upper()}**")

    # ---------------- 3️⃣ Feature Selection ----------------
    with st.expander("3️⃣ Feature Selection", expanded=False):
        feature_cols = list(st.multiselect("Select Feature Columns", [c for c in df.columns if c != target_col],
                                           default=[c for c in df.columns if c != target_col]))
        if not feature_cols:
            st.info("No features selected. Skipping model training.")

    # ---------------- Encode categorical features ----------------
    X = df[feature_cols].copy() if feature_cols else pd.DataFrame()
    y = df[target_col].copy() if feature_cols else pd.Series(dtype='float')

    le_dict = {}
    if problem_type == "classification":
        if y.dtype == "object" or str(y.dtype).startswith("category"):
            le_y = LabelEncoder()
            y = le_y.fit_transform(y)
            le_dict[target_col] = le_y

    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype).startswith("category"):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            le_dict[col] = le

    # ---------------- 4️⃣ Train/Test Split ----------------
    if not X.empty:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Determine adaptive CV
        if problem_type == "classification":
            unique_counts = np.bincount(y_train)
            min_class_samples = min(unique_counts)
            cv = min(3, min_class_samples) if min_class_samples > 1 else 2
            cv_strategy = StratifiedKFold(n_splits=cv)
        else:
            cv = min(3, len(y_train))
            cv_strategy = KFold(n_splits=cv)

        # ---------------- 5️⃣ Model Training, Tuning & Evaluation ----------------
        with st.expander("4️⃣ Model Training, Hyperparameter Tuning & Evaluation", expanded=False):
            results = []
            feature_importances = {}

            if problem_type == "regression":
                models = {
                    "Linear Regression": (LinearRegression(), {}),
                    "Ridge": (Ridge(), {"alpha": [0.01, 0.1, 1, 10]}),
                    "Lasso": (Lasso(), {"alpha": [0.01, 0.1, 1, 10]}),
                    "Decision Tree": (DecisionTreeRegressor(), {"max_depth": [3, 5, 7, None]}),
                    "Random Forest": (RandomForestRegressor(), {"n_estimators": [50, 100], "max_depth": [3, 5, None]}),
                    "Gradient Boosting": (GradientBoostingRegressor(), {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]})
                }

                for name, (model, params) in models.items():
                    if params and len(y_train) >= cv:
                        grid = GridSearchCV(model, params, cv=cv_strategy, scoring='r2')
                        grid.fit(X_train, y_train)
                        best_model = grid.best_estimator_
                    else:
                        best_model = model.fit(X_train, y_train)

                    preds = best_model.predict(X_test)
                    r2 = r2_score(y_test, preds)
                    mae = mean_absolute_error(y_test, preds)
                    rmse = mean_squared_error(y_test, preds, squared=False)
                    results.append((name, r2, mae, rmse))

                    # Feature importance for tree-based
                    if hasattr(best_model, "feature_importances_"):
                        feature_importances[name] = best_model.feature_importances_

                res_df = pd.DataFrame(results, columns=["Model", "R2 Score", "MAE", "RMSE"])
                st.write("### Regression Model Performance")
                st.dataframe(res_df.sort_values(by="R2 Score", ascending=False))

                best_model_name = res_df.sort_values(by="R2 Score", ascending=False).iloc[0]["Model"]
                st.success(f"Best Model: **{best_model_name}**")

                # Plot Actual vs Predicted
                best_model = models[best_model_name][0].fit(X_train, y_train) if not models[best_model_name][1] else GridSearchCV(models[best_model_name][0], models[best_model_name][1], cv=cv_strategy).fit(X_train, y_train).best_estimator_
                preds_best = best_model.predict(X_test)
                fig, ax = plt.subplots(figsize=(5,3))
                ax.scatter(y_test, preds_best, color='blue', s=20)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title(f"Actual vs Predicted - {best_model_name}")
                fig.tight_layout()
                st.pyplot(fig)

                # Feature importance plot
                if best_model_name in feature_importances:
                    fig, ax = plt.subplots(figsize=(5,3))
                    importance = feature_importances[best_model_name]
                    sns.barplot(x=importance, y=X.columns, ax=ax)
                    ax.set_title(f"Feature Importance - {best_model_name}")
                    fig.tight_layout()
                    st.pyplot(fig)

            else:  # classification
                models = {
                    "Logistic Regression": (LogisticRegression(max_iter=1000), {"C": [0.01, 0.1, 1, 10]}),
                    "Decision Tree": (DecisionTreeClassifier(), {"max_depth": [3, 5, 7, None]}),
                    "Random Forest": (RandomForestClassifier(), {"n_estimators": [50, 100], "max_depth": [3, 5, None]}),
                    "Gradient Boosting": (GradientBoostingClassifier(), {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]})
                }

                for name, (model, params) in models.items():
                    if params and len(y_train) >= cv:
                        grid = GridSearchCV(model, params, cv=cv_strategy, scoring='accuracy')
                        grid.fit(X_train, y_train)
                        best_model = grid.best_estimator_
                    else:
                        best_model = model.fit(X_train, y_train)

                    preds = best_model.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average="weighted")
                    results.append((name, acc, f1))

                    # Feature importance for tree-based
                    if hasattr(best_model, "feature_importances_"):
                        feature_importances[name] = best_model.feature_importances_

                res_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1 Score"])
                st.write("### Classification Model Performance")
                st.dataframe(res_df.sort_values(by="Accuracy", ascending=False))

                best_model_name = res_df.sort_values(by="Accuracy", ascending=False).iloc[0]["Model"]
                st.success(f"Best Model: **{best_model_name}**")

                # Confusion Matrix
                best_model = models[best_model_name][0].fit(X_train, y_train) if not models[best_model_name][1] else GridSearchCV(models[best_model_name][0], models[best_model_name][1], cv=cv_strategy).fit(X_train, y_train).best_estimator_
                preds_best = best_model.predict(X_test)
                cm = confusion_matrix
