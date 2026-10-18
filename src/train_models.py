"""
Machine Learning Training Pipeline:
Attendance Pattern Analysis & Academic Performance Prediction

Models:
1. Linear Regression (Baseline Interpretability)
2. Random Forest Regressor (GPA Prediction)
3. Logistic Regression & Random Forest Classifier (At-Risk Early Warning)
4. K-Means Clustering (Behavioral Pattern Clustering)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

def run_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "student_attendance_performance.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns.")
    
    # 1. Statistical Analysis
    corr_matrix = df[[
        "attendance_rate_pct", "morning_lecture_attendance_pct", "friday_attendance_pct",
        "consecutive_absence_streak", "weekly_study_hours", "assignment_completion_pct",
        "lms_weekly_logins", "prior_gpa", "midterm_score", "final_exam_score", "final_gpa"
    ]].corr()
    
    attendance_gpa_corr = float(corr_matrix.loc["attendance_rate_pct", "final_gpa"])
    midterm_gpa_corr = float(corr_matrix.loc["midterm_score", "final_gpa"])
    streak_gpa_corr = float(corr_matrix.loc["consecutive_absence_streak", "final_gpa"])
    
    print(f"-> Pearson Correlation (Attendance vs Final GPA): {attendance_gpa_corr:.4f}")
    print(f"-> Pearson Correlation (Absence Streak vs Final GPA): {streak_gpa_corr:.4f}")
    
    # Tier breakdown
    tiers = [
        {"name": "< 65% (Critical)", "min": 0, "max": 65},
        {"name": "65% - 74.9% (Warning)", "min": 65, "max": 75},
        {"name": "75% - 84.9% (Satisfactory)", "min": 75, "max": 85},
        {"name": "85% - 94.9% (Good)", "min": 85, "max": 95},
        {"name": ">= 95% (Excellent)", "min": 95, "max": 100.1}
    ]
    
    tier_stats = []
    for t in tiers:
        subset = df[(df["attendance_rate_pct"] >= t["min"]) & (df["attendance_rate_pct"] < t["max"])]
        if len(subset) > 0:
            tier_stats.append({
                "tier": t["name"],
                "count": int(len(subset)),
                "avg_gpa": round(float(subset["final_gpa"].mean()), 2),
                "avg_final_exam": round(float(subset["final_exam_score"].mean()), 1),
                "high_risk_pct": round(float((subset["risk_level"] == "High Risk").mean() * 100), 1)
            })
            
    # 2. Features for Regression
    features = [
        "attendance_rate_pct",
        "morning_lecture_attendance_pct",
        "friday_attendance_pct",
        "consecutive_absence_streak",
        "weekly_study_hours",
        "assignment_completion_pct",
        "lms_weekly_logins",
        "prior_gpa",
        "midterm_score"
    ]
    
    X = df[features]
    y_reg = df["final_gpa"]
    y_clf = df["risk_level"]
    
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    
    # 3. GPA Regression Models
    # A) Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train_reg)
    lr_preds = lr.predict(X_test)
    lr_r2 = float(r2_score(y_test_reg, lr_preds))
    lr_mae = float(mean_absolute_error(y_test_reg, lr_preds))
    lr_rmse = float(np.sqrt(mean_squared_error(y_test_reg, lr_preds)))
    
    # B) Random Forest Regressor
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf_reg.fit(X_train, y_train_reg)
    rf_preds = rf_reg.predict(X_test)
    rf_r2 = float(r2_score(y_test_reg, rf_preds))
    rf_mae = float(mean_absolute_error(y_test_reg, rf_preds))
    rf_rmse = float(np.sqrt(mean_squared_error(y_test_reg, rf_preds)))
    
    # Feature importances
    feature_importances = []
    for f, imp, coef in zip(features, rf_reg.feature_importances_, lr.coef_):
        feature_importances.append({
            "feature": f,
            "rf_importance": round(float(imp) * 100, 2),
            "linear_coefficient": round(float(coef), 4)
        })
    feature_importances.sort(key=lambda x: x["rf_importance"], reverse=True)
    
    # 4. Classification: Early Warning Risk Predictor
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_clf.fit(X_train, y_train_clf)
    clf_preds = rf_clf.predict(X_test)
    clf_acc = float(accuracy_score(y_test_clf, clf_preds))
    
    classes = list(rf_clf.classes_)
    cm = confusion_matrix(y_test_clf, clf_preds, labels=classes)
    
    # 5. K-Means Clustering on Attendance Behaviors
    cluster_features = [
        "attendance_rate_pct",
        "morning_lecture_attendance_pct",
        "friday_attendance_pct",
        "consecutive_absence_streak"
    ]
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(df[cluster_features])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["cluster_id"] = kmeans.fit_predict(X_cluster_scaled)
    
    cluster_profiles = []
    for c_id in range(5):
        c_subset = df[df["cluster_id"] == c_id]
        cluster_profiles.append({
            "cluster_id": int(c_id),
            "size": int(len(c_subset)),
            "pct_of_total": round(float(len(c_subset) / len(df) * 100), 1),
            "avg_attendance": round(float(c_subset["attendance_rate_pct"].mean()), 1),
            "avg_morning_attn": round(float(c_subset["morning_lecture_attendance_pct"].mean()), 1),
            "avg_friday_attn": round(float(c_subset["friday_attendance_pct"].mean()), 1),
            "avg_streak": round(float(c_subset["consecutive_absence_streak"].mean()), 1),
            "avg_gpa": round(float(c_subset["final_gpa"].mean()), 2),
            "most_common_pattern": str(c_subset["attendance_pattern"].mode()[0])
        })
    
    # 6. Save Model Metrics & Summary
    results = {
        "dataset_summary": {
            "total_students": len(df),
            "avg_attendance": round(float(df["attendance_rate_pct"].mean()), 2),
            "avg_gpa": round(float(df["final_gpa"].mean()), 2),
            "avg_midterm": round(float(df["midterm_score"].mean()), 2),
            "avg_final_exam": round(float(df["final_exam_score"].mean()), 2),
            "high_risk_students_pct": round(float((df["risk_level"] == "High Risk").mean() * 100), 2)
        },
        "correlations": {
            "attendance_to_gpa": round(attendance_gpa_corr, 4),
            "midterm_to_gpa": round(midterm_gpa_corr, 4),
            "streak_to_gpa": round(streak_gpa_corr, 4)
        },
        "tier_analysis": tier_stats,
        "regression_models": {
            "linear_regression": {
                "r2_score": round(lr_r2, 4),
                "mae": round(lr_mae, 4),
                "rmse": round(lr_rmse, 4),
                "intercept": round(float(lr.intercept_), 4)
            },
            "random_forest": {
                "r2_score": round(rf_r2, 4),
                "mae": round(rf_mae, 4),
                "rmse": round(rf_rmse, 4)
            }
        },
        "classification_model": {
            "algorithm": "Random Forest Classifier",
            "accuracy": round(clf_acc * 100, 2),
            "classes": classes,
            "confusion_matrix": cm.tolist()
        },
        "feature_importances": feature_importances,
        "clusters": cluster_profiles,
        "sample_students": df.head(100).to_dict(orient="records")
    }
    
    metrics_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Exported metrics and model data to {metrics_path}")
    
    # Save scikit-learn models for inference
    joblib.dump(lr, os.path.join(models_dir, "linear_regressor.joblib"))
    joblib.dump(rf_reg, os.path.join(models_dir, "rf_regressor.joblib"))
    joblib.dump(rf_clf, os.path.join(models_dir, "rf_classifier.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "cluster_scaler.joblib"))
    joblib.dump(kmeans, os.path.join(models_dir, "kmeans_cluster.joblib"))
    print("Saved all trained models (.joblib) to models/ directory.")

if __name__ == "__main__":
    run_pipeline()
