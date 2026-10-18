"""
Statistical Machine Learning & Pattern Analysis Engine
Topic: Analysis of Student Attendance Patterns & Academic Performance

Library Modules Used:
1. pandas - High-performance data manipulation, tier binning, and aggregation
2. numpy  - Linear algebra, vectorized array operations, and regression modeling
"""

import os
import json
import numpy as np
import pandas as pd

def load_data(filepath):
    """
    Loads student records into a pandas DataFrame.
    (Utilizes 'pandas' library module for fast data parsing and type inference)
    """
    df = pd.read_csv(filepath)
    return df

def calculate_correlations(df):
    """
    Computes Pearson correlation coefficients using numpy and pandas.
    """
    attendance = df["attendance_rate_pct"].to_numpy()
    gpa = df["final_gpa"].to_numpy()
    midterm = df["midterm_score"].to_numpy()
    streak = df["consecutive_absence_streak"].to_numpy()
    study = df["weekly_study_hours"].to_numpy()

    # numpy.corrcoef returns 2x2 correlation matrix
    r_att_gpa = float(np.corrcoef(attendance, gpa)[0, 1])
    r_mid_gpa = float(np.corrcoef(midterm, gpa)[0, 1])
    r_strk_gpa = float(np.corrcoef(streak, gpa)[0, 1])
    r_study_gpa = float(np.corrcoef(study, gpa)[0, 1])

    return {
        "attendance_to_gpa": round(r_att_gpa, 4),
        "midterm_to_gpa": round(r_mid_gpa, 4),
        "streak_to_gpa": round(r_strk_gpa, 4),
        "study_to_gpa": round(r_study_gpa, 4)
    }

def analyze_attendance_tiers(df):
    """
    Uses pandas.cut and groupby to analyze GPA and risk across 5 attendance brackets.
    """
    bins = [0, 65, 75, 85, 95, 100.1]
    labels = [
        "< 65% (Critical)",
        "65% - 74.9% (Warning)",
        "75% - 84.9% (Satisfactory)",
        "85% - 94.9% (Good)",
        ">= 95% (Excellent)"
    ]
    df["tier"] = pd.cut(df["attendance_rate_pct"], bins=bins, labels=labels, right=False)

    tier_stats = []
    grouped = df.groupby("tier", observed=False)
    for name, group in grouped:
        if len(group) > 0:
            high_risk_count = int((group["risk_level"] == "High Risk").sum())
            tier_stats.append({
                "tier": str(name),
                "count": int(len(group)),
                "avg_gpa": round(float(group["final_gpa"].mean()), 2),
                "avg_final_exam": round(float(group["final_exam_score"].mean()), 1),
                "high_risk_pct": round(float((high_risk_count / len(group)) * 100), 1)
            })
    return tier_stats

def analyze_archetypes(df):
    """
    Uses pandas aggregation to analyze behavioral student patterns.
    """
    archetype_stats = []
    total_students = len(df)
    grouped = df.groupby("attendance_pattern")

    for pattern_name, group in grouped:
        high_risk_count = int((group["risk_level"] == "High Risk").sum())
        archetype_stats.append({
            "archetype": str(pattern_name),
            "count": int(len(group)),
            "pct": round(float(len(group) / total_students * 100), 1),
            "avg_attendance": round(float(group["attendance_rate_pct"].mean()), 1),
            "avg_morning_attn": round(float(group["morning_lecture_attendance_pct"].mean()), 1),
            "avg_friday_attn": round(float(group["friday_attendance_pct"].mean()), 1),
            "avg_gpa": round(float(group["final_gpa"].mean()), 2),
            "avg_midterm": round(float(group["midterm_score"].mean()), 1),
            "high_risk_pct": round(float((high_risk_count / len(group)) * 100), 1)
        })

    # Sort descending by average GPA
    archetype_stats.sort(key=lambda x: x["avg_gpa"], reverse=True)
    return archetype_stats

def train_multivariate_regression(df, feature_names):
    """
    Solves Ordinary Least Squares (OLS) regression using numpy linear algebra:
    beta = (X^T X)^(-1) X^T y
    """
    n = len(df)
    split_idx = int(n * 0.8)

    # 1. Feature Matrix with Bias Intercept (numpy.ones)
    X_raw = df[feature_names].to_numpy()
    bias = np.ones((n, 1))
    X = np.hstack([bias, X_raw])
    y = df["final_gpa"].to_numpy()

    # Train / Test Split
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 2. Linear Algebra Solver using numpy.linalg.lstsq
    beta, residuals, rank, s = np.linalg.lstsq(X_train, y_train, rcond=None)

    # 3. Vectorized Evaluation on Test Set
    y_pred = X_test @ beta

    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    ss_res = np.sum((y_test - y_pred) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot))
    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

    # Feature weights
    feature_weights = []
    for idx, f in enumerate(feature_names):
        coef = float(beta[idx + 1])
        feature_weights.append({
            "feature": f,
            "coefficient": round(coef, 4),
            "impact_direction": "Positive" if coef >= 0 else "Negative"
        })

    model_result = {
        "algorithm": "Multivariate Linear Model (NumPy OLS)",
        "r2_score": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "intercept": round(float(beta[0]), 4),
        "weights": {feature_names[i]: round(float(beta[i + 1]), 4) for i in range(len(feature_names))}
    }

    return model_result, feature_weights

def evaluate_models():
    """
    Orchestrates the statistical analytics pipeline and exports JSON metrics.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "student_attendance_performance.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"[pandas] Loading dataset from: {data_path}")
    df = load_data(data_path)
    n = len(df)
    print(f"[pandas] Loaded DataFrame with shape: {df.shape}")

    # 1. Correlations via NumPy
    correlations = calculate_correlations(df)
    print(f"[numpy] Attendance to GPA Pearson Correlation (r): {correlations['attendance_to_gpa']}")
    print(f"[numpy] Absence Streak to GPA Correlation (r): {correlations['streak_to_gpa']}")

    # 2. Tier & Archetype Analytics via Pandas
    tier_stats = analyze_attendance_tiers(df)
    archetype_stats = analyze_archetypes(df)

    # 3. Model Training via NumPy Linear Algebra
    feature_names = [
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

    regression_model, feature_weights = train_multivariate_regression(df, feature_names)
    print(f"[numpy] Model Fit R² Score: {regression_model['r2_score']} | MAE: {regression_model['mae']} | RMSE: {regression_model['rmse']}")

    # 4. Summary Payload
    metrics = {
        "dataset_summary": {
            "total_students": int(n),
            "avg_attendance": round(float(df["attendance_rate_pct"].mean()), 2),
            "avg_gpa": round(float(df["final_gpa"].mean()), 2),
            "avg_midterm": round(float(df["midterm_score"].mean()), 2),
            "avg_final_exam": round(float(df["final_exam_score"].mean()), 2),
            "high_risk_students_pct": round(float((df["risk_level"] == "High Risk").mean() * 100), 2)
        },
        "correlations": correlations,
        "tier_analysis": tier_stats,
        "archetype_analysis": archetype_stats,
        "regression_model": regression_model,
        "feature_weights": feature_weights,
        "sample_students": df.head(120).to_dict(orient="records")
    }

    out_file = os.path.join(models_dir, "model_metrics.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[output] Successfully saved comprehensive model metrics to: {out_file}")
    return metrics

if __name__ == "__main__":
    evaluate_models()
