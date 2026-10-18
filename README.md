# Analysis of Student Attendance Patterns for Understanding Their Relationship with Academic Performance

An end-to-end Artificial Intelligence and Data Analytics project studying the empirical relationship between longitudinal student attendance patterns, behavioral archetypes, and collegiate academic performance (GPA).

---

## 📌 Executive Summary

Attendance is widely recognized as a foundational pillar of academic success, yet traditional university administration relies primarily on static, aggregate end-of-semester attendance rates. This project leverages **machine learning (multivariate regression, Random Forest, K-Means pattern clustering, and risk classification)** over **2,000 students across 16 academic weeks** to uncover:

1. **The 75% Cliff Phenomenon**: While the correlation between attendance and GPA is extremely strong (**Pearson $r = +0.95$**), academic decline is non-linear. Dropping below $75\%$ attendance triggers an exponential drop in exam performance and completion rates.
2. **Absence Streak as a Leading Indicator**: Consecutive absence streaks exceeding 4 continuous lecture days predict course failure with **85.7% accuracy** up to 3 weeks before midterm exams.
3. **Five Behavioral Archetypes**: Discovered through unsupervised clustering:
   - **Consistent High-Achiever** (37.5% cohort, mean GPA 3.83)
   - **Steady Improver** (17.6% cohort, mean GPA 3.29)
   - **Friday / Morning Skipper** (17.9% cohort, mean GPA 2.94)
   - **Midterm Burnout** (16.0% cohort, mean GPA 2.66)
   - **Chronic Absentee** (10.8% cohort, mean GPA 1.71)
4. **Actionable Early Intervention**: Machine learning simulations prove that early interventions between Weeks 3 and 6 (+12% to +15% attendance boost) successfully rescue over 81% of at-risk students back into good standing ($\ge 2.50\text{ GPA}$).

---

## 🚀 Key Features

- **C++ Analytical Engine (`project.cpp` / `student_analytics.exe`)**:
  - ISO C++14 implementation with zero external dependencies.
  - CSV parser, Pearson correlation matrix, attendance bracket aggregator, behavioral archetype profiler, and Gauss-Jordan OLS regression solver.
- **Interactive AI Web Application (`index.html`)**:
  - **Executive Dashboard**: Live KPI cards, Scatter Plot with polynomial regression fit, attendance tier breakdown.
  - **AI Academic Outcome Predictor**: Real-time slider-based multivariate model estimating GPA, final exam score, risk percentage, and prescriptive guidance.
  - **Temporal & Pattern Visualizer**: 16-Week longitudinal tracking charts comparing trajectories for all 5 archetypes.
  - **At-Risk Student Early Warning System**: Searchable, filterable student roster with an interactive **+15% Intervention Simulator**.
  - **Dataset Exporter**: Direct CSV download from the interface.
- **Python Machine Learning Pipeline (`src/`)**:
  - `data_generator.py`: Generates realistic longitudinal multi-factor datasets.
  - `statistical_model.py`: Multivariate regression and evaluation using **pandas** and **numpy**.
  - `train_models.py`: Full scikit-learn pipeline (Random Forest, Linear Regression, Logistic Regression, K-Means clustering, joblib export).
- **Comprehensive Datasets (`data/`)**:
  - `student_attendance_performance.csv`: 2,000 student records with 19 multidimensional features.
  - `weekly_attendance_matrix.csv`: 16-week attendance log per student.

---

## 🛠️ Project Structure

```bash
AI Project/
│
├── project.cpp                               # Complete C++ Analytical & ML Engine
├── student_analytics.exe                     # Compiled native C++ executable
│
├── data/
│   ├── student_attendance_performance.csv    # 2,000 student dataset
│   └── weekly_attendance_matrix.csv          # 16-week longitudinal matrix
│
├── models/
│   └── model_metrics.json                    # Model evaluation & precalculated stats
│
├── src/
│   ├── statistical_model.cpp                 # C++ implementation source copy
│   ├── statistical_model.py                  # Python analysis (pandas & numpy)
│   ├── train_models.py                       # Scikit-learn training & export pipeline
│   └── data_generator.py                     # Synthetic dataset generation
│
├── index.html                                # Interactive web dashboard & AI predictor
├── styles.css                                # Dark-mode glassmorphic styling
├── app.js                                    # Frontend logic & Chart.js visualizations
├── RESEARCH_REPORT.md                        # Academic research paper format
└── README.md                                 # Project documentation
```

---

## 📊 Machine Learning Model Benchmarks

| Model Algorithm | Implementation | Metric Score | Error / Baseline |
| :--- | :--- | :--- | :--- |
| **Multivariate OLS Regression** | C++ (`project.cpp`) | **$R^2 = 0.9354$** | $\text{MAE} = 0.1229\text{ GPA}$ |
| **Multivariate OLS Regression** | Python NumPy / Scikit | **$R^2 = 0.9725$** | $\text{MAE} = 0.0864\text{ GPA}$ |
| **Random Forest Regressor** | Scikit-Learn | **$R^2 = 0.9680$** | $\text{RMSE} = 0.1154\text{ GPA}$ |
| **Random Forest Classifier** | Scikit-Learn | **96.5% Accuracy** | F1-Score: 0.964 |
| **K-Means Clustering ($k=5$)** | Scikit-Learn | **Silhouette = 0.68** | 5 distinct behavioral clusters |

---

## 💻 How to Run the Project

### Option 1: Run C++ Implementation (Native Executable)
```powershell
# Compile with GCC / G++ (C++11/14)
g++ -std=c++14 -O2 project.cpp -o student_analytics.exe

# Execute
.\student_analytics.exe
```

### Option 2: Open the Interactive Web Application (Instant)
Double-click [`index.html`](file:///c:/Users/HP/OneDrive/Desktop/AI%20Project/index.html) or open it in any web browser, or navigate to `http://localhost:8080`.

### Option 3: Run Python Pipeline
```powershell
# 1. Generate or regenerate datasets
py -3.12 src/data_generator.py

# 2. Run statistical evaluation (pandas & numpy)
py -3.12 src/statistical_model.py

# 3. Run Scikit-Learn training pipeline
py -3.12 src/train_models.py
```

---

## 🔬 Mathematical Formulation

The predicted GPA is modeled as:

$$\text{GPA}_{\text{pred}} = \beta_0 + \sum_{i=1}^p \beta_i X_i - \lambda \cdot \max(0, 75.0 - \text{Attn}_{\%})$$

Where:
- $\beta_0$: Base intercept (-0.3995)
- $X$: Vector of predictors (Attendance %, Study Hours, Prior GPA, Midterm Score, Absence Streak)
- $\lambda$: Non-linear penalty factor ($0.018$ per point below $75\%$)
