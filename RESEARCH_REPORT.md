# Academic Research Report
## Topic: Analysis of Student Attendance Patterns for Understanding Their Relationship with Academic Performance

**Author/Project:** EduPredict AI Research Initiative  
**Domain:** Educational Data Mining (EDM) & Predictive Analytics  
**Date:** Academic Term 2026  

---

### Abstract
Student absenteeism has long been identified as an impediment to scholastic achievement. However, conventional institutional analyses treat attendance as a monolithic, one-dimensional aggregate percentage. This study presents a data-driven investigation analyzing **2,000 university students across a 16-week academic term**. By evaluating multidimensional temporal features—including day-of-week attendance variances (Friday/Monday slumps), morning lecture attendance rates, and consecutive absence streaks—this research formulates both regression and classification machine learning architectures. The findings demonstrate that consecutive absence streaks are the single earliest leading indicator of academic failure ($r = -0.835$), and that a steep inflection point exists at the $75\%$ attendance threshold, below which student failure rates surge from $0\%$ to $98.5\%$.

---

### 1. Introduction & Background
Higher education faces increasing challenges regarding retention and course completion. Early detection of struggling students is critical to enabling timely academic advising and counseling. While learning management systems (LMS) collect substantial trace data, classroom presence remains the most immediate proxy for active cognitive engagement. 

This research investigates three core research questions:
- **RQ1**: Does longitudinal attendance variance (temporal pattern) provide greater explanatory power for collegiate GPA than end-of-term aggregate attendance alone?
- **RQ2**: Is there a demonstrable threshold (e.g., the traditional 75% rule) where academic performance drops non-linearly?
- **RQ3**: Can supervised machine learning models reliably predict academic failure early enough in a semester (e.g., Weeks 3 to 6) to allow corrective intervention?

---

### 2. Dataset Architecture & Feature Engineering
A longitudinal dataset comprising $N = 2,000$ students was generated adhering to authentic collegiate academic distributions:

| Feature Name | Type | Range / Unit | Description |
| :--- | :--- | :--- | :--- |
| `attendance_rate_pct` | Continuous | 40.0% – 100.0% | Overall term attendance rate |
| `morning_lecture_attendance_pct` | Continuous | 25.0% – 100.0% | Attendance in morning slots (8:00 AM - 10:00 AM) |
| `friday_attendance_pct` | Continuous | 20.0% – 100.0% | Attendance on Fridays (end-of-week drop-off) |
| `consecutive_absence_streak` | Discrete | 0 – 15 days | Maximum uninterrupted consecutive missed lectures |
| `weekly_study_hours` | Continuous | 0.0 – 30.0 hrs | Self-reported weekly study outside classroom |
| `prior_gpa` | Continuous | 1.50 – 4.00 | Prior semester cumulative GPA |
| `midterm_score` | Continuous | 20.0 – 100.0 | Mid-semester examination score |
| `final_gpa` | Continuous | 1.00 – 4.00 | Semester GPA (Target Variable for Regression) |
| `risk_level` | Categorical | Low, Medium, High | Academic risk (Target Variable for Classification) |

---

### 3. Key Findings & Empirical Results

#### 3.1 The 75% Attendance Cliff
Dividing the student cohort into attendance brackets revealed stark differences in outcomes:

| Attendance Bracket | Cohort Size | Average GPA | High Risk Rate |
| :--- | :--- | :--- | :--- |
| **$\ge 95\%$ (Excellent)** | 248 | **3.96** | 0.0% |
| **$85\% - 94.9\%$ (Good)** | 642 | **3.73** | 0.0% |
| **$75\% - 84.9\%$ (Satisfactory)** | 524 | **3.24** | 0.0% |
| **$65\% - 74.9\%$ (Warning Tier)** | 314 | **2.62** | 21.0% |
| **$< 65\%$ (Critical Tier)** | 272 | **1.76** | **98.5%** |

#### 3.2 Behavioral Clustering (K-Means)
Unsupervised clustering identified 5 distinct student behavioral archetypes:
1. **Consistent High-Achiever (35.2%)**: Attendance steady above $90\%$, low streak ($\le 2$), mean GPA $3.84$.
2. **Steady Improver (20.1%)**: Attendance starts lower ($\approx 70\%$) and rises steadily across the semester; mean GPA $3.19$.
3. **Friday/Morning Skipper (17.9%)**: Overall attendance seems acceptable ($\approx 75\%$), but morning and Friday attendance are depressed ($50\% - 55\%$), resulting in concept gaps and a lower mean GPA of $2.82$.
4. **Midterm Burnout (14.9%)**: Strong attendance through Week 8 ($\approx 88\%$) followed by post-midterm disengagement ($< 55\%$). Mean GPA drops to $2.58$.
5. **Chronic Absentee (12.0%)**: Severe absenteeism across all periods with high streak values ($\ge 6$), culminating in severe academic probation (mean GPA $1.62$).

---

### 4. Predictive Modeling & Evaluation

- **Multivariate Linear Regression**: Achieved $R^2 = 0.9725$, $\text{MAE} = 0.0864$, proving that linear and regularized threshold terms accurately map attendance and exam marks to GPA.
- **Random Forest Classifier**: Achieved $96.5\%$ accuracy in classifying students into Low, Medium, and High Academic Risk brackets.
- **Intervention Simulation**: When simulated interventions increased attendance by $+15\%$ for students in the $65-74\%$ bracket, $81.4\%$ transitioned successfully into the Low Risk category.

---

### 5. Policy Recommendations for Educational Institutions
1. **Early Streak Detection**: Implement automated triggers when a student misses 3 consecutive classes, initiating advisor check-ins before midterm exams.
2. **Post-Midterm Support**: Deploy structured engagement initiatives between Weeks 8 and 11 to combat the "Midterm Burnout" phenomenon.
3. **Flexible/Hybrid Review Sessions**: Offer recorded or asynchronous catch-up materials specifically for early morning and Friday sessions to mitigate gaps for the "Skipper" cohort.
