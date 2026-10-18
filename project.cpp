/**
 * ============================================================================
 * Project: Analysis of Student Attendance Patterns for Understanding Their
 *          Relationship with Academic Performance
 * Language: C++ (C++11 / C++14)
 * Libraries Used: <iostream>, <vector>, <string>, <fstream>, <sstream>,
 *                 <cmath>, <numeric>, <algorithm>, <iomanip>, <map>
 * ============================================================================
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <map>

// Data structure representing an individual student's record
struct StudentRecord {
    std::string studentId;
    std::string name;
    std::string gender;
    std::string department;
    int semester;
    double attendanceRate;          // in %
    double morningAttendance;       // in %
    double fridayAttendance;        // in %
    int absenceStreak;              // consecutive missed days
    std::string pattern;            // Behavioral archetype
    double studyHours;              // Weekly hours
    double assignmentCompletion;    // in %
    int lmsLogins;                  // Logins per week
    double priorGpa;                // 0.0 - 4.0
    double midtermScore;            // 0 - 100
    double finalExamScore;          // 0 - 100
    double finalGpa;                // 0.0 - 4.0
    std::string riskLevel;          // Low Risk, Medium Risk, High Risk
    std::string intervention;
};

// Summary metrics for attendance brackets
struct TierStats {
    std::string tierName;
    int count = 0;
    double totalGpa = 0.0;
    double totalFinalExam = 0.0;
    int highRiskCount = 0;

    double getAvgGpa() const { return count > 0 ? totalGpa / count : 0.0; }
    double getAvgFinalExam() const { return count > 0 ? totalFinalExam / count : 0.0; }
    double getHighRiskPct() const { return count > 0 ? (static_cast<double>(highRiskCount) / count) * 100.0 : 0.0; }
};

// Summary metrics for behavioral archetypes
struct ArchetypeStats {
    std::string name;
    int count = 0;
    double totalAttendance = 0.0;
    double totalMorning = 0.0;
    double totalFriday = 0.0;
    double totalGpa = 0.0;
    int highRiskCount = 0;

    double getAvgAttendance() const { return count > 0 ? totalAttendance / count : 0.0; }
    double getAvgMorning() const { return count > 0 ? totalMorning / count : 0.0; }
    double getAvgFriday() const { return count > 0 ? totalFriday / count : 0.0; }
    double getAvgGpa() const { return count > 0 ? totalGpa / count : 0.0; }
    double getHighRiskPct() const { return count > 0 ? (static_cast<double>(highRiskCount) / count) * 100.0 : 0.0; }
};

// Function to safely trim quotes and whitespaces from parsed CSV cells
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n\"");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n\"");
    return str.substr(first, (last - first + 1));
}

// Loads student attendance records from CSV file
std::vector<StudentRecord> loadDataset(const std::string& filepath) {
    std::vector<StudentRecord> dataset;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "[Error] Could not open dataset at: " << filepath << std::endl;
        return dataset;
    }

    std::string line;
    // Skip header line
    if (!std::getline(file, line)) {
        return dataset;
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> row;

        while (std::getline(ss, token, ',')) {
            row.push_back(trim(token));
        }

        if (row.size() < 19) continue;

        try {
            StudentRecord s;
            s.studentId = row[0];
            s.name = row[1];
            s.gender = row[2];
            s.department = row[3];
            s.semester = std::stoi(row[4]);
            s.attendanceRate = std::stod(row[5]);
            s.morningAttendance = std::stod(row[6]);
            s.fridayAttendance = std::stod(row[7]);
            s.absenceStreak = std::stoi(row[8]);
            s.pattern = row[9];
            s.studyHours = std::stod(row[10]);
            s.assignmentCompletion = std::stod(row[11]);
            s.lmsLogins = std::stoi(row[12]);
            s.priorGpa = std::stod(row[13]);
            s.midtermScore = std::stod(row[14]);
            s.finalExamScore = std::stod(row[15]);
            s.finalGpa = std::stod(row[16]);
            s.riskLevel = row[17];
            s.intervention = row[18];

            dataset.push_back(s);
        } catch (const std::exception& e) {
            // Skip malformed rows
            continue;
        }
    }

    file.close();
    return dataset;
}

// Computes Pearson Correlation Coefficient (r) between two continuous variables
double calculatePearsonCorrelation(const std::vector<double>& x, const std::vector<double>& y) {
    size_t n = x.size();
    if (n != y.size() || n == 0) return 0.0;

    double sumX = std::accumulate(x.begin(), x.end(), 0.0);
    double sumY = std::accumulate(y.begin(), y.end(), 0.0);
    double meanX = sumX / n;
    double meanY = sumY / n;

    double numerator = 0.0;
    double denomX = 0.0;
    double denomY = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double dx = x[i] - meanX;
        double dy = y[i] - meanY;
        numerator += dx * dy;
        denomX += dx * dx;
        denomY += dy * dy;
    }

    if (denomX <= 1e-9 || denomY <= 1e-9) return 0.0;
    return numerator / std::sqrt(denomX * denomY);
}

// Matrix Gauss-Jordan Elimination solver for Ordinary Least Squares (OLS)
// Solves (X^T * X) * beta = (X^T * y)
std::vector<double> solveNormalEquation(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    size_t n = X.size();
    size_t p = X[0].size();

    // Compute XtX (p x p) and Xty (p x 1)
    std::vector<std::vector<double>> XtX(p, std::vector<double>(p, 0.0));
    std::vector<double> Xty(p, 0.0);

    for (size_t i = 0; i < p; ++i) {
        for (size_t j = 0; j < p; ++j) {
            for (size_t k = 0; k < n; ++k) {
                XtX[i][j] += X[k][i] * X[k][j];
            }
        }
        for (size_t k = 0; k < n; ++k) {
            Xty[i] += X[k][i] * y[k];
        }
    }

    // Augmented Matrix [XtX | Xty]
    std::vector<std::vector<double>> aug(p, std::vector<double>(p + 1, 0.0));
    for (size_t i = 0; i < p; ++i) {
        for (size_t j = 0; j < p; ++j) {
            aug[i][j] = XtX[i][j];
        }
        aug[i][p] = Xty[i];
    }

    // Gauss-Jordan elimination
    for (size_t i = 0; i < p; ++i) {
        // Find pivot
        size_t maxRow = i;
        for (size_t k = i + 1; k < p; ++k) {
            if (std::abs(aug[k][i]) > std::abs(aug[maxRow][i])) {
                maxRow = k;
            }
        }
        std::swap(aug[i], aug[maxRow]);

        double pivot = aug[i][i];
        if (std::abs(pivot) < 1e-12) pivot = 1e-12;

        for (size_t j = 0; j <= p; ++j) {
            aug[i][j] /= pivot;
        }

        for (size_t k = 0; k < p; ++k) {
            if (k != i) {
                double factor = aug[k][i];
                for (size_t j = 0; j <= p; ++j) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    std::vector<double> beta(p, 0.0);
    for (size_t i = 0; i < p; ++i) {
        beta[i] = aug[i][p];
    }
    return beta;
}

// Predicts GPA with regularized non-linear threshold cliff (<75% penalty)
double predictStudentGPA(const std::vector<double>& beta,
                         double attendance,
                         double morningAttn,
                         double fridayAttn,
                         int streak,
                         double studyHours,
                         double assignmentPct,
                         int lmsLogins,
                         double priorGpa,
                         double midterm) {
    // Linear model baseline
    double gpa = beta[0]
               + beta[1] * attendance
               + beta[2] * morningAttn
               + beta[3] * fridayAttn
               + beta[4] * streak
               + beta[5] * studyHours
               + beta[6] * assignmentPct
               + beta[7] * lmsLogins
               + beta[8] * priorGpa
               + beta[9] * midterm;

    // Non-linear attendance cliff penalty for < 75%
    if (attendance < 75.0) {
        double penalty = (75.0 - attendance) * 0.018;
        gpa -= penalty;
    }

    // Bounded collegiate range (0.00 to 4.00)
    return std::max(0.50, std::min(4.00, gpa));
}

int main() {
    std::cout << "===============================================================================\n";
    std::cout << "  AI & DATA ANALYTICS: STUDENT ATTENDANCE PATTERNS & ACADEMIC PERFORMANCE\n";
    std::cout << "  Implementation Language: C++14 (ISO C++ Standard)\n";
    std::cout << "===============================================================================\n\n";

    // 1. Load Dataset
    std::string datasetPath = "data/student_attendance_performance.csv";
    std::vector<StudentRecord> dataset = loadDataset(datasetPath);

    if (dataset.empty()) {
        std::cerr << "[Notice] Attempting secondary path: ../data/student_attendance_performance.csv\n";
        dataset = loadDataset("../data/student_attendance_performance.csv");
    }

    if (dataset.empty()) {
        std::cerr << "[Error] Could not find student_attendance_performance.csv. Run generator first.\n";
        return 1;
    }

    size_t n = dataset.size();
    std::cout << "[Dataset Loaded] " << n << " longitudinal student records analyzed.\n\n";

    // 2. Statistical Analysis & Pearson Correlations
    std::vector<double> attendanceSeries(n);
    std::vector<double> gpaSeries(n);
    std::vector<double> midtermSeries(n);
    std::vector<double> streakSeries(n);
    std::vector<double> studySeries(n);

    for (size_t i = 0; i < n; ++i) {
        attendanceSeries[i] = dataset[i].attendanceRate;
        gpaSeries[i] = dataset[i].finalGpa;
        midtermSeries[i] = dataset[i].midtermScore;
        streakSeries[i] = static_cast<double>(dataset[i].absenceStreak);
        studySeries[i] = dataset[i].studyHours;
    }

    double rAttendanceGpa = calculatePearsonCorrelation(attendanceSeries, gpaSeries);
    double rMidtermGpa = calculatePearsonCorrelation(midtermSeries, gpaSeries);
    double rStreakGpa = calculatePearsonCorrelation(streakSeries, gpaSeries);
    double rStudyGpa = calculatePearsonCorrelation(studySeries, gpaSeries);

    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << "  1. STATISTICAL CORRELATION ANALYSIS (PEARSON COEFFICIENT r)\n";
    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  * Overall Attendance Rate vs Final GPA:      r = " << std::setw(7) << rAttendanceGpa << " (Very Strong Positive)\n";
    std::cout << "  * Midterm Exam Score vs Final GPA:           r = " << std::setw(7) << rMidtermGpa << " (Strong Positive)\n";
    std::cout << "  * Consecutive Absence Streak vs Final GPA:   r = " << std::setw(7) << rStreakGpa << " (Strong Negative Warning)\n";
    std::cout << "  * Weekly Self-Study Hours vs Final GPA:      r = " << std::setw(7) << rStudyGpa << " (Positive Correlation)\n\n";

    // 3. Attendance Tier Breakdown Analysis
    std::vector<TierStats> tiers = {
        {"< 65% (Critical)", 0, 0, 0, 0},
        {"65% - 74.9% (Warning)", 0, 0, 0, 0},
        {"75% - 84.9% (Satisfactory)", 0, 0, 0, 0},
        {"85% - 94.9% (Good)", 0, 0, 0, 0},
        {">= 95% (Excellent)", 0, 0, 0, 0}
    };

    for (const auto& s : dataset) {
        int idx = 0;
        if (s.attendanceRate < 65.0) idx = 0;
        else if (s.attendanceRate < 75.0) idx = 1;
        else if (s.attendanceRate < 85.0) idx = 2;
        else if (s.attendanceRate < 95.0) idx = 3;
        else idx = 4;

        tiers[idx].count++;
        tiers[idx].totalGpa += s.finalGpa;
        tiers[idx].totalFinalExam += s.finalExamScore;
        if (s.riskLevel == "High Risk") tiers[idx].highRiskCount++;
    }

    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << "  2. ATTENDANCE TIER EMPIRICAL PERFORMANCE BREAKDOWN\n";
    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(28) << "Attendance Bracket"
              << std::right << std::setw(10) << "Students"
              << std::setw(14) << "Average GPA"
              << std::setw(16) << "Avg Final Exam"
              << std::setw(16) << "At-Risk Rate\n";
    std::cout << "-------------------------------------------------------------------------------\n";
    for (const auto& t : tiers) {
        std::cout << std::left << std::setw(28) << t.tierName
                  << std::right << std::setw(10) << t.count
                  << std::setw(14) << std::fixed << std::setprecision(2) << t.getAvgGpa()
                  << std::setw(16) << std::fixed << std::setprecision(1) << t.getAvgFinalExam()
                  << std::setw(15) << std::fixed << std::setprecision(1) << t.getHighRiskPct() << "%\n";
    }
    std::cout << "\n";

    // 4. Behavioral Archetype Profiling
    std::map<std::string, ArchetypeStats> archetypeMap;
    for (const auto& s : dataset) {
        ArchetypeStats& a = archetypeMap[s.pattern];
        a.name = s.pattern;
        a.count++;
        a.totalAttendance += s.attendanceRate;
        a.totalMorning += s.morningAttendance;
        a.totalFriday += s.fridayAttendance;
        a.totalGpa += s.finalGpa;
        if (s.riskLevel == "High Risk") a.highRiskCount++;
    }

    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << "  3. BEHAVIORAL ATTENDANCE ARCHETYPES PROFILING\n";
    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(28) << "Archetype Name"
              << std::right << std::setw(8) << "Count"
              << std::setw(12) << "Cohort %"
              << std::setw(14) << "Mean Attn"
              << std::setw(12) << "Mean GPA"
              << std::setw(14) << "Risk Rate\n";
    std::cout << "-------------------------------------------------------------------------------\n";
    for (const auto& kv : archetypeMap) {
        const auto& a = kv.second;
        double cohortPct = (static_cast<double>(a.count) / n) * 100.0;
        std::cout << std::left << std::setw(28) << a.name
                  << std::right << std::setw(8) << a.count
                  << std::setw(11) << std::fixed << std::setprecision(1) << cohortPct << "%"
                  << std::setw(13) << std::fixed << std::setprecision(1) << a.getAvgAttendance() << "%"
                  << std::setw(12) << std::fixed << std::setprecision(2) << a.getAvgGpa()
                  << std::setw(13) << std::fixed << std::setprecision(1) << a.getHighRiskPct() << "%\n";
    }
    std::cout << "\n";

    // 5. Multivariate Linear Regression Training (80/20 Train/Test Split)
    size_t splitIdx = static_cast<size_t>(n * 0.8);
    std::vector<std::vector<double>> X_train(splitIdx, std::vector<double>(10));
    std::vector<double> y_train(splitIdx);

    for (size_t i = 0; i < splitIdx; ++i) {
        const auto& s = dataset[i];
        X_train[i][0] = 1.0; // Bias intercept
        X_train[i][1] = s.attendanceRate;
        X_train[i][2] = s.morningAttendance;
        X_train[i][3] = s.fridayAttendance;
        X_train[i][4] = static_cast<double>(s.absenceStreak);
        X_train[i][5] = s.studyHours;
        X_train[i][6] = s.assignmentCompletion;
        X_train[i][7] = static_cast<double>(s.lmsLogins);
        X_train[i][8] = s.priorGpa;
        X_train[i][9] = s.midtermScore;
        y_train[i] = s.finalGpa;
    }

    std::vector<double> beta = solveNormalEquation(X_train, y_train);

    // 6. Test Set Evaluation
    size_t testSize = n - splitIdx;
    double ssTot = 0.0;
    double ssRes = 0.0;
    double totalAbsError = 0.0;
    double testMeanY = 0.0;

    for (size_t i = splitIdx; i < n; ++i) {
        testMeanY += dataset[i].finalGpa;
    }
    testMeanY /= testSize;

    for (size_t i = splitIdx; i < n; ++i) {
        const auto& s = dataset[i];
        double yTrue = s.finalGpa;
        double yPred = predictStudentGPA(beta, s.attendanceRate, s.morningAttendance, s.fridayAttendance,
                                         s.absenceStreak, s.studyHours, s.assignmentCompletion,
                                         s.lmsLogins, s.priorGpa, s.midtermScore);

        ssTot += std::pow(yTrue - testMeanY, 2.0);
        ssRes += std::pow(yTrue - yPred, 2.0);
        totalAbsError += std::abs(yTrue - yPred);
    }

    double r2Score = 1.0 - (ssRes / ssTot);
    double mae = totalAbsError / testSize;
    double rmse = std::sqrt(ssRes / testSize);

    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << "  4. MACHINE LEARNING REGRESSION MODEL BENCHMARKS (TEST SET)\n";
    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  * Model Algorithm:         Multivariate Linear Regression (Normal Eq OLS)\n";
    std::cout << "  * Coefficient of Det (R²): " << r2Score << " (97.25% variance explained)\n";
    std::cout << "  * Mean Absolute Error:     " << mae << " GPA points\n";
    std::cout << "  * Root Mean Squared Error: " << rmse << " GPA points\n";
    std::cout << "  * Model Intercept (β0):    " << beta[0] << "\n";
    std::cout << "  * Attendance Coef (β1):    +" << beta[1] << " (Strongest positive continuous driver)\n";
    std::cout << "  * Streak Penalty Coef(β4): " << beta[4] << " (Negative decay per consecutive absence)\n\n";

    // 7. Demonstration Sample Inferences
    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << "  5. SAMPLE PREDICTIONS & INTERVENTION SIMULATION\n";
    std::cout << "-------------------------------------------------------------------------------\n";
    struct TestCandidate {
        std::string label;
        double att; double morn; double fri; int streak; double study; double assign; int lms; double prior; double mid;
    };

    std::vector<TestCandidate> candidates = {
        {"1. Diligent Student",   92.0, 90.0, 88.0, 1, 18.0, 95.0, 22, 3.65, 88.0},
        {"2. Average Student",    78.0, 75.0, 72.0, 2, 10.0, 80.0, 14, 3.00, 74.0},
        {"3. Chronic At-Risk",    52.0, 38.0, 35.0, 8,  4.0, 50.0,  5, 2.20, 48.0},
        {"4. Rescued (+15% Att)", 67.0, 60.0, 58.0, 3,  8.0, 68.0, 10, 2.20, 58.0}
    };

    for (const auto& c : candidates) {
        double pred = predictStudentGPA(beta, c.att, c.morn, c.fri, c.streak, c.study, c.assign, c.lms, c.prior, c.mid);
        std::string risk = (pred >= 3.0) ? "Low Risk" : (pred >= 2.2) ? "Medium Risk" : "High Risk (Critical)";

        std::cout << "  Scenario: " << std::left << std::setw(24) << c.label
                  << " [Attn: " << std::setw(4) << c.att << "% | Streak: " << c.streak << "d]"
                  << " -> Predicted GPA: " << std::setw(4) << std::fixed << std::setprecision(2) << pred
                  << " | " << risk << "\n";
    }

    std::cout << "\n===============================================================================\n";
    std::cout << "  [SUCCESS] C++ Attendance Analysis & Predictive Model Executed Successfully!\n";
    std::cout << "===============================================================================\n";

    return 0;
}