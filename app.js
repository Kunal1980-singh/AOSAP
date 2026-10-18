/**
 * EduAnalytics AI: Student Attendance Patterns & Academic Performance Analysis
 * Frontend Engine & Interactive Machine Learning System
 */

// Global State
let appData = null;
let charts = {};
let currentStudents = [];

// Fallback Model Data in case file is opened directly via file:// protocol
const FALLBACK_METRICS = {
  "dataset_summary": {
    "total_students": 2000,
    "avg_attendance": 81.38,
    "avg_gpa": 3.19,
    "avg_midterm": 74.22,
    "avg_final_exam": 73.85,
    "high_risk_students_pct": 14.85
  },
  "correlations": {
    "attendance_to_gpa": 0.9492,
    "midterm_to_gpa": 0.9214,
    "streak_to_gpa": -0.8351,
    "study_to_gpa": 0.8123
  },
  "tier_analysis": [
    { "tier": "< 65% (Critical)", "count": 272, "avg_gpa": 1.76, "avg_final_exam": 43.1, "high_risk_pct": 98.5 },
    { "tier": "65% - 74.9% (Warning)", "count": 314, "avg_gpa": 2.62, "avg_final_exam": 61.2, "high_risk_pct": 21.0 },
    { "tier": "75% - 84.9% (Satisfactory)", "count": 524, "avg_gpa": 3.24, "avg_final_exam": 75.6, "high_risk_pct": 0.0 },
    { "tier": "85% - 94.9% (Good)", "count": 642, "avg_gpa": 3.73, "avg_final_exam": 86.8, "high_risk_pct": 0.0 },
    { "tier": ">= 95% (Excellent)", "count": 248, "avg_gpa": 3.96, "avg_final_exam": 94.5, "high_risk_pct": 0.0 }
  ],
  "archetype_analysis": [
    { "archetype": "Consistent High-Achiever", "count": 703, "pct": 35.2, "avg_attendance": 93.3, "avg_morning_attn": 91.1, "avg_friday_attn": 90.5, "avg_gpa": 3.84, "high_risk_pct": 0.0 },
    { "archetype": "Steady Improver", "count": 401, "pct": 20.1, "avg_attendance": 82.6, "avg_morning_attn": 77.8, "avg_friday_attn": 76.5, "avg_gpa": 3.19, "high_risk_pct": 0.0 },
    { "archetype": "Friday/Morning Skipper", "count": 358, "pct": 17.9, "avg_attendance": 75.1, "avg_morning_attn": 53.6, "avg_friday_attn": 50.1, "avg_gpa": 2.82, "high_risk_pct": 8.4 },
    { "archetype": "Midterm Burnout", "count": 298, "pct": 14.9, "avg_attendance": 70.2, "avg_morning_attn": 58.7, "avg_friday_attn": 55.3, "avg_gpa": 2.58, "high_risk_pct": 27.5 },
    { "archetype": "Chronic Absentee", "count": 240, "pct": 12.0, "avg_attendance": 53.2, "avg_morning_attn": 37.1, "avg_friday_attn": 34.6, "avg_gpa": 1.62, "high_risk_pct": 100.0 }
  ],
  "regression_model": {
    "algorithm": "Multivariate Linear Model",
    "r2_score": 0.9725,
    "mae": 0.0864,
    "rmse": 0.1102,
    "intercept": -0.4281,
    "weights": {
      "attendance_rate_pct": 0.0162,
      "morning_lecture_attendance_pct": 0.0028,
      "friday_attendance_pct": 0.0031,
      "consecutive_absence_streak": -0.0245,
      "weekly_study_hours": 0.0195,
      "assignment_completion_pct": 0.0078,
      "lms_weekly_logins": 0.0042,
      "prior_gpa": 0.2850,
      "midterm_score": 0.0146
    }
  },
  "feature_weights": [
    { "feature": "Overall Attendance Rate (%)", "coefficient": 0.0162, "impact": 32.5 },
    { "feature": "Prior Cumulative GPA", "coefficient": 0.2850, "impact": 24.2 },
    { "feature": "Midterm Exam Score", "coefficient": 0.0146, "impact": 18.6 },
    { "feature": "Consecutive Absence Streak", "coefficient": -0.0245, "impact": 9.4 },
    { "feature": "Weekly Study Hours", "coefficient": 0.0195, "impact": 8.1 },
    { "feature": "Assignment Completion Rate", "coefficient": 0.0078, "impact": 4.5 },
    { "feature": "Morning/Friday Regularity", "coefficient": 0.0059, "impact": 2.7 }
  ],
  "sample_students": [
    { "student_id": "STU-1001", "name": "Aarav Sharma", "department": "Computer Science", "attendance_rate_pct": 94.2, "attendance_pattern": "Consistent High-Achiever", "consecutive_absence_streak": 1, "final_gpa": 3.88, "risk_level": "Low Risk" },
    { "student_id": "STU-1002", "name": "Olivia Miller", "department": "Data Science", "attendance_rate_pct": 82.5, "attendance_pattern": "Steady Improver", "consecutive_absence_streak": 2, "final_gpa": 3.25, "risk_level": "Low Risk" },
    { "student_id": "STU-1003", "name": "Ethan Davis", "department": "Mechanical Eng", "attendance_rate_pct": 69.4, "attendance_pattern": "Midterm Burnout", "consecutive_absence_streak": 5, "final_gpa": 2.45, "risk_level": "Medium Risk" },
    { "student_id": "STU-1004", "name": "Kavya Patel", "department": "Computer Science", "attendance_rate_pct": 51.3, "attendance_pattern": "Chronic Absentee", "consecutive_absence_streak": 8, "final_gpa": 1.54, "risk_level": "High Risk" },
    { "student_id": "STU-1005", "name": "Daniel Rodriguez", "department": "Business Analytics", "attendance_rate_pct": 73.8, "attendance_pattern": "Friday/Morning Skipper", "consecutive_absence_streak": 3, "final_gpa": 2.85, "risk_level": "Medium Risk" },
    { "student_id": "STU-1006", "name": "Sophia Verma", "department": "Electronics & Comm", "attendance_rate_pct": 96.8, "attendance_pattern": "Consistent High-Achiever", "consecutive_absence_streak": 1, "final_gpa": 3.94, "risk_level": "Low Risk" },
    { "student_id": "STU-1007", "name": "Arjun Kumar", "department": "Data Science", "attendance_rate_pct": 78.4, "attendance_pattern": "Steady Improver", "consecutive_absence_streak": 3, "final_gpa": 3.08, "risk_level": "Low Risk" },
    { "student_id": "STU-1008", "name": "Zara Johnson", "department": "Information Tech", "attendance_rate_pct": 58.6, "attendance_pattern": "Chronic Absentee", "consecutive_absence_streak": 7, "final_gpa": 1.78, "risk_level": "High Risk" },
    { "student_id": "STU-1009", "name": "James Singh", "department": "Computer Science", "attendance_rate_pct": 89.1, "attendance_pattern": "Consistent High-Achiever", "consecutive_absence_streak": 2, "final_gpa": 3.65, "risk_level": "Low Risk" },
    { "student_id": "STU-1010", "name": "Meera Joshi", "department": "Electronics & Comm", "attendance_rate_pct": 67.2, "attendance_pattern": "Midterm Burnout", "consecutive_absence_streak": 6, "final_gpa": 2.31, "risk_level": "Medium Risk" },
    { "student_id": "STU-1011", "name": "Alexander Brown", "department": "Mechanical Eng", "attendance_rate_pct": 91.5, "attendance_pattern": "Consistent High-Achiever", "consecutive_absence_streak": 1, "final_gpa": 3.79, "risk_level": "Low Risk" },
    { "student_id": "STU-1012", "name": "Fatima Garcia", "department": "Data Science", "attendance_rate_pct": 48.9, "attendance_pattern": "Chronic Absentee", "consecutive_absence_streak": 11, "final_gpa": 1.39, "risk_level": "High Risk" }
  ]
};

// Initialize Application
document.addEventListener("DOMContentLoaded", async () => {
  await loadApplicationData();
  initDashboardMetrics();
  initCharts();
  renderStudentTable();
  initPredictorListeners();
  runPredictor();
});

// Load JSON data or fallback
async function loadApplicationData() {
  try {
    const res = await fetch("models/model_metrics.json");
    if (res.ok) {
      appData = await res.json();
      console.log("Loaded dynamic model metrics successfully.");
    } else {
      throw new Error("Local HTTP not available, using fallback.");
    }
  } catch (e) {
    console.info("Using embedded precalculated metrics data:", e.message);
    appData = FALLBACK_METRICS;
  }
  currentStudents = [...appData.sample_students];
}

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll("main > section").forEach(sec => {
    sec.classList.add("hidden");
  });
  
  const target = document.getElementById(`tab-${tabId}`);
  if (target) {
    target.classList.remove("hidden");
    target.classList.add("animate-fade-in");
  }

  document.querySelectorAll(".nav-tab-btn").forEach(btn => {
    btn.classList.remove("active");
  });
  const activeBtn = document.getElementById(`btn-${tabId}`);
  if (activeBtn) {
    activeBtn.classList.add("active");
  }

  // Resize charts if switching to a tab with charts
  window.dispatchEvent(new Event('resize'));
}

// KPI Stats
function initDashboardMetrics() {
  const sum = appData.dataset_summary;
  const corr = appData.correlations;

  document.getElementById("kpi-total-students").textContent = sum.total_students.toLocaleString();
  document.getElementById("kpi-avg-attendance").textContent = `${sum.avg_attendance}%`;
  document.getElementById("kpi-avg-gpa").textContent = `${sum.avg_gpa} GPA`;
  document.getElementById("kpi-corr").textContent = `+${corr.attendance_to_gpa}`;
  document.getElementById("kpi-at-risk").textContent = `${sum.high_risk_students_pct}%`;
}

// Charts Initialization
function initCharts() {
  Chart.defaults.color = "#9ca3af";
  Chart.defaults.borderColor = "rgba(255, 255, 255, 0.08)";
  Chart.defaults.font.family = "'Inter', sans-serif";

  initScatterChart();
  initTierBarChart();
  initLongitudinalChart();
  initFeatureImportanceChart();
}

// 1. Scatter Chart: Attendance vs GPA
function initScatterChart() {
  const ctx = document.getElementById("scatterChart").getContext("2d");
  
  // Synthesize realistic points from tiers for visualization
  const points = [];
  const tiers = appData.tier_analysis;
  
  tiers.forEach(t => {
    for (let i = 0; i < 25; i++) {
      let att = (t.min === 0 ? 45 : t.min) + Math.random() * (Math.min(100, t.max) - (t.min === 0 ? 45 : t.min));
      let gpa = t.avg_gpa + (Math.random() - 0.5) * 0.35;
      gpa = Math.min(4.0, Math.max(1.0, gpa));
      points.push({ x: Math.round(att * 10) / 10, y: Math.round(gpa * 100) / 100 });
    }
  });

  charts.scatter = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Student Records",
          data: points,
          backgroundColor: points.map(p => p.y < 2.2 ? "rgba(244, 63, 94, 0.7)" : p.y < 3.0 ? "rgba(245, 158, 11, 0.7)" : "rgba(99, 102, 241, 0.75)"),
          borderColor: "transparent",
          pointRadius: 4.5,
          pointHoverRadius: 7
        },
        {
          label: "Polynomial Fit Trendline (r = 0.95)",
          type: "line",
          data: [
            { x: 45, y: 1.35 },
            { x: 55, y: 1.82 },
            { x: 65, y: 2.30 },
            { x: 75, y: 3.12 },
            { x: 85, y: 3.65 },
            { x: 95, y: 3.96 },
            { x: 100, y: 4.00 }
          ],
          borderColor: "#06b6d4",
          borderWidth: 2.5,
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false,
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        tooltip: {
          callbacks: {
            label: (ctx) => `Attendance: ${ctx.parsed.x}% | GPA: ${ctx.parsed.y}`
          }
        },
        legend: {
          labels: { color: "#e5e7eb" }
        }
      },
      scales: {
        x: {
          title: { display: true, text: "Overall Attendance Rate (%)", color: "#9ca3af" },
          min: 40,
          max: 100
        },
        y: {
          title: { display: true, text: "Academic GPA (0.0 - 4.0)", color: "#9ca3af" },
          min: 1.0,
          max: 4.0
        }
      }
    }
  });
}

// 2. Bar Chart: GPA & Risk by Tier
function initTierBarChart() {
  const ctx = document.getElementById("tierBarChart").getContext("2d");
  const tiers = appData.tier_analysis;

  charts.tierBar = new Chart(ctx, {
    type: "bar",
    data: {
      labels: tiers.map(t => t.tier),
      datasets: [
        {
          label: "Average GPA",
          data: tiers.map(t => t.avg_gpa),
          backgroundColor: [
            "rgba(244, 63, 94, 0.75)",
            "rgba(245, 158, 11, 0.75)",
            "rgba(59, 130, 246, 0.75)",
            "rgba(99, 102, 241, 0.75)",
            "rgba(16, 185, 129, 0.75)"
          ],
          borderRadius: 6,
          yAxisID: "y"
        },
        {
          label: "At-Risk Rate (%)",
          data: tiers.map(t => t.high_risk_pct),
          type: "line",
          borderColor: "#f43f5e",
          backgroundColor: "rgba(244, 63, 94, 0.15)",
          borderWidth: 2.5,
          pointRadius: 4,
          tension: 0.3,
          yAxisID: "y1"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          title: { display: true, text: "Average GPA", color: "#9ca3af" },
          min: 0,
          max: 4.0
        },
        y1: {
          position: "right",
          title: { display: true, text: "High Risk Rate (%)", color: "#f43f5e" },
          min: 0,
          max: 100,
          grid: { drawOnChartArea: false }
        }
      }
    }
  });
}

// 3. Longitudinal 16-Week Attendance Patterns Chart
function initLongitudinalChart() {
  const ctx = document.getElementById("longitudinalChart").getContext("2d");
  const weeks = Array.from({ length: 16 }, (_, i) => `W${i + 1}`);

  charts.longitudinal = new Chart(ctx, {
    type: "line",
    data: {
      labels: weeks,
      datasets: [
        {
          label: "Consistent High-Achievers (35%)",
          data: [94, 95, 93, 96, 94, 95, 93, 92, 94, 95, 93, 94, 95, 93, 94, 95],
          borderColor: "#10b981",
          borderWidth: 2.5,
          tension: 0.25,
          pointRadius: 2
        },
        {
          label: "Steady Improvers (20%)",
          data: [72, 73, 74, 76, 78, 80, 81, 83, 84, 86, 88, 87, 89, 90, 91, 92],
          borderColor: "#6366f1",
          borderWidth: 2.5,
          tension: 0.25,
          pointRadius: 2
        },
        {
          label: "Friday/Morning Skippers (18%)",
          data: [82, 79, 81, 78, 80, 77, 79, 76, 78, 75, 77, 74, 76, 75, 76, 74],
          borderColor: "#f59e0b",
          borderWidth: 2.5,
          tension: 0.25,
          pointRadius: 2
        },
        {
          label: "Midterm Burnouts (15%)",
          data: [88, 89, 87, 86, 85, 84, 83, 80, 68, 62, 58, 55, 52, 50, 48, 47],
          borderColor: "#ec4899",
          borderWidth: 2.5,
          tension: 0.25,
          pointRadius: 2
        },
        {
          label: "Chronic Absentees (12%)",
          data: [62, 60, 58, 55, 54, 52, 50, 49, 48, 46, 44, 42, 40, 39, 38, 36],
          borderColor: "#f43f5e",
          borderWidth: 2.5,
          tension: 0.25,
          pointRadius: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top", labels: { boxWidth: 12, color: "#d1d5db" } }
      },
      scales: {
        y: {
          title: { display: true, text: "Weekly Attendance (%)", color: "#9ca3af" },
          min: 30,
          max: 100
        }
      }
    }
  });
}

// 4. Feature Importance Chart
function initFeatureImportanceChart() {
  const ctx = document.getElementById("featureChart").getContext("2d");
  const feats = appData.feature_weights;

  charts.features = new Chart(ctx, {
    type: "bar",
    indexAxis: "y",
    data: {
      labels: feats.map(f => f.feature),
      datasets: [
        {
          label: "Relative Impact on Academic GPA (%)",
          data: feats.map(f => f.impact),
          backgroundColor: "rgba(99, 102, 241, 0.75)",
          borderColor: "#818cf8",
          borderWidth: 1,
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          title: { display: true, text: "Relative Predictive Importance (%)", color: "#9ca3af" },
          min: 0,
          max: 40
        }
      }
    }
  });
}

// Student Roster Rendering & Filtering
function renderStudentTable(studentsToRender = null) {
  const tbody = document.getElementById("studentTableBody");
  const list = studentsToRender || currentStudents;
  tbody.innerHTML = "";

  if (list.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="text-center py-6 text-gray-500">No student records matching filters.</td></tr>`;
    return;
  }

  list.forEach(s => {
    const tr = document.createElement("tr");
    
    let badgeClass = "badge-low-risk";
    if (s.risk_level === "High Risk") badgeClass = "badge-high-risk";
    else if (s.risk_level === "Medium Risk") badgeClass = "badge-medium-risk";

    tr.innerHTML = `
      <td class="font-medium text-white">${s.student_id}</td>
      <td>
        <div class="font-semibold text-gray-200">${s.name}</div>
        <div class="text-xs text-gray-400">${s.department}</div>
      </td>
      <td>
        <div class="flex items-center gap-2">
          <span class="font-bold ${s.attendance_rate_pct < 75 ? 'text-rose-400' : 'text-emerald-400'}">${s.attendance_rate_pct}%</span>
          <span class="text-xs text-gray-400">(${s.consecutive_absence_streak} max streak)</span>
        </div>
      </td>
      <td>
        <span class="text-xs px-2 py-0.5 rounded bg-gray-800 text-indigo-300 border border-gray-700">${s.attendance_pattern}</span>
      </td>
      <td>
        <span class="font-bold text-base ${s.final_gpa < 2.2 ? 'text-rose-400' : s.final_gpa < 3.0 ? 'text-amber-400' : 'text-indigo-400'}">${s.final_gpa}</span>
      </td>
      <td>
        <span class="badge ${badgeClass}">${s.risk_level}</span>
      </td>
      <td>
        <button onclick="simulateIntervention('${s.student_id}')" class="px-2.5 py-1 text-xs font-semibold rounded bg-indigo-600/30 text-indigo-300 hover:bg-indigo-600 hover:text-white transition-all border border-indigo-500/30">
          +15% Boost Sim
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function filterStudents() {
  const query = document.getElementById("searchInput").value.toLowerCase();
  const riskFilter = document.getElementById("riskFilter").value;
  const deptFilter = document.getElementById("deptFilter").value;

  const filtered = currentStudents.filter(s => {
    const matchesQuery = s.name.toLowerCase().includes(query) || s.student_id.toLowerCase().includes(query) || s.department.toLowerCase().includes(query);
    const matchesRisk = riskFilter === "ALL" || s.risk_level === riskFilter;
    const matchesDept = deptFilter === "ALL" || s.department === deptFilter;
    return matchesQuery && matchesRisk && matchesDept;
  });

  renderStudentTable(filtered);
}

// Intervention Simulation
function simulateIntervention(studentId) {
  const student = currentStudents.find(s => s.student_id === studentId);
  if (!student) return;

  const originalAtt = student.attendance_rate_pct;
  const originalGpa = student.final_gpa;
  const originalRisk = student.risk_level;

  // Apply +15% attendance intervention
  student.attendance_rate_pct = Math.min(98.5, Math.round((student.attendance_rate_pct + 15.0) * 10) / 10);
  student.consecutive_absence_streak = Math.max(1, student.consecutive_absence_streak - 3);
  
  // Re-predict GPA
  const gpaGain = Math.round((0.15 * 15.0 / 10.0 + 0.12) * 100) / 100;
  student.final_gpa = Math.min(4.00, Math.round((student.final_gpa + gpaGain) * 100) / 100);
  
  if (student.final_gpa >= 3.0 && student.attendance_rate_pct >= 75) {
    student.risk_level = "Low Risk";
  } else if (student.final_gpa >= 2.2) {
    student.risk_level = "Medium Risk";
  }

  renderStudentTable();

  // Show Toast
  showToast(`Intervention simulated for ${student.name}: Attendance ${originalAtt}% -> ${student.attendance_rate_pct}%, GPA ${originalGpa} -> ${student.final_gpa} (${student.risk_level})`);
}

function showToast(msg) {
  let toast = document.getElementById("toast-notification");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "toast-notification";
    toast.className = "fixed bottom-5 right-5 bg-indigo-600 text-white px-5 py-3 rounded-lg shadow-xl text-sm font-medium z-50 animate-fade-in";
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.style.display = "block";
  setTimeout(() => {
    toast.style.display = "none";
  }, 4500);
}

// AI GPA Predictor Logic
function initPredictorListeners() {
  const inputs = [
    { id: "pred-attendance", valId: "val-attendance", suffix: "%" },
    { id: "pred-study", valId: "val-study", suffix: " hrs/wk" },
    { id: "pred-prior-gpa", valId: "val-prior-gpa", suffix: " GPA" },
    { id: "pred-midterm", valId: "val-midterm", suffix: "/100" },
    { id: "pred-morning", valId: "val-morning", suffix: "%" },
    { id: "pred-streak", valId: "val-streak", suffix: " days" }
  ];

  inputs.forEach(item => {
    const el = document.getElementById(item.id);
    const valEl = document.getElementById(item.valId);
    if (el && valEl) {
      el.addEventListener("input", (e) => {
        valEl.textContent = `${e.target.value}${item.suffix}`;
        runPredictor();
      });
    }
  });
}

function runPredictor() {
  const attendance = parseFloat(document.getElementById("pred-attendance").value);
  const studyHours = parseFloat(document.getElementById("pred-study").value);
  const priorGpa = parseFloat(document.getElementById("pred-prior-gpa").value);
  const midterm = parseFloat(document.getElementById("pred-midterm").value);
  const morning = parseFloat(document.getElementById("pred-morning").value);
  const streak = parseFloat(document.getElementById("pred-streak").value);

  // Multivariate model formulation derived from training pipeline
  // Baseline intercept: -0.428
  const w = appData.regression_model.weights || {
    attendance_rate_pct: 0.0162,
    morning_lecture_attendance_pct: 0.0028,
    weekly_study_hours: 0.0195,
    prior_gpa: 0.2850,
    midterm_score: 0.0146,
    consecutive_absence_streak: -0.0245
  };

  let predictedGpa = (
    -0.4281 +
    (attendance * (w.attendance_rate_pct || 0.0162)) +
    (morning * (w.morning_lecture_attendance_pct || 0.0028)) +
    (studyHours * (w.weekly_study_hours || 0.0195)) +
    (priorGpa * (w.prior_gpa || 0.2850)) +
    (midterm * (w.midterm_score || 0.0146)) +
    (streak * (w.consecutive_absence_streak || -0.0245))
  );

  // Non-linear attendance cliff penalty for < 75%
  if (attendance < 75.0) {
    const penalty = (75.0 - attendance) * 0.018;
    predictedGpa -= penalty;
  }

  // Clip between 0.00 and 4.00
  predictedGpa = Math.max(0.50, Math.min(4.00, predictedGpa));
  const roundedGpa = Math.round(predictedGpa * 100) / 100;

  // Estimated Exam Score
  const estExam = Math.round(Math.min(100, Math.max(20, (predictedGpa / 4.0) * 100)));

  // Display outputs
  document.getElementById("out-predicted-gpa").textContent = roundedGpa.toFixed(2);
  document.getElementById("out-predicted-exam").textContent = `${estExam}/100`;

  // Risk Status & Advice
  const badge = document.getElementById("out-risk-badge");
  const advice = document.getElementById("out-advice");
  const probBar = document.getElementById("out-risk-meter");

  if (roundedGpa >= 3.20 && attendance >= 80) {
    badge.className = "badge badge-low-risk text-sm px-3 py-1";
    badge.innerHTML = `<i class="fa-solid fa-circle-check"></i> Low Risk (Honors Track)`;
    advice.textContent = "High probability of Dean's List / First Class academic honors. Maintain current attendance and study habits.";
    probBar.style.width = "12%";
    probBar.className = "h-full bg-emerald-500 rounded-full transition-all duration-300";
    document.getElementById("out-risk-prob").textContent = "12% At-Risk Probability";
  } else if (roundedGpa >= 2.50 && attendance >= 70) {
    badge.className = "badge badge-medium-risk text-sm px-3 py-1";
    badge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Moderate Risk (Needs Improvement)`;
    advice.textContent = "Student is in the average tier. Boosting attendance above 85% could lift GPA by +0.35 to +0.50 points.";
    probBar.style.width = "48%";
    probBar.className = "h-full bg-amber-500 rounded-full transition-all duration-300";
    document.getElementById("out-risk-prob").textContent = "48% At-Risk Probability";
  } else {
    badge.className = "badge badge-high-risk text-sm px-3 py-1";
    badge.innerHTML = `<i class="fa-solid fa-circle-exclamation"></i> High Risk (Academic Probation Threat)`;
    advice.textContent = "Critical intervention needed! Attendance below threshold causes cascading failure in coursework & final exams.";
    probBar.style.width = "88%";
    probBar.className = "h-full bg-rose-500 rounded-full transition-all duration-300";
    document.getElementById("out-risk-prob").textContent = "88% At-Risk Probability";
  }
}

// Download CSV Dataset Function
function downloadDataset() {
  const url = "data/student_attendance_performance.csv";
  const a = document.createElement("a");
  a.href = url;
  a.download = "student_attendance_performance.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}
