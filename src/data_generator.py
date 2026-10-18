"""
Student Attendance & Academic Performance Dataset Generator
Generates realistic longitudinal and multidimensional attendance patterns
correlated with academic performance metrics.
"""

import os
import random
import math
import csv

def generate_student_dataset(n_samples=2000, seed=42):
    random.seed(seed)
    
    first_names = [
        "Aarav", "Aditi", "Alexander", "Amara", "Arjun", "Charlotte", "Daniel", "Diya",
        "Ethan", "Fatima", "Gabriel", "Grace", "Hassan", "Ishaan", "James", "Kavya",
        "Liam", "Meera", "Noah", "Olivia", "Priya", "Rahul", "Rohan", "Sophia",
        "Tanya", "Vikram", "William", "Zara", "Lucas", "Elena", "Marcus", "Ananya"
    ]
    last_names = [
        "Sharma", "Smith", "Patel", "Johnson", "Verma", "Williams", "Gupta", "Brown",
        "Kumar", "Jones", "Singh", "Garcia", "Rao", "Miller", "Reddy", "Davis",
        "Kapoor", "Rodriguez", "Chopra", "Martinez", "Joshi", "Hernandez", "Mehta", "Lopez"
    ]
    
    departments = ["Computer Science", "Data Science", "Electronics & Comm", "Mechanical Eng", "Business Analytics", "Information Tech"]
    patterns = [
        ("Consistent High-Achiever", 0.35),
        ("Steady Improver", 0.20),
        ("Midterm Burnout", 0.15),
        ("Friday/Morning Skipper", 0.18),
        ("Chronic Absentee", 0.12)
    ]
    
    pattern_choices = [p[0] for p in patterns]
    pattern_weights = [p[1] for p in patterns]
    
    data = []
    
    for i in range(1, n_samples + 1):
        student_id = f"STU-{1000 + i}"
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        gender = random.choice(["Female", "Male", "Non-Binary"])
        dept = random.choice(departments)
        semester = random.randint(1, 8)
        
        # Select archetype
        archetype = random.choices(pattern_choices, weights=pattern_weights, k=1)[0]
        
        # Base attendance characteristics depending on archetype
        if archetype == "Consistent High-Achiever":
            base_attendance = random.uniform(88.0, 98.5)
            morning_attn = max(75.0, base_attendance - random.uniform(0.0, 5.0))
            friday_attn = max(75.0, base_attendance - random.uniform(0.0, 6.0))
            max_streak = random.randint(1, 2)
            study_hours = random.uniform(14.0, 28.0)
            prior_gpa = min(4.0, random.uniform(3.3, 3.95))
            assignment_pct = random.uniform(88.0, 99.0)
            lms_logins = random.randint(15, 30)
            
        elif archetype == "Steady Improver":
            base_attendance = random.uniform(76.0, 89.0)
            morning_attn = base_attendance - random.uniform(2.0, 8.0)
            friday_attn = base_attendance - random.uniform(3.0, 9.0)
            max_streak = random.randint(2, 4)
            study_hours = random.uniform(10.0, 20.0)
            prior_gpa = random.uniform(2.6, 3.4)
            assignment_pct = random.uniform(75.0, 92.0)
            lms_logins = random.randint(10, 22)
            
        elif archetype == "Midterm Burnout":
            base_attendance = random.uniform(62.0, 78.0)
            morning_attn = base_attendance - random.uniform(8.0, 15.0)
            friday_attn = base_attendance - random.uniform(10.0, 20.0)
            max_streak = random.randint(4, 8)
            study_hours = random.uniform(6.0, 14.0)
            prior_gpa = random.uniform(2.8, 3.5)  # had decent prior ability
            assignment_pct = random.uniform(55.0, 75.0)
            lms_logins = random.randint(5, 14)
            
        elif archetype == "Friday/Morning Skipper":
            base_attendance = random.uniform(68.0, 82.0)
            morning_attn = base_attendance - random.uniform(15.0, 28.0)
            friday_attn = base_attendance - random.uniform(18.0, 32.0)
            max_streak = random.randint(2, 5)
            study_hours = random.uniform(8.0, 16.0)
            prior_gpa = random.uniform(2.5, 3.3)
            assignment_pct = random.uniform(68.0, 85.0)
            lms_logins = random.randint(8, 18)
            
        else:  # Chronic Absentee
            base_attendance = random.uniform(42.0, 64.0)
            morning_attn = base_attendance - random.uniform(10.0, 22.0)
            friday_attn = base_attendance - random.uniform(12.0, 25.0)
            max_streak = random.randint(6, 15)
            study_hours = random.uniform(2.0, 8.0)
            prior_gpa = random.uniform(1.8, 2.7)
            assignment_pct = random.uniform(35.0, 60.0)
            lms_logins = random.randint(2, 9)
            
        # Ensure values stay within realistic bounds
        base_attendance = round(min(100.0, max(35.0, base_attendance)), 1)
        morning_attn = round(min(100.0, max(20.0, morning_attn)), 1)
        friday_attn = round(min(100.0, max(20.0, friday_attn)), 1)
        study_hours = round(study_hours, 1)
        prior_gpa = round(prior_gpa, 2)
        assignment_pct = round(assignment_pct, 1)
        
        # Internal & Midterm Score (0-100)
        # Strongly influenced by attendance + study hours + prior GPA
        midterm_noise = random.gauss(0, 4.5)
        raw_midterm = (
            (base_attendance * 0.40) +
            (study_hours * 1.5) +
            ((prior_gpa / 4.0) * 35.0) +
            midterm_noise
        )
        midterm_score = round(min(100.0, max(25.0, raw_midterm)), 1)
        
        # Final Exam Score (0-100)
        # Attendance penalty threshold: below 75% attendance causes disproportionate degradation
        threshold_penalty = 0.0
        if base_attendance < 75.0:
            threshold_penalty = (75.0 - base_attendance) * 0.45
            
        final_noise = random.gauss(0, 4.0)
        raw_final = (
            (base_attendance * 0.35) +
            (midterm_score * 0.30) +
            (assignment_pct * 0.20) +
            (study_hours * 0.8) -
            threshold_penalty +
            final_noise
        )
        final_exam_score = round(min(100.0, max(20.0, raw_final)), 1)
        
        # Final Academic Performance: GPA (0.0 to 4.0)
        # Composite score
        composite_score = (
            (final_exam_score * 0.50) +
            (midterm_score * 0.30) +
            (assignment_pct * 0.15) +
            ((base_attendance / 100.0) * 5.0)
        )
        
        # Standard collegiate GPA conversion with smooth polynomial curve
        gpa = (composite_score / 100.0) * 4.0
        # Additional adjustment for realistic variance
        gpa = round(min(4.00, max(1.10, gpa + random.gauss(0, 0.08))), 2)
        
        # Risk Classification
        if gpa < 2.20 or base_attendance < 65.0 or max_streak >= 6:
            risk_level = "High Risk"
            intervention = "Urgent: Academic Probation & Mandatory Advisory"
        elif gpa < 3.00 or base_attendance < 75.0:
            risk_level = "Medium Risk"
            intervention = "Attendance Alert & Peer Tutoring Recommended"
        else:
            risk_level = "Low Risk"
            intervention = "On Track: Standard Mentorship"
            
        data.append({
            "student_id": student_id,
            "name": name,
            "gender": gender,
            "department": dept,
            "semester": semester,
            "attendance_rate_pct": base_attendance,
            "morning_lecture_attendance_pct": morning_attn,
            "friday_attendance_pct": friday_attn,
            "consecutive_absence_streak": max_streak,
            "attendance_pattern": archetype,
            "weekly_study_hours": study_hours,
            "assignment_completion_pct": assignment_pct,
            "lms_weekly_logins": lms_logins,
            "prior_gpa": prior_gpa,
            "midterm_score": midterm_score,
            "final_exam_score": final_exam_score,
            "final_gpa": gpa,
            "risk_level": risk_level,
            "recommended_intervention": intervention
        })
        
    return data

def save_csv(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    keys = data[0].keys()
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved {len(data)} student records to {filepath}")

def generate_weekly_patterns(data, filepath):
    """
    Generates longitudinal weekly attendance (Week 1 to Week 16) for each student
    demonstrating realistic progression curves.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    weekly_records = []
    
    for row in data:
        base = row["attendance_rate_pct"]
        archetype = row["attendance_pattern"]
        sid = row["student_id"]
        
        week_rates = []
        for w in range(1, 17):
            if archetype == "Consistent High-Achiever":
                rate = base + random.uniform(-3.0, 3.0)
            elif archetype == "Steady Improver":
                # Starts lower, rallies over the semester
                progress = (w - 1) / 15.0
                rate = (base - 10.0) + (progress * 18.0) + random.uniform(-4.0, 4.0)
            elif archetype == "Midterm Burnout":
                # High till week 8, collapses after midterms
                if w <= 8:
                    rate = base + 12.0 + random.uniform(-3.0, 3.0)
                else:
                    rate = base - 16.0 - ((w - 8) * 1.5) + random.uniform(-4.0, 4.0)
            elif archetype == "Friday/Morning Skipper":
                rate = base + math.sin(w) * 5.0 + random.uniform(-4.0, 4.0)
            else:  # Chronic Absentee
                rate = base - (w * 0.8) + random.uniform(-6.0, 6.0)
                
            rate = round(min(100.0, max(15.0, rate)), 1)
            week_rates.append(rate)
            
        rec = {"student_id": sid, "name": row["name"], "archetype": archetype}
        for w_idx, r in enumerate(week_rates, 1):
            rec[f"week_{w_idx}"] = r
        weekly_records.append(rec)
        
    keys = weekly_records[0].keys()
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(weekly_records)
    print(f"Saved longitudinal weekly tracking to {filepath}")

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    students = generate_student_dataset(n_samples=2000, seed=42)
    save_csv(students, os.path.join(out_dir, "student_attendance_performance.csv"))
    generate_weekly_patterns(students, os.path.join(out_dir, "weekly_attendance_matrix.csv"))
