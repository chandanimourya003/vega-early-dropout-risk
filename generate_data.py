import pandas as pd
import numpy as np
import argparse

np.random.seed(42)

def generate_students(n=500):

    departments = ['CSE','ECE','ME','CE','EE']

    attendance = np.random.normal(70,18,n).clip(20,100)
    internal_marks = np.random.normal(55,20,n).clip(0,100)
    semester_gpa = np.random.normal(6.0,1.7,n).clip(2,10)
    fee_paid = np.random.choice([0,1],n,p=[0.25,0.75])
    lms_engagement = np.random.normal(60,25,n).clip(0,100)
    scholarship = np.random.choice([0,1],n,p=[0.6,0.4])
    backlogs = np.random.randint(0,6,n)
    family_income = np.random.choice(['Low','Medium','High'],n,p=[0.35,0.45,0.20])
    dept = np.random.choice(departments,n)

    # realistic dropout logic
    risk = (
        (100-attendance)*0.35 +
        (100-internal_marks)*0.20 +
        (10-semester_gpa)*6 +
        (1-fee_paid)*18 +
        (100-lms_engagement)*0.12 +
        backlogs*6 +
        (family_income=='Low')*12 -
        scholarship*8
    )

    dropout = (risk + np.random.normal(0,10,n) > 60).astype(int)

    df = pd.DataFrame({
        "student_id":[f"STU{str(i+1).zfill(4)}" for i in range(n)],
        "department":dept,
        "attendance_pct":attendance.round(1),
        "internal_marks":internal_marks.round(1),
        "semester_gpa":semester_gpa.round(2),
        "fee_paid":fee_paid,
        "lms_engagement":lms_engagement.round(1),
        "scholarship":scholarship,
        "backlogs":backlogs,
        "family_income":family_income,
        "dropout":dropout
    })

    return df

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows",type=int,default=500)
    args = parser.parse_args()

    data = generate_students(args.rows)
    data.to_csv("students.csv",index=False)

    print("students.csv generated successfully")
    print(f"Total Students: {len(data)}")
    print(f"Dropout Rate: {data['dropout'].mean()*100:.2f}%")