import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import json

def run_ai(csv_path):

    df = pd.read_csv(csv_path)

    if "dropout" not in df.columns:
        raise Exception("CSV must contain dropout column")

    y = df["dropout"]
    X = df.drop(columns=["dropout"])

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    model = RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42)

    pipe = Pipeline([
        ("prep", pre),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X)[:, 1]
    df["risk_prob"] = probs

    df["risk_level"] = pd.cut(
        probs,
        bins=[0, 0.4, 0.7, 1],
        labels=["Low", "Medium", "High"]
    )

    summary = {
        "total_students": len(df),
        "high_risk": int((df["risk_level"] == "High").sum()),
        "medium_risk": int((df["risk_level"] == "Medium").sum()),
        "low_risk": int((df["risk_level"] == "Low").sum()),
        "accuracy": float(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]))
    }

    students = df.to_dict(orient="records")

    output = {
        "summary": summary,
        "students": students
    }

    with open("ai_output.json", "w") as f:
        json.dump(output, f, indent=2)