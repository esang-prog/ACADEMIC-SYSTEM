
"""
Intelligent Student Academic Support System (starter prototype)
CS ELECT 2 - Intelligent Systems Midterm

HOW IT WORKS
1. clean_dataset()  -> finds and fixes data-quality problems, and logs every decision.
2. classify()       -> rule-based, points system. More risk points = more support needed.
3. validate_input() -> rejects impossible values (e.g. attendance 150, negative hours).
4. main()           -> menu: single student / batch CSV / clean dataset report.

IMPORTANT: Adjust the CONFIG section to match YOUR instructor's dataset
(column names, ranges, thresholds). Everything else reads from CONFIG, so a
"change request" is usually a one-line edit here.

Requires: pip install pandas
"""

import re
import sys
import pandas as pd

# =====================================================================
# CONFIG  (edit this section to match your dataset and your justifications)
# =====================================================================

# Variables used in the decision. Replace with your dataset's columns.
# Each has: valid range, and (high_risk_below, moderate_risk_below) thresholds.
# Below the first threshold -> 2 risk points; below the second -> 1; else 0.
FEATURES = {
    "attendance": {
        "label": "Attendance rate (%)",
        "range": (0, 100),
        "thresholds": (75, 85),
        "reason": "attendance is low",
    },
    "quiz_avg": {
        "label": "Quiz average (%)",
        "range": (0, 100),
        "thresholds": (60, 75),
        "reason": "quiz performance is low",
    },
    "study_hours": {
        "label": "Study hours per week",
        "range": (0, 100),
        "thresholds": (5, 10),
        "reason": "weekly study time is low",
    },
    "assignment_completion": {
        "label": "Assignment completion (%)",
        "range": (0, 100),
        "thresholds": (60, 80),
        "reason": "assignment completion is low",
    },
}

# Total risk points -> support level (max points = 2 x number of features = 8)
HIGH_SUPPORT_MIN = 5       # 5 or more points  -> HIGH
MODERATE_SUPPORT_MIN = 3   # 3 to 4 points     -> MODERATE, below 3 -> LOW

# Points given when a value is missing/unknown (1 = cautious "neutral" risk).
MISSING_VALUE_POINTS = 1

# How to treat missing values when CLEANING a dataset: "median" or "leave"
# "median": fill with column median (robust to outliers).
# "leave" : keep as missing; classify() then applies MISSING_VALUE_POINTS.
MISSING_STRATEGY = "median"

# If your dataset uses different column names, map them here:
#   {"their_column_name": "attendance", ...}
COLUMN_ALIASES = {
    # "attendance_rate": "attendance",
    # "quiz_average": "quiz_avg",
}

RECOMMENDATIONS = {
    "HIGH": "Immediate intervention: schedule a meeting with an adviser, "
            "enroll in tutoring, and monitor progress weekly.",
    "MODERATE": "Targeted support: encourage peer study groups, check in "
                "monthly, and share study-skills resources.",
    "LOW": "Minimal support needed: keep up the good work and offer "
           "optional enrichment resources.",
}

# =====================================================================
# DATA CLEANING
# =====================================================================

def _to_number(value):
    """Convert messy text like ' 85% ', '85 %', 'N/A', '' into a float or NaN."""
    if pd.isna(value):
        return float("nan")
    text = str(value).strip().lower()
    if text in ("", "n/a", "na", "none", "null", "-", "?"):
        return float("nan")
    match = re.search(r"-?\d+(\.\d+)?", text)
    return float(match.group()) if match else float("nan")


def clean_dataset(df):
    """Returns (cleaned_df, log). The log lists every problem found and the decision made."""
    log = []
    df = df.copy()
    log.append(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    # 1. Standardize column names (inconsistent formats)
    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]
    df = df.rename(columns=COLUMN_ALIASES)

    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        log.append(f"WARNING: expected columns not found: {missing_cols}. "
                   f"Update FEATURES or COLUMN_ALIASES in CONFIG.")

    # 2. Strip stray whitespace in text columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": pd.NA})

    # 3. Remove duplicates (same student recorded twice)
    before = len(df)
    id_col = next((c for c in ("student_id", "id") if c in df.columns), None)
    df = df.drop_duplicates(subset=[id_col] if id_col else None, keep="first")
    removed = before - len(df)
    log.append(f"Duplicates: removed {removed} duplicate row(s)"
               f"{' by ' + id_col if id_col else ' (full-row match)'}. "
               f"Reason: duplicates would count one student twice and bias the data.")

    # 4. Convert to numbers, then flag out-of-range values as invalid
    for col, cfg in FEATURES.items():
        if col not in df.columns:
            continue
        df[col] = df[col].apply(_to_number)
        low, high = cfg["range"]
        invalid = df[col].notna() & ((df[col] < low) | (df[col] > high))
        if invalid.sum():
            log.append(f"Invalid values in '{col}': {int(invalid.sum())} outside {low}-{high}; "
                       f"set to missing. Reason: impossible values cannot be trusted or corrected by guessing.")
        df.loc[invalid, col] = float("nan")

    # 5. Handle missing values
    for col in FEATURES:
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        if MISSING_STRATEGY == "median":
            med = df[col].median()
            df[col] = df[col].fillna(med)
            log.append(f"Missing in '{col}': {n_missing}; filled with median ({med:.1f}). "
                       f"Reason: median is not distorted by extreme values.")
        else:
            log.append(f"Missing in '{col}': {n_missing}; left missing. "
                       f"Reason: avoid inventing data; classifier gives neutral risk points.")

    log.append(f"Cleaned dataset has {len(df)} rows.")
    return df.reset_index(drop=True), log

# =====================================================================
# VALIDATION (for user input)
# =====================================================================

def validate_input(name, raw):
    """Returns (value, error). Blank input = unknown (value None, no error)."""
    cfg = FEATURES[name]
    raw = str(raw).strip()
    if raw == "":
        return None, None
    try:
        value = float(raw.replace("%", ""))
    except ValueError:
        return None, f"'{raw}' is not a number."
    low, high = cfg["range"]
    if value != value:  # NaN check
        return None, "NaN is not a valid value."
    if value < low or value > high:
        return None, f"{cfg['label']} must be between {low} and {high} (got {value:g})."
    return value, None

# =====================================================================
# DECISION MECHANISM (rule-based points system)
# =====================================================================

def classify(student):
    """student: dict of feature -> number or None/NaN. Returns a result dict with explanation."""
    points = 0
    reasons = []
    for name, cfg in FEATURES.items():
        value = student.get(name)
        if value is None or value != value:  # missing
            points += MISSING_VALUE_POINTS
            reasons.append(f"{cfg['label']} is missing (+{MISSING_VALUE_POINTS})")
            continue
        high_cut, mod_cut = cfg["thresholds"]
        if value < high_cut:
            points += 2
            reasons.append(f"{cfg['reason']}: {value:g} < {high_cut} (+2)")
        elif value < mod_cut:
            points += 1
            reasons.append(f"{cfg['reason']} (borderline): {value:g} < {mod_cut} (+1)")

    if points >= HIGH_SUPPORT_MIN:
        level = "HIGH"
    elif points >= MODERATE_SUPPORT_MIN:
        level = "MODERATE"
    else:
        level = "LOW"

    max_points = 2 * len(FEATURES)
    return {
        "level": level,
        "points": points,
        "max_points": max_points,
        "reasons": reasons or ["All indicators are at healthy levels."],
        "recommendation": RECOMMENDATIONS[level],
    }


def print_result(result):
    print(f"\nAcademic Support Level: {result['level']}  "
          f"(risk points {result['points']}/{result['max_points']})")
    print("Why:")
    for r in result["reasons"]:
        print(f"  - {r}")
    print(f"Recommendation: {result['recommendation']}\n")

# =====================================================================
# APPLICATION MODES
# =====================================================================

def single_student():
    print("\nEnter student data (leave blank if unknown).")
    student = {}
    for name, cfg in FEATURES.items():
        while True:
            value, error = validate_input(name, input(f"  {cfg['label']}: "))
            if error:
                print(f"  ! Invalid: {error} Please try again.")
            else:
                student[name] = value
                break
    print_result(classify(student))


def batch_mode():
    path = input("Path to CSV file: ").strip().strip('"')
    try:
        raw = pd.read_csv(path)
    except Exception as e:
        print(f"Could not read file: {e}")
        return
    df, log = clean_dataset(raw)
    print("\n--- Cleaning log ---")
    print("\n".join(f"* {line}" for line in log))

    results = df.apply(lambda row: classify(row.to_dict()), axis=1)
    df["support_level"] = [r["level"] for r in results]
    df["risk_points"] = [r["points"] for r in results]
    df["explanation"] = ["; ".join(r["reasons"]) for r in results]
    df["recommendation"] = [r["recommendation"] for r in results]

    out = "results.csv"
    df.to_csv(out, index=False)
    print(f"\nSupport level counts:\n{df['support_level'].value_counts()}")
    print(f"\nFull results saved to {out}")


def cleaning_report():
    path = input("Path to CSV file: ").strip().strip('"')
    try:
        raw = pd.read_csv(path)
    except Exception as e:
        print(f"Could not read file: {e}")
        return
    print("\n--- Raw data overview ---")
    print(raw.dtypes)
    print("\nMissing values per column:\n", raw.isna().sum())
    print("\nExact duplicate rows:", int(raw.duplicated().sum()))
    df, log = clean_dataset(raw)
    print("\n--- Cleaning log ---")
    print("\n".join(f"* {line}" for line in log))
    df.to_csv("cleaned_dataset.csv", index=False)
    print("\nCleaned data saved to cleaned_dataset.csv")


def main():
    actions = {"1": single_student, "2": batch_mode, "3": cleaning_report}
    while True:
        print("=== Intelligent Student Academic Support System ===")
        print("1. Classify one student")
        print("2. Classify a CSV of students (batch / hidden test)")
        print("3. Clean a dataset and show the data-quality report")
        print("4. Exit")
        choice = input("Choose: ").strip()
        if choice == "4":
            sys.exit(0)
        actions.get(choice, lambda: print("Invalid choice.\n"))()


if __name__ == "__main__":
    main()

