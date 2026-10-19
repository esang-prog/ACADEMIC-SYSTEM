"""
Intelligent Student Academic Support System - Modern UI (Streamlit)

Install:  pip install streamlit pandas
Run:      streamlit run app_web.py
Keep this file in the SAME folder as academic_support.py (it holds the logic).
Thresholds, columns and ranges are edited in the CONFIG of academic_support.py.
"""

import pandas as pd
import streamlit as st

import academic_support as core

COLORS = {"HIGH": "#c62828", "MODERATE": "#ef6c00", "LOW": "#2e7d32"}
ICONS = {"HIGH": "🔴", "MODERATE": "🟠", "LOW": "🟢"}

st.set_page_config(page_title="Academic Support System", page_icon="🎓", layout="centered")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def result_card(result):
    color = COLORS[result["level"]]
    st.markdown(
        f"""
        <div style="background:{color};padding:22px;border-radius:14px;
                    text-align:center;color:white;margin:8px 0 14px 0;">
            <div style="font-size:14px;letter-spacing:2px;opacity:.9;">ACADEMIC SUPPORT LEVEL</div>
            <div style="font-size:42px;font-weight:800;">{ICONS[result['level']]} {result['level']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(result["points"] / result["max_points"],
                text=f"Risk points: {result['points']} of {result['max_points']}")


def rules_table():
    rows = []
    for cfg in core.FEATURES.values():
        high_cut, mod_cut = cfg["thresholds"]
        rows.append({"Variable": cfg["label"], "+2 points if below": high_cut,
                     "+1 point if below": mod_cut})
    return pd.DataFrame(rows)


def template_csv():
    cols = ["student_id"] + list(core.FEATURES)
    sample = [
        ["S001", 95, 88, 15, 95],
        ["S002", 60, 45, 2, 40],
        ["S003", "", 70, 8, "75%"],
    ]
    return pd.DataFrame(sample, columns=cols).to_csv(index=False).encode("utf-8")


def color_level(value):
    return (f"background-color:{COLORS[value]};color:white;font-weight:bold"
            if value in COLORS else "")


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("How decisions are made")
    st.write("Each variable adds **risk points**. More points means more support is needed.")
    st.dataframe(rules_table(), hide_index=True, use_container_width=True)
    st.write(f"🔴 **HIGH**: {core.HIGH_SUPPORT_MIN}+ points")
    st.write(f"🟠 **MODERATE**: {core.MODERATE_SUPPORT_MIN}-{core.HIGH_SUPPORT_MIN - 1} points")
    st.write(f"🟢 **LOW**: below {core.MODERATE_SUPPORT_MIN} points")
    st.caption(f"A missing value counts as {core.MISSING_VALUE_POINTS} point (cautious, neutral risk).")

st.title("🎓 Student Academic Support System")
st.caption("A rule-based decision support prototype. Enter a student's data to get a support level, "
           "the reasons behind it, and a recommendation.")

tab_single, tab_batch, tab_clean = st.tabs(["👤 Single Student", "📋 Batch (CSV)", "🧹 Data Cleaning"])

# ---------------------------------------------------------------------
# Tab 1: single student
# ---------------------------------------------------------------------
with tab_single:
    with st.form("student_form"):
        st.subheader("Student information")
        student_id = st.text_input("Student ID")

        year_level = st.selectbox(
            "Year Level",
            [1, 2, 3, 4]
        )

        gpa = st.number_input(
            "Previous GPA",
            min_value=1.0,
            max_value=5.0,
            step=0.01
        )
        cols = st.columns(2)
        values = {}
        for i, (name, cfg) in enumerate(core.FEATURES.items()):
            lo, hi = cfg["range"]
            with cols[i % 2]:
                values[name] = st.number_input(
                    cfg["label"], min_value=float(lo), max_value=float(hi), value=None,
                    step=1.0, placeholder=f"{lo}-{hi}, blank if unknown")
        submitted = st.form_submit_button("Determine support level", type="primary",
                                          use_container_width=True)

    if submitted:
        student, errors = {}, []
        for name, raw in values.items():
            # Same validation function as the rest of the system (defense in depth)
            value, error = core.validate_input(name, "" if raw is None else str(raw))
            if error:
                errors.append(error)
            student[name] = value
        if errors:
            for e in errors:
                st.error(e)
        else:
            result = core.classify(student)
            result_card(result)
            st.subheader("Why this decision")
            for reason in result["reasons"]:
                st.markdown(f"- {reason}")
            st.info(f"**Recommendation:** {result['recommendation']}")

# ---------------------------------------------------------------------
# Tab 2: batch
# ---------------------------------------------------------------------
with tab_batch:
    st.subheader("Classify many students at once")
    st.download_button("⬇️ Download CSV template", template_csv(), "student_template.csv",
                       "text/csv")
    file = st.file_uploader("Upload a CSV file", type="csv", key="batch_file")

    if file:
        try:
            raw = pd.read_csv(file)
            df, log = core.clean_dataset(raw)
        except Exception as e:
            st.error(f"Could not process this file: {e}")
            st.stop()

        results = [core.classify(row.to_dict()) for _, row in df.iterrows()]
        df["support_level"] = [r["level"] for r in results]
        df["risk_points"] = [r["points"] for r in results]
        df["explanation"] = ["; ".join(r["reasons"]) for r in results]
        df["recommendation"] = [r["recommendation"] for r in results]

        counts = df["support_level"].value_counts().reindex(["HIGH", "MODERATE", "LOW"], fill_value=0)
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 High", int(counts["HIGH"]))
        c2.metric("🟠 Moderate", int(counts["MODERATE"]))
        c3.metric("🟢 Low", int(counts["LOW"]))
        st.bar_chart(counts)

        view = df.drop(columns=["explanation", "recommendation"])
        styler = view.style
        styler = (styler.map if hasattr(styler, "map") else styler.applymap)(
            color_level, subset=["support_level"])
        st.dataframe(styler, use_container_width=True)

        st.subheader("Why did a student get their level?")
        id_col = "student_id" if "student_id" in df.columns else None
        pick = st.selectbox(
            "Choose a student", df.index,
            format_func=lambda i: str(df.loc[i, id_col]) if id_col else f"Row {i + 1}")
        st.write(f"**{df.loc[pick, 'support_level']}**: {df.loc[pick, 'explanation']}")
        st.info(df.loc[pick, "recommendation"])

        with st.expander("Data cleaning log"):
            for entry in log:
                st.markdown(f"- {entry}")

        st.download_button("⬇️ Download results CSV", df.to_csv(index=False).encode("utf-8"),
                           "results.csv", "text/csv", type="primary")

# ---------------------------------------------------------------------
# Tab 3: data cleaning report
# ---------------------------------------------------------------------
with tab_clean:
    st.subheader("Data quality report")
    file2 = st.file_uploader("Upload the dataset to inspect", type="csv", key="clean_file")

    if file2:
        try:
            raw = pd.read_csv(file2)
            cleaned, log = core.clean_dataset(raw)
        except Exception as e:
            st.error(f"Could not process this file: {e}")
            st.stop()

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(raw))
        c2.metric("Missing cells", int(raw.isna().sum().sum()))
        c3.metric("Duplicate rows", int(raw.duplicated().sum()))

        st.markdown("**Column types and missing values (before cleaning)**")
        overview = pd.DataFrame({"type": raw.dtypes.astype(str), "missing": raw.isna().sum()})
        st.dataframe(overview, use_container_width=True)

        st.markdown("**Cleaning decisions (problem → decision → reason)**")
        for entry in log:
            st.markdown(f"- {entry}")

        st.markdown("**Cleaned data**")
        st.dataframe(cleaned, use_container_width=True)
        st.download_button("⬇️ Download cleaned CSV", cleaned.to_csv(index=False).encode("utf-8"),
                           "cleaned_dataset.csv", "text/csv", type="primary")
