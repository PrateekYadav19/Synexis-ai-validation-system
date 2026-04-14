import streamlit as st
from streamlit_option_menu import option_menu
import re
import pandas as pd
from collections import Counter
import numpy as np
import altair as alt
# --------------------------------------------------------------------
# APP CONFIG
# --------------------------------------------------------------------
st.set_page_config(page_title="Synexis", layout="wide")


# --------------------------------------------------------------------
# AGENT 1 – VALIDATION
# --------------------------------------------------------------------
def simulate_website_active(row) -> bool:
    """
    Demo-friendly website status:
    Use last digit of NPI to decide if website is 'active'.
    This avoids slow/failing HTTP calls on fake URLs.
    """
    npi = str(row.get("NPI", "")).strip()
    if not npi.isdigit():
        return False
    last_digit = int(npi[-1])
    # ~70% active
    return last_digit % 3 != 0

def validate_provider_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Phone_Valid"] = False
    df["NPI_Valid"] = False
    df["Website_Active"] = False
    df["Address_Valid"] = False

    phone_pattern = re.compile(r"^[0-9\-\.\(\) xX+]{7,25}$") 
    def valid_address(addr: str) -> bool:
        return len(addr) > 10 and "," in addr

    for idx, row in df.iterrows():
        phone = str(row.get("Phone", "")).strip()
        df.at[idx, "Phone_Valid"] = bool(phone_pattern.match(phone))

        npi = str(row.get("NPI", "")).strip()
        df.at[idx, "NPI_Valid"] = npi.isdigit() and len(npi) == 10

        df.at[idx, "Website_Active"] = simulate_website_active(row)

        addr = str(row.get("Address", "")).strip()
        df.at[idx, "Address_Valid"] = valid_address(addr)

    return df

# --------------------------------------------------------------------
# AGENT 2 – ENRICHMENT (SIMULATED AI)
# --------------------------------------------------------------------
def enrich_provider_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Website_Title"] = ""
    df["AI_Specialty_Prediction"] = ""
    df["AI_Category"] = ""
    df["AI_Keywords"] = ""
    df["AI_Summary"] = ""

    for idx, row in df.iterrows():
        provider = str(row.get("Provider Name", "Provider")).strip()
        specialty = str(row.get("Specialty", "General Medicine")).strip()
        state = str(row.get("State", "NA")).strip()
        clinic = str(row.get("Clinic Name", "Health Clinic")).strip()

        website_title = f"{clinic} | {specialty} in {state}"

        primary_care_specialties = ["Family Medicine", "Pediatrics", "Internal Medicine"]
        if specialty in primary_care_specialties:
            category = "Primary Care Clinic"
        else:
            category = "Specialist Clinic"

        keywords = f"{specialty}, {state}, {clinic.split()[0]}, healthcare, doctor"

        summary = (
            f"{provider} is a {specialty} provider at {clinic} based in {state}, "
            f"serving patients as a {category.lower()}."
        )

        df.at[idx, "Website_Title"] = website_title
        df.at[idx, "AI_Specialty_Prediction"] = specialty
        df.at[idx, "AI_Category"] = category
        df.at[idx, "AI_Keywords"] = keywords
        df.at[idx, "AI_Summary"] = summary

    return df

# --------------------------------------------------------------------
# AGENT 3 – QUALITY SCORING (EXTREME, TUNED)
# --------------------------------------------------------------------
def compute_quality_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Confidence_Score"] = 100
    df["Trust_Level"] = "High"
    df["Anomaly_Flag"] = False
    df["Anomaly_Notes"] = ""

    npi_counts = Counter(df.get("NPI", []))
    phone_counts = Counter(df.get("Phone", []))
    website_counts = Counter(df.get("Website", []))

    valid_states = {"CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"}

    rng = np.random.default_rng(42)

    for idx, row in df.iterrows():
        # start at 92 with small noise
        score = 92 + rng.integers(-3, 4)
        notes = []

        # serious rules first
        if not row.get("NPI_Valid", False):
            score -= 30
            notes.append("Invalid NPI")

        if not row.get("Phone_Valid", False):
            score -= 20
            notes.append("Invalid phone format")

        if not row.get("Address_Valid", False):
            score -= 12
            notes.append("Suspicious address")

        if not row.get("Website_Active", False):
            score -= 10
            notes.append("Website not reachable")

        specialty = str(row.get("Specialty", "")).strip()
        if specialty == "" or specialty.lower() == "nan":
            score -= 10
            notes.append("Missing specialty")

        state = str(row.get("State", "")).strip()
        if state not in valid_states:
            score -= 6
            notes.append("Unrecognized state code")

        license_no = str(row.get("License Number", "")).strip()
        if not re.match(r"^LIC-\d{4,6}$", license_no):
            score -= 5
            notes.append("Unusual license number format")

        # duplicates
        if npi_counts.get(row.get("NPI"), 0) > 1:
            score -= 20
            notes.append("Duplicate NPI across providers")

        if phone_counts.get(row.get("Phone"), 0) > 3:
            score -= 8
            notes.append("Phone shared across many providers")

        if website_counts.get(row.get("Website"), 0) > 5:
            score -= 6
            notes.append("Website shared across many providers")

        if not str(row.get("Website", "")).strip():
            score -= 6
            notes.append("Missing website URL")

        ai_cat = str(row.get("AI_Category", "")).strip()
        primary_care_specialties = ["Family Medicine", "Pediatrics", "Internal Medicine"]
        if specialty in primary_care_specialties and ai_cat == "Specialist Clinic":
            score -= 6
            notes.append("AI category mismatch: should be primary care")
        if specialty not in primary_care_specialties and ai_cat == "Primary Care Clinic":
            score -= 6
            notes.append("AI category mismatch: should be specialist")

        if len(str(row.get("AI_Summary", "")).strip()) < 25:
            score -= 4
            notes.append("Weak AI summary")

        provider = str(row.get("Provider Name", "")).strip()
        if len(provider) < 5:
            score -= 3
            notes.append("Suspicious provider name")

        clinic = str(row.get("Clinic Name", "")).strip()
        if len(clinic) < 5:
            score -= 3
            notes.append("Suspicious clinic name")

        # clamp
        score = max(0, min(100, score))

        if score >= 80:
            trust = "High"
        elif score >= 55:
            trust = "Moderate"
        else:
            trust = "Low"

        # anomaly only for real issues
        anomaly_flag = (
            score < 80
            or (not row.get("NPI_Valid", False))
            or (not row.get("Phone_Valid", False))
        )

        df.at[idx, "Confidence_Score"] = score
        df.at[idx, "Trust_Level"] = trust
        df.at[idx, "Anomaly_Flag"] = anomaly_flag
        df.at[idx, "Anomaly_Notes"] = "; ".join(notes)

    return df


# --------------------------------------------------------------------
# HYBRID FUTURISTIC + PROFESSIONAL CSS
# --------------------------------------------------------------------
st.markdown(
    """
<style>

/* ==============================
FONTS
============================== */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* ==============================
GLOBAL BASE
============================== */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #E6F4FF !important;
}

/* App background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #02101f 0, #020814 45%, #000000 100%);
}

/* ==============================
HEADER + ANIMATIONS
============================== */
.synexis-header {
    text-align: center;
    padding: 18px 0 10px 0;
}

.synexis-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 40px;
    letter-spacing: 0.35em;
    color: #E6F4FF;
    animation: neon-breathe 4s ease-in-out infinite;
}

.synexis-subtitle {
    font-size: 18px;
    color: #A8EAFF;
}

@keyframes neon-breathe {
    0% {
        text-shadow: 0 0 6px #00EAFF;
        opacity: 0.85;
    }
    50% {
        text-shadow: 0 0 22px #00EAFF, 0 0 40px rgba(0,234,255,0.6);
        opacity: 1;
    }
    100% {
        text-shadow: 0 0 6px #00EAFF;
        opacity: 0.85;
    }
}

.glow-divider {
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, transparent, #00EAFF, transparent);
    margin: 14px 0 6px 0;
    animation: divider-flow 3s linear infinite;
}

@keyframes divider-flow {
    0% {
        opacity: 0.4;
        box-shadow: 0 0 6px rgba(0,234,255,0.4);
    }
    50% {
        opacity: 1;
        box-shadow: 0 0 20px rgba(0,234,255,0.9);
    }
    100% {
        opacity: 0.4;
        box-shadow: 0 0 6px rgba(0,234,255,0.4);
    }
}

/* ==============================
CARDS & SECTIONS
============================== */
.card {
    padding: 22px;
    background: rgba(4, 25, 45, 0.9);
    border: 1px solid rgba(0, 234, 255, 0.4);
    border-radius: 18px;
    box-shadow: 0 0 26px rgba(0, 234, 255, 0.12);
}

.section-title {
    font-weight: 700;
    font-size: 20px;
    color: #00EAFF;
    text-decoration: underline;
    text-underline-offset: 4px;
    margin-bottom: 12px;
}

/* ==============================
SIDEBAR FIX (CRITICAL)
============================== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #010A14, #00060D) !important;
    border-right: 1px solid rgba(0, 234, 255, 0.25);
}

[data-testid="stSidebar"] * {
    color: #B8DFFF !important;
    font-weight: 500;
}

[data-testid="stSidebar"] a {
    color: #9FBED6 !important;
    transition: all 0.3s ease;
}

[data-testid="stSidebar"] a:hover {
    color: #00EAFF !important;
    text-shadow: 0 0 6px rgba(0,234,255,0.8);
}

/* Active menu item */
[data-testid="stSidebar"] .nav-link-selected {
    background: linear-gradient(90deg, #00EAFF, #00BBD4) !important;
    color: #000 !important;
    box-shadow: 0 0 18px rgba(0,234,255,0.9);
    border-radius: 10px;
}

/* Sidebar icon animation */
[data-testid="stSidebar"] svg {
    transition: transform 0.3s ease, filter 0.3s ease;
}

[data-testid="stSidebar"] a:hover svg {
    transform: scale(1.15);
    filter: drop-shadow(0 0 6px #00EAFF);
}

/* ==============================
INPUTS & CONTROLS
============================== */
input, textarea, select {
    color: #FFFFFF !important;
    background-color: #04192D !important;
}

::placeholder {
    color: #9AAFC3 !important;
}

/* ==============================
TABLES / DATAFRAMES
============================== */
thead tr th {
    background-color: #04192D !important;
    color: #00EAFF !important;
    font-weight: 600;
}

tbody tr td {
    background-color: #020B17 !important;
    color: #EAF6FF !important;
}

tbody tr:nth-child(even) td {
    background-color: #031B2E !important;
}

tbody tr:hover td {
    background-color: #063A5A !important;
}

/* ==============================
METRICS
============================== */
[data-testid="stMetricValue"] {
    color: #00EAFF !important;
}

[data-testid="stMetricLabel"] {
    color: #A8EAFF !important;
}

/* ==============================
BUTTONS
============================== */
button {
    color: #000000 !important;
    background-color: #00EAFF !important;
    border-radius: 10px;
    font-weight: 600;
}

button:hover {
    background-color: #4FF3FF !important;
}

</style>

""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------
st.markdown(
    """
<div class="synexis-header">
    <div class="synexis-title">S Y N E X I S</div>
    <div class="synexis-subtitle">Autonomous AI Provider Validation System</div>
    <div class="glow-divider"></div>
</div>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Navigation")
    selected = option_menu(
        menu_title=None,
        options=[
            "Dashboard",
            "Upload Data",
            "Validation Engine",
            "Enrichment Engine",
            "Quality Checks",
            "Final Directory",
        ],
        icons=[
            "speedometer2",
            "cloud-upload",
            "robot",
            "cpu",
            "patch-check",
            "check2-circle",
        ],
        default_index=0,
        styles={
            "container": {"background-color": "transparent"},
            "icon": {"color": "#00eaff"},
            "nav-link": {"color": "#e5f7ff"},
            "nav-link-selected": {"background-color": "#00eaff", "color": "#000"},
        },
    )


def prettify_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    return df.rename(columns=mapping)


# --------------------------------------------------------------------
# PAGE ROUTING
# --------------------------------------------------------------------

# --------------------- DASHBOARD ---------------------
if selected == "Dashboard":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>🛰 AI Operations Overview</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Providers Loaded", len(st.session_state.get("provider_data", [])))
    with col2:
        st.metric("Enriched Profiles", len(st.session_state.get("enriched_data", [])))
    with col3:
        st.metric("Quality Scored Profiles", len(st.session_state.get("quality_data", [])))

    st.info(
        "Synexis orchestrates a 4-agent pipeline: Validation → AI Enrichment → Quality Scoring → Final Directory."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------- UPLOAD DATA ---------------------
elif selected == "Upload Data":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>📤 Upload Provider Data</div>",
        unsafe_allow_html=True,
    )

    csv_file = st.file_uploader("Upload Provider CSV", type=["csv"])

    if csv_file:
        df = pd.read_csv(csv_file)
        st.success(f"✔ Loaded {df.shape[0]} provider records into Synexis.")

        st.session_state["provider_data"] = df

        st.markdown("#### 🔍 Data Preview")
        display_df = prettify_columns(
            df[
                [
                    "Provider Name",
                    "Specialty",
                    "State",
                    "Phone",
                    "NPI",
                    "Website",
                    "Clinic Name",
                ]
            ],
            {
                "Provider Name": "Provider Name",
                "Specialty": "Specialty",
                "State": "State",
                "Phone": "Phone",
                "NPI": "NPI",
                "Website": "Website",
                "Clinic Name": "Clinic Name",
            },
        )
        st.dataframe(display_df, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------- VALIDATION ENGINE ---------------------
elif selected == "Validation Engine":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>🤖 Data Validation Agent</div>",
        unsafe_allow_html=True,
    )

    if "provider_data" not in st.session_state:
        st.warning("⚠ Please upload provider data first from the **Upload Data** page.")
    else:
        df = st.session_state["provider_data"]

        st.write(
            "This agent validates phone numbers, NPIs, address sanity and simulates website reachability."
        )
        if st.button("Run Validation"):
            with st.spinner("🔍 Running validation across all providers..."):
                validated_df = validate_provider_data(df)
                st.session_state["validated_data"] = validated_df

            st.success("✔ Validation completed.")

        if "validated_data" in st.session_state:
            validated_df = st.session_state["validated_data"]
            st.markdown("#### 📊 Validation Results")

            display_df = prettify_columns(
                validated_df[
                    [
                        "Provider Name",
                        "Specialty",
                        "State",
                        "Phone",
                        "NPI",
                        "Website",
                        "Phone_Valid",
                        "NPI_Valid",
                        "Website_Active",
                        "Address_Valid",
                    ]
                ],
                {
                    "Provider Name": "Provider Name",
                    "Specialty": "Specialty",
                    "State": "State",
                    "Phone": "Phone",
                    "NPI": "NPI",
                    "Website": "Website",
                    "Phone_Valid": "Phone Valid",
                    "NPI_Valid": "NPI Valid",
                    "Website_Active": "Website Active",
                    "Address_Valid": "Address Valid",
                },
            )
            st.dataframe(display_df, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------- ENRICHMENT ENGINE ---------------------
elif selected == "Enrichment Engine":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>🧠 Enrichment Agent (AI-Powered)</div>",
        unsafe_allow_html=True,
    )

    base_df = None
    if "validated_data" in st.session_state:
        base_df = st.session_state["validated_data"]
        st.info("Using validated data as input for enrichment.")
    elif "provider_data" in st.session_state:
        base_df = st.session_state["provider_data"]
        st.info("Using raw data as input for enrichment (no validation applied yet).")
    else:
        st.warning("⚠ Please upload provider data first from the **Upload Data** page.")

    if base_df is not None:
        if st.button("Run AI Enrichment"):
            with st.spinner("🧠 Synexis AI is enriching provider profiles..."):
                enriched_df = enrich_provider_data(base_df)
                st.session_state["enriched_data"] = enriched_df

            st.success("✔ Enrichment completed.")

        if "enriched_data" in st.session_state:
            enriched_df = st.session_state["enriched_data"]
            st.markdown("#### 📊 Enriched Provider Directory (AI Fields)")

            display_df = prettify_columns(
                enriched_df[
                    [
                        "Provider Name",
                        "Specialty",
                        "State",
                        "Website",
                        "Website_Title",
                        "AI_Specialty_Prediction",
                        "AI_Category",
                        "AI_Keywords",
                        "AI_Summary",
                    ]
                ],
                {
                    "Provider Name": "Provider Name",
                    "Specialty": "Specialty",
                    "State": "State",
                    "Website": "Website",
                    "Website_Title": "Website Title",
                    "AI_Specialty_Prediction": "AI Specialty Prediction",
                    "AI_Category": "AI Category",
                    "AI_Keywords": "AI Keywords",
                    "AI_Summary": "AI Summary",
                },
            )
            st.dataframe(display_df, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------- QUALITY CHECKS ---------------------
elif selected == "Quality Checks":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>📊 AI Quality & Risk Analytics</div>",
        unsafe_allow_html=True,
    )

    base_df = None
    if "enriched_data" in st.session_state:
        base_df = st.session_state["enriched_data"]
        st.info("Using enriched data for quality scoring.")
    elif "validated_data" in st.session_state:
        base_df = st.session_state["validated_data"]
        st.info("Using validated data for quality scoring.")
    elif "provider_data" in st.session_state:
        base_df = st.session_state["provider_data"]
        st.info("Using raw data for quality scoring (limited signals).")
    else:
        st.warning("⚠ Please upload provider data first from the **Upload Data** page.")

    if base_df is not None:
        if st.button("Run Quality Checks"):
            with st.spinner("📈 Computing confidence scores and detecting anomalies..."):
                quality_df = compute_quality_scores(base_df)
                st.session_state["quality_data"] = quality_df

            st.success("✔ Quality analysis completed.")

        if "quality_data" in st.session_state:
            quality_df = st.session_state["quality_data"]

            # Controls
            st.markdown("#### 🔎 AI Analytics Controls")
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                search = st.text_input("Search provider / clinic / state")
            with c2:
                trust_filter = st.selectbox(
                    "Filter by Trust Level", ["All", "High", "Moderate", "Low"]
                )
            with c3:
                min_score = st.slider("Minimum Confidence Score", 0, 100, 60, step=5)

            filtered = quality_df.copy()

            if search:
                s = search.lower()
                filtered = filtered[
                    filtered["Provider Name"].str.lower().str.contains(s)
                    | filtered["Clinic Name"].str.lower().str.contains(s)
                    | filtered["State"].astype(str).str.lower().str.contains(s)
                ]

            if trust_filter != "All":
                filtered = filtered[filtered["Trust_Level"] == trust_filter]

            filtered = filtered[filtered["Confidence_Score"] >= min_score]

            # -------- Charts (Altair with explicit types) --------
            st.markdown("#### 📊 AI Analytics Dashboard")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                trust_counts = (
                    quality_df["Trust_Level"]
                    .value_counts()
                    .reset_index()
                    .rename(columns={"index": "Trust_Level", "Trust_Level": "Count"})
                )
                if not trust_counts.empty:
                    chart = (
                        alt.Chart(trust_counts)
                        .mark_bar()
                        .encode(
                            x=alt.X("Trust_Level:N", title="Trust Level"),
                            y=alt.Y("Count:Q", title="Providers"),
                            tooltip=["Trust_Level:N", "Count:Q"],
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(chart, use_container_width=True)

            with chart_col2:
                spec_counts = (
                    quality_df["Specialty"]
                    .value_counts()
                    .reset_index()
                    .rename(columns={"index": "Specialty", "Specialty": "Count"})
                    .head(7)
                )
                if not spec_counts.empty:
                    chart2 = (
                        alt.Chart(spec_counts)
                        .mark_bar()
                        .encode(
                            x=alt.X("Specialty:N", title="Specialty"),
                            y=alt.Y("Count:Q", title="Providers"),
                            tooltip=["Specialty:N", "Count:Q"],
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(chart2, use_container_width=True)

            # High-risk table
            st.markdown("#### ⚠ High-Risk Providers (Low Trust)")
            risky = quality_df[quality_df["Trust_Level"] == "Low"]
            if risky.empty:
                st.info("No low-trust providers detected in the current dataset.")
            else:
                risky_disp = prettify_columns(
                    risky[
                        [
                            "Provider Name",
                            "Specialty",
                            "State",
                            "Confidence_Score",
                            "Trust_Level",
                            "Anomaly_Notes",
                        ]
                    ],
                    {
                        "Provider Name": "Provider Name",
                        "Specialty": "Specialty",
                        "State": "State",
                        "Confidence_Score": "Confidence Score",
                        "Trust_Level": "Trust Level",
                        "Anomaly_Notes": "Anomaly Notes",
                    },
                )
                st.dataframe(risky_disp, use_container_width=True)

            # Filtered directory
            st.markdown("#### 📂 Full Quality-Scored Directory (Filtered View)")
            disp = prettify_columns(
                filtered[
                    [
                        "Provider Name",
                        "Specialty",
                        "State",
                        "Phone",
                        "NPI",
                        "Website",
                        "Confidence_Score",
                        "Trust_Level",
                        "Anomaly_Flag",
                        "Anomaly_Notes",
                    ]
                ],
                {
                    "Provider Name": "Provider Name",
                    "Specialty": "Specialty",
                    "State": "State",
                    "Phone": "Phone",
                    "NPI": "NPI",
                    "Website": "Website",
                    "Confidence_Score": "Confidence Score",
                    "Trust_Level": "Trust Level",
                    "Anomaly_Flag": "Anomaly Flag",
                    "Anomaly_Notes": "Anomaly Notes",
                },
            )
            st.dataframe(disp, use_container_width=True)

            # Provider profile
            st.markdown("#### 🧾 Provider Profile View")
            if not filtered.empty:
                names = filtered["Provider Name"].tolist()
                selected_name = st.selectbox("Select Provider", names)
                profile = filtered[filtered["Provider Name"] == selected_name].iloc[0]

                st.markdown(
                    f"""
                    <div class="card">
                        <div class="section-title">📌 {profile['Provider Name']}</div>
                        <b>Clinic:</b> {profile['Clinic Name']}<br>
                        <b>Specialty:</b> {profile['Specialty']}<br>
                        <b>State:</b> {profile['State']}<br>
                        <b>Phone:</b> {profile['Phone']}<br>
                        <b>NPI:</b> {profile['NPI']}<br>
                        <b>Website:</b> {profile['Website']}<br><br>
                        <b>Confidence Score:</b> {profile['Confidence_Score']}<br>
                        <b>Trust Level:</b> {profile['Trust_Level']}<br>
                        <b>Anomalies:</b> {profile['Anomaly_Notes'] or "None"}<br>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Export
            st.markdown("#### 📥 Export Quality-Scored Directory")
            csv_bytes = quality_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download as CSV",
                data=csv_bytes,
                file_name="synexis_quality_scored_directory.csv",
                mime="text/csv",
            )

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------- FINAL DIRECTORY -------------------
elif selected == "Final Directory":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>📁 Final Provider Directory</div>",
        unsafe_allow_html=True,
    )

    final_df = None
    if "quality_data" in st.session_state:
        final_df = st.session_state["quality_data"]
        st.success("Showing final directory with validation, enrichment & quality scores.")
    elif "enriched_data" in st.session_state:
        final_df = st.session_state["enriched_data"]
        st.info("Showing enriched directory (no quality scores yet).")
    elif "validated_data" in st.session_state:
        final_df = st.session_state["validated_data"]
        st.info("Showing validated directory (no enrichment or quality scoring yet).")
    elif "provider_data" in st.session_state:
        final_df = st.session_state["provider_data"]
        st.info("Showing raw uploaded directory.")
    else:
        st.warning("⚠ Please upload and process provider data first.")

    if final_df is not None:
        st.dataframe(final_df, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
