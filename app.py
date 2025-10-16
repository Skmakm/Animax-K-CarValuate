import streamlit as st
st.set_page_config(page_title="Animax K CarValuate", page_icon="🚗", layout="wide")

from auth import run_authentication


run_authentication()
if not st.session_state.get("authenticated", False):
    st.stop()



import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
from datetime import datetime
from sklearn.linear_model import LinearRegression
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, PageBreak, Image as RLImage
)
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


import plotly.express as px
import plotly.graph_objects as go


from streamlit_option_menu import option_menu


MODEL_FILE = "LinearRegressionModel.pkl"
DATA_FILE = "Cars.csv"
HISTORY_FILE = "prediction_history.csv"
LOGO_FILE = "assets/logo.png"  # Place your logo here

if not os.path.exists(MODEL_FILE):
    st.error(f"Model file not found: {MODEL_FILE}. Place your trained model there.")
    st.stop()
if not os.path.exists(DATA_FILE):
    st.error(f"Dataset not found: {DATA_FILE}. Place your dataset there.")
    st.stop()

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

df = pd.read_csv(DATA_FILE)

PALETTE = {
    "primary": "#2b6cb0",
    "accent": "#38a169",
    "muted": "#718096",
    "highlight": "#dd6b20",
    "table_header": colors.HexColor("#2b6cb0"),
    "table_odd": colors.whitesmoke,
    "table_even": colors.lightgrey,
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.grid": True,
    "grid.color": "#e6e6e6",
})

st.markdown(f"""
<style>
/* Card look for KPI containers */
.card {{
  border-radius: 10px;
  background: white;
  padding: 16px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.04);
  border: 1px solid #eef2f6;
  margin-bottom: 12px;
}}
.metric-big {{ font-size:26px; font-weight:700; color:{PALETTE['accent']}; }}
.small-muted {{ color: #6b7280; font-size:13px; }}
.footer {{
    color: #9aa4b2;
    text-align:center;
    padding-top:12px;
    padding-bottom:6px;
    font-size:13px;
}}
/* Sidebar tweaks */
[data-testid="stSidebar"] .css-1d391kg {{ padding-top: 1rem; }}
</style>
""", unsafe_allow_html=True)


def predict_price_from_features(features_df):
    """Return predicted price in lakhs (float)."""
    try:
        p = model.predict(features_df)[0]
        return float(p)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

def cap_rupee_value(v, cap=50_000_000):
    """Cap rupee value to a maximum (default 5 crore)."""
    try:
        return min(int(v), int(cap))
    except Exception:
        return int(cap)

def dynamic_price_cap_rupees():
    """Compute a dynamic max rupees from dataset (2x max observed), capped at 5 Cr."""
    try:
        ds_max = int(df["Price"].max() * 100_000 * 2)  # dataset price in lakhs -> rupees
        return cap_rupee_value(ds_max, cap=50_000_000)
    except Exception:
        return 50_000_000

def generate_emi_schedule(loan_amount, annual_rate, tenure_years):
    """Generate EMI value and amortization schedule list of dicts."""
    r = (annual_rate / 100) / 12
    n = int(tenure_years * 12)
    if r > 0:
        emi = loan_amount * r * ((1 + r) * n) / (((1 + r) * n) - 1)
    else:
        emi = loan_amount / n
    schedule = []
    balance = loan_amount
    for m in range(1, n + 1):
        interest_comp = balance * r
        principal_comp = emi - interest_comp
        balance = max(balance - principal_comp, 0)
        schedule.append({
            "Month": m,
            "EMI": round(emi, 2),
            "Principal": round(principal_comp, 2),
            "Interest": round(interest_comp, 2),
            "Balance": round(balance, 2)
        })
    return emi, schedule

def save_fig_to_bytes(fig, fmt="png", dpi=150):
    """Save matplotlib figure to BytesIO and return buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf

def styled_table_for_pdf(df_table):
    """Return a ReportLab Table object styled for the PDF."""
    data = [df_table.columns.tolist()] + df_table.values.tolist()
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALETTE["table_header"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PALETTE["table_odd"], PALETTE["table_even"]]),
    ]))
    return t


if "history" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        try:
            st.session_state["history"] = pd.read_csv(HISTORY_FILE).to_dict("records")
        except Exception:
            st.session_state["history"] = []
    else:
        st.session_state["history"] = []

def save_to_history(record):
    """
    Save a prediction record into session history and CSV.
    Caps the predicted price (lakhs) to a safe upper bound to avoid storing insane values.
    """
    try:
        
        max_lakhs = 500
        val = float(record.get("Predicted Price (₹ lakhs)", 0))
        record["Predicted Price (₹ lakhs)"] = round(min(val, max_lakhs), 2)
    except Exception:
        record["Predicted Price (₹ lakhs)"] = 0.0

    st.session_state["history"].insert(0, record)
    pd.DataFrame(st.session_state["history"]).to_csv(HISTORY_FILE, index=False)


with st.sidebar:
    
    if os.path.exists(LOGO_FILE):
        st.image(LOGO_FILE, width=160)
    st.markdown("### Animax K CarValuate")
    
    choice = option_menu(
        menu_title=None,
        options=[
            "Home Dashboard",
            "Single Car Prediction",
            "Compare Multiple Cars",
            "Price Trends & Forecasting",
            "Loan/EMI Calculator",
            "Maintenance Cost Estimator",
            "Car Valuation Report",
            "Market Insights & History"
        ],
        icons=["house", "robot", "list", "graph-up", "wallet2", "wrench", "file-earmark-pdf", "bar-chart"],
        menu_icon="cast",
        default_index=0,
    )

if choice == "Home Dashboard":
    
    if os.path.exists(LOGO_FILE): 
        col_logo, col_title = st.columns([1, 3])
        with col_logo: 
            st.image(LOGO_FILE, width=120) 
        with col_title:
            st.markdown("## Animax K CarValuate")
            st.markdown("<p class='small-muted'>AI-powered valuation, comparisons, forecasting, and professional reports.</p>", unsafe_allow_html=True) 
    else:
        st.markdown("## Animax K CarValuate")
        st.markdown("<p class='small-muted'>AI-powered valuation, comparisons, forecasting, and professional reports.</p>", unsafe_allow_html=True)
        
    
    st.markdown("<div class='card'><h2>Animax K CarValuate</h2><p class='small-muted'>AI-powered valuation, comparisons, forecasting, and professional reports.</p></div>", unsafe_allow_html=True)
    st.title("Market Overview")


    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='card'><h4>Vehicles in Dataset</h4><div class='metric-big'>{len(df):,}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='card'><h4>Avg Price (Lakhs)</h4><div class='metric-big'>{df['Price'].mean():.2f}</div></div>", unsafe_allow_html=True)
    with col3:
        try:
            top_company = df['Company'].mode()[0]
        except Exception:
            top_company = "N/A"
        st.markdown(f"<div class='card'><h4>Top Company</h4><div class='metric-big'>{top_company}</div></div>", unsafe_allow_html=True)

    
    st.subheader("Average Price by Company (Top 10)")
    avg_company = df.groupby("Company")["Price"].mean().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(avg_company, x="Company", y="Price", text="Price", title="Top 10 Companies by Avg Price")
    fig.update_layout(margin=dict(t=40, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    
    st.subheader("Depreciation Trend (Avg Price by Year)")
    avg_year = df.groupby("Year")["Price"].mean().reset_index()
    fig2 = px.line(avg_year, x="Year", y="Price", markers=True, title="Average Price by Year")
    fig2.update_layout(margin=dict(t=40, l=10, r=10))
    st.plotly_chart(fig2, use_container_width=True)


elif choice == "Single Car Prediction":
    st.header("Single Car Prediction")
    st.markdown("Enter details below. Tooltips provide help for input fields.")
    st.sidebar.subheader("Prediction inputs")

    
    company = st.sidebar.selectbox("Company", sorted(df["Company"].unique()))
    car_name = st.sidebar.selectbox("Model", sorted(df[df["Company"] == company]["Name"].unique()))
    location = st.sidebar.selectbox("Location", sorted(df["Location"].unique()))
    year = st.sidebar.number_input("Year", int(df["Year"].min()), int(df["Year"].max()), value=2015)
    kms_driven = st.sidebar.number_input("Kms Driven", 0, 500000, 20000, step=500)
    fuel_type = st.sidebar.selectbox("Fuel Type", df["Fuel_type"].unique())
    owner = st.sidebar.selectbox("Owner Type", df["Owner"].unique())
    label = st.sidebar.selectbox("Label/Variant", df["Label"].unique())

    st.markdown("<div class='card'><h4>Single Car Prediction</h4><p class='small-muted'>Use the sidebar to change car details and click Predict.</p></div>", unsafe_allow_html=True)

    features = pd.DataFrame({
        "Name": [car_name],
        "Label": [label],
        "Location": [location],
        "Kms_driven": [kms_driven],
        "Fuel_type": [fuel_type],
        "Owner": [owner],
        "Year": [year],
        "Company": [company],
    })

    if st.sidebar.button("Predict Price"):
        with st.spinner("Estimating price..."):
            price_lakhs = predict_price_from_features(features)
            if price_lakhs is not None:
                st.success(f"Estimated Price: ₹ {price_lakhs:,.2f} lakhs")
                # Save to history (preserve CSV format)
                save_to_history({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Company": company,
                    "Model": car_name,
                    "Year": year,
                    "Predicted Price (₹ lakhs)": round(price_lakhs, 2)
                })

                # Market snapshot comparison (Plotly)
                overall_avg = df["Price"].mean()
                company_avg = df[df["Company"] == company]["Price"].mean() if company in df["Company"].unique() else np.nan
                model_avg = df[df["Name"] == car_name]["Price"].mean() if car_name in df["Name"].unique() else np.nan
                cmp_df = pd.DataFrame({
                    "Metric": ["Predicted", "Company Avg", "Model Avg", "Overall Avg"],
                    "Price": [price_lakhs, company_avg, model_avg, overall_avg]
                }).dropna()

                st.subheader("Market Comparison")
                fig_cmp = px.bar(cmp_df, x="Metric", y="Price", text="Price", title="Comparison (₹ lakhs)")
                fig_cmp.update_layout(margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(fig_cmp, use_container_width=True)


elif choice == "Compare Multiple Cars":
    st.header("Compare Multiple Cars")
    st.markdown("Provide details for each car in the sidebar-like inputs below.")

    num_cars = st.number_input("Number of cars to compare", 1, 5, 2)
    car_inputs = []
    
    for i in range(int(num_cars)):
        st.markdown(f"**Car {i+1} details**")
        c1, c2, c3 = st.columns(3)
        with c1:
            company_i = st.selectbox(f"Company (Car {i+1})", sorted(df["Company"].unique()), key=f"cmp{i}")
            model_i = st.selectbox(f"Model (Car {i+1})", sorted(df[df["Company"] == company_i]["Name"].unique()), key=f"mod{i}")
        with c2:
            year_i = st.number_input(f"Year (Car {i+1})", int(df["Year"].min()), int(df["Year"].max()), 2015, key=f"y{i}")
            kms_i = st.number_input(f"Kms (Car {i+1})", 0, 500000, 20000, step=500, key=f"k{i}")
        with c3:
            fuel_i = st.selectbox(f"Fuel (Car {i+1})", df["Fuel_type"].unique(), key=f"f{i}")
            owner_i = st.selectbox(f"Owner (Car {i+1})", df["Owner"].unique(), key=f"o{i}")
        car_inputs.append({
            "Name": model_i, "Label": "", "Location": "",  
            "Kms_driven": kms_i, "Fuel_type": fuel_i, "Owner": owner_i,
            "Year": year_i, "Company": company_i
        })

    if st.button("Compare Prices"):
        rows = []
        with st.spinner("Predicting prices for selected cars..."):
            for c in car_inputs:
                p = predict_price_from_features(pd.DataFrame([c]))
                rows.append({
                    "Company": c["Company"], "Model": c["Name"], "Year": c["Year"],
                    "Predicted Price (₹ lakhs)": round(p, 2) if p is not None else np.nan
                })
        res_df = pd.DataFrame(rows)
        st.subheader("Comparison Results")
        st.dataframe(res_df, use_container_width=True)

        
        fig_bar = px.bar(res_df, x="Model", y="Predicted Price (₹ lakhs)", text="Predicted Price (₹ lakhs)",
                         title="Predicted Prices Comparison")
        fig_bar.update_layout(margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_bar, use_container_width=True)

elif choice == "Price Trends & Forecasting":
    st.header("Price Trends & Forecasting")
    st.markdown("View historical average prices and forecast future years using a simple linear model (good for quick estimates).")

    trend_mode = st.radio("View trend by", ["By Model", "By Company"])
    n_future = st.slider("Forecast years ahead", 1, 10, 3)

    if trend_mode == "By Model":
        company_sel = st.selectbox("Company", sorted(df["Company"].unique()))
        model_sel = st.selectbox("Model", sorted(df[df["Company"] == company_sel]["Name"].unique()))
        hist = df[(df["Company"] == company_sel) & (df["Name"] == model_sel)].groupby("Year")["Price"].mean().reset_index()
    else:
        company_sel = st.selectbox("Company", sorted(df["Company"].unique()))
        hist = df[df["Company"] == company_sel].groupby("Year")["Price"].mean().reset_index()

    if hist.empty:
        st.warning("Not enough historical data to display trends.")
    else:
        
        fig = px.line(hist, x="Year", y="Price", markers=True, title="Historical Average Price by Year")
        st.plotly_chart(fig, use_container_width=True)

        
        if len(hist) >= 3:
            lr = LinearRegression().fit(hist[["Year"]].values, hist["Price"].values)
            last_year = int(hist["Year"].max())
            future_years = np.arange(last_year + 1, last_year + 1 + n_future)
            preds = lr.predict(future_years.reshape(-1, 1))
            forecast_df = pd.DataFrame({"Year": future_years, "Forecast_Price (₹ lakhs)": np.round(preds, 2)})
            st.subheader("Forecast")
            st.dataframe(forecast_df, use_container_width=True)
            st.download_button("Download Forecast CSV", data=forecast_df.to_csv(index=False).encode("utf-8"),
                               file_name=f"forecast_{company_sel}_{model_sel if trend_mode=='By Model' else company_sel}.csv",
                               mime="text/csv")
        else:
            st.info("At least 3 years of historical data required for forecasting.")


elif choice == "Loan/EMI Calculator":
    st.header("Loan / EMI Calculator")
    st.markdown("Estimate EMI and view amortization schedule.")

    max_rupees = dynamic_price_cap_rupees()
    default_price = None
    if st.session_state["history"]:
        try:
            last_price_lakhs = st.session_state["history"][0]["Predicted Price (₹ lakhs)"]
            default_price = cap_rupee_value(last_price_lakhs * 100_000, cap=max_rupees)
        except Exception:
            default_price = None

    car_price = st.number_input("Car Price (₹)", min_value=50_000, max_value=max_rupees,
                                        value=default_price if default_price else 1_000_000, step=50_000)
    loan_percent = st.slider("Loan Amount (% of Price)", 50, 100, 80)
    interest_rate = st.number_input("Annual Interest Rate (%)", 1.0, 25.0, 9.5, step=0.1)
    tenure_years = st.slider("Loan Tenure (years)", 1, 10, 5)

    loan_amount = car_price * (loan_percent / 100)
    emi, schedule = generate_emi_schedule(loan_amount, interest_rate, tenure_years)
    total_payment = emi * int(tenure_years * 12)
    total_interest = total_payment - loan_amount

    st.markdown("<div class='card'><h4>Loan Summary</h4></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Loan Amount (₹)", f"{loan_amount:,.0f}")
    c2.metric("Monthly EMI (₹)", f"{emi:,.0f}")
    c3.metric("Total Interest (₹)", f"{total_interest:,.0f}")

    emi_df = pd.DataFrame(schedule)
    
    fig_bal = px.line(emi_df, x="Month", y="Balance", title="Loan Balance Over Time", markers=True)
    st.plotly_chart(fig_bal, use_container_width=True)

    st.subheader("EMI Schedule (first 24 months)")
    st.dataframe(emi_df.head(24), use_container_width=True)
    st.download_button("Download EMI Schedule (CSV)", data=emi_df.to_csv(index=False).encode("utf-8"),
                       file_name="emi_schedule.csv", mime="text/csv")

elif choice == "Maintenance Cost Estimator":
    st.header("Maintenance Cost Estimator")
    st.markdown("Estimate annual maintenance cost based on assumptions.")

    max_rupees = dynamic_price_cap_rupees()
    default_price = None
    if st.session_state["history"]:
        try:
            last_price_lakhs = st.session_state["history"][0]["Predicted Price (₹ lakhs)"]
            default_price = cap_rupee_value(last_price_lakhs * 100_000, cap=max_rupees)
        except Exception:
            default_price = None

    car_price = st.number_input("Car Price (₹)", min_value=50_000, max_value=max_rupees,
                                        value=default_price if default_price else 1_000_000, step=50_000)
    car_age = st.slider("Car Age (years)", 0, 20, 3)
    mileage = st.number_input("Annual Mileage (km)", min_value=0, max_value=300000, value=15000, step=1000)

    
    base_cost = 0.025 * car_price
    age_factor = base_cost * (0.05 * car_age)
    mileage_factor = base_cost * (0.1 if mileage > 15000 else 0.05)
    annual_cost = base_cost + age_factor + mileage_factor

    st.write(f"Base Cost (2.5% of price): ₹ {base_cost:,.0f}")
    st.write(f"Age Factor: ₹ {age_factor:,.0f}")
    st.write(f"Mileage Factor: ₹ {mileage_factor:,.0f}")
    st.subheader("Estimated Annual Maintenance Cost")
    st.markdown(f"<div class='card'><div class='metric-big'>₹ {annual_cost:,.0f}</div></div>", unsafe_allow_html=True)

    years = np.arange(1, 6)
    costs = [annual_cost * (1 + 0.08 * y) for y in years]
    proj_df = pd.DataFrame({"Years Ahead": years, "Projected Cost (₹)": np.round(costs, 0)})

    with st.expander("Projection details"):
        st.dataframe(proj_df)
        fig_proj = px.bar(proj_df, x="Years Ahead", y="Projected Cost (₹)", title="Maintenance Cost Projection (Next 5 Years)")
        st.plotly_chart(fig_proj, use_container_width=True)


elif choice == "Car Valuation Report":
    st.header("Car Valuation Report")
    st.markdown("Use inputs to generate a branded PDF valuation report (includes charts).")

    
    company = st.selectbox("Company", sorted(df["Company"].unique()))
    car_name = st.selectbox("Model", sorted(df[df["Company"] == company]["Name"].unique()))
    location = st.selectbox("Location", sorted(df["Location"].unique()))
    year = st.number_input("Year", int(df["Year"].min()), int(df["Year"].max()), 2015)
    kms_driven = st.number_input("Kms Driven", 0, 500000, 20000, step=500)
    fuel_type = st.selectbox("Fuel Type", df["Fuel_type"].unique())
    owner = st.selectbox("Owner Type", df["Owner"].unique())
    label = st.selectbox("Label/Variant", df["Label"].unique())

    st.sidebar.subheader("Report assumptions")
    maint_age = st.sidebar.slider("Assumed car age (years)", 0, 20, 3)
    maint_mileage = st.sidebar.number_input("Assumed annual mileage (km)", min_value=0, max_value=300000, value=15000, step=1000)
    report_loan_percent = st.sidebar.slider("Loan % for EMI summary", 0, 100, 80)
    report_interest = st.sidebar.number_input("Interest % for EMI summary", 1.0, 25.0, 9.5, step=0.1)
    report_tenure = st.sidebar.slider("Tenure (yrs) for EMI summary", 1, 10, 5)

    if st.button("Generate Valuation Report (PDF)"):
        with st.spinner("Creating PDF, please wait..."):
            features = pd.DataFrame({
                "Name":[car_name], "Label":[label], "Location":[location],
                "Kms_driven":[kms_driven], "Fuel_type":[fuel_type], "Owner":[owner],
                "Year":[year], "Company":[company]
            })
            price_lakhs = predict_price_from_features(features)
            if price_lakhs is None:
                st.error("Prediction failed — cannot create report.")
            else:
                predicted_price_rupees = cap_rupee_value(int(price_lakhs * 100_000), cap=max(dynamic_price_cap_rupees(), 50_000_000))

                
                base_cost = 0.025 * predicted_price_rupees
                age_factor = base_cost * (0.05 * maint_age)
                mileage_factor = base_cost * (0.1 if maint_mileage > 15000 else 0.05)
                annual_cost = base_cost + age_factor + mileage_factor
                years_arr = np.arange(1, 6)
                maint_costs = [annual_cost * (1 + 0.08 * y) for y in years_arr]
                maint_df = pd.DataFrame({"Years Ahead": years_arr, "Projected Cost (₹)": np.round(maint_costs, 0)})

                
                loan_amount = predicted_price_rupees * (report_loan_percent / 100)
                emi_val, emi_schedule = generate_emi_schedule(loan_amount, report_interest, report_tenure)
                emi_df_report = pd.DataFrame(emi_schedule)

                
                fig_maint, ax_m = plt.subplots()
                ax_m.bar(maint_df["Years Ahead"], maint_df["Projected Cost (₹)"], color=PALETTE["highlight"])
                ax_m.set_title("Maintenance Cost Projection (Next 5 Years)")
                ax_m.set_xlabel("Years Ahead")
                ax_m.set_ylabel("Projected Annual Cost (₹)")
                ax_m.grid(axis="y", linestyle="--", linewidth=0.5)
                maint_buf = save_fig_to_bytes(fig_maint)

                
                fig_loan, ax_l = plt.subplots()
                ax_l.plot(emi_df_report["Month"], emi_df_report["Balance"], marker="o", color=PALETTE["primary"])
                ax_l.set_title("Loan Balance Over Time")
                ax_l.set_xlabel("Month")
                ax_l.set_ylabel("Remaining Balance (₹)")
                ax_l.grid(True, linestyle="--", linewidth=0.5)
                loan_buf = save_fig_to_bytes(fig_loan)

                
                hist = df[df["Name"] == car_name].groupby("Year")["Price"].mean().reset_index()
                if not hist.empty:
                    fig_price, ax_p = plt.subplots()
                    ax_p.plot(hist["Year"], hist["Price"], marker="o", color=PALETTE["primary"])
                    ax_p.set_title("Historical Avg Price")
                    ax_p.set_xlabel("Year")
                    ax_p.set_ylabel("Price (lakhs)")
                    ax_p.grid(True, linestyle="--", linewidth=0.5)
                    price_buf = save_fig_to_bytes(fig_price)
                else:
                    price_buf = None

                
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
                styles = getSampleStyleSheet()
                styleH = styles["Heading1"]
                styleH2 = styles["Heading2"]
                normal = styles["Normal"]

                elements = []
                
                if os.path.exists(LOGO_FILE):
                    elements.append(RLImage(LOGO_FILE, width=2*inch, height=2*inch))
                elements.append(Paragraph("Car Valuation Report", styleH))
                elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal))
                elements.append(Paragraph(" ", normal))
                elements.append(Paragraph(f"{company} {car_name} ({year})", styleH2))
                elements.append(Paragraph(f"Location: {location} | Kms: {kms_driven:,} | Owner: {owner}", normal))
                elements.append(Paragraph(f"Predicted Price: ₹ {price_lakhs:,.2f} lakhs (₹ {predicted_price_rupees:,.0f})", normal))
                elements.append(PageBreak())

            
                elements.append(Paragraph("Table of Contents", styleH2))
                toc = ["1. Valuation & Market Comparison", "2. Recommendations", "3. Price Trends (chart)",
                       "4. Maintenance (table + chart)", "5. Loan / EMI (table + chart)"]
                for line in toc:
                    elements.append(Paragraph(line, normal))
                elements.append(PageBreak())

                
                elements.append(Paragraph("1. Valuation & Market Comparison", styleH2))
                overall_avg = df["Price"].mean()
                company_avg = df[df["Company"] == company]["Price"].mean() if company in df["Company"].unique() else np.nan
                model_avg = df[df["Name"] == car_name]["Price"].mean() if car_name in df["Name"].unique() else np.nan
                val_table = pd.DataFrame({
                    "Item": ["Predicted Price (lakhs)", "Predicted Price (₹)", "Market Avg (lakhs)", "Company Avg (lakhs)", "Model Avg (lakhs)"],
                    "Value": [f"{price_lakhs:,.2f}", f"{predicted_price_rupees:,.0f}", f"{overall_avg:,.2f}", f"{company_avg:,.2f}", f"{model_avg:,.2f}"]
                })
                elements.append(styled_table_for_pdf(val_table))
                elements.append(PageBreak())

            
                elements.append(Paragraph("2. Recommendations", styleH2))
                tips = []
                if price_lakhs > company_avg:
                    tips.append("Your car is valued above company average — market conditions favor selling.")
                if price_lakhs < company_avg:
                    tips.append("The car is valued below company average — consider negotiating if buying or inspect for issues if selling.")
                if year <= datetime.now().year - 8:
                    tips.append("Older than 8 years — depreciation may accelerate; highlight maintenance history when selling.")
                if kms_driven > 150000:
                    tips.append("High mileage — resale value will be reduced; provide records of recent major services.")
                if not tips:
                    tips.append("No specific flags — standard market conditions.")
                for t in tips:
                    elements.append(Paragraph(f"- {t}", normal))
                elements.append(PageBreak())

            
                elements.append(Paragraph("3. Price Trends", styleH2))
                if price_buf:
                    elements.append(RLImage(price_buf, width=6.5*inch, height=3*inch))
                else:
                    elements.append(Paragraph("No historical price trend available for this model.", normal))
                elements.append(PageBreak())

            
                elements.append(Paragraph("4. Maintenance", styleH2))
                elements.append(styled_table_for_pdf(maint_df.rename(columns={"Years Ahead": "Years Ahead", "Projected Cost (₹)": "Projected Cost (₹)"})))
                elements.append(Paragraph(" ", normal))
                elements.append(RLImage(maint_buf, width=6.5*inch, height=3*inch))
                elements.append(PageBreak())

            
                elements.append(Paragraph("5. Loan / EMI Summary", styleH2))
                emi_summary = pd.DataFrame({
                    "Metric": ["Loan Amount (₹)", "Monthly EMI (₹)", "Total Payment (₹)", "Total Interest (₹)"],
                    "Value": [f"{loan_amount:,.2f}", f"{emi_val:,.2f}", f"{emi_val * int(report_tenure * 12):,.2f}", f"{emi_val * int(report_tenure * 12) - loan_amount:,.2f}"]
                })
                elements.append(styled_table_for_pdf(emi_summary))
                elements.append(Paragraph(" ", normal))
                emi_preview = emi_df_report.head(12)
                elements.append(styled_table_for_pdf(emi_preview.rename(columns=lambda c: str(c))))
                elements.append(Paragraph(" ", normal))
                elements.append(RLImage(loan_buf, width=6.5*inch, height=3*inch))

                
                doc.build(elements)
                pdf_value = buffer.getvalue()
                buffer.close()

                st.success("Valuation report generated.")
                st.download_button("Download Valuation Report (PDF)", data=pdf_value,
                                   file_name=f"car_valuation_report_{company}_{car_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                   mime="application/pdf")


elif choice == "Market Insights & History":
    st.header("Market Insights & Prediction History")
    st.sidebar.subheader("Market filters (insights)")

    company_filter = st.sidebar.multiselect("Filter by Company (insights)", sorted(df["Company"].unique()))
    fuel_filter = st.sidebar.multiselect("Filter by Fuel Type (insights)", sorted(df["Fuel_type"].unique()))
    year_range = st.sidebar.slider("Year range (insights)", int(df["Year"].min()), int(df["Year"].max()), (2010, 2020))

    df_filtered = df.copy()
    if company_filter:
        df_filtered = df_filtered[df_filtered["Company"].isin(company_filter)]
    if fuel_filter:
        df_filtered = df_filtered[df_filtered["Fuel_type"].isin(fuel_filter)]
    df_filtered = df_filtered[(df_filtered["Year"] >= year_range[0]) & (df_filtered["Year"] <= year_range[1])]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'><h4>Price distribution</h4></div>", unsafe_allow_html=True)
        fig_hist = px.histogram(df_filtered, x="Price", nbins=30, title="Car Price Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.markdown("<div class='card'><h4>Average price by company (top 10)</h4></div>", unsafe_allow_html=True)
        avg_company = df_filtered.groupby("Company")["Price"].mean().sort_values(ascending=False).head(10)
        fig_bar = px.bar(avg_company.reset_index(), x="Company", y="Price", title="Avg Price by Company (Top 10)")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.write("Depreciation over years")
    avg_year = df_filtered.groupby("Year")["Price"].mean().reset_index()
    st.line_chart(avg_year.set_index("Year")["Price"])

    st.markdown("---")
    st.subheader("Prediction History")
    if st.session_state["history"]:
        history_df = pd.DataFrame(st.session_state["history"])
        if "Timestamp" in history_df.columns:
            history_df = history_df.sort_values(by="Timestamp", ascending=False)

        tab_hist1, tab_hist2 = st.tabs(["History Table", "Downloads"])

        with tab_hist1:
            st.dataframe(history_df, use_container_width=True)


        buf_hist = io.BytesIO()
        doc_h = SimpleDocTemplate(buf_hist, pagesize=letter)
        styles = getSampleStyleSheet()
        elems = []
        elems.append(Paragraph("Prediction History", styles["Title"]))
        elems.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
        elems.append(Paragraph(" ", styles["Normal"]))
        elems.append(styled_table_for_pdf(history_df))
        doc_h.build(elems)
        pdf_hist = buf_hist.getvalue()
        buf_hist.close()

        with tab_hist2:
            st.download_button("Download History (CSV)", data=history_df.to_csv(index=False).encode("utf-8"),
                               file_name="prediction_history.csv", mime="text/csv")
            st.download_button("Download History (PDF)", data=pdf_hist, file_name="prediction_history.pdf",
                               mime="application/pdf")

        if st.button("Clear History"):
            st.session_state["history"] = []
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.success("Prediction history cleared!")
    else:
        st.info("No saved predictions yet. Make a prediction to add to history.")


st.markdown('<div class="footer">© 2025 Animax K CarValuate ', unsafe_allow_html=True)
