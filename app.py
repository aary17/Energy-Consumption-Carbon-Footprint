import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Wuxi Urban Energy & Carbon Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS DESIGN SYSTEM
# ============================================================

st.markdown("""
<style>

    /* Reset & Page Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1320px;
    }

    .main {
        background-color: #f8fafc;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #0f172a !important;
    }

    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        font-weight: 700 !important;
        color: #064e3b !important;
        font-size: 13px !important;
        letter-spacing: 0.5px;
    }

    /* Sidebar Radio Options Styling */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        gap: 8px;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        padding: 10px 14px;
        border-radius: 10px;
        transition: all 0.2s ease;
        cursor: pointer;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background-color: #f0fdf4 !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label p {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #1e293b !important;
    }

    /* Hero Banner */
    .hero-banner {
        padding: 32px 38px;
        border-radius: 20px;
        background: linear-gradient(135deg, #064e3b 0%, #0d2818 50%, #13382c 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        box-shadow: 0px 12px 32px -5px rgba(6, 78, 59, 0.3);
        color: white !important;
        margin-bottom: 24px;
    }

    .status-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.2);
        color: #6ee7b7 !important;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
        border: 1px solid rgba(110, 231, 183, 0.3);
    }

    .hero-banner h1 {
        font-size: 34px;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
        text-transform: uppercase;
    }

    .hero-banner p {
        font-size: 16px;
        color: #d1fae5 !important;
        margin-bottom: 0;
        line-height: 1.5;
    }

    /* Overview KPI Cards */
    .kpi-card {
        background: #ffffff;
        padding: 20px 22px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #10b981;
        box-shadow: 0px 4px 16px rgba(0,0,0,0.03);
        text-align: left;
        margin-bottom: 20px;
        transition: transform 0.2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
    }

    .kpi-icon {
        font-size: 22px;
        margin-bottom: 6px;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        color: #064e3b !important;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }

    .kpi-label {
        font-size: 13px;
        font-weight: 600;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* Section Titles */
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #10b981 !important;
        border-left: 4px solid #059669;
        padding-left: 12px;
        margin-top: 24px;
        margin-bottom: 14px;
    }

    /* Card Containers */
    .content-card {
        background: #ffffff;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0px 4px 18px rgba(0,0,0,0.03);
        margin-bottom: 20px;
        color: #0f172a !important;
    }

    .content-card h3 {
        margin-top: 0;
        color: #064e3b !important;
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 12px;
    }

    .selected-grid-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
        padding: 24px;
        border-radius: 18px;
        border: 1px solid #a7f3d0;
        box-shadow: 0px 6px 20px rgba(5, 150, 105, 0.08);
        margin-top: 20px;
        margin-bottom: 20px;
        color: #0f172a !important;
    }

    .carbon-impact-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #ffffff 100%);
        padding: 30px;
        border-radius: 18px;
        border: 1.5px solid #10b981;
        box-shadow: 0px 10px 28px rgba(5, 150, 105, 0.12);
        text-align: center;
        margin-top: 24px;
        color: #0f172a !important;
    }

    .carbon-impact-card .carbon-number {
        font-size: 44px;
        font-weight: 800;
        color: #059669 !important;
        letter-spacing: -1px;
        margin: 10px 0;
    }

    /* Button Styling */
    .stButton > button {
        border-radius: 12px;
        min-height: 48px;
        font-weight: 700;
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white !important;
        border: none;
        box-shadow: 0px 4px 14px rgba(5, 150, 105, 0.25);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #047857 0%, #065f46 100%);
        box-shadow: 0px 6px 18px rgba(5, 150, 105, 0.35);
        transform: translateY(-1px);
    }

    /* Workflow indicator */
    .workflow-bar {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 12px;
        padding: 12px 20px;
        font-weight: 700;
        color: #064e3b;
        text-align: center;
        letter-spacing: 0.5px;
        margin-bottom: 20px;
        font-size: 14px;
    }

    /* Map container wrapper */
    .map-container {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

@st.cache_data
def load_data():
    gdf = gpd.read_file("Data/Processed/map_data.geojson")
    df = pd.read_csv("Data/Processed/ml_energy_dataset.csv")
    cols_to_add = [c for c in df.columns if c not in gdf.columns]
    gdf = pd.concat([gdf, df[cols_to_add]], axis=1)
    return gdf


@st.cache_data
def load_ml_data():
    return pd.read_csv("Data/Processed/ml_energy_dataset.csv")


@st.cache_resource
def train_model():
    df = load_ml_data()
    features = [
        "building_count",
        "total_footprint_m2",
        "mean_height_m",
        "building_density",
        "area_residential_m2",
        "area_commercial_m2",
        "area_office_m2",
        "area_education_m2",
        "area_industrial_m2",
        "area_other_m2",
        "floor_area_total_m2",
        "single_family_m2",
        "multi_family_m2",
        "facility_sales_m2",
        "facility_office_m2",
        "facility_neighborhood_m2",
        "facility_education_m2",
        "facility_industrial_m2",
        "parks_green_m2",
        "water_area_m2",
        "road_area_m2",
        "subway_influence_m2",
        "bus_routes_cnt"
    ]

    X = df[features]
    y = df["total_kwh"]

    model = GradientBoostingRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X, y)
    return model, features


gdf = load_data()
gdf["grid_id_display"] = gdf["grid_id_main"].astype(str)


# ============================================================
# MODERN SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0px 20px 0px; border-bottom: 1px solid #e2e8f0; margin-bottom: 20px;">
        <div style="font-size: 24px; font-weight: 800; color: #064e3b; display: flex; align-items: center; gap: 8px;">
            <span>🌍</span> Wuxi Explorer
        </div>
        <div style="font-size: 13px; color: #059669; font-weight: 600; margin-top: 2px;">
            Urban Energy & Carbon Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    section = st.radio(
        "NAVIGATION",
        [
            "🗺️ Explore Wuxi",
            "🧮 Carbon Calculator",
            "📊 Model Insights"
        ],
        key="main_navigation"
    )

    st.markdown("""
    <div style="margin-top: 160px; padding: 16px; background-color: #f0fdf4; border-radius: 14px; border: 1px solid #a7f3d0; text-align: center;">
        <div style="font-size: 12px; font-weight: 700; color: #064e3b;">📍 Wuxi, China</div>
        <div style="font-size: 11px; color: #059669; margin-top: 2px;">250 m Urban Grid Analysis</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HERO HEADER & TOP OVERVIEW METRICS
# ============================================================

st.markdown("""
<div class="hero-banner">
    <div class="status-badge">● ML-powered urban analysis</div>
    <h1>WUXI URBAN ENERGY & CARBON EXPLORER</h1>
    <p>Explore urban energy consumption, machine-learning predictions and carbon emissions across Wuxi.</p>
</div>
""", unsafe_allow_html=True)

# Overview KPI Cards
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">🗺️</div>
        <div class="kpi-value">{len(gdf):,}</div>
        <div class="kpi-label">Grid Cells</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    total_gwh = gdf['total_kwh'].sum() / 1e9
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">⚡</div>
        <div class="kpi-value">{total_gwh:.2f}B kWh</div>
        <div class="kpi-label">Total Energy</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    total_carbon_m = gdf['carbon_tonnes_co2e'].sum() / 1e6
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">🌱</div>
        <div class="kpi-value">{total_carbon_m:.2f}M tonnes</div>
        <div class="kpi-label">Carbon Footprint</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-icon">📐</div>
        <div class="kpi-value">250 m</div>
        <div class="kpi-label">Grid Resolution</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ============================================================
# PAGE 1: 🗺️ EXPLORE WUXI
# ============================================================

if section == "🗺️ Explore Wuxi":

    st.markdown(
        '<div class="section-header">Explore Wuxi</div>',
        unsafe_allow_html=True
    )
    st.caption("Select a grid cell to inspect its urban characteristics.")

    map_type = st.radio(
        "Select Layer",
        ["⚡ Energy Consumption", "🌱 Carbon Footprint"],
        horizontal=True
    )

    color_column = "total_kwh" if map_type == "⚡ Energy Consumption" else "carbon_tonnes_co2e"

    # Map Rendering
    geojson = gdf.__geo_interface__
    minx, miny, maxx, maxy = gdf.total_bounds
    center = {"lat": (miny + maxy) / 2, "lon": (minx + maxx) / 2}

    fig = px.choropleth_map(
        gdf,
        geojson=geojson,
        locations=gdf["grid_id_display"],
        featureidkey="properties.grid_id_main",
        color=color_column,
        custom_data=["grid_id_main"],
        map_style="open-street-map",
        zoom=11,
        center=center,
        opacity=0.75,
        color_continuous_scale="YlOrRd"
    )

    unit_label = "kWh" if map_type == "⚡ Energy Consumption" else "tonnes CO₂e"
    fig.update_traces(
        hovertemplate=f"<b>Grid ID:</b> %{{customdata[0]}}<br><b>{map_type.split()[1]}:</b> %{{z:,.2f}} {unit_label}<extra></extra>"
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=620
    )

    map_event = st.plotly_chart(
        fig,
        use_container_width=True,
        key="wuxi_map",
        on_select="rerun",
        selection_mode="points"
    )

    grid_options = ["-- Select a grid --"] + sorted(gdf["grid_id_main"].astype(str).tolist())

    # Map click handling
    if map_event and "selection" in map_event and map_event["selection"].get("points"):
        points = map_event["selection"]["points"]
        if len(points) > 0:
            clicked_point = points[0]
            clicked_grid = None

            customdata = clicked_point.get("customdata")
            if isinstance(customdata, list) and len(customdata) > 0:
                clicked_grid = str(customdata[0])
            elif isinstance(customdata, dict):
                if 0 in customdata:
                    clicked_grid = str(customdata[0])
                elif "0" in customdata:
                    clicked_grid = str(customdata["0"])
                elif len(customdata) > 0:
                    clicked_grid = str(next(iter(customdata.values())))
            elif customdata is not None:
                clicked_grid = str(customdata)

            if not clicked_grid or clicked_grid not in grid_options:
                loc = clicked_point.get("location")
                if loc and str(loc) in grid_options:
                    clicked_grid = str(loc)
                elif "point_number" in clicked_point:
                    idx = clicked_point["point_number"]
                    if isinstance(idx, int) and 0 <= idx < len(gdf):
                        clicked_grid = str(gdf.iloc[idx]["grid_id_main"])

            if clicked_grid and clicked_grid in grid_options:
                st.session_state["grid_selector"] = clicked_grid

    st.divider()

    selected_grid = st.selectbox(
        "Select Grid ID",
        grid_options,
        key="grid_selector"
    )

    # --------------------------------------------------------
    # SELECTED GRID PANEL
    # --------------------------------------------------------

    if selected_grid == "-- Select a grid --":
        st.info("👆 Click a grid cell on the map or select a Grid ID above to inspect its details.")
    else:
        matching_rows = gdf[gdf["grid_id_main"].astype(str) == selected_grid]
        if len(matching_rows) == 0:
            st.error(f"Grid ID '{selected_grid}' not found.")
        else:
            row = matching_rows.iloc[0]

            st.markdown(f"""
            <div class="selected-grid-card">
                <div style="font-size: 13px; font-weight: 700; color: #059669; text-transform: uppercase; letter-spacing: 0.5px;">SELECTED GRID PANEL</div>
                <h2 style="margin: 4px 0 0 0; color: #064e3b;">Grid #{selected_grid}</h2>
            </div>
            """, unsafe_allow_html=True)

            # Energy Metrics
            st.markdown('<div class="section-header">⚡ Energy Metrics</div>', unsafe_allow_html=True)
            e1, e2, e3 = st.columns(3)
            with e1:
                st.metric("Actual Energy", f"{row['total_kwh']:,.0f} kWh")
            with e2:
                st.metric("ML Predicted Energy", f"{row['predicted_total_kwh']:,.0f} kWh")
            with e3:
                diff = row['predicted_total_kwh'] - row['total_kwh']
                st.metric("Prediction Difference", f"{diff:,.0f} kWh")

            # Building Profile
            st.markdown('<div class="section-header">🏢 Building Profile</div>', unsafe_allow_html=True)
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.metric("Buildings", f"{row['building_count']:,.0f}")
            with b2:
                st.metric("Average Height", f"{row['mean_height_m']:.1f} m")
            with b3:
                st.metric("Building Density", f"{row['building_density']:.3f}")
            with b4:
                st.metric("Total Floor Area", f"{row['floor_area_total_m2']:,.0f} m²")

            # Land Use
            st.markdown('<div class="section-header">🏙️ Land Use & Composition</div>', unsafe_allow_html=True)
            l1, l2, l3, l4 = st.columns(4)
            with l1:
                st.metric("Residential", f"{row['area_residential_m2']:,.0f} m²")
            with l2:
                st.metric("Commercial", f"{row['area_commercial_m2']:,.0f} m²")
            with l3:
                st.metric("Office", f"{row['area_office_m2']:,.0f} m²")
            with l4:
                st.metric("Industrial", f"{row['area_industrial_m2']:,.0f} m²")

            # Transport & Infrastructure
            st.markdown('<div class="section-header">🛣️ Transport & Infrastructure</div>', unsafe_allow_html=True)
            t1, t2, t3, t4, t5 = st.columns(5)
            with t1:
                st.metric("Road Area", f"{row['road_area_m2']:,.0f} m²")
            with t2:
                st.metric("Bus Routes", f"{row['bus_routes_cnt']:,.0f}")
            with t3:
                st.metric("Subway Influence", f"{row['subway_influence_m2']:,.0f} m²")
            with t4:
                st.metric("Green Area", f"{row['parks_green_m2']:,.0f} m²")
            with t5:
                st.metric("Water Area", f"{row['water_area_m2']:,.0f} m²")

            # Carbon Calculation Action
            st.divider()
            if st.button("🌱 Calculate Carbon Footprint", use_container_width=True):
                emission_factor = 0.5777
                predicted_energy = max(float(row["predicted_total_kwh"]), 0)
                carbon_tonnes = (predicted_energy * emission_factor) / 1000

                st.session_state["carbon_result"] = {
                    "grid": selected_grid,
                    "energy": predicted_energy,
                    "carbon": carbon_tonnes
                }

            if "carbon_result" in st.session_state and st.session_state["carbon_result"]["grid"] == selected_grid:
                res = st.session_state["carbon_result"]
                st.markdown(f"""
                <div class="carbon-impact-card">
                    <div style="font-size: 14px; font-weight: 700; color: #059669; text-transform: uppercase; letter-spacing: 0.5px;">CARBON FOOTPRINT</div>
                    <div class="carbon-number">{res['carbon']:,.2f} tonnes CO₂e</div>
                    <div style="font-size: 14px; color: #475467;">Based on predicted energy consumption of <strong>{res['energy']:,.0f} kWh</strong></div>
                </div>
                """, unsafe_allow_html=True)


# ============================================================
# PAGE 2: 🧮 CARBON CALCULATOR
# ============================================================

elif section == "🧮 Carbon Calculator":

    st.markdown('<div class="section-header">Urban Energy & Carbon Calculator</div>', unsafe_allow_html=True)
    st.caption("Enter urban and building characteristics to estimate energy consumption using Gradient Boosting ML.")

    st.markdown("""
    <div class="workflow-bar">
        INPUT DATA &nbsp;&nbsp;→&nbsp;&nbsp; ML PREDICTION &nbsp;&nbsp;→&nbsp;&nbsp; CARBON ESTIMATION
    </div>
    """, unsafe_allow_html=True)

    model, features = train_model()

    # Form Organized Sections
    c_bld, c_land = st.columns(2)

    with c_bld:
        st.markdown('<div class="content-card"><h3>🏢 Building Characteristics</h3>', unsafe_allow_html=True)
        building_count = st.number_input("Building Count", min_value=0.0, value=10.0)
        total_footprint = st.number_input("Total Footprint (m²)", min_value=0.0, value=5000.0)
        mean_height = st.number_input("Average Height (m)", min_value=0.0, value=15.0)
        building_density = st.number_input("Building Density", min_value=0.0, value=0.30)
        floor_area = st.number_input("Total Floor Area (m²)", min_value=0.0, value=20000.0)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_land:
        st.markdown('<div class="content-card"><h3>🏙️ Land Use</h3>', unsafe_allow_html=True)
        residential = st.number_input("Residential Area (m²)", min_value=0.0, value=5000.0)
        commercial = st.number_input("Commercial Area (m²)", min_value=0.0, value=2000.0)
        office = st.number_input("Office Area (m²)", min_value=0.0, value=1000.0)
        education = st.number_input("Education Area (m²)", min_value=0.0, value=500.0)
        industrial = st.number_input("Industrial Area (m²)", min_value=0.0, value=1000.0)
        area_other = st.number_input("Other Land Use Area (m²)", min_value=0.0, value=500.0)
        st.markdown('</div>', unsafe_allow_html=True)

    c_comp, c_env = st.columns(2)

    with c_comp:
        st.markdown('<div class="content-card"><h3>🏘️ Building Composition & Facilities</h3>', unsafe_allow_html=True)
        single_family = st.number_input("Single Family Area (m²)", min_value=0.0, value=5000.0)
        multi_family = st.number_input("Multi Family Area (m²)", min_value=0.0, value=10000.0)
        facility_sales = st.number_input("Sales Facility Area (m²)", min_value=0.0, value=1000.0)
        facility_office = st.number_input("Office Facility Area (m²)", min_value=0.0, value=1000.0)
        facility_neighborhood = st.number_input("Neighborhood Facility Area (m²)", min_value=0.0, value=500.0)
        facility_education = st.number_input("Education Facility Area (m²)", min_value=0.0, value=500.0)
        facility_industrial = st.number_input("Industrial Facility Area (m²)", min_value=0.0, value=500.0)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_env:
        st.markdown('<div class="content-card"><h3>🌳 Environment & Transportation</h3>', unsafe_allow_html=True)
        parks_green = st.number_input("Parks / Green Area (m²)", min_value=0.0, value=1000.0)
        water_area = st.number_input("Water Area (m²)", min_value=0.0, value=500.0)
        road_area = st.number_input("Road Area (m²)", min_value=0.0, value=3000.0)
        subway = st.number_input("Subway Influence Area (m²)", min_value=0.0, value=0.0)
        bus_routes = st.number_input("Bus Routes Count", min_value=0.0, value=2.0)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    if st.button("⚡ Predict Energy & Calculate Carbon", use_container_width=True):
        input_data = pd.DataFrame([{
            "building_count": building_count,
            "total_footprint_m2": total_footprint,
            "mean_height_m": mean_height,
            "building_density": building_density,
            "area_residential_m2": residential,
            "area_commercial_m2": commercial,
            "area_office_m2": office,
            "area_education_m2": education,
            "area_industrial_m2": industrial,
            "area_other_m2": area_other,
            "floor_area_total_m2": floor_area,
            "single_family_m2": single_family,
            "multi_family_m2": multi_family,
            "facility_sales_m2": facility_sales,
            "facility_office_m2": facility_office,
            "facility_neighborhood_m2": facility_neighborhood,
            "facility_education_m2": facility_education,
            "facility_industrial_m2": facility_industrial,
            "parks_green_m2": parks_green,
            "water_area_m2": water_area,
            "road_area_m2": road_area,
            "subway_influence_m2": subway,
            "bus_routes_cnt": bus_routes
        }])

        input_data = input_data[features]
        predicted_energy = max(model.predict(input_data)[0], 0)
        carbon_tonnes = (predicted_energy * 0.5777) / 1000

        st.session_state["manual_prediction"] = {
            "energy": predicted_energy,
            "carbon": carbon_tonnes
        }

    if "manual_prediction" in st.session_state:
        res = st.session_state["manual_prediction"]
        st.success("Prediction completed successfully!")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="content-card" style="text-align: center; border-top: 4px solid #059669;">
                <div style="font-size: 13px; font-weight: 700; color: #059669; text-transform: uppercase;">⚡ PREDICTED ENERGY</div>
                <div style="font-size: 34px; font-weight: 800; color: #064e3b; margin: 10px 0;">{res['energy']:,.0f} kWh</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="content-card" style="text-align: center; border-top: 4px solid #10b981;">
                <div style="font-size: 13px; font-weight: 700; color: #059669; text-transform: uppercase;">🌱 ESTIMATED CARBON</div>
                <div style="font-size: 34px; font-weight: 800; color: #059669; margin: 10px 0;">{res['carbon']:,.2f} tonnes CO₂e</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# PAGE 3: 📊 MODEL INSIGHTS
# ============================================================

else:

    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    st.caption("Comprehensive evaluation of machine learning model metrics, prediction accuracy, SHAP explainability, and carbon footprint distribution.")

    # 4 Summary Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("🏆 Best Model", "Gradient Boosting")
    with m2:
        st.metric("📈 R² Score", "0.9901")
    with m3:
        st.metric("📉 MAE", "116,126 kWh")
    with m4:
        st.metric("📐 RMSE", "211,618 kWh")

    st.divider()

    # R2 Comparison
    st.markdown('<div class="section-header">🏆 Model R² Performance Comparison</div>', unsafe_allow_html=True)

    r2_data = pd.DataFrame({
        "Model": ["Gradient Boosting", "Extra Trees", "Random Forest", "XGBoost"],
        "R2": [0.990105, 0.986401, 0.984771, 0.979414]
    })
    r2_sorted = r2_data.sort_values("R2", ascending=True)

    fig_r2 = px.bar(
        r2_sorted,
        x="R2",
        y="Model",
        orientation="h",
        text=r2_sorted["R2"].apply(lambda v: f"{v:.4f}"),
        color="R2",
        color_continuous_scale="Viridis",
        labels={"R2": "R² Score", "Model": "Model"}
    )
    fig_r2.update_traces(textposition="outside", hovertemplate="<b>%{y}</b><br>R² Score: %{x:.6f}<extra></extra>")
    fig_r2.update_layout(height=380, margin=dict(l=20, r=40, t=30, b=30), xaxis=dict(range=[0.97, 1.0]), coloraxis_showscale=False)
    st.plotly_chart(fig_r2, use_container_width=True)

    st.info("💡 **Insight:** Gradient Boosting achieved the highest R² (0.9901) and the lowest MAE and RMSE among the four evaluated models.")

    st.divider()

    # MAE & RMSE Error Comparison
    st.markdown('<div class="section-header">📉 Error Metrics — MAE & RMSE</div>', unsafe_allow_html=True)
    c_mae, c_rmse = st.columns(2)

    error_data = pd.DataFrame({
        "Model": ["Gradient Boosting", "Extra Trees", "XGBoost", "Random Forest"],
        "MAE": [116126.47, 128713.85, 131273.56, 135764.22],
        "RMSE": [211617.72, 248087.99, 305236.32, 262532.24]
    })

    with c_mae:
        mae_sorted = error_data.sort_values("MAE", ascending=True)
        fig_mae = px.bar(
            mae_sorted,
            x="Model",
            y="MAE",
            text=mae_sorted["MAE"].apply(lambda v: f"{v:,.0f} kWh"),
            title="Mean Absolute Error (MAE)",
            color="MAE",
            color_continuous_scale="Blues_r"
        )
        fig_mae.update_traces(textposition="outside", hovertemplate="<b>%{x}</b><br>MAE: %{y:,.2f} kWh<extra></extra>")
        fig_mae.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=30), yaxis=dict(range=[0, max(error_data["MAE"]) * 1.25]), coloraxis_showscale=False)
        st.plotly_chart(fig_mae, use_container_width=True)

    with c_rmse:
        rmse_sorted = error_data.sort_values("RMSE", ascending=True)
        fig_rmse = px.bar(
            rmse_sorted,
            x="Model",
            y="RMSE",
            text=rmse_sorted["RMSE"].apply(lambda v: f"{v:,.0f} kWh"),
            title="Root Mean Squared Error (RMSE)",
            color="RMSE",
            color_continuous_scale="Oranges_r"
        )
        fig_rmse.update_traces(textposition="outside", hovertemplate="<b>%{x}</b><br>RMSE: %{y:,.2f} kWh<extra></extra>")
        fig_rmse.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=30), yaxis=dict(range=[0, max(error_data["RMSE"]) * 1.25]), coloraxis_showscale=False)
        st.plotly_chart(fig_rmse, use_container_width=True)

    st.divider()

    # Actual vs Predicted Scatter
    st.markdown('<div class="section-header">🎯 Actual vs Predicted Energy</div>', unsafe_allow_html=True)
    st.caption("Points closer to the diagonal line indicate higher prediction accuracy across the Wuxi grid cells.")

    try:
        results_df = pd.read_csv("Data/Processed/carbon_footprint_results.csv")
        if "total_kwh" in results_df.columns and "predicted_total_kwh" in results_df.columns:
            fig_scatter = px.scatter(
                results_df,
                x="total_kwh",
                y="predicted_total_kwh",
                opacity=0.6,
                labels={"total_kwh": "Actual Energy (kWh)", "predicted_total_kwh": "Predicted Energy (kWh)"}
            )
            max_val = max(float(results_df["total_kwh"].max()), float(results_df["predicted_total_kwh"].max()))
            fig_scatter.add_shape(type="line", line=dict(dash="dash", color="red", width=2), x0=0, y0=0, x1=max_val, y1=max_val)
            fig_scatter.update_traces(marker=dict(size=6, color="#059669"), hovertemplate="<b>Actual:</b> %{x:,.0f} kWh<br><b>Predicted:</b> %{y:,.0f} kWh<extra></extra>")
            fig_scatter.update_layout(height=480, margin=dict(l=20, r=20, t=30, b=30))
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.image("Data/Processed/actual_vs_predicted.png", use_container_width=True)
    except Exception:
        try:
            st.image("Data/Processed/actual_vs_predicted.png", use_container_width=True)
        except Exception:
            st.info("Actual vs predicted visualization not found.")

    st.divider()

    # SHAP Explainability
    st.markdown('<div class="section-header">🧠 Explainable AI — SHAP</div>', unsafe_allow_html=True)
    st.caption("SHAP shows how strongly each urban and building feature contributes to the model's energy-consumption prediction.")

    st.markdown("""
    <div style="background-color: #f0fdf4; border: 1px solid #a7f3d0; border-radius: 12px; padding: 14px 18px; margin-bottom: 16px; color: #064e3b;">
    <strong>Key Feature Drivers Identified by SHAP Analysis:</strong><br>
    <code>floor_area_total_m2</code> &nbsp;•&nbsp; 
    <code>single_family_m2</code> &nbsp;•&nbsp; 
    <code>multi_family_m2</code> &nbsp;•&nbsp; 
    <code>total_footprint_m2</code> &nbsp;•&nbsp; 
    <code>building_count</code> &nbsp;•&nbsp; 
    <code>facility_industrial_m2</code> &nbsp;•&nbsp; 
    <code>area_residential_m2</code> &nbsp;•&nbsp; 
    <code>mean_height_m</code> &nbsp;•&nbsp; 
    <code>building_density</code>
    </div>
    """, unsafe_allow_html=True)

    try:
        st.image("Data/Processed/shap_summary.png", use_container_width=True)
    except Exception:
        st.info("SHAP visualization not found.")

    st.divider()

    # Carbon Distribution
    st.markdown('<div class="section-header">🌱 Carbon Footprint Distribution</div>', unsafe_allow_html=True)
    st.caption("Distribution of estimated carbon footprint (tonnes CO₂e) across all 3,999 Wuxi 250 m grid cells.")

    try:
        st.image("Data/Processed/carbon_distribution.png", use_container_width=True)
    except Exception:
        st.info("Carbon distribution visualization not found.")