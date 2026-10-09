import streamlit as st
import pandas as pd
import numpy as np
import folium

from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------------------------------
# GEOSHIELD AI - HACKATHON PROTOTYPE
# ------------------------------------------

st.set_page_config(
    page_title="GeoShield AI",
    page_icon="🌍",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f5f8fc;
}
[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #dce5ef;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("🌍 GeoShield AI")
st.subheader("Intelligent Geospatial Risk Mapping")
st.caption(
    "AI + Machine Learning + GIS | Educational hackathon prototype"
)

# ------------------------------------------
# 1. CREATE DEMONSTRATION DATA
# ------------------------------------------

@st.cache_data
def create_demo_data():
    rng = np.random.default_rng(42)
    n = 1200

    rainfall = rng.uniform(0, 250, n)
    elevation = rng.uniform(0, 100, n)
    distance = rng.uniform(0, 20, n)

    # Illustrative rule for creating training labels.
    # This is NOT a real-world flood-risk formula.
    score = (
        0.45 * rainfall / 250
        + 0.30 * (1 - elevation / 100)
        + 0.25 * (1 - distance / 20)
        + rng.normal(0, 0.07, n)
    )

    risk = np.where(
        score >= 0.62, "High",
        np.where(score >= 0.40, "Medium", "Low")
    )

    return pd.DataFrame({
        "Rainfall": rainfall,
        "Elevation": elevation,
        "Distance": distance,
        "Risk": risk
    })

data = create_demo_data()

X = data[["Rainfall", "Elevation", "Distance"]]
y = data["Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

@st.cache_resource
def train_model():
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

model = train_model()

test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

# ------------------------------------------
# 2. SIDEBAR INPUTS
# ------------------------------------------

st.sidebar.header("📍 Location Risk Analyzer")

st.sidebar.write(
    "Change the example environmental values "
    "to explore the model's predictions."
)

rain = st.sidebar.slider(
    "Illustrative rainfall (mm)",
    min_value=0,
    max_value=250,
    value=150
)

height = st.sidebar.slider(
    "Elevation (m)",
    min_value=0,
    max_value=100,
    value=20
)

water_distance = st.sidebar.slider(
    "Distance from water body (km)",
    min_value=0.0,
    max_value=20.0,
    value=3.0,
    step=0.5
)

# Demo map center and sample coordinates.
# These are not verified hazard locations.
center_lat = 16.5062
center_lon = 80.6480

st.sidebar.caption(
    "Map uses demonstration coordinates near Vijayawada. "
    "Risk values are synthetic and not official warnings."
)

# ------------------------------------------
# 3. RUN PREDICTION
# ------------------------------------------

input_data = pd.DataFrame([{
    "Rainfall": rain,
    "Elevation": height,
    "Distance": water_distance
}])

prediction = model.predict(input_data)[0]
probabilities = model.predict_proba(input_data)[0]
classes = list(model.classes_)

# These scores represent model outputs, not
# calibrated real-world flood probabilities.
model_scores = dict(zip(classes, probabilities))

colors = {
    "Low": "green",
    "Medium": "orange",
    "High": "red"
}

# ------------------------------------------
# 4. TOP DASHBOARD
# ------------------------------------------

st.markdown("### 📊 Risk Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Estimated Category", prediction)
c2.metric("Rainfall Input", f"{rain} mm")
c3.metric("Elevation Input", f"{height} m")
c4.metric("Water Distance", f"{water_distance} km")

st.progress(
    float(model_scores.get(prediction, 0)),
    text=(
        "Model score for predicted category: "
        f"{model_scores.get(prediction, 0):.1%}"
    )
)

st.caption(
    "The score is the model's class probability estimate, "
    "not the probability that a flood will occur."
)

# ------------------------------------------
# 5. INTERACTIVE GEOSPATIAL MAP
# ------------------------------------------

st.markdown("### 🗺️ Interactive Risk Map")

map_obj = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles="OpenStreetMap"
)

# Add illustrative sample locations.
rng = np.random.default_rng(7)

for i in range(30):
    lat = center_lat + rng.uniform(-0.07, 0.07)
    lon = center_lon + rng.uniform(-0.07, 0.07)

    sample = pd.DataFrame([{
        "Rainfall": rng.uniform(0, 250),
        "Elevation": rng.uniform(0, 100),
        "Distance": rng.uniform(0, 20)
    }])

    category = model.predict(sample)[0]

    folium.CircleMarker(
        location=[lat, lon],
        radius=7,
        color=colors[category],
        fill=True,
        fill_color=colors[category],
        fill_opacity=0.75,
        tooltip=f"Demo point {i + 1}: {category} risk"
    ).add_to(map_obj)

folium.Marker(
    [center_lat, center_lon],
    tooltip="Demo analysis center",
    popup=(
        f"Predicted category: {prediction}<br>"
        "Illustrative model result only"
    ),
    icon=folium.Icon(
        color=colors[prediction],
        icon="info-sign"
    )
).add_to(map_obj)

st_folium(
    map_obj,
    width=None,
    height=480,
    returned_objects=[]
)

st.markdown(
    "🟢 Low risk · 🟠 Medium risk · 🔴 High risk"
)

st.caption(
    "Colored points are generated demonstration samples, "
    "not measured flood-risk zones."
)

# ------------------------------------------
# 6. SAFETY RECOMMENDATIONS
# ------------------------------------------

st.markdown("### 🛡️ Preparedness Guidance")

if prediction == "High":
    st.error(
        "The demo model assigns this input to its High category."
    )
    st.write(
        "- Check official weather and disaster-management alerts."
    )
    st.write(
        "- Follow instructions from local emergency authorities."
    )
    st.write(
        "- Avoid floodwater and do not drive through flooded roads."
    )

elif prediction == "Medium":
    st.warning(
        "The demo model assigns this input to its Medium category."
    )
    st.write(
        "- Monitor official rainfall and weather updates."
    )
    st.write(
        "- Keep essential supplies and emergency contacts ready."
    )
    st.write(
        "- Review local evacuation guidance if an alert is issued."
    )

else:
    st.success(
        "The demo model assigns this input to its Low category."
    )
    st.write(
        "- Continue monitoring official weather information."
    )
    st.write(
        "- Keep a basic emergency plan available."
    )
    st.write(
        "- Remember that a Low model category does not mean "
        "an area is safe from flooding."
    )

# ------------------------------------------
# 7. MODEL AND DATA ANALYTICS
# ------------------------------------------

st.markdown("### 🧠 Model Information")

left, right = st.columns(2)

with left:
    st.write("**Algorithm:** Random Forest Classifier")
    st.write(f"**Synthetic training records:** {len(X_train)}")
    st.write(f"**Synthetic test records:** {len(X_test)}")
    st.write(
        f"**Test accuracy on synthetic data:** "
        f"{test_accuracy:.1%}"
    )
    st.caption(
        "This accuracy measures agreement with generated labels, "
        "not accuracy on real flood events."
    )

with right:
    st.write("**Input feature importance**")
    importance = pd.Series(
        model.feature_importances_,
        index=[
            "Rainfall",
            "Elevation",
            "Distance from water"
        ]
    ).sort_values(ascending=False)

    st.bar_chart(importance)

st.markdown("### 📈 Demonstration Dataset")

chart_data = data["Risk"].value_counts().reindex(
    ["Low", "Medium", "High"],
    fill_value=0
)

st.bar_chart(chart_data)

with st.expander("View sample training data"):
    st.dataframe(data.head(20), use_container_width=True)

st.download_button(
    label="Download demonstration dataset (CSV)",
    data=data.to_csv(index=False),
    file_name="geoshield_demo_data.csv",
    mime="text/csv"
)

# ------------------------------------------
# 8. PROJECT FOOTER
# ------------------------------------------

st.divider()

st.markdown("""
**GeoShield AI | Hackathon Prototype**

Built using Python, Streamlit, Folium, Pandas,
and Scikit-learn.

**Disclaimer:** This application uses synthetic training
data and illustrative coordinates. It is not a validated
flood forecasting system and must not be used for
emergency decisions.
""")
