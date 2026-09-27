import os
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="NutriVision AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. CUSTOM UI / CSS
# =========================================================
st.markdown("""
<style>
    /* Global */
    .stApp {
        background: linear-gradient(135deg, #f7fbf8 0%, #eef7f1 100%);
    }

    [data-testid="stHeader"] {
        background: rgba(255,255,255,0);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Hero */
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #12372a 0%, #1f6f4a 55%, #4f9d69 100%);
        color: white;
        box-shadow: 0 12px 35px rgba(18,55,42,.18);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: 2.7rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .hero p {
        font-size: 1.05rem;
        margin: .55rem 0 0;
        opacity: .9;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 750;
        color: #12372a;
        margin: 1.2rem 0 .8rem;
    }

    /* Cards */
    .info-card {
        background: rgba(255,255,255,.92);
        border: 1px solid rgba(31,111,74,.10);
        border-radius: 18px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 8px 25px rgba(31,111,74,.07);
    }

    .stat-label {
        color: #66756d;
        font-size: .86rem;
        font-weight: 600;
    }

    .stat-value {
        color: #12372a;
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: .15rem;
    }

    .food-name {
        font-size: 2rem;
        font-weight: 800;
        color: #12372a;
        margin-bottom: .2rem;
    }

    .confidence {
        display: inline-block;
        background: #e5f5eb;
        color: #1f6f4a;
        border-radius: 999px;
        padding: .35rem .75rem;
        font-weight: 700;
        font-size: .85rem;
    }

    .tip {
        background: #fff8e7;
        border-left: 4px solid #e7ad34;
        padding: .8rem 1rem;
        border-radius: 10px;
        color: #6c541c;
        margin-top: .8rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        min-height: 2.8rem;
        font-weight: 700;
        border: 1px solid rgba(31,111,74,.18);
        transition: .2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(31,111,74,.16);
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,.82);
        border-radius: 18px;
        padding: .5rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #12372a;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .side-brand {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: .2rem;
    }

    .side-sub {
        opacity: .75;
        font-size: .9rem;
        margin-bottom: 1.5rem;
    }

    /* Hide Streamlit decoration */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. DYNAMIC PATH CONFIGURATION
# =========================================================
base_dir = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(base_dir, "models", "food_classifier_frozen_best.keras")
NUTRITION_PATH = os.path.join(base_dir, "data", "nutrition.csv")
MEAL_LOG_PATH = os.path.join(base_dir, "data", "meal_log.csv")

# =========================================================
# 4. FOOD CLASSES
# =========================================================
selected_classes = [
    "samosa",
    "fried_rice",
    "pizza",
    "hamburger",
    "omelette",
    "french_fries",
    "pancakes",
    "lasagna",
    "sushi",
    "tacos"
]

# =========================================================
# 5. LOAD TRAINED MODEL
# =========================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("⚠️ Model could not be loaded.")
    st.code(str(e))
    st.stop()

# =========================================================
# 6. LOAD NUTRITION DATABASE
# =========================================================
try:
    nutrition_df = pd.read_csv(NUTRITION_PATH)
except Exception as e:
    st.error("⚠️ Nutrition database could not be loaded.")
    st.code(str(e))
    st.stop()

# =========================================================
# 7. LOAD / CREATE MEAL LOG
# =========================================================
if os.path.exists(MEAL_LOG_PATH):
    meal_log = pd.read_csv(MEAL_LOG_PATH)
else:
    meal_log = pd.DataFrame(columns=[
        "date", "time", "meal_type", "food",
        "confidence", "calories", "protein", "carbs", "fat"
    ])
    os.makedirs(os.path.dirname(MEAL_LOG_PATH), exist_ok=True)
    meal_log.to_csv(MEAL_LOG_PATH, index=False)

# =========================================================
# 8. SESSION STATE
# =========================================================
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# =========================================================
# 9. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown('<div class="side-brand">🥗 NutriVision AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-sub">AI-powered food recognition & nutrition tracking</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 🧠 Supported Foods")
    for food in selected_classes:
        st.write("• " + food.replace("_", " ").title())

    st.divider()
    st.markdown("### ⚙️ About")
    st.caption(
        "Upload a food image, let the AI identify it, "
        "view nutrition information, and save it to your daily log."
    )

# =========================================================
# 10. HERO HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🥗 NutriVision AI</h1>
    <p>Recognize your food. Understand your nutrition. Track your day.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# 11. TOP SUMMARY
# =========================================================
today = datetime.now().strftime("%Y-%m-%d")
today_meals = meal_log[meal_log["date"] == today]

top1, top2, top3, top4 = st.columns(4)

with top1:
    st.markdown(
        f'<div class="info-card"><div class="stat-label">🍽️ Meals Today</div>'
        f'<div class="stat-value">{len(today_meals)}</div></div>',
        unsafe_allow_html=True
    )

with top2:
    calories_today = today_meals["calories"].sum() if not today_meals.empty else 0
    st.markdown(
        f'<div class="info-card"><div class="stat-label">🔥 Calories</div>'
        f'<div class="stat-value">{calories_today:.0f} kcal</div></div>',
        unsafe_allow_html=True
    )

with top3:
    protein_today = today_meals["protein"].sum() if not today_meals.empty else 0
    st.markdown(
        f'<div class="info-card"><div class="stat-label">💪 Protein</div>'
        f'<div class="stat-value">{protein_today:.1f} g</div></div>',
        unsafe_allow_html=True
    )

with top4:
    carbs_today = today_meals["carbs"].sum() if not today_meals.empty else 0
    st.markdown(
        f'<div class="info-card"><div class="stat-label">🍞 Carbs</div>'
        f'<div class="stat-value">{carbs_today:.1f} g</div></div>',
        unsafe_allow_html=True
    )

# =========================================================
# 12. IMAGE ANALYSIS
# =========================================================
st.markdown('<div class="section-title">📸 Analyze Your Food</div>', unsafe_allow_html=True)

upload_col, preview_col = st.columns([1, 1], gap="large")

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload a food image",
        type=["jpg", "jpeg", "png"],
        help="Use a clear image of one food item for better recognition."
    )

    meal_type = st.selectbox(
        "Meal Type",
        ["Breakfast", "Lunch", "Dinner", "Snack"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.uploaded_image = image

        if st.button("🔍 Identify Food", use_container_width=True):
            with st.spinner("AI is analyzing your food..."):
                image_resized = image.resize((224, 224))
                image_array = np.array(image_resized, dtype=np.float32)
                image_array = np.expand_dims(image_array, axis=0)
                image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                    image_array
                )

                predictions = model.predict(image_array, verbose=0)
                predicted_index = int(np.argmax(predictions[0]))

                # Safety check in case model output size changes
                if predicted_index >= len(selected_classes):
                    st.error("Model output classes do not match the configured food classes.")
                    st.stop()

                predicted_food = selected_classes[predicted_index]
                confidence = float(predictions[0][predicted_index] * 100)

                st.session_state.prediction = {
                    "food": predicted_food,
                    "confidence": confidence,
                    "meal_type": meal_type
                }

with preview_col:
    if st.session_state.uploaded_image is not None:
        st.image(
            st.session_state.uploaded_image,
            caption="Uploaded Food",
            use_container_width=True
        )
    else:
        st.markdown("""
        <div class="info-card" style="height:310px; display:flex;
        align-items:center; justify-content:center; text-align:center;">
            <div>
                <div style="font-size:3rem;">📷</div>
                <h3 style="color:#12372a;">Your food image will appear here</h3>
                <p style="color:#66756d;">Upload an image to start AI analysis.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# 13. PREDICTION RESULT
# =========================================================
prediction = st.session_state.prediction

if prediction:
    predicted_food = prediction["food"]
    confidence = prediction["confidence"]

    food_info_df = nutrition_df[nutrition_df["food"] == predicted_food]

    st.markdown('<div class="section-title">✨ AI Result</div>', unsafe_allow_html=True)

    if food_info_df.empty:
        st.warning("Nutrition information was not found for this food.")
    else:
        food_info = food_info_df.iloc[0]

        result_col, nutrition_col = st.columns([1, 1], gap="large")

        with result_col:
            st.markdown(
                f'<div class="info-card">'
                f'<div class="food-name">{predicted_food.replace("_", " ").title()}</div>'
                f'<span class="confidence">AI Confidence: {confidence:.2f}%</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.progress(min(max(confidence / 100, 0.0), 1.0))

            if confidence < 60:
                st.markdown(
                    '<div class="tip">💡 Low confidence. Try a clearer image with the food '
                    'centered and well lit.</div>',
                    unsafe_allow_html=True
                )

        with nutrition_col:
            n1, n2, n3, n4 = st.columns(4)

            with n1:
                st.metric("🔥 Calories", f"{food_info['calories']} kcal")
            with n2:
                st.metric("💪 Protein", f"{food_info['protein']} g")
            with n3:
                st.metric("🍞 Carbs", f"{food_info['carbs']} g")
            with n4:
                st.metric("🥑 Fat", f"{food_info['fat']} g")

        st.write("")
        if st.button("➕ Add to Today's Meal Log", use_container_width=True):
            now = datetime.now()

            new_meal = {
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "meal_type": prediction["meal_type"],
                "food": predicted_food,
                "confidence": round(confidence, 2),
                "calories": food_info["calories"],
                "protein": food_info["protein"],
                "carbs": food_info["carbs"],
                "fat": food_info["fat"]
            }

            meal_log = pd.concat(
                [meal_log, pd.DataFrame([new_meal])],
                ignore_index=True
            )
            meal_log.to_csv(MEAL_LOG_PATH, index=False)

            st.success("✅ Meal added to your daily log!")
            st.rerun()

# =========================================================
# 14. DAILY DASHBOARD
# =========================================================
st.markdown('<div class="section-title">📊 Today\'s Nutrition Dashboard</div>', unsafe_allow_html=True)

today_meals = meal_log[meal_log["date"] == today]

if today_meals.empty:
    st.info("No meals recorded today. Analyze a food image and add your first meal.")
else:
    total_calories = today_meals["calories"].sum()
    total_protein = today_meals["protein"].sum()
    total_carbs = today_meals["carbs"].sum()
    total_fat = today_meals["fat"].sum()

    dashboard1, dashboard2 = st.columns([1, 1], gap="large")

    with dashboard1:
        st.markdown("#### 📈 Nutrition Summary")
        chart_data = pd.DataFrame({
            "Nutrition": ["Calories", "Protein", "Carbs", "Fat"],
            "Amount": [total_calories, total_protein, total_carbs, total_fat]
        })
        st.bar_chart(chart_data.set_index("Nutrition"), use_container_width=True)

    with dashboard2:
        st.markdown("#### 🕒 Meal History")

        display_df = today_meals[
            ["time", "meal_type", "food", "confidence", "calories"]
        ].copy()

        display_df["food"] = (
            display_df["food"]
            .str.replace("_", " ", regex=False)
            .str.title()
        )

        display_df.columns = [
            "Time", "Meal Type", "Food", "Confidence (%)", "Calories"
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

# =========================================================
# 15. FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "🥗 NutriVision AI • Food recognition powered by your trained TensorFlow model"
)
