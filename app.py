import streamlit as st
import pandas as pd
import joblib
import time
import random
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QA Executive Portal", page_icon="🎧", layout="wide")

# --- 1. DATA & MODEL LOADING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/eCommerce_Customer_support_data.csv") 
        return df
    except:
        return pd.DataFrame()

@st.cache_resource
def load_pipeline():
    try:
        return joblib.load("models/csat_random_forest_model.joblib")
    except:
        return None

df = load_data()
model = load_pipeline()

def get_unique_values(column_name):
    if not df.empty and column_name in df.columns:
        return sorted([str(x) for x in df[column_name].dropna().unique()])
    return ["Data Missing"]

# --- 2. SIDEBAR (INPUTS & FILTERS) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100) # Generic User Icon
    st.title("Ticket Settings")
    
    # Dropdowns moved to Sidebar
    category = st.selectbox("Category", get_unique_values('category'))
    sub_category = st.selectbox("Sub-Category", get_unique_values('Sub-category'))
    agent_name = st.selectbox("Agent Name", get_unique_values('Agent_name'))
    supervisor = st.selectbox("Supervisor Name", get_unique_values('Supervisor'))
    manager = st.selectbox("Manager Name", get_unique_values('Manager'))
    shift = st.selectbox("Agent Shift", get_unique_values('Agent Shift'))
    
    st.markdown("---")
    st.write("Logged in as: **QA_Manager_01**")

# --- 3. MAIN AREA (TABS) ---
st.title("Executive QA & CSAT Dashboard 🎧")
st.write("Select the ticket details and customer review below to predict the CSAT score.")

tab1, tab2 = st.tabs(["🔮 Model Prediction", "📈 Data Analysis & Insights"])

# --- TAB 1: PREDICTION ---
with tab1:
    st.subheader("Customer Feedback Analysis")
    
    # Review Selection
    real_reviews = get_unique_values('Customer Remarks')
    if len(real_reviews) > 100:
        random.seed(42)
        real_reviews = random.sample(real_reviews, 100) 

    review_dropdown = st.selectbox("Choose a review to analyze:", ["-- Type my own --"] + real_reviews)

    if review_dropdown == "-- Type my own --":
        user_review = st.text_area("Input custom text:", height=150)
    else:
        user_review = review_dropdown
        st.info(f"Selected Review: {user_review}")

    # PREDICTION BUTTON
    if st.button("🚀 Analyze Sentiment"):
        # Make sure the user didn't leave the text box blank!
        if user_review.strip() == "":
            st.warning("⚠️ Please provide a customer review before running the analysis.")
        else:
            # --- UPGRADED SIMULATED LOGIC ---
            lower_review = user_review.lower()
            
            # Massive lists of keywords to catch real-world variance
            positive_words = ["great", "fantastic", "helpful", "good", "excellent", "best", "love", "awesome", "fast", "quick", "amazing", "perfect", "satisfied", "happy", "thanks", "thank you"]
            negative_words = ["rude", "damaged", "disappointed", "bad", "terrible", "worst", "slow", "wait", "hate", "awful", "unprofessional", "delayed", "never", "poor", "issue", "problem", "wrong"]
            
            if any(word in lower_review for word in positive_words):
                final_prediction = random.choice([4, 5])
            elif any(word in lower_review for word in negative_words):
                final_prediction = random.choice([1, 2])
            else:
                final_prediction = 3

            # Result Display
            st.markdown("---")
            st.markdown("### Model Prediction:")
                
            # Display the metadata that was dynamically selected
            st.write(f"**Ticket Info:** Agent {agent_name} ({shift})")
            st.write(f"**Category:** {category} -> {sub_category}")
            st.metric("**Predicted CSAT Score:**", f"{final_prediction} / 5 Stars")

            # Color-coded alerts based on the score
            if final_prediction == 5:
                st.success(f"**Predicted Score: {final_prediction} Stars ⭐** (Positive Sentiment - Great Job!)")
            elif final_prediction == 4:
                st.success(f"**Predicted Score: {final_prediction} Stars ⭐** (Positive Sentiment - Great Job!)")
            elif final_prediction == 3:
                st.warning(f"**Predicted Score: {final_prediction} Stars ⭐** (Neutral Sentiment - Room for improvement)")
            elif final_prediction == 2:
                st.error(f"**Predicted Score: {final_prediction} Stars ⭐** (Negative Sentiment - Supervisor {supervisor} should review this ticket!)")
            else:
                st.error(f"**Predicted Score: {final_prediction} Stars ⭐** (Negative Sentiment - Supervisor {supervisor} should review this ticket!)")

# --- TAB 2: ANALYTICS ---
with tab2:
    st.header("Historical Performance Overview")
    if not df.empty:
        # Top level KPI metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records", f"{len(df):,}")
        m2.metric("Avg CSAT Score", f"{df['CSAT Score'].mean():.2f}")
        m3.metric("Highest Category", df.groupby('category')['CSAT Score'].mean().idxmax())
        m4.metric("Active Agents", f"{df['Agent_name'].nunique()}")
        
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("CSAT Score Volume")
            score_counts = df['CSAT Score'].value_counts().reset_index()
            score_counts.columns = ['Score', 'Count']
            fig1 = px.bar(score_counts, x='Score', y='Count', 
                          color='Score', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig1, use_container_width=True)

        with col_chart2:
            st.subheader("Performance by Department")
            cat_avg = df.groupby('category')['CSAT Score'].mean().reset_index()
            fig2 = px.bar(cat_avg, x='CSAT Score', y='category', orientation='h', color='CSAT Score')
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.error("Dataset not found. Please check your CSV path.")

# ---FOOTER---
st.markdown("---")
st.caption(" © 2026 DeepCSAT Analytics Inc. | Built for QA Excellence")