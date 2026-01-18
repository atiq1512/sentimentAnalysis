from textblob import TextBlob
import pandas as pd
import streamlit as st
from cleantext import clean
import matplotlib.pyplot as plt

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------- TITLE --------------------
st.title("📊 Sentiment Analysis Dashboard")
st.markdown(
    """
    This dashboard analyzes **text sentiment** using **TextBlob**.
    Users can analyze **individual text** or **CSV/Excel files**.
    """
)

# -------------------- SIDEBAR --------------------
st.sidebar.header("📌 Instructions")
st.sidebar.markdown("""
1. Enter text to analyze sentiment  
2. Upload CSV or Excel file  
3. Download analyzed results  
""")

# -------------------- TABS --------------------
tab1, tab2 = st.tabs(["📝 Text Analysis", "📂 File Analysis"])

# =====================================================
# 📝 TAB 1 — TEXT ANALYSIS
# =====================================================
with tab1:
    st.subheader("Analyze Individual Text")

    col1, col2 = st.columns(2)

    with col1:
        text = st.text_area("Enter text for sentiment analysis:")

        if text:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            st.metric("Polarity", round(polarity, 2))
            st.metric("Subjectivity", round(subjectivity, 2))

    with col2:
        clean_text = st.text_area("Clean text:")

        if clean_text:
            st.write(
                clean(
                    clean_text,
                    clean_all=False,
                    extra_spaces=True,
                    stopwords=True,
                    lowercase=True,
                    numbers=True,
                    punct=True
                )
            )

# =====================================================
# 📂 TAB 2 — FILE ANALYSIS
# =====================================================
with tab2:
    st.subheader("Analyze CSV / Excel File")

    upl = st.file_uploader("Upload file", type=["csv", "xlsx"])

    def score(x):
        return TextBlob(str(x)).sentiment.polarity

    def analyze(x):
        if x >= 0.5:
            return "Positive"
        elif x <= -0.5:
            return "Negative"
        else:
            return "Neutral"

    if upl:
        if upl.name.endswith(".csv"):
            df = pd.read_csv(upl)
        else:
            df = pd.read_excel(upl)

        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

        text_col = df.columns[0]
        df[text_col] = df[text_col].fillna("")

        df["Polarity"] = df[text_col].apply(score)
        df["Sentiment"] = df["Polarity"].apply(analyze)

        # -------------------- DISPLAY DATA --------------------
        st.subheader("Preview of Analyzed Data")
        st.dataframe(df.head(10), use_container_width=True)

        # -------------------- VISUALIZATION --------------------
        st.subheader("Sentiment Distribution")

        sentiment_count = df["Sentiment"].value_counts()

        fig, ax = plt.subplots()
        sentiment_count.plot(kind="bar", ax=ax)
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.set_title("Sentiment Distribution")

        st.pyplot(fig)

        # -------------------- DOWNLOAD BUTTON --------------------
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode("utf-8")

        csv = convert_df(df)

        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="sentiment_results.csv",
            mime="text/csv"
        )
