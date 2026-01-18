from textblob import TextBlob
import pandas as pd
import streamlit as st
from cleantext import clean

# --------------------------------------------------
# PAGE CONFIG (DESIGN)
# --------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="💬",
    layout="centered"
)

# --------------------------------------------------
# TITLE & DESCRIPTION
# --------------------------------------------------
st.title("💬 Sentiment Analysis Dashboard")
st.markdown(
    """
    This dashboard performs **sentiment analysis** on:
    - 📝 Individual text
    - 📂 Uploaded CSV files  

    Sentiment is classified as **Positive, Neutral, or Negative**
    using **TextBlob polarity scores**.
    """
)

st.divider()

# --------------------------------------------------
# TEXT ANALYSIS SECTION
# --------------------------------------------------
with st.expander("📝 Analyze Text", expanded=True):

    text = st.text_area("Enter text for sentiment analysis:")

    if text:
        blob = TextBlob(str(text))
        polarity = round(blob.sentiment.polarity, 2)
        subjectivity = round(blob.sentiment.subjectivity, 2)

        col1, col2 = st.columns(2)
        col1.metric("Polarity", polarity)
        col2.metric("Subjectivity", subjectivity)

        if polarity >= 0.5:
            st.success("😊 Sentiment: Positive")
        elif polarity <= -0.5:
            st.error("☹️ Sentiment: Negative")
        else:
            st.info("😐 Sentiment: Neutral")

    st.subheader("🧹 Clean Text")
    pre = st.text_input("Enter text to clean:")

    if pre:
        cleaned = clean(
            pre,
            clean_all=False,
            extra_spaces=True,
            stopwords=True,
            lowercase=True,
            numbers=True,
            punct=True
        )
        st.code(cleaned, language="text")

# --------------------------------------------------
# CSV ANALYSIS SECTION
# --------------------------------------------------
with st.expander("📂 Analyze CSV File", expanded=True):

    st.markdown(
        """
        **CSV Requirements:**
        - Must contain a column named **`tweets`**
        """
    )

    upl = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    # ---------------- SENTIMENT FUNCTIONS ----------------
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
        # Load file safely
        if upl.name.endswith(".csv"):
            df = pd.read_csv(upl)
        else:
            df = pd.read_excel(upl)

        # Remove unwanted index column if exists
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

        # Check required column
        if "tweets" not in df.columns:
            st.error("❌ Column 'tweets' not found in file.")
        else:
            # Apply sentiment analysis
            df["score"] = df["tweets"].apply(score)
            df["analysis"] = df["score"].apply(analyze)

            st.success("✅ Sentiment analysis completed!")
            st.dataframe(df.head(10), use_container_width=True)

            # ---------------- DOWNLOAD RESULT ----------------
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode("utf-8")

            csv = convert_df(df)

            st.download_button(
                label="⬇️ Download sentiment results as CSV",
                data=csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv",
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption("📊 Sentiment Analysis Dashboard | Built with Streamlit & TextBlob")
