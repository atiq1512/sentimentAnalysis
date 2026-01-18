# =================================================
# IMDB SENTIMENT ANALYSIS DASHBOARD - STREAMLIT
# =================================================

from textblob import TextBlob
import pandas as pd
import streamlit as st
from cleantext import clean

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

# --------------------------------------------------
# TITLE & DESCRIPTION
# --------------------------------------------------
st.title("💬 IMDB Sentiment Analysis Dashboard")
st.markdown(
    """
    This dashboard performs **sentiment analysis** on:
    - 📝 Individual reviews
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

    text = st.text_area("Enter IMDB review for sentiment analysis:")

    if text:
        # Clean the input text (compatible with latest cleantext)
        cleaned_text = clean(
            str(text),
            lower=True,
            no_urls=True,
            no_numbers=True,
            no_punct=True,
            no_emoji=True,
            no_special=True,
            extra_spaces=True
        )

        blob = TextBlob(cleaned_text)
        polarity = round(blob.sentiment.polarity, 2)
        subjectivity = round(blob.sentiment.subjectivity, 2)

        col1, col2 = st.columns(2)
        col1.metric("Polarity", polarity)
        col2.metric("Subjectivity", subjectivity)

        # Adjust thresholds for IMDB reviews
        if polarity > 0.1:
            st.success("😊 Sentiment: Positive")
        elif polarity < -0.1:
            st.error("☹️ Sentiment: Negative")
        else:
            st.info("😐 Sentiment: Neutral")

# --------------------------------------------------
# CSV ANALYSIS SECTION
# --------------------------------------------------
with st.expander("📂 Analyze CSV File", expanded=True):

    st.markdown(
        """
        **CSV Requirements:**
        - Must contain a column named **`review`**
        """
    )

    upl = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    # ---------------- SENTIMENT FUNCTIONS ----------------
    def score(x):
        return TextBlob(str(x)).sentiment.polarity

    def analyze(x):
        if x > 0.1:
            return "Positive"
        elif x < -0.1:
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
        if "review" not in df.columns:
            st.error("❌ Column 'review' not found in file.")
        else:
            # Clean text for all reviews (latest cleantext API)
            df["review_clean"] = df["review"].apply(lambda x: clean(
                str(x),
                lower=True,
                no_urls=True,
                no_numbers=True,
                no_punct=True,
                no_emoji=True,
                no_special=True,
                extra_spaces=True
            ))

            # Apply sentiment analysis
            df["score"] = df["review_clean"].apply(score)
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
                file_name="IMDB_Sentiment_Results.csv",
                mime="text/csv",
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption("📊 IMDB Sentiment Analysis Dashboard | Built with Streamlit & TextBlob")
