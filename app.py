import streamlit as st
from summariser import Summarizer

st.set_page_config(
    page_title="News Article Summarizer",
    page_icon="📰",
    layout="wide"
)

# --- Caching the Summarizer Model ---
@st.cache_resource # Use st.cache_resource for non-data objects like models
def load_summarizer_model():
    """Loads the summarization model and caches it."""
    model = Summarizer() 
    return model

# --- Main Application UI ---
def main():
    st.title("📰 News Article Summarizer")
    st.markdown("Enter a news article below and get a concise summary.")

    with st.spinner("Loading summarization model... This may take a moment on first run."):
        summarizer = load_summarizer_model()

    # --- Input Area ---
    st.subheader("Enter News Article Text:")
    article_text = st.text_area("Paste your article here:", height=300, key="article_input")

    # --- Summarization Parameters (Optional UI for these) ---
    col1, col2 = st.columns(2)
    with col1:
        min_summary_length = st.slider("Minimum Summary Length:", min_value=10, max_value=150, value=30, step=5)
    with col2:
        max_summary_length = st.slider("Maximum Summary Length:", min_value=50, max_value=500, value=150, step=10)


    # --- Summarize Button ---
    if st.button("✨ Summarize Article", type="primary"):
        if not article_text.strip():
            st.warning("Please enter some text to summarize.")
        else:
            with st.spinner("Generating summary... Please wait."):
                try:
                    summary = summarizer.summarize(
                        article_text, 
                        # min_length_per_chunk=, 
                        # max_length_per_chunk=,
                        overall_min_length=min_summary_length,
                        overall_max_length=max_summary_length
                    )
                    st.subheader("Generated Summary:")
                    st.success(summary)
                    # st.text_area("Summary", value=summary, height=150, disabled=True)
                except Exception as e:
                    st.error(f"An error occurred during summarization: {e}")
                    st.error("Please ensure the model is loaded correctly and the input text is valid.")
    
    st.markdown("---")
    st.markdown("Powered by google/pegasus-cnn_dailymail.")


if __name__ == "__main__":
    main()