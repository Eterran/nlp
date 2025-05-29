import streamlit as st
from summariser import Summarizer

@st.cache_resource
def load_summarizer_model():
    model = Summarizer() 
    return model

def main():
    st.title("📰 Multilingual News Article Summarizer")
    st.markdown("Enter a news article below. If non-English, it will be translated to English, summarized, and the summary translated back.")

    with st.spinner("Loading models... This may take longer on first run."):
        summarizer = load_summarizer_model()
    
    article_text = st.text_area("Paste your article here (supports multiple languages):", height=250, key="article_input")
    
    col1, col2 = st.columns(2)
    with col1:
        min_summary_length = st.slider("Minimum Final Summary Length:", min_value=10, max_value=150, value=30, step=5)
    with col2:
        max_summary_length = st.slider("Maximum Final Summary Length:", min_value=50, max_value=500, value=150, step=10)

    show_intermediate = st.checkbox("Show intermediate translation and English summary", value=True)

    if st.button("✨ Summarize Article", type="primary"):
        if not article_text.strip():
            st.warning("Please enter some text to summarize.")
        else:
            with st.spinner("Processing and generating summary... This can take a while for long non-English texts."):
                try:
                    summary_results = summarizer.summarize(
                        article_text,
                        overall_min_length=min_summary_length,
                        overall_max_length=max_summary_length,
                        # min/max_length_per_chunk are using defaults from engine
                    )

                    if summary_results.get('error'):
                        st.error(summary_results['error'])
                    else:
                        if summary_results.get('detected_language_ld'):
                            st.info(f"Detected Input Language: **{summary_results['detected_language_ld']}**")

                        if show_intermediate and summary_results.get('english_translation'):
                            st.subheader("Intermediate: Translated to English (for Summarization)")
                            st.text_area("English Translation", value=summary_results['english_translation'], height=200, disabled=True, key="eng_trans")
                        
                        if show_intermediate and summary_results.get('english_summary'):
                            st.subheader("Intermediate: English Summary (from Pegasus)")
                            st.text_area("English Summary", value=summary_results['english_summary'], height=150, disabled=True, key="eng_sum")

                        st.subheader("Final Summary")
                        if summary_results.get('final_summary'):
                            st.success(summary_results['final_summary'])
                        else:
                            st.warning("No final summary was generated.")
                            
                except Exception as e:
                    st.error(f"A critical error occurred in the application: {e}")
                    import traceback
                    st.exception(traceback.format_exc())
    
    st.markdown("---")
    st.markdown("Powered by google/pegasus-cnn_dailymail.")


if __name__ == "__main__":
    main()