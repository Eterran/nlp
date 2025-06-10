import streamlit as st
from summariser import Summarizer
from evaluation_dashboard import show_evaluation_page


@st.cache_resource
def load_summarizer_model():
    model = Summarizer()
    return model


def show_summarizer_page():
    """Main summarizer page function."""
    st.title("📰 Multilingual News Article Summarizer")
    st.markdown(
        "Enter a news article below. If non-English, it will be translated to English, summarized, and the summary translated back."
    )

    with st.spinner("Loading models... This may take longer on first run."):
        summarizer = load_summarizer_model()

    article_text = st.text_area(
        "Paste your article here (supports multiple languages):",
        height=250,
        key="article_input",
    )

    col1, col2 = st.columns(2)
    with col1:
        min_summary_length = st.slider(
            "Minimum Final Summary Length (optional):",
            min_value=0,
            max_value=150,
            value=0,
            step=5,
            help="Set to 0 to let the model decide."
        )
    with col2:
        max_summary_length = st.slider(
            "Maximum Final Summary Length (optional):",
            min_value=0,
            max_value=500,
            value=0,
            step=10,
            help="Set to 0 to let the model decide."
        )

    show_intermediate = st.checkbox(
        "Show intermediate translation and English summary", value=True
    )

    if st.button("✨ Summarize Article", type="primary"):
        if not article_text.strip():
            st.warning("Please enter some text to summarize.")
        else:
            with st.spinner(
                "Processing and generating summary... This can take a while for long non-English texts."
            ):
                try:
                    # Convert 0 to None before passing to the summarizer
                    effective_min_length = None if min_summary_length == 0 else min_summary_length
                    effective_max_length = None if max_summary_length == 0 else max_summary_length

                    summary_results = summarizer.summarize(
                        article_text,
                        overall_min_length=effective_min_length,
                        overall_max_length=effective_max_length,
                    )

                    if summary_results.get('error'):
                        st.error(summary_results['error'])
                    else:
                        detected_lang = summary_results.get('detected_language_raw')
                        confidence = summary_results.get('detected_language_confidence')
                        
                        if detected_lang:
                            lang_display_text = f"Detected Input Language: **{detected_lang}**"
                            if confidence is not None:
                                lang_display_text += f" (Confidence: {confidence:.2f})"
                            st.info(lang_display_text)
                            if summary_results.get('translation_performed') and \
                               (detected_lang != 'en' and detected_lang != 'eng'):
                                st.caption("Translation to English was performed before summarization.")
                            elif summary_results.get('translation_performed') and \
                                 (detected_lang == 'en' or detected_lang == 'eng'):
                                st.caption("Input detected as English with low confidence; an English-to-English 'translation' pass was performed for normalization before summarization.")

                        if show_intermediate and summary_results.get(
                            "english_translation"
                        ):
                            st.subheader(
                                "Intermediate: Translated to English (for Summarization)"
                            )
                            st.text_area(
                                "English Translation",
                                value=summary_results["english_translation"],
                                height=200,
                                disabled=True,
                                key="eng_trans",
                            )

                        if show_intermediate and summary_results.get("english_summary"):
                            st.subheader("Intermediate: English Summary (from Pegasus)")
                            st.text_area(
                                "English Summary",
                                value=summary_results["english_summary"],
                                height=150,
                                disabled=True,
                                key="eng_sum",
                            )

                        st.subheader("Final Summary")
                        if summary_results.get("final_summary"):
                            st.success(summary_results["final_summary"])
                        else:
                            st.warning("No final summary was generated.")

                except Exception as e:
                    st.error(f"A critical error occurred in the application: {e}")
                    import traceback

                    st.exception(traceback.format_exc())

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 16px;'>"
        "Powered by google/pegasus-cnn_dailymail<br>"
        "<strong>Made by Group 31:</strong> Loh Lit Hoong, John Ong Ming Hom, Liew Jin Sze, Kueh Pang Lang"
        "</div>",
        unsafe_allow_html=True,
    )


def main():
    """Main application with navigation."""
    st.set_page_config(
        page_title="Multilingual News Summarizer", page_icon="📰", layout="wide"
    )

    custom_font_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@200..700&display=swap');

        /* Apply to general elements first */
        html, body, [class*="st-"],
        .stTextArea textarea, .stTextInput input,
        .stButton button, .stSelectbox select, .stSlider label,
        p, span, div, li /* Common text holding elements */
        {
            font-family: 'Oswald', sans-serif;
        }

        /* Specifically target headings, with !important */
        h1, h2, h3, h4, h5, h6,
        [data-testid="stAppViewContainer"] h1, /* Main page title */
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stSidebarContent"] h1, /* Sidebar title */
        [data-testid="stSidebarContent"] h2,
        [data-testid="stSidebarContent"] h3,
        div[data-testid="stHeading"] > h1, /* For st.title() */
        div[data-testid="stHeading"] > h2, /* For st.header() */
        div[data-testid="stHeading"] > h3  /* For st.subheader() */
        {
            font-family: 'Oswald', sans-serif !important;
        }
    </style>
    """
    st.markdown(custom_font_css, unsafe_allow_html=True)

    # Navigation sidebar
    with st.sidebar:
        st.title("🧭 Navigation")
        page = st.selectbox(
            "Select Page:",
            ["📰 News Summarizer", "📊 Evaluation Dashboard"],
            index=0,
            key="navigation",
        )

        st.markdown("---")
        st.markdown("**About this App:**")
        st.markdown("• Multilingual news summarization")
        st.markdown(
            "• Powered by [Pegasus](https://huggingface.co/google/pegasus-cnn_dailymail) & [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M)"
        )
        st.markdown("• Evaluation with ROUGE metrics")

    # Display selected page
    if page == "📰 News Summarizer":
        show_summarizer_page()
    elif page == "📊 Evaluation Dashboard":
        show_evaluation_page()


if __name__ == "__main__":
    main()
