from transformers import pipeline, PegasusForConditionalGeneration, PegasusTokenizer

def main():
    model_name = "google/pegasus-cnn_dailymail"

    try:
        print(f"Loading tokenizer: {model_name}...")
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded successfully!")

        print(f"Loading model: {model_name}...")
        model = PegasusForConditionalGeneration.from_pretrained(model_name)
        print("Model loaded successfully!")
        
        print("Creating summarization pipeline...")
        # use GPU (if it is available)
        summarizer_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer, device=0) 
        print("Pipeline created successfully!")

    except Exception as e:
        print(f"Error during model/tokenizer loading or pipeline creation: {e}")
        # Print more detailed error if possible
        import traceback
        traceback.print_exc()
        return

    article_text = """
    Scientists have discovered a new species of glowing frog in the Amazon rainforest.
    The frog, which has been named 'Luminos Hyalinobatrachium', emits a faint blue light
    from its translucent skin. Researchers believe this bioluminescence might be used
    for communication or camouflage in the dense jungle environment. The discovery
    highlights the incredible biodiversity still being uncovered in the region and
    underscores the importance of conservation efforts to protect these unique ecosystems.
    Further studies are planned to understand the exact mechanism and purpose of the glow.
    """

    print("\nOriginal Article:")
    print(article_text)

    try:
        print("\nGenerating summary...")
        summary = summarizer_pipeline(article_text, max_length=60, min_length=20, do_sample=False)
        
        if summary and isinstance(summary, list) and 'summary_text' in summary[0]:
            print("\nGenerated Summary:")
            print(summary[0]['summary_text'])
        else:
            print("Could not generate summary or unexpected output format.")
            print("Output from pipeline:", summary)

    except Exception as e:
        print(f"Error during summarization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()