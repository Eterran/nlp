from transformers import PegasusForConditionalGeneration, PegasusTokenizer, pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer as NLLBTokenizer
from transformers import M2M100ForConditionalGeneration, M2M100Config
from langdetect import detect, LangDetectException
import torch

def get_language_code(text: str) -> tuple[str | None, str | None]:
    """Detects language and returns a tuple: (langdetect_code, nllb_compatible_code)."""
    try:
        lang_code_ld = detect(text)
        
        # --- NLLB Language Code Mapping ---
        # NLLB uses Flores-200 codes
        nllb_lang_map = {
            'en': 'eng_Latn',    # English
            'es': 'spa_Latn',    # Spanish
            'fr': 'fra_Latn',    # French
            'de': 'deu_Latn',    # German
            'it': 'ita_Latn',    # Italian
            'pt': 'por_Latn',    # Portuguese
            'zh-cn': 'zho_Hans', # Chinese (Simplified)
            'zh-tw': 'zho_Hant', # Chinese (Traditional)
            'ja': 'jpn_Jpan',    # Japanese
            'ko': 'kor_Hang',    # Korean
            'ar': 'ara_Arab',    # Arabic
            'hi': 'hin_Deva',    # Hindi
            'ru': 'rus_Cyrl',    # Russian
            'bn': 'ben_Beng',    # Bengali
            'pa': 'pan_Guru',    # Punjabi
            'ur': 'urd_Arab',    # Urdu
            'ta': 'tam_Taml',    # Tamil
            'te': 'tel_Telu',    # Telugu
            'ml': 'mal_Mlym',    # Malayalam
            'gu': 'guj_Gujr',    # Gujarati
            'mr': 'mar_Deva',    # Marathi
            'id': 'ind_Latn',    # Indonesian
            'vi': 'vie_Latn',    # Vietnamese
            'th': 'tha_Thai',    # Thai
            'tr': 'tur_Latn',    # Turkish
            'fa': 'pes_Arab',    # Persian
            'uk': 'ukr_Cyrl',    # Ukrainian
            'pl': 'pol_Latn',    # Polish
            'nl': 'nld_Latn',    # Dutch
            'ro': 'ron_Latn',    # Romanian
            'cs': 'ces_Latn',    # Czech
            'sv': 'swe_Latn',    # Swedish
            'fi': 'fin_Latn',    # Finnish
            'da': 'dan_Latn',    # Danish
            'no': 'nob_Latn',    # Norwegian (Bokmål)
            'el': 'ell_Grek',    # Greek
            'he': 'heb_Hebr',    # Hebrew
            'hu': 'hun_Latn',    # Hungarian
            'bg': 'bul_Cyrl',    # Bulgarian
            'sr': 'srp_Cyrl',    # Serbian (Cyrillic)
            'hr': 'hrv_Latn',    # Croatian
            'sk': 'slk_Latn',    # Slovak
            'sl': 'slv_Latn',    # Slovenian
            'et': 'est_Latn',    # Estonian
            'lv': 'lav_Latn',    # Latvian
            'lt': 'lit_Latn',    # Lithuanian
            'sw': 'swh_Latn',    # Swahili
            'am': 'amh_Ethi',    # Amharic
            'yo': 'yor_Latn',    # Yoruba
            'ig': 'ibo_Latn',    # Igbo
            'zu': 'zul_Latn',    # Zulu
            'xh': 'xho_Latn',    # Xhosa
            'my': 'mya_Mymr',    # Burmese
            'km': 'khm_Khmr',    # Khmer
            'lo': 'lao_Laoo',    # Lao
            'ne': 'npi_Deva',    # Nepali
            'si': 'sin_Sinh',    # Sinhala
            'az': 'azj_Latn',    # Azerbaijani (North)
            'kk': 'kaz_Cyrl',    # Kazakh
            'uz': 'uzn_Latn',    # Uzbek (Northern)
            'mn': 'khk_Cyrl',    # Mongolian
            'ps': 'pbt_Arab',    # Pashto
            'tg': 'tgk_Cyrl',    # Tajik
            'tk': 'tuk_Latn',    # Turkmen
            'so': 'som_Latn',    # Somali
        }

        # Fallback for unlisted langdetect codes
        nllb_code = nllb_lang_map.get(lang_code_ld)
        if not nllb_code and '-' in lang_code_ld: # Try stripping region for NLLB map
             nllb_code = nllb_lang_map.get(lang_code_ld.split('-')[0])

        if not nllb_code:
            print(f"Warning: No NLLB mapping for langdetect code '{lang_code_ld}'. Translation might fail.")
        
        return lang_code_ld, nllb_code
    
    except LangDetectException:
        print("Language could not be detected reliably.")
        return None, None

class Summarizer:
    def __init__(self, 
                 summarizer_model_name="google/pegasus-cnn_dailymail", # "facebook/mbart-large-50"
                 translator_model_name="facebook/nllb-200-distilled-600M"): # NLLB model
        
        self.summarizer_model_name = summarizer_model_name
        self.translator_model_name = translator_model_name
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.pegasus_tokenizer = None
        self.pegasus_model = None
        self.translator_tokenizer = None
        self.translator_model = None

        if "pegasus" in self.summarizer_model_name.lower():
            self.effective_input_token_limit = 512
        elif "bart" in self.summarizer_model_name.lower():
            self.effective_input_token_limit = 1024
        else:
            self.effective_input_token_limit = 512
        
        self._load_models() # Combined loading

    def _load_models(self):
        try:
            print(f"Loading Pegasus tokenizer: {self.summarizer_model_name}...")
            self.pegasus_tokenizer = PegasusTokenizer.from_pretrained(self.summarizer_model_name)
            print(f"{self.summarizer_model_name} tokenizer loaded.")
            print(f"Loading Pegasus model: {self.summarizer_model_name}...")
            self.pegasus_model = PegasusForConditionalGeneration.from_pretrained(self.summarizer_model_name).to(self.device)
            print(f"{self.summarizer_model_name} model loaded.")

            print(f"Loading NLLB tokenizer: {self.translator_model_name}...")
            self.translator_tokenizer = NLLBTokenizer.from_pretrained(self.translator_model_name)
            print(f"NLLB tokenizer type: {type(self.translator_tokenizer)}")
            print("NLLB tokenizer loaded.")

            print(f"Loading NLLB model config using M2M100Config: {self.translator_model_name}...")
            translator_config = M2M100Config.from_pretrained(self.translator_model_name)
            print(f"Explicit NLLB/M2M100 config type: {type(translator_config)}")

            print(f"Loading NLLB model using M2M100ForConditionalGeneration: {self.translator_model_name}...")
            self.translator_model = M2M100ForConditionalGeneration.from_pretrained(
                self.translator_model_name,
                config=translator_config
            ).to(self.device)
            print(f"NLLB model type: {type(self.translator_model)}")
            print("NLLB model loaded.")

            # --- DEBUGGING ---
            # if self.translator_model and hasattr(self.translator_model, 'config'):
            #     print(f"NLLB model config type: {type(self.translator_model.config)}") # Should be M2M100Config
            #     print(f"NLLB model config keys: {list(self.translator_model.config.to_dict().keys())}")
            #     if hasattr(self.translator_model.config, 'lang_code_to_id'):
            #         print("SUCCESS: 'lang_code_to_id' FOUND in translator_model.config")
            #     else:
            #         print("FAILURE: 'lang_code_to_id' NOT FOUND in translator_model.config")
            # else:
            #     print("NLLB model or its config is None after loading attempts.")
            # --- END DEBUGGING ---

        except Exception as e:
            print(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    def _translate_text(self, text_to_translate: str, src_nllb_lang: str, tgt_nllb_lang: str) -> str | None:
        if not self.translator_model or not self.translator_tokenizer:
            print("Translator model/tokenizer not loaded.")
            return None
        if not src_nllb_lang: # This is the NLLB code for the source language
            print(f"Missing NLLB source language code for translation. Cannot translate.")
            return text_to_translate # Or None, depending on how you want to handle

        try:
            # Set the source language for the NLLB tokenizer
            # This is crucial for NLLB when translating FROM a non-English language
            self.translator_tokenizer.src_lang = src_nllb_lang
            
            inputs = self.translator_tokenizer(
                text_to_translate, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, # Pad to max_length of model or batch, helps with potential length issues
                # max_length=512 # Optional: you can set a max_length for the tokenizer input here too
            ).to(self.device)
            
            # Get the target language token ID using convert_tokens_to_ids
            # This is the method shown in the official documentation
            try:
                target_lang_token_id = self.translator_tokenizer.convert_tokens_to_ids(tgt_nllb_lang)
            except Exception as e_conv:
                print(f"Error converting target language code '{tgt_nllb_lang}' to ID: {e_conv}")
                # This might happen if tgt_nllb_lang is not a known special token for the tokenizer.
                # Ensure your NLLB code mapping produces valid codes recognized by the tokenizer.
                return None

            if target_lang_token_id == self.translator_tokenizer.unk_token_id:
                print(f"Warning: Target language code '{tgt_nllb_lang}' was converted to UNK token ID. "
                        f"This means the tokenizer doesn't recognize it as a language code. Check your mapping.")
                # Forcing generation to UNK is not useful.
                return None

            print(f"DEBUG: Source NLLB Lang: {src_nllb_lang}, Target NLLB Lang: {tgt_nllb_lang}, Target Lang Token ID: {target_lang_token_id}")

            generated_tokens = self.translator_model.generate(
                **inputs,
                forced_bos_token_id=target_lang_token_id,
                max_length=1024,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
            translated_text = self.translator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translated_text
        except Exception as e:
            print(f"Error during translation from {src_nllb_lang} to {tgt_nllb_lang}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _summarize_english_text(self, text_chunk: str, min_length: int, max_length: int) -> str:
        inputs = self.pegasus_tokenizer(
            text_chunk, return_tensors="pt", truncation=True, max_length=self.effective_input_token_limit
        ).to(self.device)
        summary_ids = self.pegasus_model.generate(
            inputs["input_ids"], num_beams=4, min_length=min_length, max_length=max_length, early_stopping=True
        )
        summary_text = self.pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        summary_text = summary_text.replace("<n>", "\n").strip()
        return summary_text
    
    def summarize(self, text: str, min_length_per_chunk: int = 20, max_length_per_chunk: int = 80,
                      overall_min_length: int = 30, overall_max_length: int = 150) -> dict:
        """
        Returns a dictionary containing:
        {
            'final_summary': str | None,
            'detected_language_ld': str | None, (langdetect code)
            'english_translation': str | None, (if input was non-English)
            'english_summary': str | None, (summary of the (translated) English text)
            'error': str | None (if an error occurred)
        }
        """
        # Initialize return dictionary
        result = {
            'final_summary': None,
            'detected_language_ld': None,
            'english_translation': None,
            'english_summary': None,
            'error': None
        }

        if not all([self.pegasus_model, self.pegasus_tokenizer, self.translator_model, self.translator_tokenizer]):
            result['error'] = "Error: Core models not loaded."
            return result
        if not text.strip():
            result['error'] = "Error: Input text is empty."
            return result

        original_text_to_process = text
        detected_lang_ld, detected_lang_nllb = get_language_code(text)
        result['detected_language_ld'] = detected_lang_ld

        if not detected_lang_ld:
            result['error'] = "Error: Could not detect language."
            return result
        
        if detected_lang_ld != 'en' and detected_lang_nllb:
            print(f"Original language: {detected_lang_ld} ({detected_lang_nllb}). Translating to English...")
            english_text_translation = self._translate_text(original_text_to_process, detected_lang_nllb, "eng_Latn")
            if not english_text_translation or english_text_translation.startswith("Error"):
                result['error'] = f"Error: Translation to English failed for lang {detected_lang_ld}."
                if english_text_translation: result['error'] += f" ({english_text_translation})" # append translator error
                return result
            
            original_text_to_process = english_text_translation # This is now the English text
            result['english_translation'] = original_text_to_process
            print("Translation to English complete.")
        elif detected_lang_ld != 'en' and not detected_lang_nllb:
                print(f"Warning: Original language is {detected_lang_ld}, but no NLLB code mapping. Summary will likely be poor or fail.")
                result['error'] = f"Error: Cannot translate from {detected_lang_ld} due to missing NLLB code mapping. Cannot proceed to summarization."
                return result


        # --- Summarization of English text (original_text_to_process is now English) ---
        all_input_ids = self.pegasus_tokenizer.encode(original_text_to_process, add_special_tokens=False)
        total_tokens = len(all_input_ids)
        english_summary_text = None

        if total_tokens <= self.effective_input_token_limit:
            english_summary_text = self._summarize_english_text(original_text_to_process, overall_min_length, overall_max_length)
        else:
            print(f"Input text for summarizer has {total_tokens} tokens. Applying chunking...")
            chunk_size = self.effective_input_token_limit - 50 
            overlap_size = 50 
            chunks_texts = []
            start_idx = 0
            while start_idx < total_tokens:
                end_idx = min(start_idx + chunk_size, total_tokens)
                chunk_token_ids = all_input_ids[start_idx:end_idx]
                chunk_text_for_summary = self.pegasus_tokenizer.decode(chunk_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                chunks_texts.append(chunk_text_for_summary)
                if end_idx == total_tokens: break
                start_idx += (chunk_size - overlap_size)
                if start_idx >= end_idx : break
            
            if not chunks_texts:
                result['error'] = "Error: Failed to create any chunks from the (potentially translated) text."
                return result
            
            chunk_summaries_list = []
            for i_chunk, chunk_text_to_summarize in enumerate(chunks_texts):
                print(f"Summarizing (English) chunk {i_chunk+1}/{len(chunks_texts)}...")
                chunk_sum = self._summarize_english_text(chunk_text_to_summarize, min_length_per_chunk, max_length_per_chunk)
                if chunk_sum.startswith("Error"): # Check if individual chunk summarization failed
                    result['error'] = f"Error summarizing chunk {i_chunk+1}: {chunk_sum}"
                    return result
                chunk_summaries_list.append(chunk_sum)
            english_summary_text = " ".join(chunk_summaries_list)
        
        if not english_summary_text or english_summary_text.startswith("Error"):
            result['error'] = f"Error during English summarization: {english_summary_text if english_summary_text else 'Unknown error'}"
            return result
        
        result['english_summary'] = english_summary_text

        # --- Translate summary back ---
        final_summary_text = english_summary_text
        if detected_lang_ld != 'en' and detected_lang_nllb:
            print(f"Translating summary back to {detected_lang_ld} ({detected_lang_nllb})...")
            translated_summary_back = self._translate_text(english_summary_text, "eng_Latn", detected_lang_nllb)
            if translated_summary_back and not translated_summary_back.startswith("Error"):
                final_summary_text = translated_summary_back
                print("Back-translation complete.")
            else:
                print(f"Warning: Back-translation to {detected_lang_ld} failed or returned error. Returning English summary.")
                if translated_summary_back: print(f"Translator error: {translated_summary_back}")
        
        result['final_summary'] = final_summary_text
        return result
    
    def _summarize_single_chunk(self, text_chunk: str, min_length: int, max_length: int) -> str:
        """Helper function to summarize a single chunk of text."""
        if not self.model or not self.tokenizer:
            return "Error: Model and/or tokenizer not loaded for chunk."

        try:
            inputs = self.tokenizer(
                text_chunk,
                return_tensors="pt",
                truncation=True, # Should not truncate if chunking is done right
                max_length=self.effective_input_token_limit 
            )
            # inputs = inputs.to(self.model.device)

            summary_ids = self.model.generate(
                inputs["input_ids"],
                num_beams=4,
                min_length=min_length,
                max_length=max_length,
                early_stopping=True
            )
            summary_text = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            return summary_text.strip()
        except Exception as e:
            import traceback
            print(f"Error summarizing chunk: {e}")
            traceback.print_exc()
            return f"[Error summarizing chunk: {e}]"

    # def summarize(self, text: str, min_length_per_chunk: int = 20, max_length_per_chunk: int = 80, 
    #               overall_min_length: int = 30, overall_max_length: int = 150) -> str:
    #     """
    #     Generates a summary for the given English text, using chunking if necessary.

    #     Args:
    #         text (str): The English text to summarize.
    #         min_length_per_chunk (int): Min length for summary of each individual chunk.
    #         max_length_per_chunk (int): Max length for summary of each individual chunk.
    #         overall_min_length (int): Desired min length of the final combined summary (not strictly enforced yet).
    #         overall_max_length (int): Desired max length of the final combined summary (not strictly enforced yet).

    #     Returns:
    #         str: The generated summary, or an error message if summarization fails.
    #     """
    #     if not self.model or not self.tokenizer:
    #         return "Error: Model and/or tokenizer not loaded."
    #     if not text.strip():
    #         return "Error: Input text is empty."

    #     # 1. Tokenize the entire text to check its length
    #     all_input_ids = self.tokenizer.encode(text, add_special_tokens=False) # Get raw token ID
    #     total_tokens = len(all_input_ids)
    #     print(f"Total tokens in input text: {total_tokens}")

    #     # Check if chunking is needed
    #     if total_tokens <= self.effective_input_token_limit:
    #         print("Input text is within token limit. Summarizing as a single chunk.")
    #         return self._summarize_single_chunk(text, overall_min_length, overall_max_length)
    #     else:
    #         print(f"Input text exceeds token limit ({total_tokens} > {self.effective_input_token_limit}). Applying chunking.")
            
    #         # 2. Create Chunks
    #         # currently: split by tokens, with some overlap to maintain context.
    #         # improvements: maybe split by sentences or paragraphs using NLTK or spaCy.
            
    #         chunk_size = self.effective_input_token_limit - 50 # Leave some room
    #         overlap_size = 50 # Number of tokens to overlap between chunks

    #         chunks = []
    #         start_idx = 0
    #         while start_idx < total_tokens:
    #             end_idx = min(start_idx + chunk_size, total_tokens)
    #             chunk_token_ids = all_input_ids[start_idx:end_idx]
    #             chunk_text = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #             chunks.append(chunk_text)
                
    #             if end_idx == total_tokens:
    #                 break
    #             start_idx += (chunk_size - overlap_size) # Move window forward
    #             if start_idx >= end_idx : # Safety break if overlap is too large or chunk_size too small
    #                 print("Warning: Chunking logic might have an issue with start/end index. Breaking.")
    #                 break
            
    #         print(f"Created {len(chunks)} chunks.")
    #         if not chunks:
    #             return "Error: Failed to create any chunks from the text."

    #         # 3. Summarize each chunk
    #         chunk_summaries = []
    #         for i, chunk_text in enumerate(chunks):
    #             print(f"Summarizing chunk {i+1}/{len(chunks)}...")
    #             # Adjust min/max length for chunk summaries; they should be shorter
    #             chunk_summary = self._summarize_single_chunk(chunk_text, 
    #                                                          min_length_per_chunk, 
    #                                                          max_length_per_chunk)
    #             if not chunk_summary.startswith("[Error"):
    #                 chunk_summaries.append(chunk_summary)
    #             print(f"Chunk {i+1} summary: {chunk_summary[:100]}...")

    #         if not chunk_summaries:
    #             return "Error: Failed to generate summaries for any chunk."

    #         # 4. Combine chunk summaries
    #         # Simple concatenation for now.
    #         # TODO: This might make the summary too long or repetitive.
    #         # Consider a second summarization pass (hierarchical) in the future.
    #         final_summary = " ".join(chunk_summaries)
            
    #         # Optional: A very naive truncation of the combined summary if it's too long
    #         # A better approach would be to summarize the combined summaries.
    #         if len(self.tokenizer.encode(final_summary)) > overall_max_length + 50 : # +50 as buffer
    #              print(f"Combined summary is too long. Truncating naively. Consider hierarchical summarization.")
    #              # This is a crude way to truncate. Better to summarize the summaries.
    #              # For now, let's just return it, or truncate by tokens
    #              final_summary_tokens = self.tokenizer.encode(final_summary, truncation=True, max_length=overall_max_length)
    #              final_summary = self.tokenizer.decode(final_summary_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    #         return final_summary.strip()

if __name__ == "__main__":
    print("Testing Summarizer Engine...")
    try:
        engine = Summarizer()
        
        short_article = "Apple today announced its new iPhone, which features a faster processor and an improved camera. The company expects strong sales."
        print(f"\nSummarizing short article:\n{short_article}")
        summary1 = engine.summarize(short_article, min_length=10, max_length=50)
        print(f"Summary 1: {summary1}")

        # long_article = """
        # Persistent rain and strong winds continued to lash the coastal regions today, leading to widespread flooding and power outages. 
        # Emergency services have been working around the clock to rescue stranded residents and clear debris from roads. 
        # The meteorological department issued further warnings, stating that the adverse weather conditions are expected to prevail for at least another 24 hours. 
        # Several rivers have breached their banks, inundating low-lying agricultural lands and residential areas. 
        # Local authorities have set up temporary shelters in schools and community halls for those displaced by the floods. 
        # Volunteers are distributing food, water, and blankets to the affected families. 
        # The full extent of the damage is yet to be assessed, but initial reports suggest significant losses to property and crops. 
        # Transportation services, including trains and buses, have been severely disrupted, with many routes canceled or indefinitely delayed. 
        # Residents are urged to stay indoors and avoid unnecessary travel. The government has pledged full support for the relief and rehabilitation efforts. 
        # Officials are closely monitoring the situation and coordinating with various agencies to ensure a swift response. 
        # The impact of this severe weather event will likely be felt for weeks to come as communities begin the arduous task of recovery and rebuilding. 
        # Additional support from neighboring states has been requested to bolster the ongoing operations. 
        # The focus remains on ensuring the safety and well-being of all citizens affected by this natural calamity.
        # """
        
        # https://edition.cnn.com/2025/05/24/health/drowning-water-safety-swimming-children-wellness
        long_article = """
        Warmer weather is finally here in the Northern Hemisphere, and with it, many pools and beaches are opening for the summer. That’s great for families who want to spend time by the water, but it’s also a good time to be reminded about the importance of water safety.

        An estimated 4,000 fatal unintentional drownings happen every year in the United States — an average of 11 drowning deaths per day — according to the Centers for Disease Control and Prevention.

        Drowning is the leading cause of death in children ages 1 to 4, and after motor vehicle accidents, it is the second leading cause of death attributed to unintentional injuries among kids ages 5 to 14.

        I wanted to speak more about water safety with CNN wellness expert Dr. Leana Wen. Wen is an emergency physician and adjunct associate professor at George Washington University who previously served as Baltimore’s health commissioner. She is also the mother of two young children, and as someone who learned to swim as an adult, she is a passionate advocate for kids — and adults — learning to swim.

        CNN: Who is most at risk of drowning, and under what circumstances?

        Dr. Leana Wen: The CDC issued an important report in 2024 about unintentional drowning deaths in the United States. Death rates were significantly higher in 2020, 2021 and 2022 than in 2019, according to the agency. Moreover, the highest rates were seen in young children ages 1 to 4. Among children in this age group, 461 died by drowning in 2022, an increase of 28% from 2019.

        The report also highlighted racial disparities, with higher rates of drowning deaths among individuals identified as non-Hispanic American Indian or Alaska Native and as non-Hispanic Black or African American. Only 45% of all adults reported having had swim lessons, and those numbers were higher among White Americans (52%) than among Black (37%) and Hispanic (28%) Americans.

        Racial disparities were also reported in a 2023 analysis from the Consumer Product Safety Commission, which found that African American children made up 21% of all drownings for kids younger than age 15 in which race and ethnicity are known. Among kids ages 5 to 14, 45% of drowning deaths occurred among African Americans.

        The CPSC analysis also contained another key data point: The vast majority (80%) of pediatric drownings in which the setting is known occurred at a residence. This means that 4 in 5 kids who drowned died in their own backyard pool or that of a friend, neighbor or family member. Of these residential drownings, 91% occurred among kids younger than 7 years old.

        CNN: Why do so many drownings happen in residential settings?

        Wen: One major reason is the difference in supervision. Many public beaches and community pools hire lifeguards whose job is to watch out for the safety of everyone in or near the water. Private pools in people’s yards often don’t have someone designated for this purpose. Sometimes older children are supervising younger children but aren’t always watching. Or adults may be supervising, but they are also busy with other tasks. In addition, some of those watching others may not know how to swim themselves.

        There may also be a false sense of security in residential settings. People may think the pool is small or not that deep or there are a lot of people around, so what can happen? Keep in mind, though, that small kids can drown in just inches of water. Serious injury or death can happen within 30 seconds. Drownings are often silent because the victim is unable to call out for help.

        CNN: How can parents and guardians prevent drownings in residential settings?

        Wen: The single most important best practice is to never leave children unsupervised near a body of water. Even if they already know how to swim, even if they are wearing a flotation device, even if the pool is shallow or small, an accident could occur — and either you or another responsible adult should always be able to see your child. The supervising adult should be actively watching the child and not distracted by chores or their smartphone. That person also should not be under the influence of alcohol or drugs.

        The adult who’s responsible must also know how to swim well enough so they are able to jump into the pool and save the child if necessary. An additional safety precaution is learning CPR and first aid for infants, children and adults, which you can do through the American Red Cross.

        More than 1 in 3 Black adults say they can’t swim. This team is trying to teach them and their kids

        If you have a pool, be very careful before allowing others to use it. If your neighbors’ children want to swim in your pool, a responsible adult must accompany them. Private swimming pools should all have childproof fencing around them. The fencing should enclose the pool, have a self-close latch out of the reach of children and be at least 4 feet high. This is required by law in most states.

        CNN: What safety precautions should people take around natural bodies of water?

        Wen: Always wear a properly fitted, US Coast Guard–approved life jacket when boating. Of all the people who drowned while boating in 2022, 85% were not wearing a life jacket, according to the CDC.

        To be safe, swim in areas where a lifeguard is on duty. Always follow lifeguard guidance about safety conditions, and stay in the area designated for swimming.

        CNN: What about teaching children how to swim — can that help with water safety?

        Wen: Yes. Kids ages 1 to 4 who took part in formal swim lessons had an 88% lower risk of drowning, according to a study in JAMA Pediatrics. The goal here isn’t necessarily to teach kids all the different strokes and get them to join a swim team; it’s to impart basic lifesaving skills, such as treading water and floating on their back.

        When you are in the water with your children, take every opportunity to remind them about water safety. Other tips include never swimming alone, always asking for permission before entering the water and never diving into unknown bodies of water headfirst. Young children should also be reminded not to reach for items in the pool, as they are at risk of falling in; they should always ask for help instead.

        Never leave children alone by the water, and remind them to ask for help if they want to reach something in the pool. travelism/E+/Getty Images
        CNN: What about parents or guardians who don’t know how to swim? Do you recommend that they also take swim lessons?

        Wen: Yes. First, adults who don’t know how to swim are more likely to have children who don’t know how to swim. This was the case for me. My parents didn’t swim, and I also never learned swimming growing up.

        Second, it’s hard for adults to properly supervise children swimming if they can’t swim themselves. It was actually a terrifying experience with my own children that prompted me to learn to swim. My children were just 1 year and 3 years old one summer when my older kid pushed the younger one into the pool.

        We were at our local community pool, and there was a lifeguard who immediately sprang into action. But I remember how terrified I felt — and how helpless. I enrolled my kids in swim lessons right away. I also found an instructor to teach me, too, because I realized I had to overcome my own fear of the water and learn basic water safety skills to protect my kids.

        Learning how to swim as an adult is a humbling experience, especially for people like me who had to first start with overcoming fear. I began literally from zero. For weeks, I worked on just getting comfortable submerging my head underwater.

        Eventually, I learned how to swim and now really enjoy being in the water. And I feel a lot more comfortable supervising my children when we are in private or community swimming spaces. I’m looking forward to our local pool opening for the summer and to spending time with my kids having a fun — and safe — time in the water.
        """
        
        print(f"\nSummarizing long article:")
        # print(long_article) #
        summary2 = engine.summarize(long_article, min_length=50, max_length=200)
        print(f"Summary 2: {summary2}")

    except Exception as e:
        print(f"Failed to initialize or use Summarizer engine: {e}")
        import traceback
        traceback.print_exc()