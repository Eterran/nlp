from transformers import PegasusForConditionalGeneration, PegasusTokenizer, pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer as NLLBTokenizer
from transformers import M2M100ForConditionalGeneration, M2M100Config
# from langdetect import detect, LangDetectException
import torch
import os

_ft_model = None
_FASTTEXT_MODEL_PATH = "lid.176.bin"
_ENGLISH_CONFIDENCE_THRESHOLD = 0.95

try:
    import fasttext
    if os.path.exists(_FASTTEXT_MODEL_PATH):
        print(f"Loading fastText language identification model from: {_FASTTEXT_MODEL_PATH}")
        _ft_model = fasttext.load_model(_FASTTEXT_MODEL_PATH)
        print("fastText model loaded successfully.")
    else:
        print(f"Warning: fastText model file not found at {_FASTTEXT_MODEL_PATH}.")
        _ft_model = None
except ImportError:
    print("Warning: fastText library not installed.")
    _ft_model = None
except Exception as e:
    print(f"Error loading fastText model: {e}")
    _ft_model = None

def get_language_code(text: str) -> tuple[str | None, str | None, float | None, bool]:
    """
    Detects language using fastText and returns a tuple:
    (detected_code_raw, nllb_compatible_code, confidence, force_translation_to_english_flag)
    The last flag is True if detected as English but confidence is below threshold.
    """
    global _ft_model
    detected_code_raw = None
    confidence = 0.0
    force_translation = False

    if not _ft_model:
        print("Error: fastText model not available for language detection.")
        return None, None, None, False 

    try:
        cleaned_text = text.replace("\n", " ").replace("\r", " ")
        if not cleaned_text.strip():
            print("Warning: Input text is empty or whitespace only for language detection.")
            return None, None, None, False

        predictions = _ft_model.predict(cleaned_text, k=1)
        if predictions and predictions[0]:
            detected_code_raw = predictions[0][0].replace('__label__', '')
            confidence = predictions[1][0]
            print(f"fastText detected: {detected_code_raw} with confidence {confidence:.4f}")

            # Check for low confidence English
            if (detected_code_raw == 'en' or detected_code_raw == 'eng') and confidence < _ENGLISH_CONFIDENCE_THRESHOLD:
                print(f"English detected with low confidence ({confidence:.4f} < {_ENGLISH_CONFIDENCE_THRESHOLD}). Flagging for potential translation.")
                force_translation = True
        else:
            print("fastText could not predict language.")
            return None, None, None, False
    except Exception as e:
        print(f"Error during fastText language detection: {e}")
        return None, None, None, False

    if not detected_code_raw:
        return None, None, None, False
        
        # --- NLLB Language Code Mapping ---
        # NLLB uses Flores-200 codes
    nllb_lang_map = {
        'en': 'eng_Latn', 'eng': 'eng_Latn',
        'es': 'spa_Latn', 'spa': 'spa_Latn',
        'fr': 'fra_Latn',
        'de': 'deu_Latn', 'ger': 'deu_Latn',
        'it': 'ita_Latn',
        'pt': 'por_Latn',
        'zh': 'zho_Hans',
        'zh-cn': 'zho_Hans', 
        'zh-tw': 'zho_Hant',
        'ja': 'jpn_Jpan', 'jpn': 'jpn_Jpan',
        'ko': 'kor_Hang', 'kor': 'kor_Hang',
        'ar': 'ara_Arab',
        'hi': 'hin_Deva',
        'ru': 'rus_Cyrl',
        'bn': 'ben_Beng',
        'pa': 'pan_Guru',
        'ur': 'urd_Arab',
        'ta': 'tam_Taml',
        'te': 'tel_Telu',
        'ml': 'mal_Mlym',
        'gu': 'guj_Gujr',
        'mr': 'mar_Deva',
        'id': 'ind_Latn', 'ind': 'ind_Latn',
        'vi': 'vie_Latn', 'vie': 'vie_Latn',
        'th': 'tha_Thai',
        'tr': 'tur_Latn',
        'fa': 'pes_Arab', 'per': 'pes_Arab',
        'uk': 'ukr_Cyrl',
        'pl': 'pol_Latn',
        'nl': 'nld_Latn', 'dut': 'nld_Latn',
        'ro': 'ron_Latn', 'rum': 'ron_Latn',
        'cs': 'ces_Latn', 'cze': 'ces_Latn',
        'sv': 'swe_Latn',
        'fi': 'fin_Latn',
        'da': 'dan_Latn',
        'no': 'nob_Latn',
        'el': 'ell_Grek', 'gre': 'ell_Grek',
        'he': 'heb_Hebr',
        'hu': 'hun_Latn',
        'bg': 'bul_Cyrl',
        'sr': 'srp_Cyrl',
        'hr': 'hrv_Latn',
        'sk': 'slk_Latn',
        'sl': 'slv_Latn',
        'et': 'est_Latn',
        'lv': 'lav_Latn',
        'lt': 'lit_Latn',
        'sw': 'swh_Latn',
        'am': 'amh_Ethi',
        'yo': 'yor_Latn',
        'ig': 'ibo_Latn',
        'zu': 'zul_Latn',
        'xh': 'xho_Latn',
        'my': 'mya_Mymr',
        'km': 'khm_Khmr',
        'lo': 'lao_Laoo',
        'ne': 'npi_Deva',
        'si': 'sin_Sinh',
        'az': 'azj_Latn',
        'kk': 'kaz_Cyrl',
        'uz': 'uzn_Latn',
        'mn': 'khk_Cyrl',
        'ps': 'pbt_Arab',
        'tg': 'tgk_Cyrl',
        'tk': 'tuk_Latn',
        'so': 'som_Latn',
    }

    nllb_code = nllb_lang_map.get(detected_code_raw)
    if not nllb_code and '-' in detected_code_raw:
         nllb_code = nllb_lang_map.get(detected_code_raw.split('-')[0])

    if not nllb_code:
        print(f"Warning: No NLLB mapping for detected language code '{detected_code_raw}'.")
    
    return detected_code_raw, nllb_code, confidence, force_translation

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
        
        self._load_models()

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
        # Refined from your version for clarity and robustness
        if not self.pegasus_model or not self.pegasus_tokenizer:
            print("Error: Pegasus model/tokenizer not loaded.")
            return "Error: Summarization model not available."
        try:
            inputs = self.pegasus_tokenizer(
                text_chunk, return_tensors="pt", truncation=True, max_length=self.effective_input_token_limit
            ).to(self.device)

            summary_ids = self.pegasus_model.generate(
                inputs["input_ids"], num_beams=4, min_length=min_length, max_length=max_length, early_stopping=True
            )
            
            summary_text_raw = self.pegasus_tokenizer.decode(
                summary_ids[0], 
                skip_special_tokens=True, # Important
                clean_up_tokenization_spaces=True
            )

            # Post-processing for <n> tokens
            processed_summary = summary_text_raw.replace("<n>", "\n")
            lines = [line.strip() for line in processed_summary.split('\n')]
            processed_summary = "\n".join(filter(None, lines)) # Removes empty lines

            return processed_summary.strip()
        except Exception as e:
            print(f"Error during Pegasus summarization of chunk: {e}")
            import traceback
            traceback.print_exc()
            return f"Error summarizing chunk: {e}"
    
    def summarize(self, text: str, min_length_per_chunk: int = 20, max_length_per_chunk: int = 80,
                      overall_min_length: int = 30, overall_max_length: int = 150) -> dict:
        result = {
            'final_summary': None, 'detected_language_raw': None, 'detected_language_confidence': None,
            'english_translation': None, 'english_summary': None, 'error': None,
            'translation_performed': False # New key to indicate if translation to English happened
        }

        if not all([self.pegasus_model, self.pegasus_tokenizer, self.translator_model, self.translator_tokenizer]):
            result['error'] = "Error: Core models not loaded."
            return result
        if _ft_model is None:
            result['error'] = "Error: Language detection model (fastText) not available."
            return result
        if not text.strip():
            result['error'] = "Error: Input text is empty."
            return result

        original_text_to_process = text
        detected_lang_raw, detected_lang_nllb, confidence, force_translation_flag = get_language_code(text)
        
        result['detected_language_raw'] = detected_lang_raw
        result['detected_language_confidence'] = confidence

        if not detected_lang_raw:
            result['error'] = "Error: Could not detect language."
            return result
        
        # Determine if translation to English is needed
        needs_translation_to_english = False
        if (detected_lang_raw != 'en' and detected_lang_raw != 'eng'): # Clearly not English
            needs_translation_to_english = True
        elif force_translation_flag: # Detected as English but low confidence
            print(f"INFO: Detected as English ('{detected_lang_raw}') but confidence {confidence} is low. Attempting 'translation' to English to normalize.")
            needs_translation_to_english = True
            # For "translation" from low-confidence English to English, NLLB needs a source NLLB code.
            # We'll use 'eng_Latn' as the source for NLLB in this specific case.
            if not detected_lang_nllb: # If 'en' or 'eng' didn't map (should not happen with good map)
                detected_lang_nllb = 'eng_Latn' 
            result['translation_performed'] = True # Mark that we are "translating"
        
        if needs_translation_to_english:
            if detected_lang_nllb: # Check if we have a valid NLLB mapping for the source
                print(f"Original language: {detected_lang_raw} (NLLB src: {detected_lang_nllb}). Translating to English (eng_Latn)...")
                english_text_translation = self._translate_text(original_text_to_process, detected_lang_nllb, "eng_Latn")
                result['translation_performed'] = True # Mark that actual translation happened
                
                if not english_text_translation or english_text_translation.startswith("Error"):
                    err_msg = f"Error: Translation to English failed for lang {detected_lang_raw}."
                    # ... (your existing error message construction) ...
                    result['error'] = err_msg
                    return result
                
                original_text_to_process = english_text_translation
                result['english_translation'] = original_text_to_process
                print("Translation to English complete.")
            else: # Non-English detected by fastText, but no NLLB mapping for it
                print(f"Critical: Original language detected as '{detected_lang_raw}', but no NLLB code mapping found. Cannot reliably translate.")
                result['error'] = f"Error: Input language '{detected_lang_raw}' is not supported for translation (no NLLB mapping)."
                return result


        # --- Summarization of English text (original_text_to_process is now English) ---
        all_input_ids = self.pegasus_tokenizer.encode(original_text_to_process, add_special_tokens=False)
        total_tokens = len(all_input_ids)
        english_summary_text = None

        if total_tokens <= self.effective_input_token_limit:
            english_summary_text = self._summarize_english_text(original_text_to_process, overall_min_length, overall_max_length)
        else:
            # (Your full chunking logic here)
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
                if chunk_sum.startswith("Error"):
                    result['error'] = f"Error summarizing chunk {i_chunk+1}: {chunk_sum.split(':', 1)[1].strip() if ':' in chunk_sum else chunk_sum}"
                    return result
                chunk_summaries_list.append(chunk_sum)
            english_summary_text = " ".join(chunk_summaries_list)
        
        if not english_summary_text or english_summary_text.startswith("Error"):
            err_msg_sum = f"Error during English summarization"
            if english_summary_text and english_summary_text.startswith("Error"):
                err_msg_sum += f": {english_summary_text.split(':', 1)[1].strip() if ':' in english_summary_text else english_summary_text}"
            elif not english_summary_text:
                err_msg_sum += ": Summarization returned no text."
            result['error'] = err_msg_sum
            return result
        
        result['english_summary'] = english_summary_text

        # --- Translate summary back ---
        final_summary_text = english_summary_text
        if result['translation_performed'] and not force_translation_flag: # if force_translation_flag, it was already "English"
            if detected_lang_raw != 'en' and detected_lang_raw != 'eng' and detected_lang_nllb:
                print(f"Translating summary back to {detected_lang_raw} (NLLB: {detected_lang_nllb})...")
                translated_summary_back = self._translate_text(english_summary_text, "eng_Latn", detected_lang_nllb)
                if translated_summary_back and not translated_summary_back.startswith("Error"):
                    final_summary_text = translated_summary_back
                    print("Back-translation complete.")
                else:
                    warning_msg = f"Warning: Back-translation to {detected_lang_raw} failed or returned error. Returning English summary."
                    if translated_summary_back and translated_summary_back.startswith("Error"):
                        warning_msg += f" Details: {translated_summary_back.split(':', 1)[1].strip() if ':' in translated_summary_back else translated_summary_back}"
                    elif not translated_summary_back:
                        warning_msg += " Back-translation function returned None."
                    print(warning_msg)
        
        result['final_summary'] = final_summary_text
        return result

if __name__ == "__main__":
    print("Testing Summarizer Engine...")
    if _ft_model is None:
        print("CRITICAL: fastText model not loaded. Please ensure 'lid.176.bin' is present and readable.")
        print("Skipping direct engine tests.")
    else:
        try:
            engine = Summarizer()
            
            print("\n--- Testing English Article ---")
            short_article_en = "Apple today announced its new iPhone, which features a faster processor and an improved camera. The company expects strong sales."
            print(f"Input (EN):\n{short_article_en}")
            summary_en_result = engine.summarize(short_article_en, overall_min_length=10, overall_max_length=50)
            print(f"Result (EN): {summary_en_result}")

            print("\n--- Testing Chinese Article (Short) ---")
            short_article_zh = "今天天气真好，阳光明媚。"
            print(f"Input (ZH):\n{short_article_zh}")
            summary_zh_result = engine.summarize(short_article_zh, overall_min_length=5, overall_max_length=30)
            print(f"Result (ZH): {summary_zh_result}")
            
            print("\n--- Testing Mixed Language Article ---")
            mixed_article = '今天天气真好 How are you? 大家都在玩'
            print(f"Input (Mixed):\n{mixed_article}")
            summary_mixed_result = engine.summarize(mixed_article, overall_min_length=5, overall_max_length=40)
            print(f"Result (Mixed): {summary_mixed_result}")


            print("\n--- Testing Long English Article ---")
            long_article_en = """
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
            print(f"Input (Long EN), num chars: {len(long_article_en)}")
            summary_long_en_result = engine.summarize(long_article_en, overall_min_length=50, overall_max_length=200)
            print(f"Result (Long EN): {summary_long_en_result}")


        except Exception as e:
            print(f"Failed to initialize or use Summarizer engine: {e}")
            import traceback
            traceback.print_exc()