# -*- coding: utf-8 -*-
"""
MCQ Generation Using T5 with Safety Checks and Performance Evaluation

Requirements:
pip install torch transformers==4.38.1 # Pinning version for stability, adjust if needed
pip install nltk scikit-learn numpy rouge-score pandas matplotlib psutil textstat sentencepiece # Added sentencepiece
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import nltk
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import string
import time
import psutil
import tracemalloc
import gc
import warnings
import matplotlib.pyplot as plt
import pandas as pd

# --- Optional Dependency Handling ---
try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False
    print("⚠️ Warning: `textstat` library not found. Readability metrics will be skipped.")
    print("   Install it using: pip install textstat")

# --- Download NLTK data ---
print("Downloading NLTK resources...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True) # Correct package name
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}. Please ensure you have an internet connection.")
    # Depending on the error, you might want to exit or continue with potential issues later.

# --- Load Safety Models ---
print("Loading safety models...")
try:
    toxicity_model = pipeline("text-classification", model="unitary/toxic-bert", truncation=True) # Added truncation=True
    bias_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", truncation=True) # Added truncation=True
    print("✅ Safety models loaded.")
except Exception as e:
    print(f"❌ Error loading safety models: {e}. Safety checks might not function correctly.")
    toxicity_model = None
    bias_model = None

# --- Enhanced Safety check function ---
def is_suitable_for_students(text):
    """Comprehensive content check for appropriateness in educational settings"""
    text = text.strip()
    if not text:
        print("⚠️ Empty paragraph provided.")
        return False

    # Check for text length
    if len(text.split()) < 20:
        print("⚠️ Text too short for meaningful MCQ generation (less than 20 words).")
        return False

    # --- Toxicity Check ---
    if toxicity_model:
        try:
            # Truncate text to model's max input size if needed (often 512 tokens)
            toxicity_result = toxicity_model(text[:512])[0] # Use pre-truncated text
            tox_label, tox_score = toxicity_result['label'].lower(), toxicity_result['score']
            toxicity_threshold = 0.60
            if tox_label == "toxic" and tox_score > toxicity_threshold:
                print(f"⚠️ Toxicity Detected ({tox_score:.2f}) — ❌ Not Suitable for Students")
                return False
        except Exception as e:
            print(f"⚠️ Error during toxicity check: {e}")
            # Optionally decide whether to proceed without toxicity check or fail
            # return False # Or just continue
    else:
        print("⚠️ Toxicity model not loaded. Skipping toxicity check.")


    # --- Comprehensive Bias Detection ---
    bias_detected = False
    bias_detection_message = ""

    # 1. Keyword checks (simple but fast)
    gender_bias_keywords = ["women are", "men are", "boys are", "girls are", "females are", "males are", "better at", "worse at", "naturally better", "suited for", "belong in", "should be", "can't do"]
    racial_bias_keywords = ["race", "racial", "racist", "ethnicity", "ethnic", "black people", "white people", "asian people", "latinos", "minorities", "majority", "immigrants", "foreigners"]
    political_bias_keywords = ["liberal", "conservative", "democrat", "republican", "left-wing", "right-wing", "socialism", "capitalism", "corrupt", "freedom", "rights", "policy", "taxes"] # Simplified
    religious_bias_keywords = ["christian", "muslim", "jewish", "hindu", "buddhist", "atheist", "religion", "religious", "faith", "belief", "worship", "sacred", "holy"]
    socioeconomic_bias_keywords = ["poor", "rich", "wealthy", "poverty", "privileged", "underprivileged", "class", "elite", "welfare", "lazy", "hardworking", "deserve", "entitled"]
    problematic_phrases = ["more aggressive", "less educated", "less intelligent", "more violent", "inferior", "superior", "better", "smarter", "worse", "dumber", "tend to be more", "tend to be less", "are naturally", "by nature", "all people", "those people", "these people", "that group", "always", "never", "inherently", "genetically"]

    text_lower = text.lower()
    all_bias_keywords = (gender_bias_keywords + racial_bias_keywords + political_bias_keywords + religious_bias_keywords + socioeconomic_bias_keywords)
    contains_bias_keywords = any(keyword in text_lower for keyword in all_bias_keywords)
    contains_problematic_phrases = any(phrase in text_lower for phrase in problematic_phrases)

    if contains_problematic_phrases:
        bias_detected = True
        bias_detection_message = "⚠️ Problematic Generalizations Detected (Keywords) — ❌ Not Suitable for Students"

    # 2. Advanced bias detection using BART model (if keywords are found or for general check)
    if bias_model and (contains_bias_keywords or not bias_detected): # Run BART if keywords found or no problematic phrases yet
        try:
            bias_threshold = 0.55 # Lower threshold to be more sensitive
            general_bias_labels = ["neutral", "biased", "discriminatory", "prejudiced", "stereotyping"]
            gender_bias_labels = ["gender neutral", "gender biased", "sexist"]
            racial_bias_labels = ["racially neutral", "racially biased", "racist"]
            political_bias_labels = ["politically neutral", "politically biased", "partisan"]

            # Run general bias detection first
            # Truncate input text for the model
            bias_result = bias_model(text[:512], candidate_labels=general_bias_labels)
            bias_label = bias_result['labels'][0].lower()
            bias_score = bias_result['scores'][0]

            specific_bias_check_done = False

            # If general check suggests bias or is uncertain and keywords were present, run specific checks
            if bias_label != "neutral" or (bias_score < 0.7 and contains_bias_keywords):
                if any(keyword in text_lower for keyword in gender_bias_keywords):
                    specific_result = bias_model(text[:512], candidate_labels=gender_bias_labels)
                    if specific_result['labels'][0] != gender_bias_labels[0] and specific_result['scores'][0] > 0.6:
                        bias_label = "gender biased"
                        bias_score = specific_result['scores'][0]
                        specific_bias_check_done = True
                        bias_detected = True

                if not specific_bias_check_done and any(keyword in text_lower for keyword in racial_bias_keywords):
                    specific_result = bias_model(text[:512], candidate_labels=racial_bias_labels)
                    if specific_result['labels'][0] != racial_bias_labels[0] and specific_result['scores'][0] > 0.6:
                        bias_label = "racially biased"
                        bias_score = specific_result['scores'][0]
                        specific_bias_check_done = True
                        bias_detected = True

                if not specific_bias_check_done and any(keyword in text_lower for keyword in political_bias_keywords):
                    specific_result = bias_model(text[:512], candidate_labels=political_bias_labels)
                    if specific_result['labels'][0] != political_bias_labels[0] and specific_result['scores'][0] > 0.6:
                        bias_label = "politically biased"
                        bias_score = specific_result['scores'][0]
                        specific_bias_check_done = True
                        bias_detected = True

            # Final decision based on BART model
            if bias_label in ["biased", "discriminatory", "prejudiced", "stereotyping",
                             "gender biased", "racially biased", "politically biased"] and bias_score > bias_threshold:
                bias_detected = True
                bias_detection_message = f"⚠️ {bias_label.title()} Content Detected (Model: {bias_score:.2f}) — ❌ Not Suitable for Students"

        except Exception as e:
            print(f"⚠️ Error during bias check: {e}")
            # Decide whether to proceed or fail
            # return False # Or just continue

    elif not bias_model:
         print("⚠️ Bias model not loaded. Skipping advanced bias check.")
         # Rely only on keyword checks if bias model failed
         if contains_bias_keywords and not bias_detected: # If problematic phrases didn't trigger already
             bias_detected = True
             bias_detection_message = "⚠️ Potential Bias Keywords Detected (Simple Check) — Review Recommended"
             # Maybe don't return False here, just warn if model isn't available? Or return False for safety.
             # For now, let's be cautious:
             print(bias_detection_message)
             return False


    if bias_detected:
        print(bias_detection_message)
        return False
    else:
        print(f"✅ Passed Safety Check — 🟢 Proceeding to Generate MCQs")
        return True


# --- Improved MCQ Generator Class ---
class ImprovedMCQGenerator:
    def __init__(self):
        # Initialize QG-specific model
        self.qg_model_name = "lmqg/t5-base-squad-qg"
        self.qg_tokenizer = None
        self.qg_model = None
        self.has_qg_model = False
        print(f"Attempting to load QG model: {self.qg_model_name}...")
        try:
            self.qg_tokenizer = AutoTokenizer.from_pretrained(self.qg_model_name)
            self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(self.qg_model_name)
            self.has_qg_model = True
            print(f"✅ Loaded QG model: {self.qg_model_name}")
        except Exception as e:
            print(f"⚠️ Could not load specialized QG model ({self.qg_model_name}): {e}. Falling back to T5 for question generation.")
            self.has_qg_model = False

        # Initialize T5 model for distractors and fallback QG
        self.t5_model_name = "google/flan-t5-base"
        self.t5_tokenizer = None
        self.t5_model = None
        print(f"Attempting to load T5 model: {self.t5_model_name}...")
        try:
            self.t5_tokenizer = T5Tokenizer.from_pretrained(self.t5_model_name)
            self.t5_model = T5ForConditionalGeneration.from_pretrained(self.t5_model_name)
            print(f"✅ Loaded T5 model: {self.t5_model_name}")
        except Exception as e:
            print(f"❌ Critical Error: Could not load T5 model ({self.t5_model_name}): {e}. MCQ generation will likely fail.")
            # You might want to raise the exception here or handle it downstream
            # raise e # Or set a flag indicating failure

        # Configuration
        self.max_length_qg = 128 # Max length for generated questions
        self.max_length_dist = 64 # Max length for generated distractors
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = text.strip()
        # Remove leading/trailing punctuation that might be artifacts
        text = text.strip(string.punctuation)
        return text

    def generate_question(self, context, answer):
        """Generate a question given a context and answer"""
        if not self.t5_model: # Check if essential model is loaded
             print("❌ Cannot generate question: T5 model not loaded.")
             return f"Which of the following relates to {answer}?" # Basic fallback

        # Find the sentence containing the answer for focused context
        sentences = sent_tokenize(context)
        relevant_sentences = []
        answer_lower = answer.lower()

        # Find sentences containing the answer (case-insensitive)
        answer_indices = [i for i, s in enumerate(sentences) if answer_lower in s.lower()]

        if answer_indices:
            first_occurrence_idx = answer_indices[0]
            # Include the sentence itself
            relevant_sentences.append(sentences[first_occurrence_idx])
            # Optionally add preceding sentence if exists
            if first_occurrence_idx > 0:
                relevant_sentences.insert(0, sentences[first_occurrence_idx - 1])
            # Optionally add succeeding sentence if exists
            if first_occurrence_idx < len(sentences) - 1:
                relevant_sentences.append(sentences[first_occurrence_idx + 1])
        else:
            # If answer not found in any sentence, use the whole context (or a sample)
            # This might happen if 'answer' is a concept, not exact text
            # Using the whole context might be too long, consider sampling or first few sentences
            # For now, let's stick to the original context if answer isn't found in a specific sentence
             focused_context = context
             # Or maybe use the first few sentences: focused_context = ' '.join(sentences[:3])

        if relevant_sentences:
            focused_context = ' '.join(relevant_sentences)
        # else: focused_context remains the original context if answer wasn't located

        generated_question = None

        # --- Try Specialized QG Model First ---
        if self.has_qg_model and self.qg_model and self.qg_tokenizer:
            try:
                input_text = f"answer: {answer} context: {focused_context} </s>" # Format often expected by QG models
                inputs = self.qg_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

                outputs = self.qg_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_length_qg,
                    num_beams=5,
                    early_stopping=True,
                    num_return_sequences=1 # Often 1 is enough if beams are used
                    # Removed sampling parameters for more deterministic QG
                )
                question_text = self.qg_tokenizer.decode(outputs[0], skip_special_tokens=True)
                question_text = self.clean_text(question_text)

                # Basic validation
                if question_text and '?' in question_text and answer.lower() not in question_text.lower():
                     generated_question = question_text
                else:
                     print("⚠️ QG model produced invalid question, falling back to T5.")

            except Exception as e:
                print(f"⚠️ Error generating question with QG model: {e}. Falling back to T5.")

        # --- Fallback to T5 Model ---
        if generated_question is None and self.t5_model and self.t5_tokenizer:
            try:
                # Using a more direct prompt for Flan-T5
                input_text = f"Generate a question for which the answer is '{answer}'. Context: {focused_context}"
                inputs = self.t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

                outputs = self.t5_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_length_qg,
                    num_beams=5,
                    top_k=100, # Slightly reduced top_k
                    top_p=0.95,
                    temperature=1.0, # Temp=1 often works well with Flan-T5 for generation
                    do_sample=True, # Sampling can yield more diverse questions
                    num_return_sequences=3, # Generate a few options
                    no_repeat_ngram_size=2
                )

                questions = [self.t5_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

                # Clean and validate questions
                valid_questions = []
                for q in questions:
                    q = self.clean_text(q)
                    if not q: continue
                    # Ensure it ends with a question mark
                    if not q.endswith('?'):
                        # If it's declarative, try to frame it as a question
                        if '.' in q: q = q.split('.')[0] # Take first sentence if multiple
                        q = q.strip() + '?'
                    # Avoid questions that contain the answer directly (case-insensitive)
                    if answer.lower() not in q.lower() and len(q.split()) > 4: # Ensure minimum length
                        valid_questions.append(q)

                if valid_questions:
                    generated_question = valid_questions[0] # Pick the first valid one
                else:
                     print("⚠️ T5 model also failed to produce a valid question.")

            except Exception as e:
                 print(f"⚠️ Error generating question with T5 model: {e}")

        # --- Final Fallback ---
        if generated_question is None:
            generated_question = f"What is the role or definition of '{answer}' according to the text?"

        return generated_question


    def extract_key_entities(self, text, n=10):
        """Extract key entities/terms from text that could serve as answers or distractors"""
        # Simple approach: Extract Noun Phrases and Named Entities
        # This is less sophisticated than the original but might be more robust
        # and less prone to TF-IDF issues with short/similar texts.
        sentences = sent_tokenize(text)
        key_phrases = set()

        for sentence in sentences:
            words = word_tokenize(sentence)
            tagged_words = pos_tag(words)

            # Extract potential noun phrases (NN, NNP, NNS, JJ NN, etc.)
            current_phrase = []
            for word, tag in tagged_words:
                # Keep proper nouns, nouns, and adjectives modifying nouns
                if tag.startswith('NN') or (tag.startswith('JJ') and current_phrase):
                    current_phrase.append(word)
                elif current_phrase:
                    # Phrase ended, check if valid
                    phrase = " ".join(current_phrase)
                    # Basic cleaning and filtering
                    phrase = phrase.strip().lower()
                    phrase = re.sub(r'^[^\w]+|[^\w]+$', '', phrase) # Remove leading/trailing punctuation
                    if len(phrase.split()) <= 5 and len(phrase) > 2 and phrase not in self.stop_words:
                         # Check if it contains at least one non-stopword
                        if any(w not in self.stop_words for w in phrase.split()):
                            key_phrases.add(phrase.strip())
                    current_phrase = [] # Reset

            # Add the last phrase if any
            if current_phrase:
                 phrase = " ".join(current_phrase)
                 phrase = phrase.strip().lower()
                 phrase = re.sub(r'^[^\w]+|[^\w]+$', '', phrase)
                 if len(phrase.split()) <= 5 and len(phrase) > 2 and phrase not in self.stop_words:
                      if any(w not in self.stop_words for w in phrase.split()):
                          key_phrases.add(phrase.strip())

        # Select up to N unique phrases, prioritize longer ones slightly
        sorted_phrases = sorted(list(key_phrases), key=lambda x: len(x.split()), reverse=True)

        # Capitalize appropriately (simple title case for multi-word, keep single words as is unless likely acronym)
        final_candidates = []
        for phrase in sorted_phrases:
            words = phrase.split()
            if len(words) > 1:
                final_candidates.append(string.capwords(phrase))
            elif phrase.isupper() and len(phrase) > 1: # Likely acronym
                 final_candidates.append(phrase)
            else: # Single word, keep original case if possible (lost here, so just use lower)
                 # A better approach would involve checking the original text case
                 final_candidates.append(phrase)


        return final_candidates[:n]


    def generate_distractors(self, answer, context, n=3):
        """Generate plausible distractors for a given answer"""
        if not self.t5_model: # Check if essential model is loaded
             print("❌ Cannot generate distractors: T5 model not loaded.")
             return [f"Option {i+1}" for i in range(n)] # Basic fallback

        distractors = []
        answer_lower = answer.lower()

        # --- Strategy 1: Extract from Context ---
        potential_distractors_from_context = self.extract_key_entities(context, n=15)
        for potential in potential_distractors_from_context:
            potential_lower = potential.lower()
            # Check if it's the answer, too similar, or already added
            if potential_lower != answer_lower and \
               answer_lower not in potential_lower and \
               potential_lower not in answer_lower and \
               potential_lower not in [d.lower() for d in distractors]:
                # Add a check for semantic similarity if needed later
                distractors.append(potential)
            if len(distractors) >= n: break # Stop if we have enough

        # --- Strategy 2: Generate with T5 if needed ---
        num_needed = n - len(distractors)
        if num_needed > 0 and self.t5_model and self.t5_tokenizer:
            try:
                # Prompt T5 for alternatives/related concepts
                input_text = f"Generate {num_needed*2} plausible incorrect options (distractors) for a multiple choice question where the correct answer is '{answer}'. Context: {context}"
                inputs = self.t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

                outputs = self.t5_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_length_dist,
                    num_beams=5,
                    top_k=75, # Adjusted K
                    top_p=0.95,
                    temperature=1.2, # Higher temp for more diverse distractors
                    do_sample=True,
                    num_return_sequences=num_needed * 2, # Generate more than needed
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )

                model_distractors = [self.t5_tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

                # Clean and validate model distractors
                for distractor in model_distractors:
                    distractor = self.clean_text(distractor)
                    distractor_lower = distractor.lower()

                    # Skip if empty, the answer, too similar, or already present
                    if not distractor or \
                       distractor_lower == answer_lower or \
                       answer_lower in distractor_lower or \
                       distractor_lower in answer_lower or \
                       distractor_lower in [d.lower() for d in distractors] or \
                       distractor_lower in [d.lower() for d in potential_distractors_from_context]: # Avoid re-adding context distractors
                        continue

                    distractors.append(distractor)
                    if len(distractors) >= n: break # Stop when enough are found

            except Exception as e:
                print(f"⚠️ Error generating distractors with T5 model: {e}")

        # --- Strategy 3: Fallback if still not enough ---
        num_needed = n - len(distractors)
        if num_needed > 0:
            # Use simple placeholders as a last resort
            fallback_options = ["Related Concept A", "Related Concept B", "An Alternative Idea", "None of the above"]
            random.shuffle(fallback_options)
            for i in range(num_needed):
                 # Ensure uniqueness against existing distractors
                 fallback = fallback_options[i % len(fallback_options)]
                 if fallback.lower() not in [d.lower() for d in distractors] and fallback.lower() != answer_lower:
                    distractors.append(fallback)
                 else:
                    # Add a variant if the standard fallback is already used
                    distractors.append(f"{fallback} (v{i+1})")


        # Return exactly n distractors
        return distractors[:n]


    def validate_mcq(self, mcq, context):
        """Validate if an MCQ meets basic quality standards"""
        question = mcq.get('question', '')
        options = mcq.get('options', [])
        answer = mcq.get('answer', '')

        if not question or not question.endswith('?'):
            print(f"Validation Fail: Invalid question format - '{question[:50]}...'")
            return False

        if len(question.split()) < 5:
            print(f"Validation Fail: Question too short - '{question}'")
            return False

        if not answer or not options or answer not in options:
             print(f"Validation Fail: Answer missing or not in options.")
             return False

        # Check if question contains the answer (allow partial overlap, but not exact match)
        if answer.lower() in question.lower() and len(answer) > 3 : # Avoid flagging short words like 'a', 'is'
            # More lenient check: allow if answer is part of a larger phrase in question
            if f" '{answer.lower()}'" not in question.lower() and \
               f' "{answer.lower()}"' not in question.lower() and \
               f" {answer.lower()}?" not in question.lower():
                 pass # Allow partial overlap
            else:
                print(f"Validation Fail: Question might contain the answer too directly - Q: '{question}' A: '{answer}'")
                # return False # Decide if this should be a hard fail

        # Check if options are unique (case-insensitive)
        if len(set(o.lower() for o in options)) != len(options):
            print(f"Validation Fail: Duplicate options found - {options}")
            return False

        # Check if options are reasonably distinct (very basic check)
        if len(options) > 1:
            if any(len(opt) < 2 for opt in options):
                 print(f"Validation Fail: Option too short - {options}")
                 return False


        # Optionally: Check if answer is actually plausible given the context
        # (This is harder, relying on the generation process for now)
        # if answer.lower() not in context.lower():
        #     print(f"Validation Warning: Answer '{answer}' not found literally in context.")
            # Decide if this is a failure or just a warning

        return True

    def generate_mcqs(self, paragraph, num_questions=5):
        """Generate multiple-choice questions from a paragraph"""
        if not self.t5_model: # Essential check
            print("❌ Cannot generate MCQs: T5 model not loaded.")
            return []

        paragraph = self.clean_text(paragraph)
        if not paragraph:
            print("❌ Cannot generate MCQs: Input paragraph is empty after cleaning.")
            return []

        mcqs = []

        # Extract potential answers (generate more candidates than needed)
        potential_answers = self.extract_key_entities(paragraph, n=num_questions * 3)
        if not potential_answers:
            print("⚠️ Could not extract any potential key entities/answers from the text.")
            return [] # Cannot proceed without potential answers

        print(f"Identified {len(potential_answers)} potential answer candidates.")

        # Shuffle potential answers to vary the selection
        random.shuffle(potential_answers)

        # Try to generate MCQs
        generated_count = 0
        attempts = 0
        max_attempts = len(potential_answers) # Try each candidate once

        while generated_count < num_questions and attempts < max_attempts:
            if not potential_answers: break # Stop if we run out of candidates
            answer = potential_answers.pop(0)
            attempts += 1

            print(f"\nAttempting MCQ for answer: '{answer}' ({attempts}/{max_attempts})")

            # Generate question
            question = self.generate_question(paragraph, answer)
            if not question:
                 print("   Skipping: Failed to generate question.")
                 continue

            # Generate distractors
            distractors = self.generate_distractors(answer, paragraph, n=3) # Aim for 3 distractors + 1 answer = 4 options
            if not distractors or len(distractors) < 3:
                 print("   Skipping: Failed to generate enough distractors.")
                 continue

            # Create initial MCQ structure
            mcq_options = [answer] + distractors
            temp_mcq = {
                'question': question,
                'options': mcq_options,
                'answer': answer
            }

            # Validate the generated MCQ
            if self.validate_mcq(temp_mcq, paragraph):
                print(f"   ✅ MCQ candidate passed validation.")
                # Shuffle options
                shuffled_options = temp_mcq['options'].copy()
                random.shuffle(shuffled_options)

                # Find the index of the correct answer in the shuffled list
                try:
                    correct_index = shuffled_options.index(answer)
                except ValueError:
                     print(f"   Critical Error: Correct answer '{answer}' not found in shuffled options: {shuffled_options}. Skipping.")
                     continue # Should not happen if validation passed, but safety check

                # Final MCQ structure
                final_mcq = {
                    'question': temp_mcq['question'],
                    'options': shuffled_options,
                    'answer': answer, # Keep the text of the answer
                    'answer_index': correct_index # Store the index
                }
                mcqs.append(final_mcq)
                generated_count += 1
            else:
                print(f"   ❌ MCQ candidate failed validation.")


        print(f"\nGenerated {len(mcqs)} MCQs out of {num_questions} requested.")
        return mcqs

# --- Helper functions for Display ---
def format_mcq(mcq, index):
    """Format a single MCQ for display"""
    if not mcq or 'question' not in mcq or 'options' not in mcq or 'answer_index' not in mcq:
        return f"Q{index+1}: Error formatting this MCQ.\n"

    question_line = f"Q{index+1}: {mcq['question']}"
    options_lines = []
    for i, option in enumerate(mcq['options']):
        options_lines.append(f"   {chr(65+i)}. {option}")
    try:
        answer_char = chr(65 + mcq['answer_index'])
        correct_option_text = mcq['options'][mcq['answer_index']]
        answer_line = f"Answer: {answer_char} ({correct_option_text})"
    except IndexError:
         answer_line = f"Answer: Error finding answer index ({mcq.get('answer_index', 'N/A')})"
    except Exception as e:
         answer_line = f"Answer: Error formatting answer ({e})"

    return "\n".join([question_line] + options_lines + [answer_line, ""]) # Add blank line for spacing

def generate_and_display_mcqs(paragraph, num_questions=5):
    """Generate and format MCQs from a paragraph"""
    try:
        generator = ImprovedMCQGenerator() # Initialize models here
        if not generator.t5_model: # Check if essential model loaded
             print("❌ Cannot proceed: Failed to load the core T5 model.")
             return "MCQ Generation Failed due to model loading error."

        print("\n--- Generating MCQs ---")
        mcqs = generator.generate_mcqs(paragraph, num_questions)

        if not mcqs:
            return "No valid MCQs were generated for this paragraph."

        print("\n--- Formatting MCQs ---")
        formatted_output = [f"Generated {len(mcqs)} MCQs from the paragraph:\n"]
        for i, mcq in enumerate(mcqs):
            formatted_output.append(format_mcq(mcq, i))

        return "\n".join(formatted_output)

    except Exception as e:
        print(f"\n❌ An unexpected error occurred during MCQ generation or formatting: {e}")
        import traceback
        traceback.print_exc()
        return f"MCQ Generation Failed: {e}"


# ================================================
# SECTION 2: PERFORMANCE METRICS
# ================================================

class MCQPerformanceMetrics:
    def __init__(self, mcq_generator_instance):
        """Initialize the performance metrics class"""
        if not isinstance(mcq_generator_instance, ImprovedMCQGenerator):
             raise ValueError("mcq_generator_instance must be an instance of ImprovedMCQGenerator")
        self.mcq_generator = mcq_generator_instance
        try:
             self.rouge = Rouge()
        except Exception as e:
             print(f"⚠️ Error initializing Rouge: {e}. ROUGE scores might not be available.")
             self.rouge = None

        # NLTK smoothing function for BLEU
        self.smoothing = SmoothingFunction().method1
        # TF-IDF Vectorizer (initialize once)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        # Fit once on dummy data to avoid errors on first use with very short texts
        try:
            self.tfidf_vectorizer.fit(["dummy text for initialization"])
        except Exception as e:
            print(f"Warning: Could not pre-fit TFIDF Vectorizer: {e}")


    def measure_execution_time(self, paragraphs, num_questions=5, repetitions=1): # Default reps=1 for faster eval
        """Measure execution time for generating MCQs"""
        if not isinstance(paragraphs, list):
            paragraphs = [paragraphs]

        execution_times = []
        total_questions_generated = 0
        total_time_spent = 0

        print(f"\nMeasuring execution time ({repetitions} repetitions)...")
        for i, paragraph in enumerate(paragraphs):
            paragraph_times = []
            paragraph_q_count = 0
            print(f"  Processing paragraph {i+1}/{len(paragraphs)}...")
            for rep in range(repetitions):
                start_time = time.time()
                # Ensure generator is initialized correctly (models loaded) before timing
                if not self.mcq_generator.t5_model:
                     print("    Skipping timing: T5 model not loaded.")
                     continue # Skip if model isn't ready

                mcqs = self.mcq_generator.generate_mcqs(paragraph, num_questions)
                end_time = time.time()

                execution_time = end_time - start_time
                paragraph_times.append(execution_time)
                total_time_spent += execution_time
                q_generated = len(mcqs)
                paragraph_q_count = q_generated # Use count from last repetition
                print(f"    Rep {rep+1}/{repetitions}: Time={execution_time:.2f}s, MCQs={q_generated}")


            if paragraph_times:
                 execution_times.append(np.mean(paragraph_times))
            total_questions_generated += paragraph_q_count # Add count from last rep


        avg_exec_time = np.mean(execution_times) if execution_times else 0
        min_exec_time = np.min(execution_times) if execution_times else 0
        max_exec_time = np.max(execution_times) if execution_times else 0
        avg_qps = total_questions_generated / total_time_spent if total_time_spent > 0 else 0

        return {
            'avg_execution_time': avg_exec_time,
            'min_execution_time': min_exec_time, # Min avg time per paragraph
            'max_execution_time': max_exec_time, # Max avg time per paragraph
            'avg_questions_per_second': avg_qps,
            'total_questions_generated_in_timing': total_questions_generated
        }

    def measure_memory_usage(self, paragraph, num_questions=5):
        """Measure peak memory usage during MCQ generation for a single paragraph"""
        print("\nMeasuring memory usage...")
        # Clear memory before test
        gc.collect()
        tracemalloc.start()

        # Generate MCQs
        try:
             # Ensure generator is initialized correctly
             if not self.mcq_generator.t5_model:
                  print("  Skipping memory measurement: T5 model not loaded.")
                  tracemalloc.stop()
                  return {'current_memory_MB': 0, 'peak_memory_MB': 0}

             print("  Generating MCQs for memory measurement...")
             self.mcq_generator.generate_mcqs(paragraph, num_questions)
             print("  MCQ generation complete.")
        except Exception as e:
            print(f"  Error during MCQ generation for memory measurement: {e}")
            # Continue to get memory usage up to the point of error

        # Get memory usage
        try:
            current, peak = tracemalloc.get_traced_memory()
        except Exception as e:
             print(f"  Error getting traced memory: {e}")
             current, peak = 0, 0

        # Stop tracking
        tracemalloc.stop()
        # Clear the collected traces
        tracemalloc.clear_traces()


        print(f"  Peak memory usage: {peak / (1024*1024):.2f} MB")
        return {
            'current_memory_MB': current / (1024 * 1024),
            'peak_memory_MB': peak / (1024 * 1024)
        }

    def compute_semantic_similarity(self, text1, text2):
        """Compute semantic similarity using TF-IDF and cosine similarity"""
        if not text1 or not text2:
             # print("Warning: Empty string provided for semantic similarity calculation.")
             return 0.0 # Handle empty strings

        try:
            # Use the pre-fitted vectorizer, just transform
            # Need to fit *and* transform here because vocabulary might change per pair
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])

            # Check if vocabulary is empty (can happen with only stopwords or very short strings)
            if tfidf_matrix.shape[1] == 0:
                 # print(f"Warning: Empty vocabulary for texts: '{text1[:50]}...', '{text2[:50]}...'. Returning 0 similarity.")
                 return 0.0

            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

            # Check for NaN results which can occur in edge cases
            if np.isnan(similarity[0][0]):
                 # print(f"Warning: NaN similarity score for texts: '{text1[:50]}...', '{text2[:50]}...'. Returning 0.")
                 return 0.0

            return similarity[0][0]

        except ValueError as ve:
             # This can happen if a text contains only stopwords after processing
             # print(f"ValueError computing semantic similarity (likely empty vocabulary): {ve}. Texts: '{text1[:50]}...', '{text2[:50]}...'. Returning 0.")
             return 0.0
        except Exception as e:
             print(f"Error computing semantic similarity: {e}. Texts: '{text1[:50]}...', '{text2[:50]}...'. Returning 0.")
             return 0.0


    def evaluate_question_quality(self, mcqs, reference_questions=None):
        """Evaluate the quality of generated questions"""
        print("\nEvaluating question quality...")
        if not mcqs:
            print("  No MCQs provided for quality evaluation.")
            return {'avg_question_length': 0, 'has_question_mark': 0, 'option_distinctiveness': 0}

        question_lengths = [len(mcq.get('question', '').split()) for mcq in mcqs if mcq.get('question')]
        has_question_mark = [int(mcq.get('question', '').strip().endswith('?')) for mcq in mcqs if mcq.get('question')]

        # Option distinctiveness - average semantic distance between options
        option_distinctiveness_scores = []
        for i, mcq in enumerate(mcqs):
            options = mcq.get('options', [])
            if len(options) < 2:
                continue

            pairwise_distinctiveness = []
            for j in range(len(options)):
                for k in range(j + 1, len(options)):
                    sim = self.compute_semantic_similarity(str(options[j]), str(options[k]))
                    # Ensure similarity is valid before calculating distinctiveness
                    if sim is not None and not np.isnan(sim):
                         pairwise_distinctiveness.append(1.0 - sim) # Distinctiveness = 1 - similarity
                    # else: print(f"  Skipping invalid similarity for option distinctiveness in MCQ {i}")


            if pairwise_distinctiveness:
                option_distinctiveness_scores.append(np.mean(pairwise_distinctiveness))
            # else: print(f"  Could not calculate distinctiveness for MCQ {i}")


        # NLP metrics (BLEU, ROUGE, Semantic Similarity) if references are provided
        bleu_scores = []
        modified_bleu_scores = []
        rouge_1_f = []
        rouge_2_f = []
        rouge_l_f = []
        semantic_similarities = []

        if reference_questions and isinstance(reference_questions, list) and len(reference_questions) > 0:
            print(f"  Comparing {len(mcqs)} generated questions against {len(reference_questions)} reference questions.")

            # Simple alignment: Compare each generated question to the reference at the same index (modulo length)
            # A more complex alignment (like finding the best match) is possible but adds complexity.
            num_refs = len(reference_questions)
            for i, mcq in enumerate(mcqs):
                 gen_q = mcq.get('question', '')
                 if not gen_q: continue

                 ref_idx = i % num_refs # Cycle through references
                 ref_q = reference_questions[ref_idx]
                 if not isinstance(ref_q, str) or not ref_q:
                      print(f"   Skipping comparison for MCQ {i}: Invalid reference question at index {ref_idx}.")
                      continue

                 # print(f"\n   Comparing Gen Q {i}: '{gen_q}'")
                 # print(f"   With Ref Q {ref_idx}: '{ref_q}'")


                 # Semantic Similarity
                 sem_sim = self.compute_semantic_similarity(gen_q, ref_q)
                 if sem_sim is not None: semantic_similarities.append(sem_sim)
                 # print(f"   Semantic Similarity: {sem_sim:.4f}")


                 # BLEU Score
                 try:
                     ref_tokens = [word_tokenize(ref_q.lower())] # Reference is a list of lists of tokens
                     hyp_tokens = word_tokenize(gen_q.lower())   # Hypothesis is a list of tokens
                     if not hyp_tokens: # Handle case where tokenization yields empty list
                         # print("   Skipping BLEU: Generated question tokenization resulted in empty list.")
                         continue

                     with warnings.catch_warnings(): # Suppress NLTK warnings about short sentences
                         warnings.simplefilter("ignore")
                         # Standard BLEU (can be 0 if no overlap)
                         # bleu_score = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method0)
                         # bleu_scores.append(bleu_score)

                         # BLEU with smoothing
                         modified_bleu = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=self.smoothing)
                         modified_bleu_scores.append(modified_bleu)
                         # print(f"   Smoothed BLEU: {modified_bleu:.4f}")

                 except Exception as e:
                     print(f"   Error calculating BLEU for MCQ {i}: {e}")


                 # ROUGE Score
                 if self.rouge:
                     try:
                         # Ensure non-empty strings for ROUGE
                         if gen_q.strip() and ref_q.strip():
                              scores = self.rouge.get_scores(gen_q, ref_q)[0]
                              rouge_1_f.append(scores['rouge-1']['f'])
                              rouge_2_f.append(scores['rouge-2']['f'])
                              rouge_l_f.append(scores['rouge-l']['f'])
                              # print(f"   ROUGE-1 F1: {scores['rouge-1']['f']:.4f}, ROUGE-L F1: {scores['rouge-l']['f']:.4f}")
                         # else: print("   Skipping ROUGE: Empty generated or reference question after stripping.")

                     except ValueError as ve:
                          # Handle potential "Hypothesis is empty" error from rouge-score
                          print(f"   ValueError calculating ROUGE for MCQ {i}: {ve}")
                     except Exception as e:
                          print(f"   Error calculating ROUGE for MCQ {i}: {e}")


        results = {
            'avg_question_length': np.mean(question_lengths) if question_lengths else 0,
            'has_question_mark_perc': np.mean(has_question_mark) * 100 if has_question_mark else 0,
            'option_distinctiveness': np.mean(option_distinctiveness_scores) if option_distinctiveness_scores else 0
        }

        # Add NLP metrics if calculated
        if modified_bleu_scores: results['avg_smoothed_bleu'] = np.mean(modified_bleu_scores)
        if rouge_1_f: results['avg_rouge1_f1'] = np.mean(rouge_1_f)
        if rouge_2_f: results['avg_rouge2_f1'] = np.mean(rouge_2_f)
        if rouge_l_f: results['avg_rougeL_f1'] = np.mean(rouge_l_f)
        if semantic_similarities: results['avg_semantic_similarity_to_ref'] = np.mean(semantic_similarities)

        print("  Question quality evaluation complete.")
        return results


    def analyze_distractor_quality(self, mcqs, context):
        """Analyze the quality of distractors"""
        print("\nAnalyzing distractor quality...")
        if not mcqs:
            print("  No MCQs provided for distractor analysis.")
            return {}

        all_distractor_similarities_to_answer = []
        all_distractor_semantic_relevance_to_context = []
        all_distractor_plausibility_scores = []
        # Removed context_presence as semantic relevance is more informative

        for i, mcq in enumerate(mcqs):
             options = mcq.get('options', [])
             answer_index = mcq.get('answer_index', -1)
             answer = mcq.get('answer', None) # Or options[answer_index] if index is valid

             if answer_index < 0 or answer_index >= len(options) or answer is None:
                  print(f"  Skipping distractor analysis for MCQ {i}: Invalid answer index or missing answer.")
                  continue

             correct_answer = answer # Use the stored answer text
             distractors = [opt for idx, opt in enumerate(options) if idx != answer_index]

             if not distractors:
                  # print(f"  No distractors found for MCQ {i}.")
                  continue

             # Analyze each distractor
             for distractor in distractors:
                 if not distractor or not correct_answer: continue # Skip if empty

                 # 1. Similarity to Correct Answer
                 sim_to_answer = self.compute_semantic_similarity(str(distractor), str(correct_answer))
                 if sim_to_answer is not None:
                      all_distractor_similarities_to_answer.append(sim_to_answer)
                      # 3. Plausibility (based on similarity to answer)
                      # Aim for moderate similarity (e.g., 0.3 - 0.7)
                      plausibility = max(0, 1.0 - abs(0.5 - sim_to_answer) / 0.5) # Normalize: 1 at 0.5, 0 at 0 or 1
                      all_distractor_plausibility_scores.append(plausibility)

                 # 2. Semantic Relevance to Context
                 relevance_to_context = self.compute_semantic_similarity(str(distractor), context)
                 if relevance_to_context is not None:
                      all_distractor_semantic_relevance_to_context.append(relevance_to_context)


        results = {
            'avg_distractor_similarity_to_answer': np.mean(all_distractor_similarities_to_answer) if all_distractor_similarities_to_answer else 0,
            'avg_distractor_relevance_to_context': np.mean(all_distractor_semantic_relevance_to_context) if all_distractor_semantic_relevance_to_context else 0,
            'avg_distractor_plausibility': np.mean(all_distractor_plausibility_scores) if all_distractor_plausibility_scores else 0
        }
        print("  Distractor quality analysis complete.")
        return results


    def calculate_readability_scores(self, mcqs):
        """Calculate readability scores for the generated questions"""
        print("\nCalculating readability scores...")
        if not HAS_TEXTSTAT:
            print("  Skipping readability scores: `textstat` library not available.")
            return {}
        if not mcqs:
            print("  No MCQs provided for readability analysis.")
            return {}

        readability_scores = {
            'flesch_reading_ease': [], 'flesch_kincaid_grade': [],
            'automated_readability_index': [], 'smog_index': [],
            'coleman_liau_index': []
        }
        failed_calculations = 0

        for i, mcq in enumerate(mcqs):
            question_text = mcq.get('question', '')
            if not question_text:
                # print(f"  Skipping readability for MCQ {i}: Empty question.")
                continue

            # Analyze question text only for simplicity, or combine with options if needed
            # full_mcq_text = question_text + "\n" + "\n".join(mcq.get('options',[]))
            text_to_analyze = question_text

            try:
                 # Need sufficient text for some scores (e.g., SMOG needs >= 30 sentences usually)
                 # textstat might handle short texts internally, but results could be less reliable.
                 if len(sent_tokenize(text_to_analyze)) < 3: # Basic check for very short text
                      # print(f"  Warning: Readability for MCQ {i} might be unreliable due to short text.")
                      pass # Proceed anyway, textstat might return default values


                 readability_scores['flesch_reading_ease'].append(textstat.flesch_reading_ease(text_to_analyze))
                 readability_scores['flesch_kincaid_grade'].append(textstat.flesch_kincaid_grade(text_to_analyze))
                 readability_scores['automated_readability_index'].append(textstat.automated_readability_index(text_to_analyze))
                 # SMOG index requires more text, might raise errors on short inputs
                 try:
                      readability_scores['smog_index'].append(textstat.smog_index(text_to_analyze))
                 except ValueError:
                      # print(f"  Could not calculate SMOG Index for MCQ {i} (likely too short).")
                      pass # Append nothing or a default value like None/NaN? Let's skip.

                 readability_scores['coleman_liau_index'].append(textstat.coleman_liau_index(text_to_analyze))

            except Exception as e:
                 failed_calculations += 1
                 print(f"  Error calculating readability for MCQ {i}: {e}")


        result = {}
        for metric, scores in readability_scores.items():
            valid_scores = [s for s in scores if s is not None and not np.isnan(s)]
            if valid_scores:
                result[f'avg_{metric}'] = np.mean(valid_scores)
            # else: result[f'avg_{metric}'] = 0 # Report 0 if no valid scores

        if failed_calculations > 0:
             print(f"  Readability calculation failed for {failed_calculations}/{len(mcqs)} MCQs.")

        print("  Readability analysis complete.")
        return result


    def evaluate_question_diversity(self, mcqs):
        """Evaluate the diversity of questions generated using semantic similarity"""
        print("\nEvaluating question diversity...")
        if not mcqs or len(mcqs) < 2:
            print("  Skipping diversity calculation: Need at least 2 MCQs.")
            return {'question_semantic_diversity': 0}

        similarities = []
        num_comparisons = 0
        for i in range(len(mcqs)):
            q1 = mcqs[i].get('question', '')
            if not q1: continue
            for j in range(i + 1, len(mcqs)):
                q2 = mcqs[j].get('question', '')
                if not q2: continue

                similarity = self.compute_semantic_similarity(q1, q2)
                if similarity is not None:
                     similarities.append(similarity)
                num_comparisons += 1

        if not similarities:
             print("  Could not compute any pairwise similarities for diversity.")
             return {'question_semantic_diversity': 0}

        avg_similarity = np.mean(similarities)
        # Diversity = 1 - average similarity
        diversity = 1.0 - avg_similarity

        print(f"  Average pairwise question similarity: {avg_similarity:.3f} ({num_comparisons} comparisons)")
        print(f"  Question diversity score: {diversity:.3f}")
        return {'question_semantic_diversity': diversity}


    def evaluate_contextual_relevance(self, mcqs, context):
        """Evaluate how relevant questions are to the overall context"""
        print("\nEvaluating contextual relevance...")
        if not mcqs:
            print("  No MCQs provided for relevance evaluation.")
            return {'avg_question_relevance_to_context': 0}
        if not context:
             print("  No context provided for relevance evaluation.")
             return {'avg_question_relevance_to_context': 0}

        relevance_scores = []
        for i, mcq in enumerate(mcqs):
            question = mcq.get('question', '')
            if not question: continue

            similarity = self.compute_semantic_similarity(question, context)
            if similarity is not None:
                 relevance_scores.append(similarity)
            # else: print(f"  Could not compute relevance for MCQ {i}.")


        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0
        print(f"  Average question relevance to context: {avg_relevance:.3f}")
        return {'avg_question_relevance_to_context': avg_relevance}


    def evaluate(self, paragraphs, num_questions=5, reference_questions=None):
        """Run a comprehensive evaluation of the MCQ generator"""
        print("\n" + "="*30 + " Starting Comprehensive Evaluation " + "="*30)
        all_metrics = {}

        if not isinstance(paragraphs, list):
             paragraphs = [paragraphs]
        if not paragraphs:
             print("❌ Evaluation failed: No paragraphs provided.")
             return {"error": "No paragraphs provided for evaluation."}

        # --- Timing Metrics ---
        try:
            timing_metrics = self.measure_execution_time(paragraphs, num_questions)
            all_metrics.update(timing_metrics)
        except Exception as e:
            print(f"❌ Error during timing measurement: {e}")
            all_metrics['timing_error'] = str(e)

        # --- Memory Metrics (on the first paragraph only) ---
        try:
            memory_metrics = self.measure_memory_usage(paragraphs[0], num_questions)
            all_metrics.update(memory_metrics)
        except Exception as e:
            print(f"❌ Error during memory measurement: {e}")
            all_metrics['memory_error'] = str(e)

        # --- Generate MCQs for Quality Analysis (use first paragraph) ---
        print("\nGenerating sample MCQs for quality analysis...")
        sample_mcqs = []
        try:
             if not self.mcq_generator.t5_model:
                  print("❌ Skipping quality analysis: T5 model not loaded.")
             else:
                  sample_mcqs = self.mcq_generator.generate_mcqs(paragraphs[0], num_questions)
                  print(f"  Generated {len(sample_mcqs)} sample MCQs.")
                  if not sample_mcqs:
                       print("  ⚠️ No sample MCQs generated, quality metrics will be limited.")

        except Exception as e:
            print(f"❌ Error generating sample MCQs for analysis: {e}")
            all_metrics['sample_generation_error'] = str(e)


        # --- Quality, Distractor, Readability, Diversity, Relevance (based on sample MCQs) ---
        if sample_mcqs:
            try:
                quality_metrics = self.evaluate_question_quality(sample_mcqs, reference_questions)
                all_metrics.update(quality_metrics)
            except Exception as e:
                print(f"❌ Error during question quality evaluation: {e}")
                all_metrics['quality_error'] = str(e)

            try:
                distractor_metrics = self.analyze_distractor_quality(sample_mcqs, paragraphs[0])
                all_metrics.update(distractor_metrics)
            except Exception as e:
                print(f"❌ Error during distractor quality analysis: {e}")
                all_metrics['distractor_error'] = str(e)

            try:
                readability_metrics = self.calculate_readability_scores(sample_mcqs)
                all_metrics.update(readability_metrics)
            except Exception as e:
                print(f"❌ Error during readability calculation: {e}")
                all_metrics['readability_error'] = str(e)

            try:
                diversity_metrics = self.evaluate_question_diversity(sample_mcqs)
                all_metrics.update(diversity_metrics)
            except Exception as e:
                print(f"❌ Error during diversity evaluation: {e}")
                all_metrics['diversity_error'] = str(e)

            try:
                relevance_metrics = self.evaluate_contextual_relevance(sample_mcqs, paragraphs[0])
                all_metrics.update(relevance_metrics)
            except Exception as e:
                print(f"❌ Error during relevance evaluation: {e}")
                all_metrics['relevance_error'] = str(e)
        else:
             print("\nSkipping detailed quality metrics as no sample MCQs were generated.")

        print("\n" + "="*30 + " Evaluation Complete " + "="*30)
        return all_metrics


    def visualize_results(self, metrics):
        """Visualize the evaluation results"""
        print("\n" + "="*30 + " Visualizing Evaluation Results " + "="*30)
        if not metrics or 'error' in metrics:
            print("❌ Cannot visualize results: Evaluation data is missing or contains errors.")
            if 'error' in metrics: print(f"   Error reported: {metrics['error']}")
            return

        # --- Display Metrics Table ---
        try:
            # Filter out non-numeric and error metrics for cleaner display
            display_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and 'error' not in k}
            if not display_metrics:
                 print("No numeric metrics available to display in table.")
                 return # Stop if no metrics to show

            metrics_df = pd.DataFrame({k: [v] for k, v in display_metrics.items()})

            # Simple formatting for printing
            formatted_metrics = {}
            for col in metrics_df.columns:
                value = metrics_df[col].iloc[0]
                if 'time' in col:
                    formatted_metrics[col] = f'{value:.2f} sec'
                elif 'memory' in col:
                    formatted_metrics[col] = f'{value:.2f} MB'
                elif '_perc' in col: # e.g., has_question_mark_perc
                    formatted_metrics[col] = f'{value:.1f}%'
                elif 'similarity' in col or 'bleu' in col or 'rouge' in col or \
                     'distinctiveness' in col or 'diversity' in col or 'relevance' in col or \
                     'plausibility' in col:
                     formatted_metrics[col] = f'{value:.3f}'
                elif 'grade' in col or 'index' in col or 'length' in col or 'questions_per' in col:
                     formatted_metrics[col] = f'{value:.2f}'
                else: # Default formatting
                     formatted_metrics[col] = f'{value:.3f}'


            # Create DataFrame from formatted strings for printing
            print_df = pd.DataFrame(formatted_metrics, index=['Value']).T
            print(print_df)

        except Exception as e:
            print(f"Error creating metrics table: {e}")


        # --- Create Plots ---
        try:
            # Use only numeric metrics available for plotting
            plottable_metrics = {k: v for k, v in display_metrics.items() if np.isfinite(v)} # Ensure finite numbers
            if not plottable_metrics:
                 print("\nNo plottable metrics available.")
                 return

            fig = plt.figure(figsize=(18, 16)) # Adjusted size
            # Create 3 rows, 2 columns grid
            gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3) # Add spacing

            # Function to create a bar chart safely
            def safe_bar_plot(ax, keys, title, colors):
                plot_data = {k: plottable_metrics.get(k, 0) for k in keys if k in plottable_metrics}
                if not plot_data:
                    ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12)
                    ax.set_title(title, fontsize=14, fontweight='bold')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    return
                labels = list(plot_data.keys())
                values = list(plot_data.values())
                bars = ax.bar(labels, values, color=colors[:len(labels)])
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                             f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

            # 1. Performance Metrics
            ax1 = fig.add_subplot(gs[0, 0])
            perf_keys = ['avg_execution_time', 'avg_questions_per_second']
            safe_bar_plot(ax1, perf_keys, 'Performance Metrics', ['#3498db', '#2ecc71'])

            # 2. Memory Usage
            ax2 = fig.add_subplot(gs[0, 1])
            mem_keys = ['current_memory_MB', 'peak_memory_MB']
            safe_bar_plot(ax2, mem_keys, 'Memory Usage (MB)', ['#9b59b6', '#34495e'])

            # 3. Question Quality & Relevance
            ax3 = fig.add_subplot(gs[1, 0])
            qual_keys = ['avg_question_length', 'has_question_mark_perc', 'option_distinctiveness',
                         'question_semantic_diversity', 'avg_question_relevance_to_context']
             # Normalize percentage for plotting scale if needed, or plot as is
            qual_plot_keys = qual_keys # Use original keys for lookup
            qual_labels = ['Avg Len', 'Mark %', 'Opt Distinct', 'Diversity', 'Relevance']
            qual_metrics = [plottable_metrics.get(k, 0) for k in qual_plot_keys]
            # Adjust scaling if necessary, e.g., divide percentage by 100 if mixing scales
            # qual_metrics[1] = qual_metrics[1] / 100.0 # Example if scaling needed
            if qual_plot_keys:
                 bars = ax3.bar(qual_labels[:len(qual_metrics)], qual_metrics, color=['#f39c12', '#d35400', '#c0392b', '#16a085', '#27ae60'])
                 ax3.set_title('Question Quality & Relevance', fontsize=14, fontweight='bold')
                 ax3.set_xticklabels(qual_labels[:len(qual_metrics)], rotation=45, ha='right')
                 ax3.grid(axis='y', linestyle='--', alpha=0.7)
                 for bar in bars:
                     height = bar.get_height()
                     ax3.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
            else:
                 ax3.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12)
                 ax3.set_title('Question Quality & Relevance', fontsize=14, fontweight='bold')


            # 4. Distractor Quality
            ax4 = fig.add_subplot(gs[1, 1])
            dist_keys = ['avg_distractor_similarity_to_answer', 'avg_distractor_relevance_to_context', 'avg_distractor_plausibility']
            dist_labels = ['Sim to Ans', 'Rel to Ctx', 'Plausibility']
            safe_bar_plot(ax4, dist_keys, 'Distractor Quality Metrics', ['#1abc9c', '#3498db', '#f1c40f'])


            # 5. NLP Metrics (BLEU/ROUGE/Similarity)
            ax5 = fig.add_subplot(gs[2, 0])
            nlp_keys = ['avg_smoothed_bleu', 'avg_semantic_similarity_to_ref',
                       'avg_rouge1_f1', 'avg_rouge2_f1', 'avg_rougeL_f1']
            nlp_labels = ['BLEU', 'Sem Sim Ref', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
            safe_bar_plot(ax5, nlp_keys, 'NLP Evaluation Metrics (vs References)', ['#3498db', '#2980b9', '#e74c3c', '#c0392b', '#d35400'])


            # 6. Readability Metrics
            ax6 = fig.add_subplot(gs[2, 1])
            read_keys = ['avg_flesch_reading_ease', 'avg_flesch_kincaid_grade',
                         'avg_automated_readability_index', 'avg_smog_index', 'avg_coleman_liau_index']
            read_labels = ['Flesch Ease', 'Kincaid Grade', 'ARI', 'SMOG', 'Coleman-Liau']
            safe_bar_plot(ax6, read_keys, 'Readability Metrics', ['#27ae60', '#2ecc71', '#16a085', '#1abc9c', '#f39c12'])


            plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
            fig.suptitle("MCQ Generator Performance Evaluation Summary", fontsize=18, fontweight='bold', y=0.99)
            plt.show()

        except Exception as e:
            print(f"\nError during results visualization: {e}")
            import traceback
            traceback.print_exc()


# ================================================
# SECTION 3: EXAMPLE USAGE & MAIN EXECUTION
# ================================================

# --- Example paragraphs ---
example_paragraphs = [
    """
    The cell is the basic structural and functional unit of all living organisms. Cells can be classified into two main types: prokaryotic and eukaryotic.
    Prokaryotic cells, found in bacteria and archaea, lack a defined nucleus and membrane-bound organelles. In contrast, eukaryotic cells, which make up plants,
    animals, fungi, and protists, contain a nucleus that houses the cell’s DNA, as well as various organelles like mitochondria and the endoplasmic reticulum.
    The cell membrane regulates the movement of substances in and out of the cell, while the cytoplasm supports the internal structures.
    """,

    """
   The Industrial Revolution was a major historical transformation that began in Great Britain in the late 18th century. It marked the shift from manual labor and
   hand-made goods to machine-based manufacturing and mass production. This shift significantly increased productivity and efficiency. The textile industry was the
   first to implement modern industrial methods, including the use of spinning machines and mechanized looms. A key innovation during this period was the development
   of steam power, notably improved by Scottish engineer James Watt. Steam engines enabled factories to operate away from rivers, which had previously been the main
   power source. Additional advancements included the invention of machine tools and the emergence of large-scale factory systems. These changes revolutionized industrial
   labor and contributed to the rise of new social classes, including the industrial working class and the capitalist class. The Industrial Revolution also led to rapid
   urbanization, a sharp rise in population, and eventually, improvements in living standards and economic growth.
    """
]

# Reference questions for the FIRST example paragraph (optional, for evaluation)
reference_questions_example1 = [
    "What is the basic structural and functional unit of all living organisms?",
    "What are the two main types of cells?",
    "Which type of cells lacks a defined nucleus and membrane-bound organelles?",
    "In which organisms are prokaryotic cells typically found?",
    "What structure regulates the movement of substances in and out of the cell?"
]


# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Part 1: Demonstrate MCQ Generation with Examples ---
    print("\n" + "="*30 + " MCQ Generator Demonstration " + "="*30)
    print("Testing with Example Paragraphs...")
    print("=" * 80)

    generated_mcqs_for_eval = [] # Store MCQs from the first example for evaluation

    for i, paragraph in enumerate(example_paragraphs):
        print(f"\n--- Example Paragraph {i + 1} ---")
        print(f"Input Text:\n{paragraph.strip()}")
        print("-" * 40)

        print("Running safety check...")
        if is_suitable_for_students(paragraph):
            # Generate and display MCQs for this example
            # Initialize generator inside the loop if you want fresh models per paragraph (memory intensive)
            # Or initialize once outside the loop if models are large
            # For this script, let's use the helper that initializes internally
            formatted_mcqs_output = generate_and_display_mcqs(paragraph, num_questions=5)
            print("\n--- Generated MCQs (Example) ---")
            print(formatted_mcqs_output)

            # Store the raw MCQs from the first paragraph for performance evaluation later
            # Need to call the generator directly to get the list, not the formatted string
            if i == 0:
                 try:
                    temp_generator = ImprovedMCQGenerator()
                    if temp_generator.t5_model:
                         generated_mcqs_for_eval = temp_generator.generate_mcqs(paragraph, num_questions=5)
                    del temp_generator # Clean up generator instance if created temporarily
                 except Exception as e:
                      print(f"Error generating MCQs for eval storage: {e}")

        else:
            print("\n--- Generated MCQs (Example) ---")
            print("❌ Content deemed unsuitable. No MCQs generated for this example.")

        print("=" * 80)


    # --- Part 2: Run Performance Evaluation ---
    # Uses the first example paragraph and its reference questions
    print("\n" + "="*30 + " Performance Evaluation " + "="*30)
    try:
        # Initialize the generator ONCE for evaluation to measure its typical performance
        print("Initializing MCQ Generator for Evaluation...")
        eval_mcq_generator = ImprovedMCQGenerator()

        if not eval_mcq_generator.t5_model:
             print("❌ Cannot run performance evaluation: Core T5 model failed to load.")
        else:
            print("Initializing Performance Metrics Evaluator...")
            metrics_evaluator = MCQPerformanceMetrics(eval_mcq_generator)

            # Run evaluation using the first example paragraph
            print(f"Running evaluation on Example Paragraph 1...")
            evaluation_results = metrics_evaluator.evaluate(
                paragraphs=example_paragraphs[0], # Evaluate only on the first paragraph
                num_questions=5,
                reference_questions=reference_questions_example1 # Use references for the first paragraph
            )

            # Visualize results
            metrics_evaluator.visualize_results(evaluation_results)

            # Print detailed results again for clarity
            print("\n--- Detailed Performance Metrics Summary ---")
            if evaluation_results and 'error' not in evaluation_results:
                 for metric, value in evaluation_results.items():
                     if 'error' not in metric: # Don't print error keys here
                         print(f"{metric}: {value}")
            elif 'error' in evaluation_results:
                 print(f"Evaluation failed with error: {evaluation_results['error']}")
            else:
                 print("Evaluation did not produce results.")

            # Clean up the evaluation generator instance if needed
            del eval_mcq_generator
            del metrics_evaluator
            gc.collect() # Suggest garbage collection

    except Exception as e:
        print(f"\n❌ An error occurred during the Performance Evaluation setup or execution: {e}")
        import traceback
        traceback.print_exc()


    # --- Part 3: Interactive Mode ---
    print("\n" + "="*30 + " Interactive MCQ Generation " + "="*30)
    print("Enter a paragraph to generate MCQs (or type 'exit' to quit):")
    # Initialize a generator for interactive use (if not already available)
    try:
         interactive_generator = ImprovedMCQGenerator()
         if not interactive_generator.t5_model:
              print("⚠️ Warning: Core T5 model not loaded. Interactive mode may fail.")
    except Exception as e:
         print(f"❌ Failed to initialize generator for interactive mode: {e}")
         interactive_generator = None # Prevent usage if failed

    while interactive_generator: # Only run loop if generator is ready
        try:
            user_input = input("> ")
            if user_input.lower() == 'exit':
                break
            if not user_input.strip():
                print("Please enter some text.")
                continue

            print("\nRunning safety check for your input...")
            if is_suitable_for_students(user_input):
                 # Use the interactive generator instance
                 print("\n--- Generating MCQs for your input ---")
                 mcqs = interactive_generator.generate_mcqs(user_input, num_questions=5)

                 if not mcqs:
                      print("No valid MCQs were generated for this paragraph.")
                 else:
                      print("\n--- Formatted MCQs ---")
                      formatted_output = [f"Generated {len(mcqs)} MCQs:\n"]
                      for i, mcq in enumerate(mcqs):
                          formatted_output.append(format_mcq(mcq, i))
                      print("\n".join(formatted_output))
            else:
                print("\n--- Generated MCQs ---")
                print("❌ Content deemed unsuitable. No MCQs generated.")

        except EOFError: # Handle Ctrl+D or unexpected end of input
             print("\nExiting interactive mode.")
             break
        except KeyboardInterrupt: # Handle Ctrl+C
             print("\nExiting interactive mode.")
             break
        except Exception as e:
             print(f"\nAn error occurred in interactive mode: {e}")
             # Optionally break or continue
             # break

    print("\nMCQ Generator script finished.")