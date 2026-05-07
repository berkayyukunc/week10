"""
EE471 - Modern Software Development Practices and Technologies
Week 10 In-Class Exercise - Task 1
Transformers and Sentiment Analysis

This script demonstrates the use of HuggingFace Transformers pipeline()
for various NLP, vision, and audio tasks.

Author: Berkay Yukunc
Date: 07/05/2026
"""

from transformers import pipeline
import warnings
import json

warnings.filterwarnings("ignore")


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_result(result) -> None:
    """Pretty-print pipeline results."""
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                print(json.dumps(item, indent=2, ensure_ascii=False))
            else:
                print(item)
    elif isinstance(result, dict):
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result)


# ============================================================================
# 1. SENTIMENT ANALYSIS
# ============================================================================
print_section("1. SENTIMENT ANALYSIS")

sentiment_classifier = pipeline("sentiment-analysis")

sentences = [
    "I've been waiting for a EE471 course my whole life.",
    "I hate EE471 course"
]

print("\nAnalyzing sentences:")
for sentence in sentences:
    result = sentiment_classifier(sentence)
    print(f"\n  Input:  \"{sentence}\"")
    print(f"  Label:  {result[0]['label']}")
    print(f"  Score:  {result[0]['score']:.4f}")


# ============================================================================
# 2. ZERO-SHOT CLASSIFICATION
# ============================================================================
print_section("2. ZERO-SHOT CLASSIFICATION")

zero_shot_classifier = pipeline("zero-shot-classification")

text = "Berkshire keeps their cash reserves at an extremely high level."
candidate_labels = ["finance", "politics", "technology", "sports", "health"]

result = zero_shot_classifier(text, candidate_labels=candidate_labels)

print(f"\n  Input:  \"{text}\"")
print(f"  Labels: {candidate_labels}")
print(f"\n  Results:")
for label, score in zip(result["labels"], result["scores"]):
    print(f"    {label:12s} -> {score:.4f}")


# ============================================================================
# 3. TEXT GENERATION
# ============================================================================
print_section("3. TEXT GENERATION")

text_generator = pipeline("text-generation", model="gpt2")

prompt = "If I continue to successfully complete all in-class exercises in EE471 course,"

results = text_generator(
    prompt,
    max_new_tokens=35,
    num_return_sequences=2,
    do_sample=True,
    temperature=0.7,
)

print(f"\n  Prompt: \"{prompt}\"")
print(f"  Max Length: 35 words | Alternatives: 2\n")
for i, res in enumerate(results, 1):
    print(f"  Alternative {i}:")
    print(f"    {res['generated_text']}\n")


# ============================================================================
# 4. FILL-MASK (MASK FILLING)
# ============================================================================
print_section("4. FILL-MASK")

unmasker = pipeline("fill-mask")

masked_sentence = "To understand generative AI, one must study <mask> well."

results = unmasker(masked_sentence, top_k=5)

print(f"\n  Input: \"{masked_sentence}\"")
print(f"\n  Top 5 predictions:")
for i, res in enumerate(results, 1):
    print(f"    {i}. \"{res['token_str'].strip()}\" (score: {res['score']:.4f})")
    print(f"       -> {res['sequence']}")


# ============================================================================
# 5. NAMED ENTITY RECOGNITION (NER)
# ============================================================================
print_section("5. NAMED ENTITY RECOGNITION (NER)")

ner = pipeline("ner", aggregation_strategy="simple")

ner_sentence = (
    "I am Nate, a research assistant in Izmir Institute of Technology, "
    "and currently living and working in beautiful city İzmir in Türkiye."
)

ner_results = ner(ner_sentence)

print(f"\n  Input: \"{ner_sentence}\"")
print(f"\n  Extracted Entities:")
for entity in ner_results:
    print(f"    [{entity['entity_group']:4s}] {entity['word']:30s} (score: {entity['score']:.4f})")

# Identify specific entities
person = [e for e in ner_results if e["entity_group"] == "PER"]
org = [e for e in ner_results if e["entity_group"] == "ORG"]
loc = [e for e in ner_results if e["entity_group"] == "LOC"]

print(f"\n  Summary:")
if person:
    print(f"    Person:       {', '.join(e['word'] for e in person)}")
if org:
    print(f"    Organization: {', '.join(e['word'] for e in org)}")
if loc:
    print(f"    Location:     {', '.join(e['word'] for e in loc)}")


# ============================================================================
# 6. QUESTION ANSWERING (Validate NER results)
# ============================================================================
print_section("6. QUESTION ANSWERING (Validating NER Results)")

qa = pipeline("question-answering")

context = (
    "I am Nate, a research assistant in Izmir Institute of Technology, "
    "and currently living and working in beautiful city İzmir in Türkiye."
)

questions = [
    "What is the name of the person?",
    "Which organization does the person work at?",
    "Where does the person live?",
    "In which country does the person live?",
]

print(f"\n  Context: \"{context}\"")
print(f"\n  Validation via Q&A:")
for q in questions:
    result = qa(question=q, context=context)
    print(f"\n    Q: {q}")
    print(f"    A: {result['answer']} (score: {result['score']:.4f})")


# ============================================================================
# 7. TEXT SUMMARIZATION
# ============================================================================
print_section("7. TEXT SUMMARIZATION")

summarizer = pipeline("summarization")

text_to_summarize = (
    "The 2008 Global Financial Crisis stands as the most severe economic collapse "
    "of the 21st century, often compared to the Great Depression of the 1930s. "
    "Triggered by the bursting of the United States housing bubble, its effects "
    "rippled across the globe, leading to the collapse of major financial "
    "institutions and a deep international recession. The crisis began with the "
    "subprime mortgage market. In the early 2000s, low interest rates and a push "
    "for homeownership led banks to issue high-risk loans to borrowers with poor credit."
)

result = summarizer(text_to_summarize, max_length=60, min_length=20)

print(f"\n  Original Text ({len(text_to_summarize.split())} words):")
print(f"    {text_to_summarize}")
print(f"\n  Summary ({len(result[0]['summary_text'].split())} words):")
print(f"    {result[0]['summary_text']}")


# ============================================================================
# 8. TRANSLATION
# ============================================================================
print_section("8. TRANSLATION (English -> Turkish)")

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-tr")

text_to_translate = (
    "The 2008 Global Financial Crisis stands as the most severe economic collapse "
    "of the 21st century, often compared to the Great Depression."
)

result = translator(text_to_translate)

print(f"\n  Original (EN): \"{text_to_translate}\"")
print(f"\n  Translation (TR): \"{result[0]['translation_text']}\"")


# ============================================================================
# 9. IMAGE CLASSIFICATION (Google ViT)
# ============================================================================
print_section("9. IMAGE CLASSIFICATION (google/vit-base-patch16-224)")

image_classifier = pipeline(
    task="image-classification",
    model="google/vit-base-patch16-224"
)

image_url = (
    "https://huggingface.co/datasets/huggingface/documentation-images/"
    "resolve/main/pipeline-cat-chonk.jpeg"
)

result = image_classifier(image_url)

print(f"\n  Image URL: {image_url}")
print(f"\n  Classification Results (Top 5):")
for i, pred in enumerate(result[:5], 1):
    print(f"    {i}. {pred['label']:50s} (score: {pred['score']:.4f})")


# ============================================================================
# 10. AUTOMATIC SPEECH RECOGNITION (OpenAI Whisper)
# ============================================================================
print_section("10. AUTOMATIC SPEECH RECOGNITION (openai/whisper-large-v3)")

transcriber = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-large-v3"
)

audio_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"

result = transcriber(audio_url)

print(f"\n  Audio URL: {audio_url}")
print(f"\n  Transcription:")
print(f"    \"{result['text']}\"")


# ============================================================================
# DONE
# ============================================================================
print_section("ALL TASKS COMPLETED SUCCESSFULLY")
print("\n  EE471 Week 10 - Task 1: All 10 subtasks executed.\n")
