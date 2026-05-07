"""
EE471 - Modern Software Development Practices and Technologies
Week 10 In-Class Exercise - Task 2
Transformers OOP Module with Gradio Web Interface

Author: Berkay Yukunc
Date: 07/05/2026
"""

import warnings
import json
import gradio as gr
from transformers import pipeline

warnings.filterwarnings("ignore")


class TransformerTasks:
    """
    Object-Oriented wrapper for 10 HuggingFace pipeline tasks.
    Implements lazy loading to save memory and reduce startup time.
    """

    def __init__(self):
        # Cache for loaded pipelines
        self.pipelines = {}

    def _get_pipeline(self, task: str, model: str = None, **kwargs):
        """Helper method to lazily load a pipeline."""
        pipeline_key = f"{task}_{model}"
        if pipeline_key not in self.pipelines:
            print(f"Loading model for {task}... This might take a moment if it's the first time.")
            if model:
                self.pipelines[pipeline_key] = pipeline(task, model=model, **kwargs)
            else:
                self.pipelines[pipeline_key] = pipeline(task, **kwargs)
        return self.pipelines[pipeline_key]

    # ==========================================
    # 1. Sentiment Analysis
    # ==========================================
    def analyze_sentiment(self, text: str) -> str:
        if not text.strip():
            return "Please enter some text."
        pipe = self._get_pipeline("sentiment-analysis")
        result = pipe(text)[0]
        return f"Label: {result['label']}\nScore: {result['score']:.4f}"

    # ==========================================
    # 2. Zero-Shot Classification
    # ==========================================
    def zero_shot_classification(self, text: str, labels_str: str) -> str:
        if not text.strip() or not labels_str.strip():
            return "Please enter both text and labels."
        labels = [l.strip() for l in labels_str.split(",") if l.strip()]
        pipe = self._get_pipeline("zero-shot-classification")
        result = pipe(text, candidate_labels=labels)
        
        output = "Classification Results:\n"
        for label, score in zip(result['labels'], result['scores']):
            output += f"  {label}: {score:.4f}\n"
        return output

    # ==========================================
    # 3. Text Generation
    # ==========================================
    def generate_text(self, text: str, max_words: int, num_alternatives: int) -> str:
        if not text.strip():
            return "Please enter a prompt."
        pipe = self._get_pipeline("text-generation", model="gpt2")
        results = pipe(
            text,
            max_new_tokens=int(max_words),
            num_return_sequences=int(num_alternatives),
            pad_token_id=50256,
            truncation=True
        )
        output = ""
        for i, res in enumerate(results, 1):
            output += f"--- Alternative {i} ---\n{res['generated_text']}\n\n"
        return output

    # ==========================================
    # 4. Fill-Mask
    # ==========================================
    def fill_mask(self, text: str) -> str:
        if not text.strip():
            return "Please enter text."
        if "<mask>" not in text:
            return "Error: Please include '<mask>' in your text so the model knows what to predict."
            
        pipe = self._get_pipeline("fill-mask", model="distilroberta-base")
        results = pipe(text)
        
        output = "Predictions:\n"
        for i, res in enumerate(results, 1):
            output += f"{i}. {res['token_str']} (Score: {res['score']:.4f}) -> {res['sequence']}\n"
        return output

    # ==========================================
    # 5. Named Entity Recognition
    # ==========================================
    def recognize_entities(self, text: str) -> str:
        if not text.strip():
            return "Please enter text."
        pipe = self._get_pipeline(
            "ner", 
            model="dbmdz/bert-large-cased-finetuned-conll03-english", 
            aggregation_strategy="simple"
        )
        results = pipe(text)
        
        if not results:
            return "No entities found."
            
        output = "Extracted Entities:\n"
        for res in results:
            output += f"[{res['entity_group']:4s}] {res['word']:30s} (score: {res['score']:.4f})\n"
        return output

    # ==========================================
    # 6. Question Answering
    # ==========================================
    def answer_question(self, context: str, question: str) -> str:
        if not context.strip() or not question.strip():
            return "Please enter both context and a question."
        pipe = self._get_pipeline("question-answering")
        result = pipe(question=question, context=context)
        return f"Answer: {result['answer']}\nScore: {result['score']:.4f}"

    # ==========================================
    # 7. Text Summarization
    # ==========================================
    def summarize_text(self, text: str) -> str:
        if not text.strip():
            return "Please enter text to summarize."
        pipe = self._get_pipeline("summarization")
        # Ensure input isn't too short for the default max_length
        input_length = len(text.split())
        max_len = min(130, max(20, int(input_length * 0.6)))
        
        result = pipe(text, max_length=max_len, min_length=10, do_sample=False)
        return result[0]['summary_text'].strip()

    # ==========================================
    # 8. Translation
    # ==========================================
    def translate_text(self, text: str) -> str:
        if not text.strip():
            return "Please enter text to translate."
        pipe = self._get_pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-tr")
        result = pipe(text)
        return result[0]['translation_text']

    # ==========================================
    # 9. Image Classification
    # ==========================================
    def classify_image(self, image) -> str:
        if image is None:
            return "Please upload an image."
        pipe = self._get_pipeline("image-classification", model="google/vit-base-patch16-224")
        results = pipe(image)
        
        output = "Top 5 Predictions:\n"
        for i, res in enumerate(results, 1):
            output += f"{i}. {res['label']} (score: {res['score']:.4f})\n"
        return output

    # ==========================================
    # 10. Automatic Speech Recognition
    # ==========================================
    def transcribe_audio(self, audio) -> str:
        if audio is None:
            return "Please record or upload an audio file."
        pipe = self._get_pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
        result = pipe(audio)
        return result['text']


# ============================================================================
# GRADIO INTERFACE
# ============================================================================
def create_gradio_app():
    tasks = TransformerTasks()
    
    with gr.Blocks(title="EE471 Week 10 - Transformers", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 EE471: HuggingFace Transformers Playground")
        gr.Markdown("Created by Berkay Yukunc. Choose a task from the tabs below to test the implemented OOP models.")
        
        with gr.Tabs():
            # Tab 1: Sentiment
            with gr.TabItem("1. Sentiment Analysis"):
                with gr.Row():
                    sent_input = gr.Textbox(lines=3, label="Input Text", value="I've been waiting for a machine learning course my whole life.")
                    sent_output = gr.Textbox(label="Result")
                sent_btn = gr.Button("Analyze", variant="primary")
                sent_btn.click(tasks.analyze_sentiment, inputs=sent_input, outputs=sent_output)
                
            # Tab 2: Zero Shot
            with gr.TabItem("2. Zero-Shot Classification"):
                with gr.Row():
                    with gr.Column():
                        zs_input = gr.Textbox(lines=2, label="Input Text", value="Berkshire keeps their cash reserves at an extremely high level.")
                        zs_labels = gr.Textbox(label="Candidate Labels (comma separated)", value="finance, politics, technology, sports, health")
                    zs_output = gr.Textbox(label="Result", lines=6)
                zs_btn = gr.Button("Classify", variant="primary")
                zs_btn.click(tasks.zero_shot_classification, inputs=[zs_input, zs_labels], outputs=zs_output)
                
            # Tab 3: Text Generation
            with gr.TabItem("3. Text Generation"):
                with gr.Row():
                    with gr.Column():
                        gen_input = gr.Textbox(lines=2, label="Prompt", value="If I continue to successfully complete all in-class exercises in EE471 course,")
                        gen_max = gr.Slider(minimum=10, maximum=100, value=35, step=1, label="Max New Words")
                        gen_alts = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Alternatives")
                    gen_output = gr.Textbox(label="Generated Output", lines=8)
                gen_btn = gr.Button("Generate", variant="primary")
                gen_btn.click(tasks.generate_text, inputs=[gen_input, gen_max, gen_alts], outputs=gen_output)
                
            # Tab 4: Fill-Mask
            with gr.TabItem("4. Fill-Mask"):
                with gr.Row():
                    fm_input = gr.Textbox(lines=2, label="Input Text (must contain <mask>)", value="To understand generative AI, one must study <mask> well.")
                    fm_output = gr.Textbox(label="Predictions", lines=6)
                fm_btn = gr.Button("Fill Mask", variant="primary")
                fm_btn.click(tasks.fill_mask, inputs=fm_input, outputs=fm_output)
                
            # Tab 5: NER
            with gr.TabItem("5. NER"):
                with gr.Row():
                    ner_input = gr.Textbox(lines=3, label="Input Text", value="I am Nate, a research assistant in Izmir Institute of Technology, and currently living and working in beautiful city İzmir in Türkiye.")
                    ner_output = gr.Textbox(label="Entities", lines=6)
                ner_btn = gr.Button("Extract Entities", variant="primary")
                ner_btn.click(tasks.recognize_entities, inputs=ner_input, outputs=ner_output)
                
            # Tab 6: Q&A
            with gr.TabItem("6. Question Answering"):
                with gr.Row():
                    with gr.Column():
                        qa_context = gr.Textbox(lines=3, label="Context", value="I am Nate, a research assistant in Izmir Institute of Technology, and currently living and working in beautiful city İzmir in Türkiye.")
                        qa_question = gr.Textbox(label="Question", value="Which organization does the person work at?")
                    qa_output = gr.Textbox(label="Answer")
                qa_btn = gr.Button("Get Answer", variant="primary")
                qa_btn.click(tasks.answer_question, inputs=[qa_context, qa_question], outputs=qa_output)
                
            # Tab 7: Summarization
            with gr.TabItem("7. Summarization"):
                with gr.Row():
                    sum_input = gr.Textbox(lines=6, label="Text to Summarize", value="The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century, often compared to the Great Depression of the 1930s. Triggered by the bursting of the United States housing bubble, its effects rippled across the globe, leading to the collapse of major financial institutions and a deep international recession. The crisis began with the subprime mortgage market. In the early 2000s, low interest rates and a push for homeownership led banks to issue high-risk loans to borrowers with poor credit.")
                    sum_output = gr.Textbox(label="Summary", lines=4)
                sum_btn = gr.Button("Summarize", variant="primary")
                sum_btn.click(tasks.summarize_text, inputs=sum_input, outputs=sum_output)
                
            # Tab 8: Translation
            with gr.TabItem("8. Translation (EN -> TR)"):
                with gr.Row():
                    tr_input = gr.Textbox(lines=3, label="English Text", value="The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century, often compared to the Great Depression.")
                    tr_output = gr.Textbox(label="Turkish Translation", lines=3)
                tr_btn = gr.Button("Translate", variant="primary")
                tr_btn.click(tasks.translate_text, inputs=tr_input, outputs=tr_output)
                
            # Tab 9: Image Classification
            with gr.TabItem("9. Image Classification"):
                with gr.Row():
                    img_input = gr.Image(type="filepath", label="Upload Image")
                    img_output = gr.Textbox(label="Predictions", lines=6)
                img_btn = gr.Button("Classify Image", variant="primary")
                img_btn.click(tasks.classify_image, inputs=img_input, outputs=img_output)
                
            # Tab 10: ASR
            with gr.TabItem("10. Speech Recognition"):
                with gr.Row():
                    asr_input = gr.Audio(type="filepath", label="Upload or Record Audio")
                    asr_output = gr.Textbox(label="Transcription", lines=4)
                asr_btn = gr.Button("Transcribe", variant="primary")
                asr_btn.click(tasks.transcribe_audio, inputs=asr_input, outputs=asr_output)

    return demo

if __name__ == "__main__":
    app = create_gradio_app()
    # Launching quietly on localhost
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
