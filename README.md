# 🧠 Multimodal Emotion AI Companion

> An open-source affective computing prototype that provides real-time emotional support through empathetic dialogue and dynamic visual avatars.

## 📌 Overview & Features
Designed to address emotional isolation in vulnerable demographics (such as international students), this Gradio-based application utilizes a hybrid generative pipeline to offer secure, non-sycophantic psychological first aid. 

* **Granular Emotion Detection:** Classifies raw user text across 28 distinct emotional states using a GoEmotions RoBERTa model to capture nuanced affect.
* **Ethical Guardrails:** The `Zephyr-7b` LLM Director Agent is strictly prompted to avoid "toxic positivity," instead mirroring user tone to deliver grounded empathy.
* **Fault-Tolerant Parsing:** Implements a robust Regex fallback protocol to gracefully handle LLM instruction decay and malformed JSONs, ensuring 100% system uptime.
* **Dynamic Avatars:** Synthesizes real-time companion visuals that reflect the user's emotional state via the `FLUX.1-schnell` text-to-image diffusion model.

## 🏗️ System Architecture
![System Architecture Diagram](<img width="682" height="681" alt="DesignMethodolgy drawio" src="https://github.com/user-attachments/assets/6134a6c1-3290-4d31-a406-277a39302291" />
) *(Note: Update this path once you upload your flowchart!)*

The application utilizes a branching, multi-agent pipeline designed for maximum resilience:
1. **Perception Layer:** The `SamLowe/roberta-base-go_emotions` NLP pipeline analyzes user input to provide an accurate affective label.
2. **Cognitive Layer:** The `Zephyr-7b-beta` LLM ingests the input and emotion label, functioning as a "Director" to generate a JSON containing parallel conversational and visual prompts.
3. **Execution Layer:** A custom parsing engine extracts the data (utilizing Regex failsafes if necessary), routing the text to the chat UI and the visual prompt to the `FLUX.1` API simultaneously.

## 🚀 Installation & Usage
This project requires Python 3.10+ and a free Hugging Face inference token.

1. Clone the repository and install the required dependencies:
   `pip install -r requirements.txt`
2. Create a `.env` file in the root directory to manage your credentials securely:
   `HUGGINGFACE_API_KEY="your_token_here"`
3. Launch the Gradio interface:
   `python app.py`

## 🔮 Limitations & Future Work
As an exploratory prototype, this system currently experiences an average 15-second latency due to shared cloud inference APIs, alongside morphological variance inherent to zero-shot diffusion models. Future iterations will prioritize:
* Transitioning to **Proprietary Model Fine-Tuning** to eliminate network latency, secure user privacy, and resolve instruction decay.
* Implementing a **Standardized Single-Agent Architecture** to enforce visual continuity (via ControlNet) and minimize the ethical surface area.
