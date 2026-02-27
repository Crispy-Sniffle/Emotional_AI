import gradio as gr
import os
import requests
import io
import json
import re
from PIL import Image
from transformers import pipeline, logging as hf_logging
from openai import OpenAI

# ==========================================
# 1. SETUP
# ==========================================
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
hf_logging.set_verbosity_error()
HF_TOKEN = os.environ.get("HF_TOKEN")
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)
IMAGE_API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"

# --- AVATAR LIST ---
AVATAR_NAMES = {
    "Monkey": "Monkey",
    "Ferret": "Ferret",
    "Dog": "Dog",
    "Rabbit": "Rabbit",
    "Panda": "Panda",
    "Parrot": "Parrot",
    "Cat": "Cat",
    "Snake": "Snake",
    "Horse": "Horse",
    "Pig": "Pig",
    "Rat": "Rat"
}

print("Loading The Emotion Model...")
emotion_classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=1)

# ==========================================
# 2. LOGIC
# ==========================================
def detect_emotion(text):
    results = emotion_classifier(text)
    # The pipeline returns a nested list: [[{'label': '...', 'score': ...}]]
    return results[0][0]['label'], results[0][0]['score']

def generate_image(prompt):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    clean_prompt = (f"{prompt}, masterpiece, best quality,"
                    f" character focus, no text, no words, "
                    f"no subtitles, no inappropriate imagery, "
                    f"and strictly NO TEXT on the image")
    try:
        response = requests.post(IMAGE_API_URL, headers=headers, json={"inputs": clean_prompt})
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        return None
    except Exception as e:
        print(f"Connection Error: {e}")
        return None

def run_director_agent(user_text, emotion, avatar_name):
    # SYSTEM PROMPT
    system_prompt = (
        f"You are a {avatar_name}. "
        f"The user is feeling: {emotion.upper()}. "
        f"Context: {user_text}\n\n"

        "INSTRUCTIONS:\n"
        "You must output a valid JSON object. Do not write any other text.\n"
        "The JSON must have exactly two keys:\n"
        "1. 'reply': A short message (max 15 words). IMPORTANT: Mirror the user's emotional tone. "
        "If they are sad, be gentle and solemn. If they are happy, be excited. Do not use toxic positivity.\n"
        "2. 'image_prompt': Generate an image of YOURSELF ({avatar_name}). The lighting, weather, "
        "and your expression MUST match the user's mood exactly (e.g., dark and rainy for sadness, bright for joy).\n\n"

        "EXAMPLE JSON FORMAT:\n"
        '{ "reply": "That sounds incredibly heavy. I am sitting right here with you.", "image_prompt": '
        '"A solemn panda sitting alone in the shadows, looking downward empathetically" }'
    )

    try:
        completion = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-beta:featherless-ai",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate JSON."}  # <-- Remind it one last time
            ],
            max_tokens=150,
            temperature=0.7  # <-- Lowered slightly for more structural consistency
        )
        full_response = completion.choices[0].message.content
        print(f"\n--- DEBUG: RAW AI RESPONSE ---\n{full_response}\n------------------------------")

        chat_reply = ""
        visual_prompt = ""
        parsed_successfully = False

        # PLAN A: Parse as JSON
        json_match = re.search(r"\{.*\}", full_response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                chat_reply = data.get("reply", "")
                visual_prompt = data.get("image_prompt", "")
                if chat_reply and visual_prompt:
                    parsed_successfully = True
            except json.JSONDecodeError:
                print("DEBUG: JSON parsing failed. Falling to Plan B.")

        # PLAN B: Manual String Splitting (The Janitor)
        if not parsed_successfully:
            print("DEBUG: Using manual string splitting.")

            # Case-insensitive search for the leakage marker
            if re.search(r"image prompt:?", full_response, re.IGNORECASE):
                # Split the text into two pieces
                parts = re.split(r"image prompt:?", full_response, flags=re.IGNORECASE)
                chat_reply = parts[0].strip()
                visual_prompt = parts[1].strip()
            else:
                # Absolute worst-case scenario fallback
                chat_reply = full_response
                visual_prompt = f"A {avatar_name} experiencing {emotion}"

        # --- FINAL CLEANUP ---
        # Strip out loose JSON artifacts just in case Plan B grabbed them
        chat_reply = chat_reply.replace('"', '').replace('{', '').replace('}', '').strip()
        # Ensure 'reply:' isn't leaking
        chat_reply = re.sub(r'^reply:\s*', '', chat_reply, flags=re.IGNORECASE).strip()

        # Boost Image Prompt to keep the avatar visible
        if avatar_name.lower() not in visual_prompt.lower():
            visual_prompt = f"Close-up of a {avatar_name}, {visual_prompt}"
        else:
            visual_prompt = f"Close-up of {visual_prompt}"

        return visual_prompt, chat_reply

    except Exception as e:
        print(f"Director Error: {e}")
        return None, "System error."
# ==========================================
# 3. GRADIO INTERFACE
# ==========================================
def process_interaction(user_input, avatar_selection, chat_history):
    if not user_input:
        return chat_history, None, "", ""

    if chat_history is None:
        chat_history = []

    # 1. Pipeline
    emotion, conf = detect_emotion(user_input)
    visual_prompt, bot_reply = run_director_agent(user_input, emotion, avatar_selection)
    generated_img = generate_image(visual_prompt)

    # 2. Update Chat (Universal Tuple Format for older Gradio versions)
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": bot_reply})

    # 3. Map the 28 GoEmotions to Emojis
    emojis = {
        "admiration": "🤩", "amusement": "😄", "anger": "😡", "annoyance": "😒",
        "approval": "👍", "caring": "🤗", "confusion": "😕", "curiosity": "🤔",
        "desire": "❤️‍🔥", "disappointment": "😞", "disapproval": "👎", "disgust": "🤢",
        "embarrassment": "😳", "excitement": "🎉", "fear": "😱", "gratitude": "🙏",
        "grief": "💔", "joy": "😊", "love": "❤️", "nervousness": "😬",
        "optimism": "🌟", "pride": "🦁", "realization": "💡", "relief": "😌",
        "remorse": "🥺", "sadness": "😢", "surprise": "😲", "neutral": "😐"
    }
    emoji = emojis.get(emotion, "😐")
    status_text = f"{emoji} Detected: {emotion.upper()} ({int(conf * 100)}%)"

    return chat_history, generated_img, status_text, ""


# --- UI SETUP ---
theme = gr.themes.Soft(primary_hue="purple", secondary_hue="indigo")

with gr.Blocks(title="Emotion AI") as demo:
    gr.Markdown("# 🤖 Emotion AI Companion\n### The 28-Emotion Engine")

    with gr.Row(variant="panel"):
        avatar_dropdown = gr.Dropdown(choices=list(AVATAR_NAMES.keys()), value="Dog", label="1. Choose Companion",
                                      interactive=True)
        emotion_status = gr.Label(value="Waiting...", label="2. Emotional State", num_top_classes=0)

    with gr.Row():
        chatbot = gr.Chatbot(label="Conversation", height=450)
        image_output = gr.Image(label="Visual Response", height=450)

    with gr.Row():
        msg_input = gr.Textbox(label="Type your message...",
                               placeholder="e.g. I am feeling really stressed about my exams...", scale=4)
        send_btn = gr.Button("✨ Send", variant="primary", scale=1)

    # WIRING
    inputs = [msg_input, avatar_dropdown, chatbot]
    outputs = [chatbot, image_output, emotion_status, msg_input]

    send_btn.click(fn=process_interaction, inputs=inputs, outputs=outputs)
    msg_input.submit(fn=process_interaction, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch(theme=theme)
