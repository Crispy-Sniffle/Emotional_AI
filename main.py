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

print("Loading Emotion Model...")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# ==========================================
# 2. LOGIC
# ==========================================
def detect_emotion(text):
    results = emotion_classifier(text)
    return results[0][0]['label'], results[0][0]['score']

def generate_image(prompt):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    clean_prompt = f"{prompt}, masterpiece, best quality, character focus, no text, no words, no subtitles, no humans, no people"
    try:
        response = requests.post(IMAGE_API_URL, headers=headers, json={"inputs": clean_prompt})
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        return None
    except Exception as e:
        print(f"Connection Error: {e}")
        return None


def run_director_agent(user_text, emotion, avatar_name):
    # 1. NEW JSON SYSTEM PROMPT
    # We explicitly ask for a JSON object with specific keys.
    system_prompt = (
        f"You are a {avatar_name}. "
        f"The user is feeling: {emotion.upper()}. "
        f"Context: {user_text}\n\n"

        "INSTRUCTIONS:\n"
        "You must output a valid JSON object. Do not write any other text.\n"
        "The JSON must have exactly two keys:\n"
        "1. 'reply': A short, casual, supportive message (max 15 words).\n"
        "2. 'image_prompt': Generate an image of YOURSELF ({avatar_name}) that matches the mood and the emotion. Be kind.\n\n"

        "EXAMPLE JSON FORMAT:\n"
        '{ "reply": "I am here for you.", "image_prompt": "The character sitting quietly in a warm room" }'
    )

    try:
        completion = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-beta:featherless-ai",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate response."}
            ],
            max_tokens=150,
            temperature=0.8
        )
        full_response = completion.choices[0].message.content

        print(f"\n--- DEBUG: RAW AI RESPONSE ---\n{full_response}\n------------------------------")

        # 2. JSON PARSING LOGIC
        # Sometimes AI adds text like "Here is the JSON:", so we use regex to find the curly braces { ... }
        try:
            # Find the first '{' and the last '}'
            json_match = re.search(r"\{.*\}", full_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)  # <--- The Magic Line

                chat_reply = data.get("reply", "I am listening...")
                visual_prompt = data.get("image_prompt", f"A {avatar_name}")
            else:
                raise ValueError("No JSON found")

        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON Parsing Failed: {e}")
            # Fallback if AI fails to write JSON
            chat_reply = full_response
            visual_prompt = f"A comforting {avatar_name}"

        # 3. CLEANUP (Still good to have)
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
        return chat_history, None, None, ""

    if chat_history is None:
        chat_history = []

    emotion, conf = detect_emotion(user_input)
    visual_prompt, bot_reply = run_director_agent(user_input, emotion, avatar_selection)
    generated_img = generate_image(visual_prompt)

    # Modern Dictionary Format
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": bot_reply})

    emojis = {"joy": "😊", "sadness": "😢", "anger": "😡", "fear": "😱", "surprise": "😲", "neutral": "😐", "disgust": "🤢"}
    emoji = emojis.get(emotion, "😐")
    status_text = f"{emoji} Detected: {emotion.upper()} ({int(conf * 100)}%)"

    return chat_history, generated_img, status_text, ""


theme = gr.themes.Soft(primary_hue="purple", secondary_hue="indigo")

with gr.Blocks(title="Emotion AI") as demo:
    gr.Markdown("# 🤖 Emotion AI Companion")

    with gr.Row(variant="panel"):
        avatar_dropdown = gr.Dropdown(choices=list(AVATAR_NAMES.keys()), value="Panda", label="1. Choose Companion",
                                      interactive=True)
        emotion_status = gr.Label(value="Waiting...", label="2. Emotional State", num_top_classes=0)

    with gr.Row():
        # I removed the avatar_images parameter to fix the "Broken Icon" issue for now.
        chatbot = gr.Chatbot(label="Conversation", height=450)
        image_output = gr.Image(label="Visual Response", height=450)

    with gr.Row():
        msg_input = gr.Textbox(label="Your Message", placeholder="Type here...", scale=4)
        send_btn = gr.Button("✨ Send", scale=1, variant="primary")

    send_btn.click(fn=process_interaction, inputs=[msg_input, avatar_dropdown, chatbot],
                   outputs=[chatbot, image_output, emotion_status, msg_input])
    msg_input.submit(fn=process_interaction, inputs=[msg_input, avatar_dropdown, chatbot],
                     outputs=[chatbot, image_output, emotion_status, msg_input])

if __name__ == "__main__":
    demo.launch(theme=theme)