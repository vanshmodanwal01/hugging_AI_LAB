import os
import json

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found.\n"
        "Please add your Hugging Face token to the .env file."
    )


# ============================================================
# HUGGING FACE MODEL
# ============================================================

MODEL = "openai/gpt-oss-120b"


# ============================================================
# CREATE HUGGING FACE INFERENCE CLIENT
# ============================================================

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)


# ============================================================
# GENERATE FLASHCARDS
# ============================================================

def generate_flashcards(topic, number_of_cards):

    prompt = f"""
Create exactly {number_of_cards} educational flashcards
about the following topic:

{topic}

Return ONLY valid JSON.

The JSON must follow exactly this format:

[
    {{
        "question": "Question here",
        "answer": "Answer here"
    }}
]

Rules:
- Create exactly {number_of_cards} flashcards.
- Questions must be clear and useful for learning.
- Answers should be concise but informative.
- Do not include Markdown.
- Do not include any text outside the JSON.
"""

    try:

        # ====================================================
        # CALL HUGGING FACE MODEL
        # ====================================================

        response = client.chat_completion(

            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert educational "
                        "flashcard generator."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.7,
            max_tokens=2000
        )

        # ====================================================
        # GET MODEL RESPONSE
        # ====================================================

        content = response.choices[0].message.content

        if not content:
            print("❌ Model returned an empty response.")
            return None

        # Remove accidental Markdown code fences
        content = content.strip()

        if content.startswith("```json"):
            content = content[7:]

        elif content.startswith("```"):
            content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        # ====================================================
        # CONVERT JSON STRING → PYTHON OBJECT
        # ====================================================

        flashcards = json.loads(content)

        return flashcards

    except Exception as error:

        print("\n❌ Hugging Face Error:")
        print(error)

        return None


# ============================================================
# DISPLAY FLASHCARDS
# ============================================================

def display_flashcards(flashcards):

    print("\n")
    print("=" * 65)
    print("                 GENERATED FLASHCARDS")
    print("=" * 65)

    for index, card in enumerate(flashcards, start=1):

        print(f"\nCARD {index}")
        print("-" * 65)

        print(f"Q: {card['question']}")
        print(f"A: {card['answer']}")

    print("\n" + "=" * 65)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 65)
    print("                  AI FLASHCARD GENERATOR")
    print("=" * 65)

    # --------------------------------------------------------
    # GET TOPIC
    # --------------------------------------------------------

    topic = input("\nEnter your topic: ").strip()

    if not topic:

        print("❌ Topic cannot be empty.")
        return

    # --------------------------------------------------------
    # GET NUMBER OF FLASHCARDS
    # --------------------------------------------------------

    try:

        number_of_cards = int(
            input("How many flashcards? [5]: ") or "5"
        )

    except ValueError:

        print("❌ Please enter a valid number.")
        return

    # --------------------------------------------------------
    # VALIDATE NUMBER
    # --------------------------------------------------------

    if number_of_cards < 1 or number_of_cards > 20:

        print("❌ Please choose between 1 and 20 flashcards.")
        return

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    print("\n🤖 Generating flashcards...")
    print("⏳ Please wait...\n")

    flashcards = generate_flashcards(
        topic,
        number_of_cards
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    if flashcards:

        display_flashcards(flashcards)

    else:

        print("\n❌ Failed to generate flashcards.")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()