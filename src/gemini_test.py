from dotenv import load_dotenv
from google import genai
import os

# Load API key from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Test Gemini
interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input="Explain the Transformer architecture in one sentence."
)

print("\nGemini response:")
print(interaction.output_text)