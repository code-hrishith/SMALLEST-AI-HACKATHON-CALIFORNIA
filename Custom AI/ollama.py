import subprocess
from retrieval import retrieve_relevant_chunks

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def ask_mistral(context, question):
    prompt = f"""
You are a concise and expert Ayurvedic advisor. Use the given CONTEXT to answer the QUESTION based on Ayurveda only.

- Reply concisely in 3-4 sentences using household Ayurvedic remedies.
- Focus ONLY on Pitta dosha balancing, natural therapies, or Ayurvedic herbs and remedies from ingredients found in general houses.
- If no relevant info is found, say: "I don't have that information at the moment."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

    try:
        result = subprocess.run(
            ["ollama", "run", "mistral","--num-predict", "100"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60  # seconds
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "Timed out while getting response from Ollama"


def main():
    import sys
    query = sys.argv[1]
    print("[Python] Received query:", query)
    
    context = retrieve_relevant_chunks(query)
    print("[Python] Retrieved context length:", len(context))

    response = ask_mistral(context, query)
    print("[Python] Final response:")
    print(response)

if __name__ == "__main__":
    main()

