import os
import chromadb
import anthropic
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rugby_laws"
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a rugby laws expert assistant. You help players, coaches, and referees
understand the Laws of the Game (Rugby Union) as published by World Rugby.

When answering questions:
- Always use the search_rugby_laws tool to find relevant law text before answering
- Cite specific law numbers (e.g. Law 11.4) in your answers
- Be precise and clear — misunderstanding laws has real consequences in a match
- If a situation is ambiguous or involves referee discretion, say so
- Keep answers concise but complete"""

TOOLS = [
    {
        "name": "search_rugby_laws",
        "description": (
            "Search the World Rugby Laws of the Game document for rules and definitions. "
            "Use this to find relevant law text before answering any question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural language search query about a rugby law or situation",
                }
            },
            "required": ["query"],
        },
    }
]


def search_laws(query: str, n_results: int = 4) -> str:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    results = collection.query(query_texts=[query], n_results=n_results)

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        label = meta.get("label", "Law ?")
        chunks.append(f"[{label}]\n{doc}")

    return "\n\n---\n\n".join(chunks)


def run_turn(client: anthropic.Anthropic, messages: list) -> str:
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Append assistant response to history
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "search_rugby_laws":
                    print(f"  [searching laws: {block.input['query']}]")
                    result = search_laws(block.input["query"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})
            continue

        # Extract final text response
        for block in response.content:
            if hasattr(block, "text"):
                return block.text

        return ""


def main():
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = []

    print("Rugby Laws Bot (type 'quit' to exit)\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})
        answer = run_turn(client, messages)
        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    main()
