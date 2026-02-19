import os
import discord
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

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)
ai_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)


def search_laws(query: str, n_results: int = 4) -> str:
    results = collection.query(query_texts=[query], n_results=n_results)
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        label = meta.get("label", "Law ?")
        chunks.append(f"[{label}]\n{doc}")
    return "\n\n---\n\n".join(chunks)


def run_turn(messages: list) -> str:
    while True:
        response = ai_client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "search_rugby_laws":
                    result = search_laws(block.input["query"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        for block in response.content:
            if hasattr(block, "text"):
                return block.text

        return ""


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")


@bot.event
async def on_message(message: discord.Message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Only respond when mentioned
    if bot.user not in message.mentions:
        return

    # Strip the mention from the message
    question = message.content.replace(f"<@{bot.user.id}>", "").strip()
    if not question:
        await message.reply("Ask me anything about the Laws of Rugby Union!")
        return

    async with message.channel.typing():
        messages = [{"role": "user", "content": question}]
        answer = run_turn(messages)

    # Discord has a 2000 character limit per message
    if len(answer) <= 2000:
        await message.reply(answer)
    else:
        # Split into chunks at newlines to avoid cutting mid-sentence
        chunks = []
        current = ""
        for line in answer.split("\n"):
            if len(current) + len(line) + 1 > 1900:
                chunks.append(current)
                current = line
            else:
                current += ("\n" if current else "") + line
        if current:
            chunks.append(current)

        for chunk in chunks:
            await message.channel.send(chunk)


bot.run(os.environ["DISCORD_BOT_TOKEN"])
