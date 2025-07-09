import os
import json
import random
import difflib
import re
from textblob import TextBlob

MEMORY_FILE = "memory.json"
CHAT_LOG_FILE = "chat_log.txt"

# Load memory as a list of facts
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        try:
            memory_data = json.load(f)
            if isinstance(memory_data, dict) and "facts" in memory_data:
                facts = memory_data["facts"]
            elif isinstance(memory_data, list):
                facts = memory_data
            else:
                facts = []
        except Exception:
            facts = []
else:
    facts = []

# Save facts to memory.json
def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump({"facts": facts}, f, indent=2)

def save_chat_log(user_input, reply):
    with open(CHAT_LOG_FILE, "a") as f:
        f.write(f"You: {user_input}\n")
        f.write(f"EchoBot: {reply}\n")

def get_keywords(text):
    blob = TextBlob(text.lower())
    keywords = [w for w in blob.words if w.isalnum()]
    return set(keywords)

def find_relevant_fact(user_input):
    input_keywords = get_keywords(user_input)
    best_fact = None
    best_overlap = 0
    for fact in facts:
        fact_keywords = get_keywords(fact)
        overlap = len(input_keywords & fact_keywords)
        if overlap > best_overlap:
            best_overlap = overlap
            best_fact = fact
    # Only return if there's some overlap
    if best_overlap > 0:
        return best_fact
    return None

def get_reply(user_input):
    user_input = user_input.strip()
    # Try to find a relevant fact
    relevant_fact = find_relevant_fact(user_input)
    if relevant_fact:
        # Paraphrase or restate the fact
        if random.random() < 0.5:
            return f"I know: {relevant_fact}"
        else:
            return relevant_fact
    # If no relevant fact, generate a new statement or question
    blob = TextBlob(user_input)
    words = blob.words
    if len(words) > 3:
        # Ask a question about the input
        return f"Why do you say: '{user_input}'?"
    elif len(words) > 0:
        # Rephrase or echo
        return f"Tell me more about '{user_input}'."
    else:
        return "I'm still learning. Can you tell me something?"

def learn_from_conversation(user_input, reply):
    # Add user input as a new fact if not already present and is not empty
    user_input = user_input.strip()
    if user_input and user_input not in facts:
        facts.append(user_input)
        save_memory()
    # Optionally, add the bot's reply as a fact if it's not a question
    if reply and not reply.endswith('?') and reply not in facts:
        facts.append(reply)
        save_memory()

def forget_fact(fact_text):
    # Remove a fact from memory
    if fact_text in facts:
        facts.remove(fact_text)
        save_memory()
        return True
    return False

if __name__ == "__main__":
    print("🤖 EchoBot (AI) is online! Type 'quit' to exit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("EchoBot: Shutting down...")
            break
        if user_input.startswith("!forget "):
            forget_phrase = user_input[8:].strip()
            if forget_fact(forget_phrase):
                print(f"EchoBot: I forgot '{forget_phrase}'. 🧠❌")
            else:
                print("EchoBot: I don't remember that fact.")
            continue
        reply = get_reply(user_input)
        print("EchoBot:", reply)
        save_chat_log(user_input, reply)
        learn_from_conversation(user_input, reply)
