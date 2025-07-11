import sys
import os
import re
from knowledge_base import init_db, add_fact, get_all_facts

# Set of already known facts (avoid duplicates)
existing_facts = set(get_all_facts())

def extract_facts_from_text(text):
    # Split into sentences with punctuation handling
    sentences = re.split(r'(?<=[.!?])\s+', text)
    facts = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        s_lower = s.lower()

        # Filter out things that aren't facts
        if len(s.split()) < 4:
            continue  # too short
        if s.endswith('?'):
            continue  # question
        if s_lower.startswith(('please', 'do ', 'can you', 'could you', 'would you', 'should you', 'let us', "let's")):
            continue  # command
        if any(word in s_lower for word in ["click", "download", "visit", "buy", "subscribe"]):
            continue  # promotional or action text
        if s in existing_facts:
            continue  # already in DB

        facts.append(s)

    return facts

def main():
    if len(sys.argv) < 2:
        print("📘 Usage: python train_from_docs.py file1.txt [file2.md ...]")
        return

    print("🧠 Initializing knowledge base...")
    init_db()
    added_total = 0

    for filename in sys.argv[1:]:
        if not os.path.isfile(filename):
            print(f"⚠️ Skipping {filename}: not a file.")
            continue

        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()

        facts = extract_facts_from_text(text)
        for fact in facts:
            add_fact(fact, category="training", source=os.path.basename(filename))

        added_total += len(facts)
        print(f"✅ Added {len(facts)} facts from '{filename}'.")
            # Add some friendly personality responses
    chatty_facts = [
        "I'm here to help you figure things out.",
        "If I make a mistake, just tell me — I'm learning too!",
        "Asking smart questions makes you smarter.",
        "Learning together is more fun, right?",
        "It's okay to be confused. Let's figure it out step by step.",
        "I try my best, but I'm not perfect. Yet.",
        "If something seems weird, just tell me — I’ll try to fix it!",
        "Talking to me helps me grow. You’re teaching me!",
        "You can ask me about math, facts, or ideas. I’ll give it my best shot!",
        "Don’t worry — even I get things wrong sometimes. 😅",
    ]
    for phrase in chatty_facts:
        add_fact(phrase, category="tone", source="personality")
        
    print(f"\n🎉 Training complete! {added_total} new facts added to EchoBot's brain.")

if __name__ == "__main__":
    main()
