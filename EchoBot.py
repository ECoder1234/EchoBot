import os
import random
import difflib
import re
import spacy
from knowledge_base import init_db, add_fact, get_all_facts

CHAT_LOG_FILE = "chat_log.txt"

init_db()
nlp = spacy.load("en_core_web_sm")

paraphrases = {}
affirmations = {"yes", "yeah", "yep", "sure", "ok", "okay", "alright", "affirmative"}
negations = {"no", "nope", "nah", "negative"}
common_words = set(["the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"])
frustration_words = {"hurt", "explode", "frustrated", "angry", "annoyed", "mad", "crazy", "confusing", "brain"}

# Save chat log
def save_chat_log(user_input, reply):
    with open(CHAT_LOG_FILE, "a") as f:
        f.write(f"You: {user_input}\n")
        f.write(f"EchoBot: {reply}\n")

def learn_new_paraphrase(user_input):
    if ' means ' in user_input:
        phrase, meaning = user_input.split(' means ', 1)
        phrase = phrase.strip().lower()
        meaning = meaning.strip()
        if phrase and meaning:
            paraphrases[phrase] = meaning
            add_fact(f"{phrase} means {meaning}", category="paraphrase", source="conversation")
            return f"Got it! I'll remember that '{phrase}' means '{meaning}'."
    return None

def get_keywords(text):
    words = text.lower().split()
    keywords = [w for w in words if w.isalnum()]
    return set(keywords)

def centrality_score(input_keywords, fact_keywords):
    if not input_keywords or not fact_keywords:
        return 0.0
    intersection = len(input_keywords & fact_keywords)
    union = len(input_keywords | fact_keywords)
    return intersection / union

def is_low_quality_fact(fact, user_input):
    # Ignore facts that are too short, look like user input, or are affirmations/negations
    if len(fact.split()) < 3:
        return True
    if fact.strip().lower() == user_input.strip().lower():
        return True
    if fact.strip().lower() in affirmations or fact.strip().lower() in negations:
        return True
    return False

def find_best_fact(user_input, min_score=0.0, exclude_facts=None):
    input_keywords = get_keywords(user_input)
    all_facts = get_all_facts()
    if exclude_facts is None:
        exclude_facts = set()
    best_fact = None
    best_score = 0.0
    main_subject = None
    doc = nlp(user_input)
    noun_chunks = list(doc.noun_chunks)
    if noun_chunks:
        main_subject = noun_chunks[0].text
    else:
        nouns = [token.text for token in doc if token.pos_ == "NOUN"]
        if nouns:
            main_subject = nouns[0]
    if not main_subject:
        words = user_input.split()
        if words:
            main_subject = max(words, key=len)
    for fact in all_facts:
        if is_low_quality_fact(fact, user_input) or fact in exclude_facts:
            continue
        fact_keywords = get_keywords(fact)
        score = centrality_score(input_keywords, fact_keywords)
        if main_subject and main_subject.lower() not in fact.lower():
            continue
        if score > best_score and score >= min_score:
            best_score = score
            best_fact = fact
    return best_fact, best_score

def fuzzy_match_paraphrase(user_input):
    if not paraphrases:
        return None
    matches = difflib.get_close_matches(user_input.lower(), paraphrases.keys(), n=1, cutoff=0.75)
    if matches:
        return paraphrases[matches[0]]
    return None

def suggest_spelling(word, vocabulary):
    if word in common_words or word in affirmations or word in negations:
        return None
    matches = difflib.get_close_matches(word, vocabulary, n=1, cutoff=0.8)
    if matches:
        return matches[0]
    return None

def get_reply(user_input):
    user_input = user_input.strip()
    # Handle affirmations/negations
    if user_input.lower() in affirmations:
        return random.choice(["Yes.", "Affirmative.", "Understood."])
    if user_input.lower() in negations:
        return random.choice(["No.", "Negative.", "Understood."])
    # Frustration detection
    if any(word in user_input.lower() for word in frustration_words):
        return random.choice([
            "I'm sorry, I know this is frustrating. I'm still learning!",
            "I understand this is confusing. Thank you for your patience.",
            "I'm trying to improve. Can you help me learn?"
        ])
    # Learn new paraphrase
    learned = learn_new_paraphrase(user_input)
    if learned:
        return learned
    # Fuzzy match paraphrase
    paraphrased = fuzzy_match_paraphrase(user_input)
    if paraphrased:
        return paraphrased
    # Math evaluation
    math_pattern = r'^\s*([-+]?\d+(?:\.\d+)?\s*[-+*/]\s*[-+]?\d+(?:\.\d+)?(\s*[-+*/]\s*[-+]?\d+(?:\.\d+)?)*\s*)$'
    if re.match(math_pattern, user_input.replace(' ', '')):
        try:
            result = eval(user_input, {"__builtins__": None}, {})
            return f"The answer is {result}."
        except Exception:
            return "Sorry, I couldn't calculate that."
    # Fact matching with centrality and main subject
    tried_facts = set()
    while True:
        best_fact, best_score = find_best_fact(user_input, min_score=0.0, exclude_facts=tried_facts)
        print(f"[EchoBot Certainty] Centrality Score: {best_score:.2f}")
        if best_fact and best_score >= 0.75:
            print(f"[EchoBot Accuracy] High confidence in answer.")
            return best_fact
        if best_fact:
            tried_facts.add(best_fact)
        else:
            break
        # Try next best fact
        if len(tried_facts) > 10:  # Avoid infinite loops
            break
    # Fallback
    words = user_input.split()
    fallback_responses = [
        "I'm not confident enough to answer that yet. Can you rephrase or give me more details?",
        "I'm still learning and not sure how to answer that. Can you help me improve?",
        "Sorry, I don't know that yet. Can you explain more or ask differently?"
    ]
    if len(words) > 3:
        return random.choice(fallback_responses)
    elif len(words) > 0:
        return random.choice(["I'm not sure how to respond to '" + user_input + "'.", "Tell me more!", "Let's talk more about that."])
    else:
        return "I'm still learning. Can you tell me something?"

def learn_from_conversation(user_input, reply):
    user_input = user_input.strip()
    if user_input:
        add_fact(user_input, category=None, source="conversation")
    if reply and not reply.endswith('?'):
        add_fact(reply, category=None, source="conversation")

if __name__ == "__main__":
    print("🤖 EchoBot (AI) is online! Type 'quit' to exit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("EchoBot: Shutting down...")
            break
        reply = get_reply(user_input)
        print("EchoBot:", reply)
        save_chat_log(user_input, reply)
        learn_from_conversation(user_input, reply)
