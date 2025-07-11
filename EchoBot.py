import os
import random
import difflib
import re
import spacy
from knowledge_base import init_db, add_fact, get_all_facts
from langdetect import detect, LangDetectException
from autocorrect import Speller
from textblob import TextBlob
from rapidfuzz import fuzz
from rich.console import Console
from rich.prompt import Prompt
import datetime
from textblob.sentiments import PatternAnalyzer
import string
import requests
from bs4 import BeautifulSoup

CHAT_LOG_FILE = "chat_log.txt"

# Initialize DB and NLP
init_db()
affirmations = {"yes", "yeah", "yep", "sure", "ok", "okay", "alright", "affirmative"}
negations = {"no", "nope", "nah", "negative"}
def clean_knowledge_base():
    import sqlite3
    conn = sqlite3.connect('knowledge_base.db')
    c = conn.cursor()
    # Remove facts that are too short or are affirmations/negations
    c.execute("DELETE FROM facts WHERE LENGTH(fact) < 4")
    for word in list(affirmations) + list(negations):
        c.execute("DELETE FROM facts WHERE LOWER(fact) = ?", (word,))
    # Remove duplicates
    c.execute("""DELETE FROM facts WHERE rowid NOT IN (
        SELECT MIN(rowid) FROM facts GROUP BY LOWER(fact)
    )""")
    conn.commit()
    conn.close()

clean_knowledge_base()
try:
    nlp = spacy.load("en_core_web_sm")
    from sentence_transformers import SentenceTransformer, util
    import numpy as np
    model = SentenceTransformer('all-MiniLM-L6-v2')
    use_transformers = True
except ImportError:
    print("sentence-transformers not available, using spacy-based matching.")
    use_transformers = False

paraphrases = {}
common_words = set(["the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"])
frustration_words = {"hurt", "explode", "frustrated", "angry", "annoyed", "mad", "crazy", "confusing", "brain"}
greetings = {"hi", "hello", "hey", "greetings", "howdy", "good morning", "good afternoon", "good evening"}
conversation_starters = {"how are you", "what's up", "how's it going", "how do you do", "how are things", "how are you doing"}

# Enhanced, engaging greetings and conversation starters
rich_greetings = [
    "Hello! 😊 How can I brighten your day?",
    "Hi there! What would you like to explore today?",
    "Hey! I'm here to chat, learn, and help. What's on your mind?",
    "Greetings! Ready for a great conversation?",
    "Howdy! Ask me anything or just say hi!"
]
rich_starters = [
    "I'm doing well, thank you! How about you? What's new in your world?",
    "I'm always learning and ready to help. How are you feeling today?",
    "I'm here to chat about anything—what's on your mind?",
    "I'm excited to talk! Tell me something interesting about yourself.",
    "I'm always ready for a good conversation. What's up?"
]

console = Console()
spell = Speller(lang='en')
pattern_analyzer = PatternAnalyzer()

SESSION_MEMORY_SIZE = 10
session_memory = []
system_prompt = (
    "You are EchoBot, an AI assistant trained to reason step-by-step, learn from users, and improve over time. "
    "Today is {date}. Use your knowledge base, session memory, and logic to answer."
)
user_persona = {'name': 'User', 'interests': []}

def set_user_persona():
    console.print("[bold cyan]Let's personalize your EchoBot experience![/bold cyan]")
    name = Prompt.ask("What's your name? (or press Enter to skip)").strip()
    if name:
        user_persona['name'] = name
    interests = Prompt.ask("What are your interests? (comma separated, or press Enter to skip)").strip()
    if interests:
        user_persona['interests'] = [i.strip() for i in interests.split(',') if i.strip()]
    console.print(f"[bold green]Welcome, {user_persona['name']}! I'll try to keep your interests in mind: {', '.join(user_persona['interests']) if user_persona['interests'] else 'none specified'}.[/bold green]")

def add_to_session_memory(user, bot):
    session_memory.append((user, bot))
    if len(session_memory) > SESSION_MEMORY_SIZE:
        session_memory.pop(0)

def summarize_session_memory():
    if not session_memory:
        return ""
    summary = []
    for i, (u, b) in enumerate(session_memory[-SESSION_MEMORY_SIZE:]):
        summary.append(f"[{i+1}] {user_persona['name']}: {u}")
        summary.append(f"[{i+1}] EchoBot: {b}")
    return '\n'.join(summary)

def show_reasoning_chain(user_input, fact, score, reasoning_steps):
    chain = [f"System prompt: {system_prompt.format(date=datetime.datetime.now().strftime('%Y-%m-%d'))}"]
    mem_summary = summarize_session_memory()
    if mem_summary:
        chain.append("Here's what we've talked about recently:")
        chain.append(mem_summary)
    chain.append(f"You ({user_persona['name']}): {user_input}")
    if fact:
        chain.append(f"I found this in my knowledge: '{fact}' (confidence: {score:.2f})")
    if reasoning_steps:
        chain.append("Here's how I thought about your question:")
        chain.extend(reasoning_steps)
    return '\n'.join(chain)

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
    return set(w for w in words if w.isalnum())

def centrality_score(input_keywords, fact_keywords):
    if not input_keywords or not fact_keywords:
        return 0.0
    intersection = len(input_keywords & fact_keywords)
    union = len(input_keywords | fact_keywords)
    return intersection / union

def is_low_quality_fact(fact, user_input):
    if len(fact.split()) < 3:
        return True
    if fact.strip().lower() == user_input.strip().lower():
        return True
    if fact.strip().lower() in affirmations or fact.strip().lower() in negations:
        return True
    return False

def find_best_fact(user_input, min_score=0.0, exclude_facts=None):
    all_facts = get_all_facts()
    if exclude_facts is None:
        exclude_facts = set()
    best_fact = None
    best_score = 0.0
    user_input_lower = user_input.strip().lower()
    if use_transformers and all_facts:
        user_emb = model.encode([user_input], convert_to_tensor=True)
        fact_embs = model.encode(all_facts, convert_to_tensor=True)
        similarities = util.cos_sim(user_emb, fact_embs)[0].cpu().numpy()
        for i, sim in enumerate(similarities):
            if all_facts[i].strip().lower() == user_input_lower:
                continue
            if all_facts[i] in exclude_facts:
                continue
            if sim > best_score and sim >= min_score:
                best_score = sim
                best_fact = all_facts[i]
        if best_score == 0.0:
            return None, 0.0
        return best_fact, float(best_score)
    else:
        input_keywords = get_keywords(user_input)
        main_subject = None
        doc = nlp(user_input)
        noun_chunks = list(doc.noun_chunks)
        if noun_chunks:
            main_subject = noun_chunks[0].text
        else:
            nouns = [token.text for token in doc if token.pos_ == "NOUN"]
            if nouns:
                main_subject = nouns[0]
        if not main_subject and input_keywords:
            main_subject = max(input_keywords, key=len)
        for fact in all_facts:
            if is_low_quality_fact(fact, user_input) or fact in exclude_facts:
                continue
            if fact.strip().lower() == user_input_lower:
                continue
            fact_keywords = get_keywords(fact)
            score = centrality_score(input_keywords, fact_keywords)
            if main_subject and main_subject.lower() not in fact.lower():
                continue
            if score > best_score and score >= min_score:
                best_score = score
                best_fact = fact
        if best_score == 0.0:
            return None, 0.0
        return best_fact, best_score

def fuzzy_match_paraphrase(user_input):
    if not paraphrases:
        return None
    matches = difflib.get_close_matches(user_input.lower(), paraphrases.keys(), n=1, cutoff=0.75)
    return paraphrases[matches[0]] if matches else None

def suggest_spelling(word, vocabulary):
    if word in common_words or word in affirmations or word in negations:
        return None
    matches = difflib.get_close_matches(word, vocabulary, n=1, cutoff=0.8)
    return matches[0] if matches else None

def dynamic_greeting(user_input):
    now = datetime.datetime.now()
    hour = now.hour
    if 'morning' in user_input.lower():
        time_phrase = 'Good morning'
    elif 'afternoon' in user_input.lower():
        time_phrase = 'Good afternoon'
    elif 'evening' in user_input.lower():
        time_phrase = 'Good evening'
    else:
        if hour < 12:
            time_phrase = 'Good morning'
        elif hour < 18:
            time_phrase = 'Good afternoon'
        else:
            time_phrase = 'Good evening'
    return f"{time_phrase}! I'm EchoBot, your AI companion. How can I help you today?"

def dynamic_starter(user_input):
    # Extract sentiment and intent
    sentiment = pattern_analyzer.analyze(user_input)[0]
    if sentiment > 0.3:
        mood = "You sound upbeat!"
    elif sentiment < -0.3:
        mood = "You sound a bit down. I'm here for you!"
    else:
        mood = "How are you feeling today?"
    return f"I'm always learning and ready to chat. {mood} What's on your mind?"

def dynamic_affirmation(user_input):
    return f"You said '{user_input}'. Absolutely, I agree!"

def dynamic_negation(user_input):
    return f"You said '{user_input}'. I understand, let's consider another perspective."

def self_reflect(last_reply, user_input):
    # Simple self-critique: was the answer relevant, clear, and helpful?
    critique = []
    if not last_reply or 'I don\'t know' in last_reply or 'not sure' in last_reply:
        critique.append("I realize my answer may not have been very helpful. Next time, I should ask for more details or clarification.")
    elif user_input.lower() in last_reply.lower():
        critique.append("I may have just repeated your input. I should try to add more value or explanation.")
    else:
        critique.append("I tried to use my knowledge and reasoning to help. If you have feedback, I'd love to learn!")
    return ' '.join(critique)

def normalize_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

def clear_knowledge_base():
    import sqlite3
    conn = sqlite3.connect('knowledge_base.db')
    c = conn.cursor()
    c.execute('DELETE FROM facts')
    conn.commit()
    conn.close()

def batch_train(pairs):
    # pairs: list of (input, response)
    for inp, resp in pairs:
        add_fact(f"{normalize_text(inp)}|||{resp}", category="starter_pair", source="batch_train")

def get_reply(user_input):
    user_input = user_input.strip()
    lower_input = user_input.lower()
    reasoning_steps = []
    norm_input = normalize_text(user_input)
    thinking_msg = "EchoBot is thinking..."
    # Special case: self-introduction
    if norm_input in ["who are you", "what are you", "who is echobot", "what is echobot", "tell me about yourself"]:
        reply = ("Hello! I'm EchoBot, your AI assistant. I'm designed to learn from our conversations, reason step-by-step, "
                 "and help you with information, ideas, and friendly chat. You can teach me new things, and I'll remember them! "
                 "Feel free to ask me anything or tell me how I can improve.")
        add_to_session_memory(user_input, reply)
        return thinking_msg + "\n" + show_reasoning_chain(user_input, reply, 1.0, ["Special case: self-introduction."]) + f"\n\n{reply}"
    # Language detection
    try:
        if len(user_input) > 6:
            lang = detect(user_input)
            if lang != 'en':
                return "Sorry, I can only understand and reply in English right now."
    except LangDetectException:
        pass
    # Autocorrect (only if not a math expression)
    math_pattern = r'^[\d\s+\-*/().]+$'
    if not re.match(math_pattern, user_input):
        corrected = spell(user_input)
        if corrected != user_input:
            reasoning_steps.append(f"Autocorrected input: '{user_input}' → '{corrected}'")
            user_input = corrected
            lower_input = user_input.lower()
    # Sentiment analysis
    sentiment = pattern_analyzer.analyze(user_input)[0]
    if sentiment < -0.5:
        import random
        return random.choice([
            "I'm sorry if things feel negative. I'm here to help!",
            "It sounds like you're upset. Want to talk about it?"
        ])
    # Teach me mode
    if lower_input == 'teach me':
        lesson = Prompt.ask("[bold cyan]What would you like to teach me? (or type 'batch' to train multiple)")
        if lesson.strip().lower() == 'batch':
            batch = []
            console.print("[bold cyan]Enter input and response pairs, one per line, separated by '|||'. Type 'done' to finish.[/bold cyan]")
            while True:
                line = Prompt.ask("[bold cyan]Pair (input|||response)[/bold cyan]")
                if line.strip().lower() == 'done':
                    break
                if '|||' in line:
                    inp, resp = line.split('|||', 1)
                    batch.append((inp.strip(), resp.strip()))
            batch_train(batch)
            return f"Batch training complete! Added {len(batch)} pairs."
        elif lesson and len(lesson) > 3:
            import sqlite3
            conn = sqlite3.connect('knowledge_base.db')
            c = conn.cursor()
            c.execute("DELETE FROM facts WHERE LOWER(fact) = ?", (norm_input,))
            c.execute("DELETE FROM facts WHERE fact LIKE ?", (f"%|||%",))
            conn.commit()
            conn.close()
            if session_memory:
                last_user_input = session_memory[-1][0]
                add_fact(f"{normalize_text(last_user_input)}|||{lesson}", category="starter_pair", source="teach_me_mode")
            add_fact(lesson, category="user_lesson", source="teach_me_mode")
            reply = f"Thank you for teaching me! I've added this to my knowledge: '{lesson}'"
            add_to_session_memory(user_input, reply)
            return reply
        else:
            return "Please provide a more detailed lesson."
    # --- Efficient DeepSeek-style starter/greeting handling ---
    all_facts = get_all_facts()
    # Build a hash for fast exact match
    starter_map = {}
    starter_pairs = []
    for fact in all_facts:
        if '|||' in fact:
            key, value = fact.split('|||', 1)
            starter_map[normalize_text(key)] = value
            starter_pairs.append((key, value))
    if norm_input in starter_map:
        value = starter_map[norm_input]
        reply = value + ("\nIf you have more to share, I'm always here to listen!" if len(value.split()) > 3 else "\nBy the way, feel free to ask me anything or tell me more about your day!")
        add_to_session_memory(user_input, reply)
        return thinking_msg + "\n" + show_reasoning_chain(user_input, value, 1.0, ["Exact match for starter/greeting (user-taught mapping). "]) + f"\n\n[Internal confidence: 1.00 | Overall confidence: 1.00]\n{reply}"
    # For similarity, only check the N most recent facts and N random facts
    import random
    N = 30
    recent_facts = all_facts[-N:]
    random_facts = random.sample(all_facts, min(N, len(all_facts))) if len(all_facts) > N else []
    facts_to_check = list(set(recent_facts + random_facts))
    best_fact = None
    best_score = 0.0
    best_secondary = 0.0
    for fact in facts_to_check:
        if is_low_quality_fact(fact, user_input) or fact.strip().lower() == user_input.lower():
            continue
        if '|||' in fact:
            continue  # Only check non-mapping facts for similarity
        # Primary: semantic similarity
        if use_transformers:
            user_emb = model.encode([user_input], convert_to_tensor=True)
            fact_emb = model.encode([fact], convert_to_tensor=True)
            sim = float(util.cos_sim(user_emb, fact_emb)[0][0])
        else:
            input_keywords = get_keywords(user_input)
            fact_keywords = get_keywords(fact)
            sim = centrality_score(input_keywords, fact_keywords)
        # Secondary: keyword overlap
        input_keywords = set(get_keywords(user_input))
        fact_keywords = set(get_keywords(fact))
        overlap = len(input_keywords & fact_keywords) / (len(input_keywords | fact_keywords) + 1e-6)
        # Combine for internal confidence
        confidence = 0.7 * sim + 0.3 * overlap
        reasoning_steps.append(f"Checked fact: '{fact}' | sim: {sim:.2f}, overlap: {overlap:.2f}, confidence: {confidence:.2f}")
        if confidence > best_score:
            best_score = confidence
            best_fact = fact
            best_secondary = overlap
    if len(user_input.split()) <= 3:
        min_conf = 0.2
    else:
        min_conf = 0.45
    if best_fact and best_score >= min_conf:
        reply = f"[Internal confidence: {best_score:.2f} | Overall confidence: {best_score:.2f}] {best_fact}"
        add_to_session_memory(user_input, reply)
        return thinking_msg + "\n" + show_reasoning_chain(user_input, best_fact, best_score, reasoning_steps) + f"\n\n{reply}"
    # --- Self-training: ask user to teach ---
    if len(user_input.split()) > 3:
        reply = random.choice([
            "I'm not sure I know the answer yet. Could you explain it to me? I'll remember!",
            "That's a great question! Can you teach me the answer so I can help others in the future?",
            "I don't have a confident answer yet. Would you like to tell me more or provide the answer?"
        ])
        add_to_session_memory(user_input, reply)
        return thinking_msg + "\n" + show_reasoning_chain(user_input, None, 0.0, reasoning_steps) + f"\n\n[Internal confidence: 0.00 | Overall confidence: 0.00]\n{reply}"
    fallback_responses = [
        "That's an interesting topic! Could you share more details or clarify your question?",
        "I'm eager to learn more about that. Can you elaborate or rephrase?",
        "I'm still learning and want to give you the best answer. Could you help me understand better?",
        "Let's dive deeper! What specifically would you like to know?"
    ]
    reply = random.choice(fallback_responses)
    add_to_session_memory(user_input, reply)
    return thinking_msg + "\n" + show_reasoning_chain(user_input, None, 0.0, reasoning_steps) + f"\n\n[Internal confidence: 0.00 | Overall confidence: 0.00]\n{reply}"

def learn_from_conversation(user_input, reply):
    # Only add facts if the user is explicitly teaching (handled in get_reply feedback/teach me)
    pass

def web_search(query, num_results=1):
    """Search the web and return a summary of the top result(s)."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        params = {'q': query}
        response = requests.get('https://www.bing.com/search', params=params, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        for li in soup.select('li.b_algo')[:num_results]:
            title = li.find('h2')
            snippet = li.find('p')
            if title and snippet:
                results.append(f"{title.text.strip()}: {snippet.text.strip()}")
        if results:
            return '\n'.join(results)
        else:
            return None
    except Exception as e:
        return None

if __name__ == "__main__":
    # Uncomment the next line to clear the knowledge base on startup
    # clear_knowledge_base()
    set_user_persona()
    console.print(f"[bold green]{system_prompt.format(date=datetime.datetime.now().strftime('%Y-%m-%d'))}[/bold green]")
    while True:
        user_input = Prompt.ask(f"[bold cyan]{user_persona['name']}[/bold cyan]").strip()
        if user_input.lower() in ["quit", "exit", "bye"]:
            console.print("[bold yellow]EchoBot: Shutting down...[/bold yellow]")
            break
        reply = get_reply(user_input)
        console.print(f"[bold magenta]EchoBot:[/bold magenta] {reply}")
        save_chat_log(user_input, reply)
        learn_from_conversation(user_input, reply)
        feedback = Prompt.ask("[bold cyan]Was this answer helpful? (yes/no/teach me)[/bold cyan]").strip().lower()
        if feedback in ["no", "teach me"]:
            correction = Prompt.ask("[bold cyan]Please provide the correct answer or more info so I can learn:[/bold cyan]")
            if correction and len(correction) > 3:
                add_fact(f"{normalize_text(user_input)}|||{correction}", category="user_feedback", source="feedback_loop")
                console.print("[bold green]Thank you! I've learned from your feedback.[/bold green]")
