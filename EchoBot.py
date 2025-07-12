import os
import random
import difflib
import re
import spacy
import Knowledge_Base as Knowledge_Base
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
import itertools
from dataclasses import dataclass
from enum import Enum
import json

CHAT_LOG_FILE = "chat_log.txt"

# Initialize DB and NLP
Knowledge_Base.init_db()
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
except (ImportError, OSError) as e:
    print(f"spaCy model or sentence-transformers not available: {e}")
    print("Using fallback keyword-based matching.")
    nlp = None
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
class SessionMemory:
    def __init__(self, size=10):
        self.size = size
        self.memory = []
    def add(self, user, bot):
        self.memory.append((user, bot))
        if len(self.memory) > self.size:
            self.memory.pop(0)
    def summary(self):
        if not self.memory:
            return ""
        summary = []
        for i, (u, b) in enumerate(self.memory[-self.size:]):
            summary.append(f"[{i+1}] {user_persona['name']}: {u}")
            summary.append(f"[{i+1}] EchoBot: {b}")
        return '\n'.join(summary)
    def get_context(self):
        return ' '.join(u for u, _ in self.memory[-self.size:])
    def clear(self):
        self.memory = []

session_memory_obj = SessionMemory(SESSION_MEMORY_SIZE)


class ReasoningStrategy(Enum):
    STEP_BY_STEP = "step_by_step"
    BREAKDOWN = "breakdown"
    ANALOGY = "analogy"
    LOGICAL_CHAIN = "logical_chain"
    CREATIVE_SOLUTION = "creative_solution"

@dataclass
class ReasoningStep:
    step_type: str
    content: str
    confidence: float
    reasoning: str

class AdvancedReasoner:
    def __init__(self):
        self.reasoning_strategies = {
            ReasoningStrategy.STEP_BY_STEP: self._step_by_step_reasoning,
            ReasoningStrategy.BREAKDOWN: self._breakdown_reasoning,
            ReasoningStrategy.ANALOGY: self._analogy_reasoning,
            ReasoningStrategy.LOGICAL_CHAIN: self._logical_chain_reasoning,
            ReasoningStrategy.CREATIVE_SOLUTION: self._creative_solution_reasoning
        }
        
    def generate_reasoning_chain(self, user_input: str, context: str = "") -> tuple[str, list[ReasoningStep]]:
        """Generate a DeepSeek R1-style reasoning chain"""
        
        # Analyze input type and choose strategy
        strategy = self._choose_strategy(user_input)
        
        # Generate reasoning steps
        steps = self.reasoning_strategies[strategy](user_input, context)
        
        # Synthesize final answer
        final_answer = self._synthesize_answer(steps, user_input)
        
        return final_answer, steps
    
    def _choose_strategy(self, user_input: str) -> ReasoningStrategy:
        """Choose the best reasoning strategy based on input"""
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ["how", "why", "explain", "process"]):
            return ReasoningStrategy.STEP_BY_STEP
        elif any(word in input_lower for word in ["break down", "analyze", "parts"]):
            return ReasoningStrategy.BREAKDOWN
        elif any(word in input_lower for word in ["like", "similar", "compare"]):
            return ReasoningStrategy.ANALOGY
        elif any(word in input_lower for word in ["if", "then", "because", "therefore"]):
            return ReasoningStrategy.LOGICAL_CHAIN
        else:
            return ReasoningStrategy.CREATIVE_SOLUTION
    
    def _step_by_step_reasoning(self, user_input: str, context: str) -> list[ReasoningStep]:
        """Step-by-step reasoning like DeepSeek R1"""
        steps = []
        
        # Step 1: Understand the question
        steps.append(ReasoningStep(
            step_type="understanding",
            content=f"First, I need to understand what '{user_input}' is asking for.",
            confidence=0.9,
            reasoning="Identifying the core question and requirements"
        ))
        
        # Step 2: Break into components
        components = self._extract_components(user_input)
        steps.append(ReasoningStep(
            step_type="analysis",
            content=f"I can break this down into: {', '.join(components)}",
            confidence=0.8,
            reasoning="Breaking complex question into manageable parts"
        ))
        
        # Step 3: Apply knowledge
        steps.append(ReasoningStep(
            step_type="knowledge_application",
            content="Now I'll apply my knowledge to each component systematically.",
            confidence=0.7,
            reasoning="Using stored knowledge and reasoning patterns"
        ))
        
        # Step 4: Synthesize
        steps.append(ReasoningStep(
            step_type="synthesis",
            content="Combining all the information into a coherent answer.",
            confidence=0.8,
            reasoning="Integrating multiple pieces of information"
        ))
        
        return steps
    
    def _breakdown_reasoning(self, user_input: str, context: str) -> list[ReasoningStep]:
        """Break down complex problems into parts"""
        steps = []
        
        # Identify key elements
        elements = self._identify_elements(user_input)
        
        for i, element in enumerate(elements, 1):
            steps.append(ReasoningStep(
                step_type="element_analysis",
                content=f"Element {i}: {element} - This requires specific consideration.",
                confidence=0.7,
                reasoning=f"Analyzing component {i} of the problem"
            ))
        
        return steps
    
    def _analogy_reasoning(self, user_input: str, context: str) -> list[ReasoningStep]:
        """Use analogies and comparisons"""
        steps = []
        
        # Find similar concepts
        analogy = self._find_analogy(user_input)
        
        steps.append(ReasoningStep(
            step_type="analogy",
            content=f"This reminds me of: {analogy}",
            confidence=0.6,
            reasoning="Using analogical reasoning to understand the concept"
        ))
        
        return steps
    
    def _logical_chain_reasoning(self, user_input: str, context: str) -> list[ReasoningStep]:
        """Logical chain of reasoning"""
        steps = []
        
        # Build logical chain
        logical_steps = self._build_logical_chain(user_input)
        
        for i, step in enumerate(logical_steps, 1):
            steps.append(ReasoningStep(
                step_type="logical_step",
                content=f"Step {i}: {step}",
                confidence=0.8,
                reasoning=f"Logical reasoning step {i}"
            ))
        
        return steps
    
    def _creative_solution_reasoning(self, user_input: str, context: str) -> list[ReasoningStep]:
        """Creative problem-solving approach"""
        steps = []
        
        # Generate creative approaches
        approaches = self._generate_creative_approaches(user_input)
        
        for approach in approaches:
            steps.append(ReasoningStep(
                step_type="creative_approach",
                content=f"Creative approach: {approach}",
                confidence=0.6,
                reasoning="Exploring creative solutions"
            ))
        
        return steps
    
    def _extract_components(self, text: str) -> list[str]:
        """Extract key components from text"""
        # Simple component extraction
        words = text.split()
        components = []
        
        # Look for question words and key nouns
        question_words = ["what", "how", "why", "when", "where", "who"]
        for word in words:
            if word.lower() in question_words or len(word) > 4:
                components.append(word)
        
        return components[:3]  # Limit to 3 components
    
    def _identify_elements(self, text: str) -> list[str]:
        """Identify key elements in the problem"""
        # Simple element identification
        return [text[:len(text)//2], text[len(text)//2:]]
    
    def _find_analogy(self, text: str) -> str:
        """Find an analogy for the given text"""
        analogies = [
            "learning to ride a bicycle",
            "cooking a complex recipe",
            "solving a puzzle",
            "building with blocks",
            "navigating a maze"
        ]
        return random.choice(analogies)
    
    def _build_logical_chain(self, text: str) -> list[str]:
        """Build a logical chain of reasoning"""
        return [
            "If we consider the basic principles...",
            "Then we can apply the relevant knowledge...",
            "This leads us to the conclusion that..."
        ]
    
    def _generate_creative_approaches(self, text: str) -> list[str]:
        """Generate creative approaches to the problem"""
        return [
            "Think about this from a different angle...",
            "What if we approach this creatively...",
            "Consider alternative perspectives..."
        ]
    
    def _synthesize_answer(self, steps: list[ReasoningStep], user_input: str) -> str:
        """Synthesize the final answer from reasoning steps"""
        if not steps:
            return "I need to think about this more carefully."
        
        # Combine the reasoning into a coherent answer
        synthesis = f"Let me think through this step by step:\n\n"
        
        for i, step in enumerate(steps, 1):
            synthesis += f"{i}. {step.content}\n"
        
        synthesis += f"\nBased on this reasoning, here's my answer to '{user_input}': "
        synthesis += "I've analyzed this systematically and considered multiple approaches. "
        synthesis += "The most logical conclusion is that this requires careful consideration "
        synthesis += "of all the factors we've discussed."
        
        return synthesis

# Global advanced reasoner instance
advanced_reasoner = AdvancedReasoner()

class Reasoner:
    def __init__(self, facts, use_semantic, model=None):
        self.facts = facts
        self.use_semantic = use_semantic
        self.model = model
    def find_relevant_facts(self, user_input, top_n=5):
        # Use semantic similarity if available, else keyword overlap
        if self.use_semantic and self.model:
            user_emb = self.model.encode([user_input], convert_to_tensor=True)
            fact_embs = self.model.encode(self.facts, convert_to_tensor=True)
            sims = util.cos_sim(user_emb, fact_embs)[0].cpu().numpy()
            top_idx = sims.argsort()[-top_n:][::-1]
            return [(self.facts[i], float(sims[i])) for i in top_idx if sims[i] > 0.2]
        else:
            input_keywords = get_keywords(user_input)
            scored = []
            for fact in self.facts:
                fact_keywords = get_keywords(fact)
                score = centrality_score(input_keywords, fact_keywords)
                if score > 0.1:
                    scored.append((fact, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_n]
    def synthesize_answer(self, user_input, relevant_facts):
        # Combine facts into a coherent answer
        if not relevant_facts:
            return None, []
        reasoning = []
        combined = []
        for fact, score in relevant_facts:
            reasoning.append(f"Used fact: '{fact}' (score: {score:.2f})")
            combined.append(fact)
        # Simple synthesis: join facts, remove duplicates
        answer = ' '.join(dict.fromkeys(itertools.chain.from_iterable(f.split('. ') for f in combined)))
        return answer, reasoning

system_prompt = (
    "You are EchoBot, an AI assistant with advanced reasoning capabilities similar to DeepSeek R1. "
    "You can think step-by-step, use multiple reasoning strategies, and generate sophisticated responses. "
    "Today is {date}. Use your knowledge base, session memory, and advanced reasoning to answer."
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
    session_memory_obj.add(user, bot)

def summarize_session_memory():
    return session_memory_obj.summary()

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

def show_enhanced_reasoning_chain(user_input, reasoning_steps, final_answer):
    """Show enhanced reasoning chain like DeepSeek R1"""
    chain = [f"🤖 EchoBot Advanced Reasoning Chain"]
    chain.append(f"System: {system_prompt.format(date=datetime.datetime.now().strftime('%Y-%m-%d'))}")
    
    mem_summary = summarize_session_memory()
    if mem_summary:
        chain.append("📚 Recent Context:")
        chain.append(mem_summary)
    
    chain.append(f"👤 {user_persona['name']}: {user_input}")
    chain.append("🧠 Step-by-Step Reasoning:")
    
    for i, step in enumerate(reasoning_steps, 1):
        chain.append(f"  {i}. [{step.step_type.upper()}] {step.content}")
        chain.append(f"     Confidence: {step.confidence:.2f} | {step.reasoning}")
    
    chain.append("💡 Final Answer:")
    chain.append(final_answer)
    
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
            Knowledge_Base.add_fact(f"{phrase} means {meaning}", category="paraphrase", source="conversation")
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
    all_facts = Knowledge_Base.get_all_facts()
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
    if nlp is not None:
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
        Knowledge_Base.add_fact(f"{normalize_text(inp)}|||{resp}", category="starter_pair", source="batch_train")

def format_deepseek_r1_trace(reasoning_steps, final_answer):
    trace = "<think>\n"
    for i, step in enumerate(reasoning_steps, 1):
        trace += f"Step {i}: {step.content}\n"
    trace += f"\nReflection: Let's check if the answer makes sense.\n"
    trace += "</think>\n"
    trace += f"\nFinal Answer: {final_answer}"
    return trace

def get_reply(user_input, show_trace=True):
    print("get_reply called with:", user_input)
    user_input = user_input.strip()
    lower_input = user_input.lower()
    norm_input = normalize_text(user_input)
    thinking_msg = "🧠 EchoBot is thinking..."
    
    # Special case: self-introduction
    if norm_input in ["who are you", "what are you", "who is echobot", "what is echobot", "tell me about yourself"]:
        # Use advanced reasoning for self-introduction
        context = session_memory_obj.get_context()
        final_answer, reasoning_steps = advanced_reasoner.generate_reasoning_chain(
            "Tell me about yourself and your capabilities", context
        )
        session_memory_obj.add(user_input, final_answer)
        return format_deepseek_r1_trace(reasoning_steps, final_answer)
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
            user_input = corrected
            lower_input = user_input.lower()
    # Sentiment analysis
    sentiment = pattern_analyzer.analyze(user_input)[0]
    if sentiment < -0.5:
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
            if session_memory_obj.memory:
                last_user_input = session_memory_obj.memory[-1][0]
                Knowledge_Base.add_fact(f"{normalize_text(last_user_input)}|||{lesson}", category="starter_pair", source="teach_me_mode")
            Knowledge_Base.add_fact(lesson, category="user_lesson", source="teach_me_mode")
            reply = f"Thank you for teaching me! I've added this to my knowledge: '{lesson}'"
            session_memory_obj.add(user_input, reply)
            return reply
        else:
            return "Please provide a more detailed lesson."
    # --- DeepSeek R1-style advanced reasoning ---
    context = session_memory_obj.get_context()
    final_answer, reasoning_steps = advanced_reasoner.generate_reasoning_chain(user_input, context)
    
    # Try to find relevant facts to enhance the reasoning
    user_keywords = extract_keywords(user_input)
    all_facts = Knowledge_Base.get_relevant_facts(user_keywords, limit=100)
    reasoner = Reasoner(all_facts, use_transformers, model if use_transformers else None)
    relevant_facts = reasoner.find_relevant_facts(user_input, top_n=3)
    
    # Enhance the answer with relevant facts if available
    if relevant_facts and len(relevant_facts) > 0:
        fact_info = "Based on my knowledge: " + " ".join([f[0] for f in relevant_facts[:2]])
        final_answer = final_answer + "\n\n" + fact_info
    
    session_memory_obj.add(user_input, final_answer)
    
    return format_deepseek_r1_trace(reasoning_steps, final_answer)

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

def is_greeting(text):
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "what's up", "howdy", "yo"]
    return any(g in text.lower() for g in greetings)

# Add a helper to extract keywords from user input (place after imports, before main logic)
STOPWORDS = set(['the','be','to','of','and','a','in','that','have','i','it','for','not','on','with','he','as','you','do','at','this','but','his','by','from','they','we','say','her','she','or','an','will','my','one','all','would','there','their','what','so','up','out','if','about','who','get','which','go','me','when','make','can','like','time','no','just','him','know','take','people','into','year','your','good','some','could','them','see','other','than','then','now','look','only','come','its','over','think','also','back','after','use','two','how','our','work','first','well','way','even','new','want','because','any','these','give','day','most','us'])
def extract_keywords(text):
    words = [w.lower() for w in text.split() if w.isalpha() and w.lower() not in STOPWORDS]
    return words[:5]  # limit to top 5 keywords for efficiency

if __name__ == "__main__":
    # Uncomment the next line to clear the knowledge base on startup
    # clear_knowledge_base()
    set_user_persona()
    console.print(f"[bold green]{system_prompt.format(date=datetime.datetime.now().strftime('%Y-%m-%d'))}[/bold green]")
    show_trace = True
    while True:
        user_input = Prompt.ask(f"[bold cyan]{user_persona['name']}[/bold cyan]").strip()
        if user_input.lower() in ["quit", "exit", "bye"]:
            console.print("[bold yellow]EchoBot: Shutting down...[/bold yellow]")
            break
        if user_input.lower() == "clear memory":
            session_memory_obj.clear()
            console.print("[bold green]Session memory cleared.[/bold green]")
            continue
        if user_input.lower() == "toggle trace":
            show_trace = not show_trace
            console.print(f"[bold green]Reasoning trace is now {'ON' if show_trace else 'OFF'}.[/bold green]")
            continue
        print(f"User input: {user_input}")
        reply = get_reply(user_input, show_trace=show_trace)
        print(f"EchoBot reply: {reply}")
        console.print(f"[bold magenta]EchoBot:[/bold magenta] {reply}")
        save_chat_log(user_input, reply)
        feedback = Prompt.ask("[bold cyan]Was this answer helpful? (yes/no/teach me)[/bold cyan]").strip().lower()
        if feedback in ["no", "teach me"]:
            correction = Prompt.ask("[bold cyan]Please provide the correct answer or more info so I can learn:[/bold cyan]")
            if correction and len(correction) > 3:
                Knowledge_Base.add_fact(f"{normalize_text(user_input)}|||{correction}", category="user_feedback", source="feedback_loop")
                console.print("[bold green]Thank you! I've learned from your feedback.[/bold green]")
