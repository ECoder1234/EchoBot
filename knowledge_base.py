import sqlite3
import os
from datetime import datetime

DB_FILE = 'knowledge_base.db'

# ✅ Create DB and table if it doesn't exist
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT UNIQUE,
            category TEXT,
            source TEXT,
            date_added TEXT
        )
    ''')
    conn.commit()
    conn.close()

# ✅ Add a new fact (ignores duplicates)
def add_fact(fact, category=None, source=None):
    if not fact.strip():
        return  # ignore empty
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    date_added = datetime.now().strftime('%Y-%m-%d')
    try:
        c.execute('''
            INSERT OR IGNORE INTO facts (fact, category, source, date_added)
            VALUES (?, ?, ?, ?)
        ''', (fact.strip(), category, source, date_added))
        conn.commit()
    except sqlite3.Error as e:
        print(f"[DB Error] Failed to add fact: {e}")
    finally:
        conn.close()

# ✅ Search facts by keyword
def search_facts(keyword):
    if not keyword.strip():
        return []
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT fact FROM facts WHERE fact LIKE ?', (f'%{keyword.strip()}%',))
    results = [row[0] for row in c.fetchall()]
    conn.close()
    return results

# ✅ Get all known facts
def get_all_facts():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT fact FROM facts')
    results = [row[0] for row in c.fetchall()]
    conn.close()
    return results

# Efficiently get relevant facts by keywords

def get_relevant_facts(keywords, limit=100):
    if not keywords:
        return []
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Build WHERE clause for multiple keywords
    where = " OR ".join(["fact LIKE ?" for _ in keywords])
    params = [f"%{kw}%" for kw in keywords]
    c.execute(f'SELECT fact FROM facts WHERE {where} LIMIT ?', (*params, limit))
    results = [row[0] for row in c.fetchall()]
    conn.close()
    return results

# ✅ Optional: run manually
if __name__ == '__main__':
    print("📚 Initializing knowledge base...")
    init_db()
    print("✅ Database ready.")
