import sqlite3
import os
from datetime import datetime

DB_FILE = 'knowledge_base.db'

# Initialize the database and create the facts table if it doesn't exist
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

# Add a fact to the database (skip duplicates)
def add_fact(fact, category=None, source=None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    date_added = datetime.now().strftime('%Y-%m-%d')
    try:
        c.execute('''
            INSERT OR IGNORE INTO facts (fact, category, source, date_added)
            VALUES (?, ?, ?, ?)
        ''', (fact, category, source, date_added))
        conn.commit()
    except Exception as e:
        print(f"Error adding fact: {e}")
    conn.close()

# Search for facts by keyword
def search_facts(keyword):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        SELECT fact FROM facts WHERE fact LIKE ?
    ''', (f'%{keyword}%',))
    results = [row[0] for row in c.fetchall()]
    conn.close()
    return results

# Get all facts
def get_all_facts():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT fact FROM facts')
    results = [row[0] for row in c.fetchall()]
    conn.close()
    return results

if __name__ == '__main__':
    init_db() 