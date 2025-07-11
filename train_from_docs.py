import sys
import os
from knowledge_base import init_db, add_fact
import re

# Usage: python train_from_docs.py file1.txt [file2.txt ...]

def extract_facts_from_text(text):
    # Split text into sentences (simple split on period, can be improved)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    facts = []
    for s in sentences:
        s = s.strip()
        # Only keep sentences that look like facts (not questions or commands)
        if s and not s.endswith('?') and not s.lower().startswith(('please', 'do ', 'can you', 'could you', 'would you', 'should you', 'let us', "let's")):
            facts.append(s)
    return facts

def main():
    if len(sys.argv) < 2:
        print('Usage: python train_from_docs.py file1.txt [file2.txt ...]')
        return
    init_db()
    for filename in sys.argv[1:]:
        if not os.path.isfile(filename):
            print(f'Skipping {filename}: not a file')
            continue
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        facts = extract_facts_from_text(text)
        for fact in facts:
            add_fact(fact, category=None, source=os.path.basename(filename))
        print(f'Added {len(facts)} facts from {filename}')

if __name__ == '__main__':
    main() 