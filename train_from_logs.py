import json
import os

MEMORY_FILE = "memory.json"
DEFAULT_CHAT_LOG_FILE = "chat_log.txt"

def train_from_log(log_file):
    try:
        with open(MEMORY_FILE, "r") as f:
            memory = json.load(f)
    except FileNotFoundError:
        memory = {}

    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"No {log_file} found.")
        return

    for i in range(len(lines) - 1):
        if lines[i].startswith("You: ") and lines[i+1].startswith("EchoBot: "):
            question = lines[i][5:].strip().lower()
            answer = lines[i+1][9:].strip()
            if question not in memory:
                memory[question] = [answer]

    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

    print(f"Training complete! Learned from {log_file}.")

def main():
    print("Welcome to EchoBot Trainer!")
    print("You can train EchoBot from chat log files.")
    print("Press Enter to use default 'chat_log.txt', or type a filename:")

    filename = input("Log file: ").strip()
    if filename == "":
        filename = DEFAULT_CHAT_LOG_FILE

    if not os.path.exists(filename):
        print(f"File '{filename}' does not exist. Please put your chat log file in the folder.")
        return

    train_from_log(filename)

if __name__ == "__main__":
    main()
