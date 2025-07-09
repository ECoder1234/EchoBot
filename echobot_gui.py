import tkinter as tk
from tkinter import scrolledtext
import threading
import time
import EchoBot

class EchoBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EchoBot AI")
        self.root.geometry("500x600")
        self.root.configure(bg="#222831")

        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Segoe UI", 12), bg="#393e46", fg="#eeeeee", state='disabled', height=25)
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.entry_frame = tk.Frame(root, bg="#222831")
        self.entry_frame.pack(fill=tk.X, padx=10, pady=10)

        self.user_input = tk.Entry(self.entry_frame, font=("Segoe UI", 12), bg="#eeeeee", fg="#222831")
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", self.send_message)

        self.send_button = tk.Button(self.entry_frame, text="Send", font=("Segoe UI", 12, "bold"), bg="#00adb5", fg="#eeeeee", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)

        self.root.bind('<Control-Return>', self.send_message)

    def send_message(self, event=None):
        user_text = self.user_input.get().strip()
        if not user_text:
            return
        self.display_message(f"You: {user_text}\n", user=True)
        self.user_input.delete(0, tk.END)
        threading.Thread(target=self.get_bot_reply, args=(user_text,)).start()

    def get_bot_reply(self, user_text):
        self.display_message("EchoBot is thinking...\n", bot=True, thinking=True)
        time.sleep(0.5)
        try:
            reply = EchoBot.get_reply(user_text)
        except Exception as e:
            self.remove_thinking_message()
            self.display_message(f"[Error] EchoBot failed: {e}\n", bot=True)
            return
        self.remove_thinking_message()
        self.type_out_message(f"EchoBot: {reply}\n", bot=True)
        try:
            EchoBot.save_chat_log(user_text, reply)
            EchoBot.learn_from_conversation(user_text, reply)
        except Exception as e:
            self.display_message(f"[Warning] Could not save chat: {e}\n", bot=True)

    def display_message(self, message, user=False, bot=False, thinking=False):
        self.chat_area.config(state='normal')
        tag = "user" if user else ("bot_thinking" if thinking else "bot")
        self.chat_area.insert(tk.END, message, tag)
        self.chat_area.see(tk.END)
        self.chat_area.config(state='disabled')

    def type_out_message(self, message, bot=False):
        self.chat_area.config(state='normal')
        tag = "bot"
        for char in message:
            self.chat_area.insert(tk.END, char, tag)
            self.chat_area.see(tk.END)
            self.chat_area.update()
            time.sleep(0.02)  # Typing speed (seconds per character)
        self.chat_area.config(state='disabled')

    def remove_thinking_message(self):
        self.chat_area.config(state='normal')
        content = self.chat_area.get("1.0", tk.END)
        lines = content.splitlines()
        # Remove the last occurrence of 'EchoBot is thinking...'
        for i in range(len(lines)-1, -1, -1):
            if lines[i].strip() == "EchoBot is thinking...":
                del lines[i]
                break
        new_content = "\n".join(lines) + "\n"
        self.chat_area.delete("1.0", tk.END)
        self.chat_area.insert(tk.END, new_content)
        self.chat_area.config(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    gui = EchoBotGUI(root)
    # Tag styles
    gui.chat_area.tag_config("user", foreground="#00adb5", font=("Segoe UI", 12, "bold"))
    gui.chat_area.tag_config("bot", foreground="#ffd369", font=("Segoe UI", 12))
    gui.chat_area.tag_config("bot_thinking", foreground="#aaaaaa", font=("Segoe UI", 12, "italic"))
    root.mainloop()
