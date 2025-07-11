import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
from tkinter import font as tkfont
import threading
import time
import EchoBot
from datetime import datetime
import json

class ModernEchoBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 EchoBot AI Assistant")
        self.root.geometry("800x700")
        self.root.configure(bg="#1a1a1a")
        self.root.minsize(600, 500)
        
        # Variables
        self.show_reasoning = tk.BooleanVar(value=True)
        self.typing_speed = tk.DoubleVar(value=0.02)
        self.auto_scroll = tk.BooleanVar(value=True)
        
        # Configure styles
        self.setup_styles()
        
        # Create main container
        self.main_frame = tk.Frame(root, bg="#1a1a1a")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create header
        self.create_header()
        
        # Create chat area
        self.create_chat_area()
        
        # Create input area
        self.create_input_area()
        
        # Create control panel
        self.create_control_panel()
        
        # Create status bar
        self.create_status_bar()
        
        # Initialize chat
        self.chat_history = []
        self.message_count = 0
        
        # Welcome message
        self.display_welcome_message()
        
        # Bind keyboard shortcuts
        self.bind_shortcuts()

    def setup_styles(self):
        """Configure modern styles for the GUI"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Modern.TFrame', background='#2d2d2d')
        style.configure('Modern.TButton', 
                      background='#00adb5', 
                      foreground='white',
                      borderwidth=0,
                      focuscolor='none')
        style.map('Modern.TButton',
                 background=[('active', '#00848a')])
        
        style.configure('Control.TButton',
                      background='#393e46',
                      foreground='white',
                      borderwidth=0)
        style.map('Control.TButton',
                 background=[('active', '#4a4f57')])

    def create_header(self):
        """Create the header section"""
        header_frame = tk.Frame(self.main_frame, bg="#2d2d2d", height=60)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(header_frame, 
                              text="🤖 EchoBot AI Assistant", 
                              font=("Segoe UI", 18, "bold"),
                              bg="#2d2d2d", 
                              fg="#00adb5")
        title_label.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Status indicator
        self.status_label = tk.Label(header_frame, 
                                   text="● Ready", 
                                   font=("Segoe UI", 10),
                                   bg="#2d2d2d", 
                                   fg="#00ff00")
        self.status_label.pack(side=tk.RIGHT, padx=20, pady=15)

    def create_chat_area(self):
        """Create the main chat area"""
        chat_frame = tk.Frame(self.main_frame, bg="#2d2d2d")
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Chat area with modern styling
        self.chat_area = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            bg="#2d2d2d",
            fg="#ffffff",
            insertbackground="#ffffff",
            selectbackground="#00adb5",
            borderwidth=0,
            relief="flat",
            padx=15,
            pady=15
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for different message types
        self.chat_area.tag_config("user", 
                                 foreground="#00adb5", 
                                 font=("Segoe UI", 11, "bold"),
                                 spacing1=10, spacing3=10)
        self.chat_area.tag_config("bot", 
                                 foreground="#ffd369", 
                                 font=("Segoe UI", 11),
                                 spacing1=10, spacing3=10)
        self.chat_area.tag_config("bot_thinking", 
                                 foreground="#aaaaaa", 
                                 font=("Segoe UI", 11, "italic"),
                                 spacing1=5, spacing3=5)
        self.chat_area.tag_config("system", 
                                 foreground="#ff6b6b", 
                                 font=("Segoe UI", 10, "italic"),
                                 spacing1=5, spacing3=5)
        self.chat_area.tag_config("reasoning", 
                                 foreground="#95e1d3", 
                                 font=("Segoe UI", 10),
                                 spacing1=5, spacing3=5)

    def create_input_area(self):
        """Create the input area"""
        input_frame = tk.Frame(self.main_frame, bg="#2d2d2d")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Input field
        self.user_input = tk.Entry(
            input_frame,
            font=("Segoe UI", 12),
            bg="#393e46",
            fg="#ffffff",
            insertbackground="#ffffff",
            relief="flat",
            borderwidth=0
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", self.send_message)
        
        # Send button
        self.send_button = tk.Button(
            input_frame,
            text="Send",
            font=("Segoe UI", 11, "bold"),
            bg="#00adb5",
            fg="#ffffff",
            relief="flat",
            borderwidth=0,
            command=self.send_message,
            cursor="hand2"
        )
        self.send_button.pack(side=tk.RIGHT)
        
        # Clear button
        self.clear_button = tk.Button(
            input_frame,
            text="Clear",
            font=("Segoe UI", 11),
            bg="#393e46",
            fg="#ffffff",
            relief="flat",
            borderwidth=0,
            command=self.clear_chat,
            cursor="hand2"
        )
        self.clear_button.pack(side=tk.RIGHT, padx=(0, 10))

    def create_control_panel(self):
        """Create the control panel"""
        control_frame = tk.Frame(self.main_frame, bg="#2d2d2d")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left side controls
        left_controls = tk.Frame(control_frame, bg="#2d2d2d")
        left_controls.pack(side=tk.LEFT)
        
        # Reasoning toggle
        self.reasoning_check = tk.Checkbutton(
            left_controls,
            text="Show Reasoning",
            variable=self.show_reasoning,
            font=("Segoe UI", 10),
            bg="#2d2d2d",
            fg="#ffffff",
            selectcolor="#00adb5",
            activebackground="#2d2d2d",
            activeforeground="#ffffff"
        )
        self.reasoning_check.pack(side=tk.LEFT, padx=(0, 15))
        
        # Auto-scroll toggle
        self.scroll_check = tk.Checkbutton(
            left_controls,
            text="Auto-scroll",
            variable=self.auto_scroll,
            font=("Segoe UI", 10),
            bg="#2d2d2d",
            fg="#ffffff",
            selectcolor="#00adb5",
            activebackground="#2d2d2d",
            activeforeground="#ffffff"
        )
        self.scroll_check.pack(side=tk.LEFT, padx=(0, 15))
        
        # Right side controls
        right_controls = tk.Frame(control_frame, bg="#2d2d2d")
        right_controls.pack(side=tk.RIGHT)
        
        # Memory clear button
        self.memory_button = tk.Button(
            right_controls,
            text="Clear Memory",
            font=("Segoe UI", 10),
            bg="#ff6b6b",
            fg="#ffffff",
            relief="flat",
            borderwidth=0,
            command=self.clear_memory,
            cursor="hand2"
        )
        self.memory_button.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Settings button
        self.settings_button = tk.Button(
            right_controls,
            text="⚙️ Settings",
            font=("Segoe UI", 10),
            bg="#393e46",
            fg="#ffffff",
            relief="flat",
            borderwidth=0,
            command=self.show_settings,
            cursor="hand2"
        )
        self.settings_button.pack(side=tk.RIGHT)

    def create_status_bar(self):
        """Create the status bar"""
        status_frame = tk.Frame(self.main_frame, bg="#1a1a1a")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Message counter
        self.message_counter = tk.Label(
            status_frame,
            text="Messages: 0",
            font=("Segoe UI", 9),
            bg="#1a1a1a",
            fg="#888888"
        )
        self.message_counter.pack(side=tk.LEFT)
        
        # Current time
        self.time_label = tk.Label(
            status_frame,
            text="",
            font=("Segoe UI", 9),
            bg="#1a1a1a",
            fg="#888888"
        )
        self.time_label.pack(side=tk.RIGHT)
        
        # Update time
        self.update_time()

    def display_welcome_message(self):
        """Display welcome message"""
        welcome_msg = """🤖 Welcome to EchoBot AI Assistant!

I'm your AI companion with advanced reasoning capabilities similar to DeepSeek R1. I can:

• Think step-by-step through complex problems
• Use multiple reasoning strategies
• Learn from our conversations
• Help you with information and ideas

Type your message below and let's start chatting!

---
"""
        self.display_message(welcome_msg, message_type="system")

    def send_message(self, event=None):
        """Send user message"""
        user_text = self.user_input.get().strip()
        if not user_text:
            return
        
        # Display user message
        timestamp = datetime.now().strftime("%H:%M")
        self.display_message(f"[{timestamp}] You: {user_text}\n", message_type="user")
        
        # Clear input
        self.user_input.delete(0, tk.END)
        
        # Update status
        self.status_label.config(text="● Thinking...", fg="#ffaa00")
        
        # Start bot response in thread
        threading.Thread(target=self.get_bot_reply, args=(user_text,), daemon=True).start()

    def get_bot_reply(self, user_text):
        """Get bot reply with enhanced features"""
        try:
            # Show thinking message
            self.display_message("🤔 EchoBot is thinking...\n", message_type="bot_thinking")
            
            # Get reply with reasoning
            reply = EchoBot.get_reply(user_text, show_trace=self.show_reasoning.get())
            
            # Remove thinking message
            self.remove_thinking_message()
            
            # Display reply with typing effect
            timestamp = datetime.now().strftime("%H:%M")
            self.type_out_message(f"[{timestamp}] EchoBot: {reply}\n", message_type="bot")
            
            # Save to chat history
            self.chat_history.append({
                "timestamp": timestamp,
                "user": user_text,
                "bot": reply,
                "type": "conversation"
            })
            
            # Update message counter
            self.message_count += 1
            self.message_counter.config(text=f"Messages: {self.message_count}")
            
            # Update status
            self.status_label.config(text="● Ready", fg="#00ff00")
            
            # Save chat log
            try:
                EchoBot.save_chat_log(user_text, reply)
            except Exception as e:
                self.display_message(f"[Warning] Could not save chat: {e}\n", message_type="system")
                
        except Exception as e:
            self.remove_thinking_message()
            self.display_message(f"[Error] EchoBot failed: {e}\n", message_type="system")
            self.status_label.config(text="● Error", fg="#ff0000")

    def display_message(self, message, message_type="bot"):
        """Display message in chat area"""
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, message, message_type)
        
        if self.auto_scroll.get():
            self.chat_area.see(tk.END)
        
        self.chat_area.config(state='disabled')

    def type_out_message(self, message, message_type="bot"):
        """Type out message with effect"""
        self.chat_area.config(state='normal')
        
        # Remove the "EchoBot is thinking..." message
        self.remove_thinking_message()
        
        # Type out the message
        for char in message:
            self.chat_area.insert(tk.END, char, message_type)
            if self.auto_scroll.get():
                self.chat_area.see(tk.END)
            self.chat_area.update()
            time.sleep(self.typing_speed.get())
        
        self.chat_area.config(state='disabled')

    def remove_thinking_message(self):
        """Remove thinking message"""
        self.chat_area.config(state='normal')
        content = self.chat_area.get("1.0", tk.END)
        lines = content.splitlines()
        
        # Remove the last occurrence of thinking message
        for i in range(len(lines)-1, -1, -1):
            if "EchoBot is thinking..." in lines[i]:
                del lines[i]
                break
        
        new_content = "\n".join(lines) + "\n"
        self.chat_area.delete("1.0", tk.END)
        self.chat_area.insert(tk.END, new_content)
        self.chat_area.config(state='disabled')

    def clear_chat(self):
        """Clear chat area"""
        if messagebox.askyesno("Clear Chat", "Are you sure you want to clear the chat?"):
            self.chat_area.config(state='normal')
            self.chat_area.delete("1.0", tk.END)
            self.chat_area.config(state='disabled')
            self.message_count = 0
            self.message_counter.config(text="Messages: 0")
            self.chat_history.clear()
            self.display_welcome_message()

    def clear_memory(self):
        """Clear bot memory"""
        if messagebox.askyesno("Clear Memory", "Are you sure you want to clear EchoBot's memory?"):
            try:
                EchoBot.session_memory_obj.clear()
                self.display_message("[System] EchoBot memory cleared successfully.\n", message_type="system")
            except Exception as e:
                self.display_message(f"[Error] Failed to clear memory: {e}\n", message_type="system")

    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.configure(bg="#2d2d2d")
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Settings content
        tk.Label(settings_window, text="EchoBot Settings", 
                font=("Segoe UI", 16, "bold"), 
                bg="#2d2d2d", fg="#00adb5").pack(pady=20)
        
        # Typing speed slider
        tk.Label(settings_window, text="Typing Speed:", 
                font=("Segoe UI", 11), 
                bg="#2d2d2d", fg="#ffffff").pack()
        
        speed_frame = tk.Frame(settings_window, bg="#2d2d2d")
        speed_frame.pack(pady=10)
        
        speed_slider = tk.Scale(speed_frame, from_=0.01, to=0.1, 
                               variable=self.typing_speed, 
                               orient=tk.HORIZONTAL,
                               bg="#2d2d2d", fg="#ffffff",
                               highlightbackground="#2d2d2d")
        speed_slider.pack()
        
        # Close button
        tk.Button(settings_window, text="Close", 
                 command=settings_window.destroy,
                 bg="#00adb5", fg="#ffffff",
                 font=("Segoe UI", 11),
                 relief="flat", borderwidth=0).pack(pady=20)

    def update_time(self):
        """Update time in status bar"""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)

    def bind_shortcuts(self):
        """Bind keyboard shortcuts"""
        self.root.bind('<Control-Return>', self.send_message)
        self.root.bind('<Control-l>', lambda e: self.clear_chat())
        self.root.bind('<Control-m>', lambda e: self.clear_memory())
        self.root.bind('<Control-r>', lambda e: self.show_reasoning.set(not self.show_reasoning.get()))

if __name__ == "__main__":
    root = tk.Tk()
    gui = ModernEchoBotGUI(root)
    root.mainloop()
