import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
from typing import List, Dict, Optional
import re
import random

class NeuralResponseGenerator:
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        """Initialize neural response generator with a small model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.generator = pipeline('text-generation', 
                                    model=self.model, 
                                    tokenizer=self.tokenizer,
                                    max_length=100,
                                    do_sample=True,
                                    temperature=0.7,
                                    top_p=0.9)
            
            self.use_neural = True
            print(f"✅ Neural response generator loaded with {model_name}")
            
        except Exception as e:
            print(f"⚠️ Neural model not available: {e}")
            self.use_neural = False
            self.generator = None
    
    def generate_response(self, user_input: str, context: str = "", 
                         reasoning_steps: List[str] = None) -> str:
        """Generate a neural response based on input and context"""
        
        if not self.use_neural:
            return self._fallback_response(user_input, context)
        
        try:
            # Build prompt with context and reasoning
            prompt = self._build_prompt(user_input, context, reasoning_steps)
            
            # Generate response
            response = self.generator(prompt, max_length=len(prompt.split()) + 50)[0]['generated_text']
            
            # Extract only the new part
            new_text = response[len(prompt):].strip()
            
            if new_text:
                return new_text
            else:
                return self._fallback_response(user_input, context)
                
        except Exception as e:
            print(f"Neural generation failed: {e}")
            return self._fallback_response(user_input, context)
    
    def _build_prompt(self, user_input: str, context: str, reasoning_steps: List[str] = None) -> str:
        """Build a prompt for the neural model"""
        prompt_parts = []
        
        # Add context if available
        if context:
            prompt_parts.append(f"Context: {context}")
        
        # Add reasoning steps if available
        if reasoning_steps:
            prompt_parts.append("Reasoning:")
            for i, step in enumerate(reasoning_steps, 1):
                prompt_parts.append(f"{i}. {step}")
        
        # Add user input
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def _fallback_response(self, user_input: str, context: str) -> str:
        """Fallback response when neural generation fails"""
        responses = [
            "I understand your question. Let me think about this carefully.",
            "That's an interesting point. I'll analyze this step by step.",
            "I see what you're asking. Let me break this down for you.",
            "Great question! I'll approach this systematically.",
            "I appreciate your question. Let me think through this."
        ]
        return random.choice(responses)

class TemplateBasedGenerator:
    """Template-based response generator as backup"""
    
    def __init__(self):
        self.templates = {
            "greeting": [
                "Hello! How can I help you today?",
                "Hi there! What would you like to explore?",
                "Greetings! I'm ready to assist you."
            ],
            "question": [
                "That's a great question. Let me think about this...",
                "I'll analyze this step by step for you.",
                "Let me break this down systematically."
            ],
            "explanation": [
                "Here's what I understand about this:",
                "Let me explain this clearly:",
                "Based on my analysis:"
            ],
            "thinking": [
                "Let me think through this carefully...",
                "I'm processing this information...",
                "Analyzing the components..."
            ]
        }
    
    def generate(self, response_type: str = "question") -> str:
        """Generate a template-based response"""
        if response_type in self.templates:
            return random.choice(self.templates[response_type])
        return random.choice(self.templates["question"])

# Global instances
neural_generator = NeuralResponseGenerator()
template_generator = TemplateBasedGenerator()

def generate_enhanced_response(user_input: str, context: str = "", 
                             reasoning_steps: List[str] = None) -> str:
    """Generate an enhanced response using neural generation or templates"""
    
    # Try neural generation first
    if neural_generator.use_neural:
        response = neural_generator.generate_response(user_input, context, reasoning_steps)
        if response and len(response) > 10:
            return response
    
    # Fallback to template-based generation
    return template_generator.generate("question") 