import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random

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
        
    def generate_reasoning_chain(self, user_input: str, context: str = "") -> Tuple[str, List[ReasoningStep]]:
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
    
    def _step_by_step_reasoning(self, user_input: str, context: str) -> List[ReasoningStep]:
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
    
    def _breakdown_reasoning(self, user_input: str, context: str) -> List[ReasoningStep]:
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
    
    def _analogy_reasoning(self, user_input: str, context: str) -> List[ReasoningStep]:
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
    
    def _logical_chain_reasoning(self, user_input: str, context: str) -> List[ReasoningStep]:
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
    
    def _creative_solution_reasoning(self, user_input: str, context: str) -> List[ReasoningStep]:
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
    
    def _extract_components(self, text: str) -> List[str]:
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
    
    def _identify_elements(self, text: str) -> List[str]:
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
    
    def _build_logical_chain(self, text: str) -> List[str]:
        """Build a logical chain of reasoning"""
        return [
            "If we consider the basic principles...",
            "Then we can apply the relevant knowledge...",
            "This leads us to the conclusion that..."
        ]
    
    def _generate_creative_approaches(self, text: str) -> List[str]:
        """Generate creative approaches to the problem"""
        return [
            "Think about this from a different angle...",
            "What if we approach this creatively...",
            "Consider alternative perspectives..."
        ]
    
    def _synthesize_answer(self, steps: List[ReasoningStep], user_input: str) -> str:
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

# Global instance
advanced_reasoner = AdvancedReasoner() 