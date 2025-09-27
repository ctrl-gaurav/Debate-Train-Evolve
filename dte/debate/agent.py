"""
Individual debate agent implementation based on original DTE codebase.

This module implements individual agents that participate in multi-agent debates
using the exact prompting and response generation strategy from the original
DTE implementation.
"""

import torch
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import time

from .prompts import DebatePromptManager, DebateResponse


class DebateAgent:
    """
    Individual agent for multi-agent debate based on original DTE implementation.

    Each agent maintains its own conversation history and can generate
    responses using the exact prompting methodology from the original codebase.
    """

    def __init__(self, agent_id: str, model_name: str, device: str = "auto",
                 model_config: dict = None, generation_config: dict = None):
        """Initialize debate agent.

        Args:
            agent_id: Unique identifier for this agent (e.g., "1", "2", "3")
            model_name: Name/path of the language model to use
            device: Device to load model on (auto, cpu, cuda, mps)
            model_config: Configuration for model loading
            generation_config: Configuration for text generation
        """
        self.agent_id = agent_id
        self.model_name = model_name
        self.device = self._setup_device(device)

        # Model configuration
        self.model_config = model_config or {}
        self.generation_config = generation_config or {
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "pad_token_id": None,  # Will be set after tokenizer load
        }

        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()

        # Agent state
        self.response_history: List[DebateResponse] = []
        self.current_round = 0

        # Prompt manager
        self.prompt_manager = DebatePromptManager()

        # Performance tracking
        self.generation_times: List[float] = []
        self.token_counts: List[int] = []

    def _setup_device(self, device: str) -> torch.device:
        """Set up computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)

    def _load_model(self) -> None:
        """Load model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                **self.model_config.get("tokenizer", {})
            )

            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Update generation config with pad token
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id

            # Load model
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
                "device_map": "auto" if self.device.type == "cuda" else None,
                "trust_remote_code": True,
                **self.model_config.get("model", {})
            }

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            # Move to device if not using device_map
            if model_kwargs.get("device_map") is None:
                self.model = self.model.to(self.device)

            # Set to evaluation mode
            self.model.eval()

        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")

    def generate_initial_response(self, query: str, task_type: str = "math") -> DebateResponse:
        """
        Generate initial response for round 0.

        This implements the exact initial prompting from the original DTE codebase.

        Args:
            query: The problem/question to solve
            task_type: Type of task (math, arc, reasoning)

        Returns:
            Structured response from the agent
        """
        # Generate prompt using prompt manager
        prompt = self.prompt_manager.create_initial_prompt(query, task_type)

        # Generate response
        response_text = self._generate_text(prompt)

        # Parse response
        response = self.prompt_manager.parse_response(
            response_text, agent_id=self.agent_id, round_num=0, task_type=task_type
        )

        # Store in history
        self.response_history.append(response)
        self.current_round = 0

        return response

    def generate_debate_response(self, query: str, answers_so_far: Dict[str, str],
                               round_num: int, task_type: str = "math") -> DebateResponse:
        """
        Generate debate response for round > 0.

        This implements the exact debate prompting from the original DTE codebase.

        Args:
            query: Original problem/question
            answers_so_far: Dictionary mapping agent IDs to their previous responses
            round_num: Current round number (1-indexed)
            task_type: Type of task

        Returns:
            Structured debate response from the agent
        """
        # Generate debate prompt using prompt manager
        prompt = self.prompt_manager.create_debate_prompt(
            query=query,
            agent_id=self.agent_id,
            round_num=round_num,
            answers_so_far=answers_so_far,
            task_type=task_type
        )

        # Generate response
        response_text = self._generate_text(prompt)

        # Parse response
        response = self.prompt_manager.parse_response(
            response_text, agent_id=self.agent_id, round_num=round_num, task_type=task_type
        )

        # Store in history
        self.response_history.append(response)
        self.current_round = round_num

        return response

    def _generate_text(self, prompt: str) -> str:
        """
        Generate text response using the language model.

        This implements the exact generation logic from the original DTE codebase.

        Args:
            prompt: Input prompt for generation

        Returns:
            Generated text response
        """
        start_time = time.time()

        try:
            # Create messages in chat format (matching original implementation)
            messages = [{"role": "user", "content": prompt}]

            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.generation_config.get("max_length", 2048) - 500,  # Leave room for generation
                padding=False
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                generation_config = GenerationConfig(**self.generation_config)
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=False
                )

            # Decode response
            generated_tokens = outputs.sequences[0][len(inputs["input_ids"][0]):]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Track performance
            generation_time = time.time() - start_time
            self.generation_times.append(generation_time)
            self.token_counts.append(len(generated_tokens))

            return response_text.strip()

        except Exception as e:
            print(f"Generation error for agent {self.agent_id}: {e}")
            return "Error: Failed to generate response"

    def get_current_response(self) -> Optional[DebateResponse]:
        """Get the most recent response from this agent."""
        return self.response_history[-1] if self.response_history else None

    def get_response_history(self) -> List[DebateResponse]:
        """Get complete response history for this agent."""
        return self.response_history.copy()

    def reset_history(self) -> None:
        """Reset agent's debate history."""
        self.response_history.clear()
        self.current_round = 0

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this agent.

        Returns:
            Dictionary with performance metrics
        """
        if not self.generation_times:
            return {
                "agent_id": self.agent_id,
                "total_generations": 0,
                "avg_generation_time": 0.0,
                "total_generation_time": 0.0,
                "avg_tokens_generated": 0.0,
                "total_tokens_generated": 0,
                "model_name": self.model_name,
                "device": str(self.device)
            }

        return {
            "agent_id": self.agent_id,
            "total_generations": len(self.generation_times),
            "avg_generation_time": sum(self.generation_times) / len(self.generation_times),
            "total_generation_time": sum(self.generation_times),
            "avg_tokens_generated": sum(self.token_counts) / len(self.token_counts) if self.token_counts else 0,
            "total_tokens_generated": sum(self.token_counts),
            "model_name": self.model_name,
            "device": str(self.device)
        }

    def update_generation_config(self, new_config: Dict[str, Any]) -> None:
        """Update generation configuration for this agent.

        Args:
            new_config: Dictionary with new generation parameters
        """
        self.generation_config.update(new_config)

    def cleanup(self) -> None:
        """Clean up model resources."""
        if hasattr(self.model, 'cpu'):
            self.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        """String representation of the agent."""
        return (f"DebateAgent(id={self.agent_id}, model={self.model_name}, "
               f"round={self.current_round}, responses={len(self.response_history)})")

    def __del__(self) -> None:
        """Cleanup when agent is destroyed."""
        try:
            self.cleanup()
        except:
            pass  # Ignore cleanup errors during destruction