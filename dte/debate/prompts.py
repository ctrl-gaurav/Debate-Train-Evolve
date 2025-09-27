"""
Debate prompting system based on original DTE implementation.

This module implements the exact prompting strategy used in the original DTE codebase,
which is more straightforward than RCR but highly effective at reducing sycophancy
and verbosity bias through structured debate instructions.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from ..utils.answer_extraction import extract_final_answer


@dataclass
class DebateResponse:
    """Structured response from a debate agent."""
    answer: str
    reasoning: str
    extracted_answer: str
    confidence: Optional[float] = None
    round_number: int = 0
    agent_id: str = ""


class DebatePromptManager:
    """
    Manager for debate prompts based on original DTE implementation.

    This implements the exact prompting strategy from the original codebase,
    which uses clear, structured instructions for initial problem solving
    and multi-agent debate without explicit RCR phases.
    """

    def __init__(self):
        """Initialize debate prompt manager."""
        pass

    def create_initial_prompt(self, query: str, task_type: str = "math") -> str:
        """
        Generate initial prompt for round 0 of debate.

        This matches the exact initial prompting from the original DTE codebase.

        Args:
            query: The question/problem to solve
            task_type: Type of task (math, arc, reasoning)

        Returns:
            Formatted initial prompt
        """
        if task_type == "math":
            return self._create_math_initial_prompt(query)
        elif task_type == "arc":
            return self._create_arc_initial_prompt(query)
        else:
            return self._create_general_initial_prompt(query)

    def _create_math_initial_prompt(self, question: str) -> str:
        """Create initial prompt for mathematical reasoning tasks."""
        return f"""Can you solve this math problem?
Your final answer must be in the format \\boxed{{answer}} at the end.

Problem: {question}"""

    def _create_arc_initial_prompt(self, question_data: Dict[str, Any]) -> str:
        """Create initial prompt for ARC-Challenge tasks."""
        question = question_data.get("question", "")
        choices = question_data.get("choices", {}).get("text", [])
        labels = question_data.get("choices", {}).get("label", [])

        choices_text = ""
        for label, choice in zip(labels, choices):
            choices_text += f"{label}. {choice}\n"

        return f"""Answer the following multiple-choice question from the ARC Challenge dataset.
Read the question and all choices carefully before answering.

Question: {question}

Choices:
{choices_text}

Please provide your reasoning and then give your final answer in the format: Answer: [letter]"""

    def _create_general_initial_prompt(self, query: str) -> str:
        """Create initial prompt for general reasoning tasks."""
        return f"""Please solve the following problem step by step.
Provide clear reasoning and a definitive final answer.

Problem: {query}"""

    def create_debate_prompt(self, query: str, agent_id: str, round_num: int,
                           answers_so_far: Dict[str, str], task_type: str = "math") -> str:
        """
        Generate debate prompt for rounds > 0.

        This implements the exact debate prompting strategy from the original
        DTE codebase, which provides clear structure without explicit RCR phases.

        Args:
            query: Original query/problem
            agent_id: ID of current agent
            round_num: Current debate round (1-indexed)
            answers_so_far: Dictionary mapping agent IDs to their responses
            task_type: Type of task

        Returns:
            Formatted debate prompt
        """
        if task_type == "math":
            return self._create_math_debate_prompt(query, agent_id, round_num, answers_so_far)
        elif task_type == "arc":
            return self._create_arc_debate_prompt(query, agent_id, round_num, answers_so_far)
        else:
            return self._create_general_debate_prompt(query, agent_id, round_num, answers_so_far)

    def _create_math_debate_prompt(self, question: str, agent_id: str, round_num: int,
                                 answers_so_far: Dict[str, str]) -> str:
        """Create debate prompt for mathematical reasoning tasks."""
        # Format previous answers for context
        context = ""
        for other_agent_id, answer in answers_so_far.items():
            if other_agent_id != agent_id:  # Skip own previous answer
                extracted = extract_final_answer(answer)
                context += f"Agent {other_agent_id} solution: {answer}\n\n"
                context += f"Agent {other_agent_id} answer: {extracted}\n\n"

        # Format own previous answer if it exists
        own_previous = ""
        if agent_id in answers_so_far:
            own_previous = f"""Your previous solution was:
{answers_so_far[agent_id]}

Your previous extracted answer was: {extract_final_answer(answers_so_far[agent_id])}"""

        prompt = f"""You are Agent {agent_id} in a multi-agent debate to solve the following math problem:

Problem: {question}

{own_previous}

Here are the solutions from other agents:
{context}

This is debate round {round_num}. Please carefully analyze all solutions including your own, identify any errors in reasoning, and provide your revised solution.

If you believe your previous answer is correct, explain why and defend it.
If you believe you made an error, explain the error and provide a corrected solution.
If you believe another agent's answer is correct, explain why you agree with it.

Your final answer must be in the format \\boxed{{answer}} at the end."""

        return prompt

    def _create_arc_debate_prompt(self, question_data: Dict[str, Any], agent_id: str,
                                round_num: int, answers_so_far: Dict[str, str]) -> str:
        """Create debate prompt for ARC-Challenge tasks."""
        question = question_data.get("question", "")
        choices = question_data.get("choices", {}).get("text", [])
        labels = question_data.get("choices", {}).get("label", [])

        choices_text = ""
        for label, choice in zip(labels, choices):
            choices_text += f"{label}. {choice}\n"

        # Format previous answers for context
        context = ""
        for other_agent_id, answer in answers_so_far.items():
            if other_agent_id != agent_id:
                extracted = self._extract_arc_answer(answer)
                context += f"Agent {other_agent_id} reasoning: {answer}\n\n"
                context += f"Agent {other_agent_id} answer: {extracted}\n\n"

        # Format own previous answer if it exists
        own_previous = ""
        if agent_id in answers_so_far:
            own_previous = f"""Your previous reasoning was:
{answers_so_far[agent_id]}

Your previous answer was: {self._extract_arc_answer(answers_so_far[agent_id])}"""

        prompt = f"""You are Agent {agent_id} in a multi-agent debate to answer the following ARC Challenge question:

Question: {question}

Choices:
{choices_text}

{own_previous}

Here are the responses from other agents:
{context}

This is debate round {round_num}. Please carefully analyze all responses including your own, identify any errors in scientific reasoning, and provide your revised answer.

If you believe your previous answer is correct, explain why and defend it.
If you believe you made an error, explain the error and provide a corrected answer.
If you believe another agent's answer is correct, explain why you agree with it.

Please provide your reasoning and then give your final answer in the format: Answer: [letter]"""

        return prompt

    def _create_general_debate_prompt(self, query: str, agent_id: str, round_num: int,
                                    answers_so_far: Dict[str, str]) -> str:
        """Create debate prompt for general reasoning tasks."""
        # Format previous answers for context
        context = ""
        for other_agent_id, answer in answers_so_far.items():
            if other_agent_id != agent_id:
                context += f"Agent {other_agent_id} response: {answer}\n\n"

        # Format own previous answer if it exists
        own_previous = ""
        if agent_id in answers_so_far:
            own_previous = f"""Your previous response was:
{answers_so_far[agent_id]}"""

        prompt = f"""You are Agent {agent_id} in a multi-agent debate to solve the following problem:

Problem: {query}

{own_previous}

Here are the responses from other agents:
{context}

This is debate round {round_num}. Please carefully analyze all responses including your own, identify any errors in reasoning, and provide your revised solution.

If you believe your previous answer is correct, explain why and defend it.
If you believe you made an error, explain the error and provide a corrected solution.
If you believe another agent's answer is correct, explain why you agree with it."""

        return prompt

    def _extract_arc_answer(self, response: str) -> str:
        """Extract ARC answer choice from response."""
        patterns = [
            r"(?:answer|choice)(?:\s+is)?\s*:?\s*([A-D])",
            r"(?:^|\s)([A-D])(?:\s|$|\.|,)",
            r"\\boxed\{([A-D])\}",
            r"\(([A-D])\)"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                return matches[-1].upper()

        return "Unable to Extract"

    def parse_response(self, response_text: str, agent_id: str = "", round_num: int = 0,
                      task_type: str = "math") -> DebateResponse:
        """
        Parse agent response into structured format.

        Args:
            response_text: Raw response text from agent
            agent_id: ID of the responding agent
            round_num: Current round number
            task_type: Type of task

        Returns:
            Parsed DebateResponse object
        """
        # Extract answer based on task type
        if task_type == "arc":
            extracted_answer = self._extract_arc_answer(response_text)
        else:
            extracted_answer = extract_final_answer(response_text)

        return DebateResponse(
            answer=extracted_answer,
            reasoning=response_text,
            extracted_answer=extracted_answer,
            round_number=round_num,
            agent_id=agent_id
        )

    def validate_response_format(self, response: DebateResponse, task_type: str = "math") -> List[str]:
        """
        Validate that response follows expected format.

        Args:
            response: Parsed response to validate
            task_type: Type of task

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check if answer was extracted successfully
        if response.extracted_answer == "Unable to Extract":
            errors.append("Could not extract answer from response")

        # Check reasoning length
        if not response.reasoning or len(response.reasoning.strip()) < 20:
            errors.append("Reasoning too short or missing")

        # Task-specific validations
        if task_type == "math":
            if "\\boxed{" not in response.reasoning:
                errors.append("Missing \\boxed{} format for math answer")
        elif task_type == "arc":
            if not re.search(r"Answer:\s*[A-D]", response.reasoning, re.IGNORECASE):
                errors.append("Missing 'Answer: [letter]' format for ARC question")

        return errors