"""
GRPO (Group Relative Policy Optimization) Training Implementation.

This module implements the GRPO training algorithm as described in the DTE paper,
eliminating the need for a separate value function through group-wise advantage estimation.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup, GenerationConfig
)
from torch.optim import AdamW

# Conditional import of PEFT
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. LoRA functionality will be disabled.")
import math
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..data.generator import TrainingExample
from ..core.logger import DTELogger
from .reward_model import DTERewardModel


@dataclass
class GRPOBatch:
    """Batch for GRPO training."""
    queries: List[str]
    responses: List[List[str]]  # responses[query_idx][response_idx]
    rewards: List[List[float]]  # rewards[query_idx][response_idx]
    advantages: List[List[float]]  # advantages[query_idx][response_idx]


class TrainingDataset(Dataset):
    """Dataset for GRPO training."""

    def __init__(self, examples: List[TrainingExample], tokenizer, max_length: int = 2048):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Format input and output in XML format to match reward functions
        query = example.query
        response = f"<reasoning>\n{example.reasoning}\n</reasoning>\n<answer>\n{example.answer}\n</answer>\n"

        # Tokenize
        query_tokens = self.tokenizer(
            query, truncation=True, max_length=self.max_length//2, padding=False
        )
        response_tokens = self.tokenizer(
            response, truncation=True, max_length=self.max_length//2, padding=False
        )

        return {
            'query': query,
            'response': response,
            'query_ids': query_tokens['input_ids'],
            'response_ids': response_tokens['input_ids'],
            'reward': example.confidence
        }


class GRPOTrainer:
    """GRPO (Group Relative Policy Optimization) Trainer.

    Implements the GRPO algorithm that eliminates the need for a separate value
    function by estimating advantages through group-wise comparisons.
    """

    def __init__(self, config_training, config_model, config_paths,
                 logger: Optional[DTELogger] = None):
        """Initialize GRPO trainer.

        Args:
            config_training: Training configuration
            config_model: Model configuration
            config_paths: Paths configuration
            logger: Optional logger for tracking progress
        """
        self.config = config_training
        self.model_config = config_model
        self.paths_config = config_paths
        self.logger = logger

        # Training parameters
        self.learning_rate = config_training.learning_rate
        self.weight_decay = config_training.weight_decay
        self.warmup_steps = config_training.warmup_steps
        self.max_epochs = config_training.max_epochs
        self.batch_size = config_training.batch_size
        self.gradient_accumulation_steps = config_training.gradient_accumulation_steps

        # GRPO parameters
        self.group_size = config_training.grpo.group_size
        self.clip_ratio = config_training.grpo.clip_ratio
        self.kl_penalty = config_training.grpo.kl_penalty

        # Device setup
        self.device = self._setup_device()

        # Model components
        self.tokenizer = None
        self.model = None
        self.reference_model = None
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')

        # Reward model
        self.reward_model = DTERewardModel()

        # Load model
        self._load_model()

    def _setup_device(self) -> torch.device:
        """Set up computation device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _load_model(self) -> None:
        """Load model, tokenizer, and set up LoRA if configured."""
        model_name = self.model_config.base_model_name

        if self.logger:
            self.logger.info(f"Loading model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
            "device_map": "auto" if self.device.type == "cuda" else None,
            "trust_remote_code": True
        }

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Set up LoRA if enabled and available
        if self.config.lora.enabled and PEFT_AVAILABLE:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora.rank,
                lora_alpha=self.config.lora.alpha,
                lora_dropout=self.config.lora.dropout,
                target_modules=self.config.lora.target_modules,
            )
            self.model = get_peft_model(self.model, lora_config)

            if self.logger:
                self.logger.info(f"Applied LoRA: rank={self.config.lora.rank}")
        elif self.config.lora.enabled and not PEFT_AVAILABLE:
            if self.logger:
                self.logger.warning("LoRA requested but PEFT not available. Training without LoRA.")

        # Create reference model for KL penalty
        self.reference_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.reference_model.eval()

        # Move to device if not using device_map
        if model_kwargs.get("device_map") is None:
            self.model = self.model.to(self.device)
            self.reference_model = self.reference_model.to(self.device)

    def train(self, training_examples: List[TrainingExample],
              validation_examples: Optional[List[TrainingExample]] = None) -> Dict[str, Any]:
        """Train model using GRPO algorithm.

        Args:
            training_examples: List of training examples from debates
            validation_examples: Optional validation examples

        Returns:
            Training metrics and statistics
        """
        if self.logger:
            with self.logger.component_context("grpo_training"):
                self.logger.info(f"Starting GRPO training with {len(training_examples)} examples")

        # Create dataset and dataloader
        train_dataset = TrainingDataset(training_examples, self.tokenizer, self.model_config.max_length)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True if self.device.type == "cuda" else False
        )

        # Set up optimizer and scheduler
        self._setup_optimizer_and_scheduler(len(train_dataloader))

        # Training loop
        training_metrics = self._training_loop(train_dataloader, validation_examples)

        if self.logger:
            self.logger.info("GRPO training completed")

            # Log final reward statistics
            if "reward_stats" in training_metrics and training_metrics["reward_stats"]:
                final_rewards = training_metrics["reward_stats"][-1]
                self.logger.info(f"Final reward stats: {final_rewards}")

        return training_metrics

    def _setup_optimizer_and_scheduler(self, num_training_steps_per_epoch: int) -> None:
        """Set up optimizer and learning rate scheduler."""
        # Get trainable parameters
        if hasattr(self.model, 'named_parameters'):
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            trainable_params = list(self.model.parameters())

        # Create optimizer
        self.optimizer = AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-8
        )

        # Calculate total training steps
        total_steps = num_training_steps_per_epoch * self.max_epochs

        # Create scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        if self.logger:
            self.logger.info(f"Optimizer: AdamW, LR: {self.learning_rate}, Steps: {total_steps}")

    def _training_loop(self, train_dataloader: DataLoader,
                      validation_examples: Optional[List[TrainingExample]]) -> Dict[str, Any]:
        """Main training loop implementing GRPO algorithm."""
        training_metrics = {
            "epoch_losses": [],
            "step_losses": [],
            "learning_rates": [],
            "kl_divergences": [],
            "advantages_stats": []
        }

        self.model.train()

        if self.logger:
            progress = self.logger.start_progress("GRPO Training", total=self.max_epochs)

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_dataloader):
                # Generate responses for GRPO
                grpo_batch = self._prepare_grpo_batch(batch)

                # Compute GRPO loss
                loss, metrics = self._compute_grpo_loss(grpo_batch)

                # Backward pass
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                # Update weights
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                # Track metrics
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1

                training_metrics["step_losses"].append(loss.item() * self.gradient_accumulation_steps)
                training_metrics["learning_rates"].append(self.scheduler.get_last_lr()[0])
                training_metrics["kl_divergences"].append(metrics["avg_kl_div"])
                training_metrics["advantages_stats"].append(metrics["advantages_stats"])

                # Track reward statistics if available
                if "reward_stats" in metrics:
                    if "reward_stats" not in training_metrics:
                        training_metrics["reward_stats"] = []
                    training_metrics["reward_stats"].append(metrics["reward_stats"])

                # Log progress
                if self.global_step % 10 == 0 and self.logger:
                    self.logger.log_training_step(
                        step=self.global_step,
                        loss=loss.item() * self.gradient_accumulation_steps,
                        metrics={
                            "lr": self.scheduler.get_last_lr()[0],
                            "kl_div": metrics["avg_kl_div"],
                            "advantages_mean": metrics["advantages_stats"]["mean"],
                            **metrics.get("reward_stats", {})
                        }
                    )

            # Calculate epoch metrics
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            training_metrics["epoch_losses"].append(avg_epoch_loss)

            if self.logger:
                self.logger.update_progress("GRPO Training", advance=1)
                self.logger.info(f"Epoch {epoch + 1}/{self.max_epochs} - Loss: {avg_epoch_loss:.4f}")

            # Save checkpoint
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                self._save_checkpoint(epoch, avg_epoch_loss)

        if self.logger:
            self.logger.finish_progress("GRPO Training")

        return training_metrics

    def _prepare_grpo_batch(self, batch: Dict[str, Any]) -> GRPOBatch:
        """Prepare batch for GRPO training by generating multiple responses."""
        queries = batch['query']
        batch_size = len(queries)

        all_responses = []
        all_rewards = []
        all_advantages = []

        with torch.no_grad():
            for i, query in enumerate(queries):
                # Generate multiple responses for this query
                responses = self._generate_responses(query, self.group_size)

                # Get ground truth if available in batch
                ground_truth = None
                if 'ground_truth' in batch and i < len(batch['ground_truth']):
                    ground_truth = batch['ground_truth'][i]
                elif 'answer' in batch and i < len(batch['answer']):
                    ground_truth = batch['answer'][i]

                # Calculate rewards for each response using all 5 DTE reward functions
                rewards = [self._calculate_reward(query, response, ground_truth) for response in responses]

                # Calculate group-relative advantages
                advantages = self._calculate_advantages(rewards)

                all_responses.append(responses)
                all_rewards.append(rewards)
                all_advantages.append(advantages)

        return GRPOBatch(
            queries=queries,
            responses=all_responses,
            rewards=all_rewards,
            advantages=all_advantages
        )

    def _generate_responses(self, query: str, num_responses: int) -> List[str]:
        """Generate multiple responses for a query."""
        responses = []

        # Tokenize query
        query_tokens = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_config.max_length // 2,
            padding=False
        ).to(self.device)

        # Generation config
        generation_config = GenerationConfig(
            max_length=self.model_config.max_length,
            temperature=self.model_config.temperature,
            top_p=self.model_config.top_p,
            top_k=self.model_config.top_k,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        for _ in range(num_responses):
            with torch.no_grad():
                outputs = self.model.generate(
                    **query_tokens,
                    generation_config=generation_config,
                    return_dict_in_generate=True
                )

                # Decode response
                generated_tokens = outputs.sequences[0][len(query_tokens["input_ids"][0]):]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                responses.append(response.strip())

        return responses

    def _calculate_reward(self, query: str, response: str, ground_truth: str = None) -> float:
        """Calculate reward for a query-response pair using all 5 DTE reward functions.

        Args:
            query: The input query/question
            response: Model response
            ground_truth: Ground truth answer (if available)

        Returns:
            Combined reward score
        """
        # Use the complete DTE reward model with all 5 reward functions
        rewards_dict = self.reward_model.calculate_all_rewards(
            query=query,
            responses=[response],
            ground_truth=ground_truth
        )

        # Use DTE-specific reward weights from config
        if hasattr(self.config.rewards, 'get_dte_weights'):
            reward_weights = self.config.rewards.get_dte_weights()
        else:
            # Fallback to default DTE weights
            reward_weights = {
                'correctness': 2.0,  # Most important
                'int': 0.5,
                'strict_format': 0.5,
                'soft_format': 0.5,
                'xmlcount': 0.5
            }

        combined_rewards = self.reward_model.combine_rewards(rewards_dict, reward_weights)
        return combined_rewards[0] if combined_rewards else 0.0

    def get_detailed_reward_breakdown(self, query: str, response: str, ground_truth: str = None) -> Dict[str, Any]:
        """Get detailed breakdown of all reward components for analysis.

        Args:
            query: The input query/question
            response: Model response
            ground_truth: Ground truth answer (if available)

        Returns:
            Detailed breakdown of all reward components
        """
        rewards_dict = self.reward_model.calculate_all_rewards(
            query=query,
            responses=[response],
            ground_truth=ground_truth
        )

        # Get statistics for each reward function
        reward_stats = self.reward_model.get_reward_statistics(rewards_dict)

        return {
            "individual_rewards": {k: v[0] for k, v in rewards_dict.items()},
            "reward_statistics": reward_stats,
            "extracted_answer": self.reward_model._extract_xml_answer(response),
            "response_length": len(response.split()),
            "has_xml_format": "<reasoning>" in response and "<answer>" in response
        }

    def _calculate_advantages(self, rewards: List[float]) -> List[float]:
        """Calculate group-relative advantages."""
        if len(rewards) <= 1:
            return [0.0] * len(rewards)

        # Group statistics
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = math.sqrt(variance + 1e-8)

        # Normalize advantages
        advantages = [(r - mean_reward) / (std_reward + 1e-8) for r in rewards]

        return advantages

    def _compute_grpo_loss(self, grpo_batch: GRPOBatch) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute GRPO loss with clipping and KL regularization."""
        total_loss = 0.0
        total_kl_div = 0.0
        all_advantages = []
        all_rewards = []

        for query, responses, rewards, advantages in zip(
            grpo_batch.queries, grpo_batch.responses,
            grpo_batch.rewards, grpo_batch.advantages
        ):
            all_rewards.extend(rewards)
            for response, advantage in zip(responses, advantages):
                # Prepare input
                full_text = f"{query}\n{response}"
                tokens = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.model_config.max_length,
                    padding=False
                ).to(self.device)

                # Get model logits
                with torch.no_grad():
                    old_logits = self.model(**tokens).logits

                current_logits = self.model(**tokens).logits

                # Calculate importance ratio
                old_log_probs = F.log_softmax(old_logits, dim=-1)
                current_log_probs = F.log_softmax(current_logits, dim=-1)

                # Simple ratio calculation (would need more sophisticated implementation)
                ratio = torch.exp(current_log_probs - old_log_probs).mean()

                # Clipped policy loss
                advantage_tensor = torch.tensor(advantage, device=self.device)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

                policy_loss = -torch.min(
                    ratio * advantage_tensor,
                    clipped_ratio * advantage_tensor
                )

                # KL divergence with reference model
                with torch.no_grad():
                    ref_logits = self.reference_model(**tokens).logits
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

                kl_div = F.kl_div(current_log_probs, ref_log_probs, reduction='mean')

                # Total loss
                loss = policy_loss + self.kl_penalty * kl_div
                total_loss += loss
                total_kl_div += kl_div.item()
                all_advantages.append(advantage)

        # Average loss
        num_responses = sum(len(responses) for responses in grpo_batch.responses)
        if num_responses > 0:
            total_loss = total_loss / num_responses
            avg_kl_div = total_kl_div / num_responses
        else:
            avg_kl_div = 0.0

        # Metrics
        metrics = {
            "avg_kl_div": avg_kl_div,
            "advantages_stats": {
                "mean": np.mean(all_advantages) if all_advantages else 0.0,
                "std": np.std(all_advantages) if all_advantages else 0.0,
                "min": np.min(all_advantages) if all_advantages else 0.0,
                "max": np.max(all_advantages) if all_advantages else 0.0
            },
            "reward_stats": {
                "mean_reward": np.mean(all_rewards) if all_rewards else 0.0,
                "std_reward": np.std(all_rewards) if all_rewards else 0.0,
                "min_reward": np.min(all_rewards) if all_rewards else 0.0,
                "max_reward": np.max(all_rewards) if all_rewards else 0.0
            }
        }

        return total_loss, metrics

    def _save_checkpoint(self, epoch: int, loss: float) -> None:
        """Save model checkpoint."""
        checkpoint_dir = f"{self.paths_config.models_dir}/checkpoint_epoch_{epoch}"

        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(checkpoint_dir)
        else:
            torch.save(self.model.state_dict(), f"{checkpoint_dir}/model.pt")

        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        training_state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "loss": loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
        }
        torch.save(training_state, f"{checkpoint_dir}/training_state.pt")

        if self.logger:
            self.logger.log_model_checkpoint(checkpoint_dir, {"loss": loss, "epoch": epoch})

    def cleanup(self) -> None:
        """Clean up model resources."""
        if hasattr(self.model, 'cpu'):
            self.model.cpu()
        if hasattr(self.reference_model, 'cpu'):
            self.reference_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self) -> None:
        """Cleanup when trainer is destroyed."""
        try:
            self.cleanup()
        except:
            pass