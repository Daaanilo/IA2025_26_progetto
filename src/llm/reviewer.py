"""
Reviewer LLM - Evaluates Helper suggestions and provides feedback
"""

import json
import torch
from typing import List, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()


class Reviewer:
    """
    Reviewer LLM that evaluates Helper's action suggestions and provides feedback.
    Uses a fine-tuned LLM (Llama 2 or Mistral) for evaluation.
    """

    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        """
        Initialize Reviewer LLM.

        Args:
            config: Configuration dictionary from reviewer_config.yaml
            device: Device to use (cuda/cpu)
        """
        self.config = config
        reviewer_config = config.get("reviewer", {})

        # Normalize device selection: prefer requested device only if available
        try:
            requested = str(device)
        except Exception:
            requested = "cpu"

        if requested.startswith("cuda") and torch.cuda.is_available():
            device_str = requested
        else:
            device_str = "cpu"

        self.device = torch.device(device_str)

        # Model paths
        self.base_model_name = reviewer_config.get(
            "base_model", "meta-llama/Llama-2-7b-chat-hf"
        )
        self.fine_tuned_model_path = reviewer_config.get(
            "fine_tuned_model", "models/reviewer_finetuned"
        )

        # Prompts
        self.system_prompt = reviewer_config.get("system_prompt", "")
        self.feedback_template = reviewer_config.get("feedback_template", "")

        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self.is_loaded = False

        # Statistics
        self.total_reviews = 0
        self.average_rating = 0.0

    def load_model(self, use_fine_tuned: bool = False):
        """
        Load the model (base or fine-tuned).

        Args:
            use_fine_tuned: Whether to load fine-tuned model
        """
        quantization_config = self._get_quantization_config()

        model_path = (
            self.fine_tuned_model_path if use_fine_tuned else self.base_model_name
        )

        print(f"Loading Reviewer model: {model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name, trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )

            self.is_loaded = True
            print("Reviewer model loaded successfully")

        except Exception as e:
            print(f"Error loading Reviewer model: {e}")
            print("Reviewer will operate in mock mode")
            self.is_loaded = False

    def _get_quantization_config(self) -> BitsAndBytesConfig:
        """Get quantization configuration for memory efficiency."""
        quant_config = self.config.get("reviewer", {}).get("quantization", {})

        # Map string dtype config to torch dtype when provided in YAML
        dtype_cfg = quant_config.get("bnb_4bit_compute_dtype", "float16")
        if isinstance(dtype_cfg, str):
            dtype_str = dtype_cfg.lower()
            if dtype_str in ("float16", "fp16"):
                compute_dtype = torch.float16
            elif dtype_str in ("float32", "fp32"):
                compute_dtype = torch.float32
            else:
                # Default fallback
                compute_dtype = torch.float16
        else:
            compute_dtype = dtype_cfg

        return BitsAndBytesConfig(
            load_in_4bit=quant_config.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_config.get(
                "bnb_4bit_use_double_quant", True
            ),
        )

    def review_actions(
        self, state_info: Dict[str, Any], suggested_actions: List[str]
    ) -> Dict[str, Any]:
        """
        Review suggested actions and provide feedback.

        Args:
            state_info: Current game state information
            suggested_actions: Actions suggested by Helper

        Returns:
            Dictionary with rating, feedback, and improved actions
        """
        if not self.is_loaded:
            # Mock mode - return dummy feedback
            return self._mock_review(suggested_actions)

        # Format prompt
        prompt = self._format_review_prompt(state_info, suggested_actions)

        # Generate review
        try:
            # Tokenize and move tensors explicitly to the chosen device
            inputs = self.tokenizer(prompt, return_tensors="pt")
            for k, v in inputs.items():
                if hasattr(v, "to"):
                    inputs[k] = v.to(self.device)

            inference_config = self.config.get("reviewer", {}).get("inference", {})

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=inference_config.get("max_new_tokens", 300),
                    temperature=inference_config.get("temperature", 0.5),
                    top_p=inference_config.get("top_p", 0.9),
                    repetition_penalty=inference_config.get("repetition_penalty", 1.1),
                    do_sample=True,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Parse feedback
            feedback = self._parse_feedback(response)

            self.total_reviews += 1
            self.average_rating = (
                self.average_rating * (self.total_reviews - 1) + feedback["rating"]
            ) / self.total_reviews

            return feedback

        except Exception as e:
            print(f"Error during review: {e}")
            return self._mock_review(suggested_actions)

    def _format_review_prompt(
        self, state_info: Dict[str, Any], suggested_actions: List[str]
    ) -> str:
        """Format the review prompt."""
        # Create state description
        state_desc = json.dumps(state_info, indent=2)
        actions_str = json.dumps(suggested_actions)

        # Format feedback template
        user_prompt = self.feedback_template.format(
            state_description=state_desc, suggested_actions=actions_str
        )

        # Combine system and user prompts
        full_prompt = f"{self.system_prompt}\n\n{user_prompt}"

        return full_prompt

    def _parse_feedback(self, response: str) -> Dict[str, Any]:
        """Parse feedback from LLM response."""
        try:
            # Try to extract JSON
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                feedback = json.loads(json_match.group())
                return {
                    "rating": feedback.get("rating", 5),
                    "strengths": feedback.get("strengths", ""),
                    "weaknesses": feedback.get("weaknesses", ""),
                    "improved_actions": feedback.get("improved_actions", []),
                    "reasoning": feedback.get("reasoning", ""),
                }
        except Exception as e:
            # If parsing fails, log debug and fall back to default parsing
            print(f"[Reviewer] Warning: feedback parsing failed: {e}")
            pass

        # Fallback parsing
        return {
            "rating": 5,
            "strengths": "Unable to parse strengths",
            "weaknesses": "Unable to parse weaknesses",
            "improved_actions": [],
            "reasoning": response[:200],
        }

    def _mock_review(self, suggested_actions: List[str]) -> Dict[str, Any]:
        """Provide mock review when model is not loaded."""
        return {
            "rating": 7,
            "strengths": "Actions appear reasonable for exploration",
            "weaknesses": "Mock review - model not loaded",
            "improved_actions": suggested_actions,
            "reasoning": "This is a placeholder review. Load the Reviewer model for actual feedback.",
        }

    def fine_tune(self, training_data_path: str, output_dir: str):
        """
        Fine-tune the Reviewer model on training data.

        Args:
            training_data_path: Path to training dataset (JSON)
            output_dir: Directory to save fine-tuned model
        """
        print("Starting Reviewer fine-tuning...")

        # Ensure model is loaded (load base if necessary)
        if self.model is None:
            print(
                "Reviewer model not loaded - attempting to load base model for fine-tuning..."
            )
            self.load_model(use_fine_tuned=False)
            if self.model is None:
                raise RuntimeError("Reviewer model could not be loaded for fine-tuning")

        # Load training data
        with open(training_data_path, "r") as f:
            data = json.load(f)

        dataset = Dataset.from_list(data)

        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)

        # LoRA configuration
        lora_config = self._get_lora_config()
        self.model = get_peft_model(self.model, lora_config)

        # Training arguments
        training_args = self._get_training_args(output_dir)

        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=training_args.max_seq_length,
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
        )

        # Train
        trainer.train()

        # Save model
        trainer.save_model(output_dir)
        print(f"Fine-tuned model saved to {output_dir}")

    def _get_lora_config(self) -> LoraConfig:
        """Get LoRA configuration."""
        lora_config = self.config.get("reviewer", {}).get("lora", {})

        return LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
            bias=lora_config.get("bias", "none"),
            task_type=lora_config.get("task_type", "CAUSAL_LM"),
        )

    def _get_training_args(self, output_dir: str) -> TrainingArguments:
        """Get training arguments."""
        ft_config = self.config.get("reviewer", {}).get("fine_tuning", {})

        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=ft_config.get("learning_rate", 2e-4),
            num_train_epochs=ft_config.get("num_epochs", 3),
            per_device_train_batch_size=ft_config.get("batch_size", 4),
            gradient_accumulation_steps=ft_config.get("gradient_accumulation_steps", 4),
            warmup_steps=ft_config.get("warmup_steps", 100),
            logging_steps=ft_config.get("logging_steps", 10),
            save_steps=ft_config.get("save_steps", 500),
            eval_steps=ft_config.get("eval_steps", 100),
            fp16=True,
            optim="paged_adamw_8bit",
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get Reviewer statistics."""
        return {
            "total_reviews": self.total_reviews,
            "average_rating": f"{self.average_rating:.2f}",
            "is_loaded": self.is_loaded,
        }
