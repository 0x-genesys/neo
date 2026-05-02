"""
Data utilities for Chain-of-Thought (CoT) fine-tuning.
Formats CoT JSONL data into the <|im_start|> messaging format.
"""
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
from pathlib import Path
import random


# Special tokens for Chain-of-Thought formatting
SPECIAL_TOKENS = {
    'im_start': '<|im_start|>',
    'im_end': '<|im_end|>',
    'system': 'system',
    'user': 'user',
    'thought': 'thought',
    'assistant': 'assistant',
}


class CoTDataset(Dataset):
    """
    Dataset for Chain-of-Thought fine-tuning.
    
    Expected JSONL format:
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant..."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "thought", "content": "Let me think step by step..."},
            {"role": "assistant", "content": "The answer is 4."}
        ]
    }
    
    Or simplified format:
    {
        "instruction": "What is 2+2?",
        "thought": "Let me think step by step...",
        "response": "The answer is 4."
    }
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize CoT dataset.
        
        Args:
            data_path: Path to JSONL file
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
            system_prompt: System prompt to prepend to all conversations
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        
        # Load data
        self.examples = self._load_data(data_path)
        
        print(f"✅ Loaded {len(self.examples)} examples from {data_path}")
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        examples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    example = json.loads(line)
                    examples.append(example)
        
        return examples
    
    def _format_message(self, role: str, content: str) -> str:
        """
        Format a single message with special tokens.
        
        Format: <|im_start|>role\ncontent<|im_end|>
        """
        return f"{SPECIAL_TOKENS['im_start']}{role}\n{content}{SPECIAL_TOKENS['im_end']}"
    
    def _format_conversation(self, example: Dict[str, Any]) -> str:
        """
        Format a conversation into the messaging format.
        
        Supports two formats:
        1. messages: List of {"role": ..., "content": ...}
        2. instruction/thought/response: Simplified format
        """
        formatted_parts = []
        
        # Add system prompt if provided
        if self.system_prompt:
            formatted_parts.append(
                self._format_message(SPECIAL_TOKENS['system'], self.system_prompt)
            )
        
        # Handle different input formats
        if 'messages' in example:
            # Standard messages format
            for message in example['messages']:
                role = message['role']
                content = message['content']
                formatted_parts.append(self._format_message(role, content))
        
        elif 'instruction' in example:
            # Simplified format
            # Add system message if not already added
            if not self.system_prompt and 'system' in example:
                formatted_parts.append(
                    self._format_message(SPECIAL_TOKENS['system'], example['system'])
                )
            
            # User instruction
            formatted_parts.append(
                self._format_message(SPECIAL_TOKENS['user'], example['instruction'])
            )
            
            # Thought process (if present)
            if 'thought' in example and example['thought']:
                formatted_parts.append(
                    self._format_message(SPECIAL_TOKENS['thought'], example['thought'])
                )
            
            # Assistant response
            formatted_parts.append(
                self._format_message(SPECIAL_TOKENS['assistant'], example['response'])
            )
        
        else:
            raise ValueError(f"Unknown example format: {example.keys()}")
        
        # Join all parts with newlines
        return '\n'.join(formatted_parts)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.
        
        Returns:
            Dictionary with:
            - input_ids: Token IDs
            - attention_mask: Attention mask
            - labels: Labels for language modeling (same as input_ids)
        """
        example = self.examples[idx]
        
        # Format conversation
        text = self._format_conversation(example)
        
        # Tokenize using encode method (TiktokenWrapper compatible)
        tokens = self.tokenizer.encode(text)
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            # Pad with pad_token_id
            pad_length = self.max_length - len(tokens)
            tokens = tokens + [self.tokenizer.pad_token_id] * pad_length
        
        # Convert to tensors
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # Validate token IDs are within vocab range
        vocab_size = len(self.tokenizer)
        if input_ids.max() >= vocab_size:
            # This should not happen, but if it does, clamp to valid range
            print(f"⚠️  Warning: Found token ID {input_ids.max().item()} >= vocab_size {vocab_size}")
            print(f"   Text preview: {text[:100]}...")
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        # Create labels (same as input_ids for causal LM)
        # Mask padding tokens in labels (-100 is ignored by loss)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def create_cot_dataset(
    train_path: str,
    val_path: Optional[str],
    tokenizer,
    max_length: int = 512,
    system_prompt: Optional[str] = None,
) -> tuple:
    """
    Create training and validation datasets.
    
    Args:
        train_path: Path to training JSONL file
        val_path: Path to validation JSONL file (optional)
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
        system_prompt: System prompt to prepend to all conversations
    
    Returns:
        (train_dataset, val_dataset) tuple
    """
    train_dataset = CoTDataset(
        train_path,
        tokenizer,
        max_length=max_length,
        system_prompt=system_prompt,
    )
    
    val_dataset = None
    if val_path:
        val_dataset = CoTDataset(
            val_path,
            tokenizer,
            max_length=max_length,
            system_prompt=system_prompt,
        )
    
    return train_dataset, val_dataset


def prepare_tokenizer(tokenizer):
    """
    Prepare tokenizer with special tokens for CoT.
    
    For tiktoken tokenizers, special tokens are already in the vocabulary.
    This function just validates and sets up padding.
    """
    # For tiktoken, special tokens like <|im_start|> and <|im_end|> are already in vocab
    # Just ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Tokenizer prepared:")
    print(f"   Added 0 special tokens (already in tiktoken vocab)")
    print(f"   Vocabulary size: {len(tokenizer)}")
    print(f"   PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"   Special tokens: {SPECIAL_TOKENS['im_start']}, {SPECIAL_TOKENS['im_end']}")
    
    return tokenizer


def create_sample_cot_data(output_path: str, num_samples: int = 100):
    """
    Create sample CoT training data for testing.
    
    Args:
        output_path: Path to save JSONL file
        num_samples: Number of samples to generate
    """
    import random
    
    # Sample templates
    templates = [
        {
            "instruction": "What is {a} + {b}?",
            "thought": "Let me add these numbers step by step. {a} + {b} = {result}",
            "response": "The answer is {result}.",
        },
        {
            "instruction": "Calculate {a} * {b}",
            "thought": "I need to multiply {a} by {b}. {a} * {b} = {result}",
            "response": "{result}",
        },
        {
            "instruction": "Is {num} even or odd?",
            "thought": "To determine if {num} is even or odd, I check if it's divisible by 2. {num} % 2 = {remainder}, so it's {parity}.",
            "response": "{num} is {parity}.",
        },
        {
            "instruction": "Write a Python function to {task}",
            "thought": "I'll create a simple function that {task}. I'll use clear variable names and add a docstring.",
            "response": "```python\n{code}\n```",
        },
    ]
    
    examples = []
    
    for i in range(num_samples):
        template = random.choice(templates)
        
        if "+" in template["instruction"]:
            a, b = random.randint(1, 100), random.randint(1, 100)
            result = a + b
            example = {
                "instruction": template["instruction"].format(a=a, b=b),
                "thought": template["thought"].format(a=a, b=b, result=result),
                "response": template["response"].format(result=result),
            }
        
        elif "*" in template["instruction"]:
            a, b = random.randint(1, 20), random.randint(1, 20)
            result = a * b
            example = {
                "instruction": template["instruction"].format(a=a, b=b),
                "thought": template["thought"].format(a=a, b=b, result=result),
                "response": template["response"].format(result=result),
            }
        
        elif "even or odd" in template["instruction"]:
            num = random.randint(1, 100)
            remainder = num % 2
            parity = "even" if remainder == 0 else "odd"
            example = {
                "instruction": template["instruction"].format(num=num),
                "thought": template["thought"].format(num=num, remainder=remainder, parity=parity),
                "response": template["response"].format(num=num, parity=parity),
            }
        
        elif "Python function" in template["instruction"]:
            tasks = [
                ("add two numbers", "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b"),
                ("check if even", "def is_even(n):\n    \"\"\"Check if number is even.\"\"\"\n    return n % 2 == 0"),
                ("reverse a string", "def reverse(s):\n    \"\"\"Reverse a string.\"\"\"\n    return s[::-1]"),
            ]
            task, code = random.choice(tasks)
            example = {
                "instruction": template["instruction"].format(task=task),
                "thought": template["thought"].format(task=task),
                "response": template["response"].format(code=code),
            }
        
        examples.append(example)
    
    # Save to JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"✅ Created {num_samples} sample examples at {output_path}")


def validate_cot_format(data_path: str) -> bool:
    """
    Validate that a JSONL file has correct CoT format.
    
    Args:
        data_path: Path to JSONL file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                example = json.loads(line)
                
                # Check format
                if 'messages' in example:
                    # Validate messages format
                    if not isinstance(example['messages'], list):
                        print(f"❌ Line {i}: 'messages' must be a list")
                        return False
                    
                    for msg in example['messages']:
                        if 'role' not in msg or 'content' not in msg:
                            print(f"❌ Line {i}: Message missing 'role' or 'content'")
                            return False
                
                elif 'instruction' in example and 'response' in example:
                    # Simplified format is valid
                    pass
                
                else:
                    print(f"❌ Line {i}: Unknown format (need 'messages' or 'instruction'+'response')")
                    return False
        
        print(f"✅ Format validation passed for {data_path}")
        return True
    
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def pull_and_process_hf_datasets(
    output_dir: str = "data/hf_cot",
    max_tokens: int = 480,
    tokenizer = None,
) -> tuple:
    """
    Pull and process HuggingFace datasets into CoT format.
    
    Sources:
    - microsoft/orca-math-word-problems-200k (5,000 samples) - Logic & Reasoning
    - databricks/databricks-dolly-15k (15,000 samples) - General Assistant
    - sahil2801/CodeAlpaca-20k (5,000 samples) - Coding & Technical
    
    Total: 25,000 samples
    
    Processing:
    - Filter by length (max 480 tokens for safety buffer)
    - Map to CoT format (User, Thought, Assistant)
    - Shuffle and merge
    - Split into train/val (90/10)
    
    Args:
        output_dir: Output directory for processed data
        max_tokens: Maximum token count per sample (default: 480)
        tokenizer: Tokenizer for length filtering (optional)
    
    Returns:
        (train_path, val_path) tuple
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ datasets library not installed. Install with: pip install datasets")
        return None, None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_examples = []
    
    print("\n" + "="*80)
    print("📥 Pulling and Processing HuggingFace Datasets")
    print("="*80 + "\n")
    
    # 1. Orca Math (5,000 samples) - Logic & Reasoning
    print("1️⃣  Processing microsoft/orca-math-word-problems-200k (Logic & Reasoning)...")
    print("   Target: 5,000 samples")
    try:
        orca_dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
        orca_samples = []
        
        for item in orca_dataset:
            if len(orca_samples) >= 5000:
                break
            
            # Map: question -> User, answer (with reasoning) -> Thought + Assistant
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            if not question or not answer:
                continue
            
            # Orca Math has detailed step-by-step solutions
            # Extract the reasoning process as "thought"
            thought = "Let me solve this step by step."
            response = answer
            
            # Try to split answer into reasoning and final answer
            if '\n' in answer:
                lines = answer.strip().split('\n')
                # Use most of the answer as thought, last line as response
                if len(lines) > 2:
                    thought = '\n'.join(lines[:-1])
                    response = lines[-1]
                elif len(lines) == 2:
                    thought = lines[0]
                    response = lines[1]
            
            example = {
                "instruction": question,
                "thought": thought,
                "response": response,
            }
            
            # Filter by length if tokenizer provided
            if tokenizer:
                text = f"System: You are a helpful assistant.\nUser: {question}\nThought: {thought}\nAssistant: {response}"
                tokens = tokenizer.encode(text)
                if len(tokens) > max_tokens:
                    continue
            
            orca_samples.append(example)
        
        all_examples.extend(orca_samples)
        print(f"   ✅ Processed {len(orca_samples)} math/logic samples")
    
    except Exception as e:
        print(f"   ⚠️  Error processing Orca Math: {e}")
    
    # 2. Databricks Dolly (ALL 15,000 samples) - General Assistant
    print("\n2️⃣  Processing databricks/databricks-dolly-15k (General Assistant)...")
    print("   Target: All 15,000 samples")
    try:
        dolly_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        dolly_samples = []
        
        for item in dolly_dataset:
            # Map: instruction + context -> User, response -> Assistant
            instruction = item.get('instruction', '')
            context = item.get('context', '')
            response = item.get('response', '')
            category = item.get('category', '')
            
            if not instruction or not response:
                continue
            
            # Combine instruction and context
            user_message = instruction
            if context:
                user_message = f"{context}\n\n{instruction}"
            
            # Create thought based on category
            if category == 'closed_qa':
                thought = "I'll analyze the context and provide a precise answer to this question."
            elif category == 'open_qa':
                thought = "Let me think about this question and provide a comprehensive answer."
            elif category == 'summarization':
                thought = "I'll read through the content and create a concise summary of the key points."
            elif category == 'information_extraction':
                thought = "I'll extract the relevant information from the provided text."
            elif category == 'creative_writing':
                thought = "I'll use creativity to craft an engaging response."
            elif category == 'general_qa':
                thought = "I'll provide a clear and helpful answer to this question."
            elif category == 'brainstorming':
                thought = "Let me generate some creative ideas for this."
            elif category == 'classification':
                thought = "I'll analyze this and provide the appropriate classification."
            else:
                thought = "I will analyze this request and provide a helpful response."
            
            example = {
                "instruction": user_message,
                "thought": thought,
                "response": response,
            }
            
            # Filter by length
            if tokenizer:
                text = f"System: You are a helpful assistant.\nUser: {user_message}\nThought: {thought}\nAssistant: {response}"
                tokens = tokenizer.encode(text)
                if len(tokens) > max_tokens:
                    continue
            
            dolly_samples.append(example)
        
        all_examples.extend(dolly_samples)
        print(f"   ✅ Processed {len(dolly_samples)} general Q&A samples")
    
    except Exception as e:
        print(f"   ⚠️  Error processing Dolly: {e}")
    
    # 3. CodeAlpaca (5,000 samples) - Coding & Technical
    print("\n3️⃣  Processing sahil2801/CodeAlpaca-20k (Coding & Technical)...")
    print("   Target: 5,000 samples")
    try:
        code_dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        code_samples = []
        
        for item in code_dataset:
            if len(code_samples) >= 5000:
                break
            
            # Map: instruction + input -> User, output -> Assistant
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            output = item.get('output', '')
            
            if not instruction or not output:
                continue
            
            # Combine instruction and input
            user_message = instruction
            if input_text:
                user_message = f"{instruction}\n\nInput: {input_text}"
            
            # Create thought for code generation
            thought = "I'll write clean, well-documented code to solve this problem."
            
            example = {
                "instruction": user_message,
                "thought": thought,
                "response": output,
            }
            
            # Filter by length
            if tokenizer:
                text = f"System: You are a helpful assistant.\nUser: {user_message}\nThought: {thought}\nAssistant: {output}"
                tokens = tokenizer.encode(text)
                if len(tokens) > max_tokens:
                    continue
            
            code_samples.append(example)
        
        all_examples.extend(code_samples)
        print(f"   ✅ Processed {len(code_samples)} code samples")
    
    except Exception as e:
        print(f"   ⚠️  Error processing CodeAlpaca: {e}")
    
    # Shuffle all examples
    print(f"\n4️⃣  Shuffling {len(all_examples)} total examples...")
    random.shuffle(all_examples)
    
    # Split into train/val (90/10)
    val_size = int(len(all_examples) * 0.1)
    val_examples = all_examples[:val_size]
    train_examples = all_examples[val_size:]
    
    # Save train
    train_path = output_dir / "train.jsonl"
    with open(train_path, 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    
    # Save val
    val_path = output_dir / "val.jsonl"
    with open(val_path, 'w', encoding='utf-8') as f:
        for example in val_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\n{'='*80}")
    print("✅ Dataset Processing Complete!")
    print(f"{'='*80}")
    print(f"\nDataset Composition:")
    print(f"  - Logic & Reasoning (Orca Math): ~5,000 samples")
    print(f"  - General Assistant (Dolly): ~15,000 samples")
    print(f"  - Coding & Technical (CodeAlpaca): ~5,000 samples")
    print(f"  - Total: {len(all_examples)} samples")
    print(f"\nSplit:")
    print(f"  - Train: {len(train_examples)} samples -> {train_path}")
    print(f"  - Val: {len(val_examples)} samples -> {val_path}")
    print(f"\nMax tokens per sample: {max_tokens}")
    print(f"Special tokens: <|im_start|>, <|im_end|>")
    print(f"{'='*80}\n")
    
    return str(train_path), str(val_path)


if __name__ == '__main__':
    # Test data creation
    print("Creating sample CoT data...")
    create_sample_cot_data("data/sample_train.jsonl", num_samples=100)
    create_sample_cot_data("data/sample_val.jsonl", num_samples=20)
    
    # Validate format
    print("\nValidating format...")
    validate_cot_format("data/sample_train.jsonl")
    validate_cot_format("data/sample_val.jsonl")
