#!/usr/bin/env python3
"""
Script to prepare fine-tuning data in Chain-of-Thought (CoT) format.
Creates sample data or converts existing datasets to CoT format.
"""
import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.finetuning.data_utils import create_sample_cot_data, validate_cot_format, pull_and_process_hf_datasets


def create_math_reasoning_data(output_path: str, num_samples: int = 1000):
    """Create math reasoning examples with CoT."""
    import random
    
    examples = []
    
    operations = [
        ('addition', lambda a, b: a + b, '+'),
        ('subtraction', lambda a, b: a - b, '-'),
        ('multiplication', lambda a, b: a * b, '*'),
        ('division', lambda a, b: a // b if b != 0 else 1, '//'),
    ]
    
    for _ in range(num_samples):
        op_name, op_func, op_symbol = random.choice(operations)
        
        if op_name == 'division':
            a = random.randint(10, 100)
            b = random.randint(2, 10)
        else:
            a = random.randint(1, 100)
            b = random.randint(1, 100)
        
        result = op_func(a, b)
        
        example = {
            "instruction": f"What is {a} {op_symbol} {b}?",
            "thought": f"Let me calculate this step by step. {a} {op_symbol} {b} = {result}",
            "response": f"The answer is {result}.",
        }
        
        examples.append(example)
    
    # Save to JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"✅ Created {num_samples} math reasoning examples at {output_path}")


def create_code_generation_data(output_path: str, num_samples: int = 500):
    """Create code generation examples with CoT."""
    import random
    
    templates = [
        {
            "task": "add two numbers",
            "thought": "I'll create a simple function that takes two parameters and returns their sum. I'll add a docstring for clarity.",
            "code": '''def add(a, b):
    """Add two numbers and return the result."""
    return a + b''',
        },
        {
            "task": "check if a number is even",
            "thought": "To check if a number is even, I'll use the modulo operator. If n % 2 equals 0, the number is even.",
            "code": '''def is_even(n):
    """Check if a number is even."""
    return n % 2 == 0''',
        },
        {
            "task": "reverse a string",
            "thought": "Python makes string reversal easy with slicing. I'll use [::-1] to reverse the string.",
            "code": '''def reverse_string(s):
    """Reverse a string."""
    return s[::-1]''',
        },
        {
            "task": "find the maximum in a list",
            "thought": "I'll iterate through the list and keep track of the maximum value seen so far.",
            "code": '''def find_max(numbers):
    """Find the maximum value in a list."""
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val''',
        },
        {
            "task": "calculate factorial",
            "thought": "I'll implement factorial recursively. Base case: factorial(0) = 1. Recursive case: factorial(n) = n * factorial(n-1).",
            "code": '''def factorial(n):
    """Calculate factorial of n."""
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)''',
        },
    ]
    
    examples = []
    
    for _ in range(num_samples):
        template = random.choice(templates)
        
        example = {
            "instruction": f"Write a Python function to {template['task']}.",
            "thought": template['thought'],
            "response": f"Here's a Python function to {template['task']}:\n\n```python\n{template['code']}\n```",
        }
        
        examples.append(example)
    
    # Save to JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"✅ Created {num_samples} code generation examples at {output_path}")


def create_general_qa_data(output_path: str, num_samples: int = 500):
    """Create general Q&A examples with CoT."""
    import random
    
    qa_pairs = [
        {
            "question": "What is the capital of France?",
            "thought": "France is a country in Western Europe. Its capital city is Paris, which is also its largest city.",
            "answer": "The capital of France is Paris.",
        },
        {
            "question": "How many days are in a leap year?",
            "thought": "A leap year occurs every 4 years and has an extra day in February. Regular years have 365 days, so leap years have 366 days.",
            "answer": "A leap year has 366 days.",
        },
        {
            "question": "What is photosynthesis?",
            "thought": "Photosynthesis is the process by which plants convert light energy into chemical energy. They use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
            "answer": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "thought": "Romeo and Juliet is a famous tragedy about two young lovers. It was written by William Shakespeare in the late 16th century.",
            "answer": "William Shakespeare wrote Romeo and Juliet.",
        },
        {
            "question": "What is the speed of light?",
            "thought": "The speed of light in a vacuum is a fundamental constant in physics. It's approximately 299,792,458 meters per second, often rounded to 3 × 10^8 m/s.",
            "answer": "The speed of light is approximately 299,792,458 meters per second (or about 3 × 10^8 m/s).",
        },
    ]
    
    examples = []
    
    for _ in range(num_samples):
        qa = random.choice(qa_pairs)
        
        example = {
            "instruction": qa['question'],
            "thought": qa['thought'],
            "response": qa['answer'],
        }
        
        examples.append(example)
    
    # Save to JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"✅ Created {num_samples} general Q&A examples at {output_path}")


def merge_datasets(input_paths: list, output_path: str):
    """Merge multiple JSONL datasets into one."""
    all_examples = []
    
    for input_path in input_paths:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_examples.append(json.loads(line))
    
    # Shuffle
    import random
    random.shuffle(all_examples)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"✅ Merged {len(all_examples)} examples to {output_path}")


def split_dataset(input_path: str, train_path: str, val_path: str, val_ratio: float = 0.1):
    """Split dataset into train and validation sets."""
    import random
    
    # Load all examples
    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    # Shuffle
    random.shuffle(examples)
    
    # Split
    val_size = int(len(examples) * val_ratio)
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]
    
    # Save train
    train_path = Path(train_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    
    # Save val
    val_path = Path(val_path)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_path, 'w', encoding='utf-8') as f:
        for example in val_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"✅ Split dataset:")
    print(f"   Train: {len(train_examples)} examples -> {train_path}")
    print(f"   Val: {len(val_examples)} examples -> {val_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare fine-tuning data in CoT format')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Sample data command
    sample_parser = subparsers.add_parser('sample', help='Create sample data')
    sample_parser.add_argument('--output', default='data/sample_cot.jsonl', help='Output path')
    sample_parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples')
    
    # Math reasoning command
    math_parser = subparsers.add_parser('math', help='Create math reasoning data')
    math_parser.add_argument('--output', default='data/math_cot.jsonl', help='Output path')
    math_parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples')
    
    # Code generation command
    code_parser = subparsers.add_parser('code', help='Create code generation data')
    code_parser.add_argument('--output', default='data/code_cot.jsonl', help='Output path')
    code_parser.add_argument('--num-samples', type=int, default=500, help='Number of samples')
    
    # General Q&A command
    qa_parser = subparsers.add_parser('qa', help='Create general Q&A data')
    qa_parser.add_argument('--output', default='data/qa_cot.jsonl', help='Output path')
    qa_parser.add_argument('--num-samples', type=int, default=500, help='Number of samples')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge multiple datasets')
    merge_parser.add_argument('--inputs', nargs='+', required=True, help='Input JSONL files')
    merge_parser.add_argument('--output', required=True, help='Output path')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val')
    split_parser.add_argument('--input', required=True, help='Input JSONL file')
    split_parser.add_argument('--train', default='data/cot_train.jsonl', help='Train output path')
    split_parser.add_argument('--val', default='data/cot_val.jsonl', help='Val output path')
    split_parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation ratio')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate CoT format')
    validate_parser.add_argument('input', help='Input JSONL file')
    
    # All command (create complete dataset)
    all_parser = subparsers.add_parser('all', help='Create complete dataset (math + code + qa)')
    all_parser.add_argument('--output-dir', default='data', help='Output directory')
    all_parser.add_argument('--math-samples', type=int, default=1000, help='Math samples')
    all_parser.add_argument('--code-samples', type=int, default=500, help='Code samples')
    all_parser.add_argument('--qa-samples', type=int, default=500, help='Q&A samples')
    all_parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation ratio')
    
    # HuggingFace datasets command
    hf_parser = subparsers.add_parser('hf', help='Pull and process HuggingFace datasets')
    hf_parser.add_argument('--output-dir', default='data/hf_cot', help='Output directory')
    hf_parser.add_argument('--max-tokens', type=int, default=480, help='Max tokens per sample')
    
    args = parser.parse_args()
    
    if args.command == 'sample':
        create_sample_cot_data(args.output, args.num_samples)
    
    elif args.command == 'math':
        create_math_reasoning_data(args.output, args.num_samples)
    
    elif args.command == 'code':
        create_code_generation_data(args.output, args.num_samples)
    
    elif args.command == 'qa':
        create_general_qa_data(args.output, args.num_samples)
    
    elif args.command == 'merge':
        merge_datasets(args.inputs, args.output)
    
    elif args.command == 'split':
        split_dataset(args.input, args.train, args.val, args.val_ratio)
    
    elif args.command == 'validate':
        validate_cot_format(args.input)
    
    elif args.command == 'all':
        print("\n" + "="*80)
        print("Creating Complete Fine-Tuning Dataset")
        print("="*80 + "\n")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create individual datasets
        math_path = output_dir / 'math_cot.jsonl'
        code_path = output_dir / 'code_cot.jsonl'
        qa_path = output_dir / 'qa_cot.jsonl'
        
        create_math_reasoning_data(str(math_path), args.math_samples)
        create_code_generation_data(str(code_path), args.code_samples)
        create_general_qa_data(str(qa_path), args.qa_samples)
        
        # Merge
        merged_path = output_dir / 'cot_all.jsonl'
        merge_datasets([str(math_path), str(code_path), str(qa_path)], str(merged_path))
        
        # Split
        train_path = output_dir / 'cot_train.jsonl'
        val_path = output_dir / 'cot_val.jsonl'
        split_dataset(str(merged_path), str(train_path), str(val_path), args.val_ratio)
        
        print("\n" + "="*80)
        print("✅ Complete Dataset Created!")
        print("="*80)
        print(f"\nFiles created:")
        print(f"  - {train_path}")
        print(f"  - {val_path}")
        print(f"\nReady for fine-tuning!")
        print("="*80 + "\n")
    
    elif args.command == 'hf':
        # Load tokenizer for length filtering
        try:
            from src.tokenizer_utils import load_tokenizer
            tokenizer = load_tokenizer()
            print("✅ Loaded tokenizer for length filtering")
        except Exception as e:
            print(f"⚠️  Could not load tokenizer: {e}")
            print("   Proceeding without length filtering")
            tokenizer = None
        
        train_path, val_path = pull_and_process_hf_datasets(
            output_dir=args.output_dir,
            max_tokens=args.max_tokens,
            tokenizer=tokenizer,
        )
        
        if train_path and val_path:
            print("\n✅ HuggingFace datasets processed successfully!")
            print(f"\nUse these files for fine-tuning:")
            print(f"  python src/finetuning/gpu_finetune.py \\")
            print(f"    --train-data {train_path} \\")
            print(f"    --val-data {val_path}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
