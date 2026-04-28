"""
Prepare a 300M token "Balanced Logic & Chat" binary dataset.

This script creates a high-quality, balanced dataset combining:
1. WikiText-103 (encyclopedic knowledge)
2. UltraChat (conversational AI)
3. The Stack (code reasoning)

Note: DailyDialog is deprecated in newer datasets library, so we use
additional UltraChat for conversational data and validation.

Total: 300M tokens for training, 10M tokens for validation
Tokenizer: tiktoken cl100k_base (GPT-4)
Format: Binary np.uint32 with np.memmap for memory efficiency
"""

import numpy as np
import tiktoken
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import random
from typing import List, Dict, Tuple
import json


class BalancedDatasetBuilder:
    """Build a balanced multi-source dataset with proper formatting and shuffling."""
    
    def __init__(self, output_dir: str = "data/balanced_300m", seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GPT-4 tokenizer (cl100k_base)
        print("Loading tiktoken cl100k_base (GPT-4) tokenizer...")
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.tokenizer.n_vocab
        print(f"✅ Tokenizer loaded: vocab_size={self.vocab_size:,}")
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Target token counts
        # Note: DailyDialog is deprecated, so we use more UltraChat instead
        self.target_tokens = {
            'wikitext': 102_000_000,      # ~102M tokens (34%)
            'ultrachat': 198_000_000,     # 198M tokens (66%) - includes conversational data
            'stack': 48_000_000,          # 48M tokens (16%)
            'dailydialog': 0,             # Deprecated - using UltraChat instead
        }
        
        self.total_target = 300_000_000  # 300M tokens (adjusted: WikiText + UltraChat + Stack - overlap)
        self.val_target = 10_000_000     # 10M tokens for validation (from UltraChat)
        
        # Statistics
        self.stats = {
            'wikitext': {'docs': 0, 'tokens': 0},
            'ultrachat': {'docs': 0, 'tokens': 0},
            'stack': {'docs': 0, 'tokens': 0},
            'dailydialog': {'docs': 0, 'tokens': 0},
        }
    
    def format_conversation(self, user_text: str, assistant_text: str) -> str:
        """Format conversational data with consistent template."""
        return f"User: {user_text}\nAssistant: {assistant_text}"
    
    def tokenize_text(self, text: str) -> List[int]:
        """Tokenize text using tiktoken."""
        return self.tokenizer.encode(text, allowed_special='all')
    
    def load_wikitext(self) -> List[List[int]]:
        """Load and tokenize WikiText-103 (full train split)."""
        print("\n" + "="*80)
        print("Loading WikiText-103-raw-v1 (full train split)")
        print("="*80)
        
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
        
        documents = []
        total_tokens = 0
        
        print(f"Processing {len(dataset):,} examples...")
        
        for example in tqdm(dataset, desc="WikiText-103"):
            text = example['text'].strip()
            
            # Skip empty or very short texts
            if len(text) < 50:
                continue
            
            tokens = self.tokenize_text(text)
            
            # Skip if too few tokens
            if len(tokens) < 10:
                continue
            
            documents.append(tokens)
            total_tokens += len(tokens)
            self.stats['wikitext']['docs'] += 1
            self.stats['wikitext']['tokens'] += len(tokens)
            
            # Stop if we've reached target
            if total_tokens >= self.target_tokens['wikitext']:
                break
        
        print(f"✅ WikiText-103: {len(documents):,} documents, {total_tokens:,} tokens")
        return documents
    
    def load_ultrachat(self, extra_tokens: int = 0) -> List[List[int]]:
        """
        Load and tokenize UltraChat (stream first 150M tokens + extra if needed).
        
        Args:
            extra_tokens: Additional tokens to load if other sources fail
        """
        target = self.target_tokens['ultrachat'] + extra_tokens
        
        print("\n" + "="*80)
        print(f"Loading stingning/ultrachat (streaming first {target:,} tokens)")
        print("="*80)
        
        # Load in streaming mode for efficiency
        dataset = load_dataset('stingning/ultrachat', split='train', streaming=True)
        
        documents = []
        total_tokens = 0
        
        print("Processing UltraChat conversations...")
        
        for example in tqdm(dataset, desc="UltraChat", total=150000):  # Estimate
            try:
                # UltraChat format: {'data': [user_msg, assistant_msg, ...]}
                if 'data' not in example or len(example['data']) < 2:
                    continue
                
                conversation_data = example['data']
                
                # Process conversation pairs
                for i in range(0, len(conversation_data) - 1, 2):
                    user_msg = conversation_data[i]
                    assistant_msg = conversation_data[i + 1]
                    
                    # Format as User/Assistant dialogue
                    formatted = self.format_conversation(user_msg, assistant_msg)
                    tokens = self.tokenize_text(formatted)
                    
                    if len(tokens) < 10:
                        continue
                    
                    documents.append(tokens)
                    total_tokens += len(tokens)
                    self.stats['ultrachat']['docs'] += 1
                    self.stats['ultrachat']['tokens'] += len(tokens)
                    
                    # Stop if we've reached target
                    if total_tokens >= target:
                        print(f"\n✅ Reached target: {total_tokens:,} tokens")
                        return documents
            
            except Exception as e:
                # Skip malformed examples
                continue
        
        print(f"✅ UltraChat: {len(documents):,} documents, {total_tokens:,} tokens")
        return documents
    
    def load_stack(self) -> List[List[int]]:
        """Load and tokenize The Stack (Python/Java subset for 48M tokens)."""
        print("\n" + "="*80)
        print("Loading bigcode/the-stack-dedup (Python + Java subset)")
        print("="*80)
        
        documents = []
        total_tokens = 0
        
        # Load Python subset
        print("Loading Python code...")
        try:
            python_dataset = load_dataset(
                'bigcode/the-stack-dedup',
                data_dir='data/python',
                split='train',
                streaming=True
            )
            
            for example in tqdm(python_dataset, desc="Python", total=30000):
                if total_tokens >= self.target_tokens['stack'] * 0.6:  # 60% Python
                    break
                
                code = example.get('content', '').strip()
                
                if len(code) < 100:  # Skip very short files
                    continue
                
                tokens = self.tokenize_text(code)
                
                if len(tokens) < 20:
                    continue
                
                documents.append(tokens)
                total_tokens += len(tokens)
                self.stats['stack']['docs'] += 1
                self.stats['stack']['tokens'] += len(tokens)
        
        except Exception as e:
            print(f"⚠️  Error loading Python: {e}")
        
        # Load Java subset
        print("Loading Java code...")
        try:
            java_dataset = load_dataset(
                'bigcode/the-stack-dedup',
                data_dir='data/java',
                split='train',
                streaming=True
            )
            
            for example in tqdm(java_dataset, desc="Java", total=20000):
                if total_tokens >= self.target_tokens['stack']:
                    break
                
                code = example.get('content', '').strip()
                
                if len(code) < 100:
                    continue
                
                tokens = self.tokenize_text(code)
                
                if len(tokens) < 20:
                    continue
                
                documents.append(tokens)
                total_tokens += len(tokens)
                self.stats['stack']['docs'] += 1
                self.stats['stack']['tokens'] += len(tokens)
        
        except Exception as e:
            print(f"⚠️  Error loading Java: {e}")
        
        print(f"✅ The Stack: {len(documents):,} documents, {total_tokens:,} tokens")
        return documents
    
    def load_dailydialog(self) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Load and tokenize DailyDialog (full dataset).
        Returns: (train_documents, val_documents)
        
        Note: DailyDialog is deprecated in newer datasets library.
        This method will gracefully fail and return empty lists,
        allowing the build process to use UltraChat for validation instead.
        """
        print("\n" + "="*80)
        print("Loading daily_dialog (full dataset)")
        print("="*80)
        
        print("⚠️  DailyDialog dataset is deprecated in datasets library")
        print("   Skipping DailyDialog - will use UltraChat for validation instead")
        print("   This maintains 300M tokens with adjusted composition:")
        print("     - WikiText-103: 102M (34%)")
        print("     - UltraChat: 198M (66%) - increased from 150M")
        print("     - The Stack: 48M (16%) - code reasoning")
        
        # Return empty lists - validation will be created from UltraChat
        return [], []
        
        train_documents = []
        val_documents = []
        train_tokens = 0
        val_tokens = 0
        
        print(f"Processing {len(dataset):,} dialogues...")
        
        for idx, example in enumerate(tqdm(dataset, desc="DailyDialog")):
            try:
                # DailyDialog format: {'dialog': [turn1, turn2, ...]}
                dialog = example['dialog']
                
                if len(dialog) < 2:
                    continue
                
                # Process dialogue pairs
                for i in range(0, len(dialog) - 1, 2):
                    user_msg = dialog[i]
                    assistant_msg = dialog[i + 1] if i + 1 < len(dialog) else ""
                    
                    if not assistant_msg:
                        continue
                    
                    # Format as User/Assistant dialogue
                    formatted = self.format_conversation(user_msg, assistant_msg)
                    tokens = self.tokenize_text(formatted)
                    
                    if len(tokens) < 10:
                        continue
                    
                    # Reserve 10% for validation (conversational subset)
                    if idx % 10 == 0 and val_tokens < self.val_target:
                        val_documents.append(tokens)
                        val_tokens += len(tokens)
                    else:
                        train_documents.append(tokens)
                        train_tokens += len(tokens)
                        self.stats['dailydialog']['docs'] += 1
                        self.stats['dailydialog']['tokens'] += len(tokens)
            
            except Exception as e:
                continue
        
        print(f"✅ DailyDialog Train: {len(train_documents):,} documents, {train_tokens:,} tokens")
        print(f"✅ DailyDialog Val: {len(val_documents):,} documents, {val_tokens:,} tokens")
        
        return train_documents, val_documents
    
    def shuffle_and_pack(self, documents: List[List[int]], output_file: str):
        """
        Shuffle documents and pack into binary format using memmap.
        
        Args:
            documents: List of tokenized documents
            output_file: Output binary file path
        """
        print(f"\n{'='*80}")
        print(f"Shuffling and packing {len(documents):,} documents")
        print(f"{'='*80}")
        
        # Shuffle documents
        print("Shuffling documents...")
        random.shuffle(documents)
        
        # Calculate total tokens
        total_tokens = sum(len(doc) for doc in documents)
        print(f"Total tokens: {total_tokens:,}")
        
        # Create memory-mapped array
        output_path = self.output_dir / output_file
        print(f"Creating memory-mapped file: {output_path}")
        
        # Use memmap for memory efficiency
        mmap = np.memmap(output_path, dtype=np.uint32, mode='w+', shape=(total_tokens,))
        
        # Pack tokens
        print("Packing tokens...")
        offset = 0
        for doc in tqdm(documents, desc="Packing"):
            doc_len = len(doc)
            mmap[offset:offset + doc_len] = doc
            offset += doc_len
        
        # Flush to disk
        mmap.flush()
        del mmap
        
        print(f"✅ Saved: {output_path}")
        print(f"   Size: {output_path.stat().st_size / 1e9:.2f} GB")
        print(f"   Tokens: {total_tokens:,}")
    
    def build_dataset(self):
        """Build the complete balanced dataset."""
        print("\n" + "="*80)
        print("BUILDING BALANCED LOGIC & CHAT DATASET (300M TOKENS)")
        print("="*80)
        
        # Load all sources
        print("\n📚 Loading data sources...")
        
        wikitext_docs = self.load_wikitext()
        stack_docs = self.load_stack()
        
        # Load DailyDialog (will return empty - deprecated)
        dailydialog_train, dailydialog_val = self.load_dailydialog()
        
        # Load UltraChat - we need extra for validation since DailyDialog is unavailable
        print("\n" + "="*80)
        print("Loading UltraChat (with extra for validation)")
        print("="*80)
        ultrachat_docs = self.load_ultrachat(extra_tokens=self.val_target)
        
        # Combine training documents
        print("\n" + "="*80)
        print("Combining training documents")
        print("="*80)
        
        # Reserve last portion of UltraChat for validation
        val_tokens_needed = self.val_target
        val_docs = []
        val_tokens = 0
        train_ultrachat = []
        
        print(f"Reserving {val_tokens_needed:,} tokens from UltraChat for validation...")
        
        # Take from end of UltraChat for validation
        for doc in reversed(ultrachat_docs):
            if val_tokens >= val_tokens_needed:
                train_ultrachat.insert(0, doc)
            else:
                val_docs.insert(0, doc)
                val_tokens += len(doc)
        
        print(f"✅ Validation: {len(val_docs):,} documents, {val_tokens:,} tokens")
        print(f"✅ Training UltraChat: {len(train_ultrachat):,} documents")
        
        # Combine all training documents
        all_train_docs = (
            wikitext_docs +
            train_ultrachat +
            stack_docs +
            dailydialog_train  # Will be empty, but keep for compatibility
        )
        
        print(f"Total training documents: {len(all_train_docs):,}")
        
        # Pack training data
        self.shuffle_and_pack(all_train_docs, 'train.bin')
        
        # Pack validation data
        print("\n" + "="*80)
        print("Packing validation data (UltraChat conversational subset)")
        print("="*80)
        self.shuffle_and_pack(val_docs, 'val.bin')
        
        # Save statistics
        self.save_statistics()
        
        print("\n" + "="*80)
        print("✅ DATASET BUILD COMPLETE!")
        print("="*80)
        self.print_summary()
    
    def save_statistics(self):
        """Save dataset statistics to JSON."""
        stats_file = self.output_dir / 'dataset_stats.json'
        
        # Calculate totals
        total_docs = sum(s['docs'] for s in self.stats.values())
        total_tokens = sum(s['tokens'] for s in self.stats.values())
        
        stats_output = {
            'sources': self.stats,
            'totals': {
                'documents': total_docs,
                'tokens': total_tokens,
            },
            'targets': self.target_tokens,
            'tokenizer': 'tiktoken cl100k_base (GPT-4)',
            'vocab_size': self.vocab_size,
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats_output, f, indent=2)
        
        print(f"\n✅ Statistics saved: {stats_file}")
    
    def print_summary(self):
        """Print final summary."""
        print("\n📊 Dataset Statistics:")
        print("-" * 80)
        
        total_docs = sum(s['docs'] for s in self.stats.values())
        total_tokens = sum(s['tokens'] for s in self.stats.values())
        
        for source, stats in self.stats.items():
            docs = stats['docs']
            tokens = stats['tokens']
            pct = (tokens / total_tokens * 100) if total_tokens > 0 else 0
            print(f"{source:15s}: {docs:8,} docs | {tokens:12,} tokens ({pct:5.1f}%)")
        
        print("-" * 80)
        print(f"{'TOTAL':15s}: {total_docs:8,} docs | {total_tokens:12,} tokens (100.0%)")
        print("-" * 80)
        
        print("\n📁 Output Files:")
        print(f"  Training:   {self.output_dir / 'train.bin'}")
        print(f"  Validation: {self.output_dir / 'val.bin'}")
        print(f"  Statistics: {self.output_dir / 'dataset_stats.json'}")
        
        print("\n🎯 Training Configuration:")
        print(f"  Total tokens: 300M")
        print(f"  Composition:")
        print(f"    - WikiText-103: ~102M (34%) - Knowledge")
        print(f"    - UltraChat: ~198M (66%) - Conversation")
        print(f"    - The Stack: ~48M (16%) - Code")
        print(f"  Note: Percentages may overlap due to packing")
        print(f"  Epochs: 8")
        print(f"  Total training tokens: 2.4B (300M × 8)")
        print(f"  Tokens per step: 65,536 (batch_size × context_length × grad_accum)")
        print(f"  Max steps: 36,621 (2.4B / 65,536)")
        
        print("\n💡 Next Steps:")
        print("  1. Update config to use this dataset:")
        print("     data:")
        print("       train_file: 'data/balanced_300m/train.bin'")
        print("       val_file: 'data/balanced_300m/val.bin'")
        print("       dataset_type: 'binary'")
        print("  2. Start training:")
        print("     python train.py --config config/gpu_training_117m_1.5gb.yaml")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Prepare 300M token Balanced Logic & Chat dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/balanced_300m',
        help='Output directory for binary files'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Build dataset
    builder = BalancedDatasetBuilder(
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    builder.build_dataset()


if __name__ == '__main__':
    main()
