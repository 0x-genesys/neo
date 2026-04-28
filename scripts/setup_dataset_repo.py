#!/usr/bin/env python3
"""
Setup script to help configure the correct HuggingFace dataset repository.

This script helps you:
1. Check if you have existing datasets on HuggingFace Hub
2. Upload your local dataset if needed
3. Update configs with the correct repository ID
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, list_repo_files
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def check_hf_auth():
    """Check if user is authenticated with HuggingFace Hub."""
    token = os.environ.get('HF_TOKEN')
    if not token:
        print("⚠️  No HuggingFace token found.")
        print("   Set your token: export HF_TOKEN=your_token_here")
        print("   Get token from: https://huggingface.co/settings/tokens")
        return None
    
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info['name']
        print(f"✅ Authenticated as: {username}")
        return username, token
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return None


def find_user_datasets(username, token):
    """Find datasets owned by the user."""
    try:
        api = HfApi(token=token)
        datasets = api.list_datasets(author=username)
        
        dataset_list = []
        for dataset in datasets:
            dataset_list.append(dataset.id)
        
        return dataset_list
    except Exception as e:
        print(f"⚠️  Could not list datasets: {e}")
        return []


def check_repository_exists(repo_id, token):
    """Check if a repository exists and list its contents."""
    try:
        files = list_repo_files(repo_id, repo_type="dataset", token=token)
        dataset_files = [f for f in files if f.endswith(('.bin', '.json'))]
        
        print(f"✅ Repository exists: {repo_id}")
        print(f"   Files: {', '.join(dataset_files)}")
        return True, dataset_files
    except Exception as e:
        if "404" in str(e):
            print(f"❌ Repository not found: {repo_id}")
        else:
            print(f"❌ Error checking repository: {e}")
        return False, []


def suggest_repository_name(username):
    """Suggest a repository name for the dataset."""
    suggestions = [
        f"{username}/balanced-300m-tokens",
        f"{username}/mix-wiki-code-chat-300m",
        f"{username}/transformer-training-data-300m"
    ]
    return suggestions


def check_local_dataset():
    """Check if local dataset exists."""
    local_paths = [
        "data/balanced_300m",
        "data/balanced_300m_old_no_cod"
    ]
    
    existing_datasets = []
    for path in local_paths:
        dataset_path = Path(path)
        if dataset_path.exists():
            train_file = dataset_path / "train.bin"
            val_file = dataset_path / "val.bin"
            stats_file = dataset_path / "dataset_stats.json"
            
            if train_file.exists():
                size_mb = train_file.stat().st_size / (1024 * 1024)
                existing_datasets.append({
                    'path': str(dataset_path),
                    'train_size_mb': size_mb,
                    'has_val': val_file.exists(),
                    'has_stats': stats_file.exists()
                })
    
    return existing_datasets


def update_config_files(repo_id):
    """Update config files with the correct repository ID."""
    config_files = [
        "config/gpu_training_117m_balanced.yaml",
        "config/gpu_training_117m_1.5gb.yaml"
    ]
    
    updated_files = []
    
    for config_file in config_files:
        config_path = Path(config_file)
        if not config_path.exists():
            continue
        
        try:
            # Read config
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Update repo_id
            if 'repo_id:' in content:
                # Replace the repo_id line
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'repo_id:' in line and 'huggingface_dataset:' in content[:content.find(line)]:
                        # Update the repo_id
                        indent = len(line) - len(line.lstrip())
                        lines[i] = ' ' * indent + f'repo_id: "{repo_id}"'
                        break
                
                # Write back
                with open(config_path, 'w') as f:
                    f.write('\n'.join(lines))
                
                updated_files.append(config_file)
        
        except Exception as e:
            print(f"⚠️  Could not update {config_file}: {e}")
    
    return updated_files


def main():
    """Main setup workflow."""
    print("🔧 HuggingFace Dataset Repository Setup")
    print("="*50)
    
    # Step 1: Check authentication
    auth_result = check_hf_auth()
    if not auth_result:
        print("\n💡 To continue, you need a HuggingFace token:")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Create a new token with 'write' permissions")
        print("3. Set it: export HF_TOKEN=your_token_here")
        print("4. Run this script again")
        return
    
    username, token = auth_result
    
    # Step 2: Check local datasets
    print(f"\n📁 Checking local datasets...")
    local_datasets = check_local_dataset()
    
    if local_datasets:
        print(f"✅ Found {len(local_datasets)} local dataset(s):")
        for dataset in local_datasets:
            print(f"  - {dataset['path']}: {dataset['train_size_mb']:.1f}MB")
            print(f"    Val file: {'✅' if dataset['has_val'] else '❌'}")
            print(f"    Stats file: {'✅' if dataset['has_stats'] else '❌'}")
    else:
        print("❌ No local datasets found.")
        print("   Build one first: python scripts/prepare_balanced_dataset.py")
        return
    
    # Step 3: Check existing repositories
    print(f"\n🔍 Checking your HuggingFace repositories...")
    user_datasets = find_user_datasets(username, token)
    
    if user_datasets:
        print(f"✅ Found {len(user_datasets)} dataset(s) on your HuggingFace profile:")
        for dataset_id in user_datasets:
            exists, files = check_repository_exists(dataset_id, token)
            if exists and any(f.endswith('.bin') for f in files):
                print(f"  🎯 {dataset_id} - Has binary files (suitable for training)")
            else:
                print(f"  - {dataset_id}")
    else:
        print("❌ No datasets found on your HuggingFace profile.")
    
    # Step 4: Interactive setup
    print(f"\n🎯 Setup Options:")
    print("1. Use existing HuggingFace repository")
    print("2. Upload local dataset to new repository")
    print("3. Exit and upload manually")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        if not user_datasets:
            print("❌ No existing repositories found.")
            return
        
        print("\nAvailable repositories:")
        for i, dataset_id in enumerate(user_datasets, 1):
            print(f"{i}. {dataset_id}")
        
        try:
            repo_choice = int(input(f"\nSelect repository (1-{len(user_datasets)}): ")) - 1
            selected_repo = user_datasets[repo_choice]
            
            # Update configs
            updated_files = update_config_files(selected_repo)
            print(f"\n✅ Updated config files: {', '.join(updated_files)}")
            print(f"🎯 Repository configured: {selected_repo}")
            
        except (ValueError, IndexError):
            print("❌ Invalid selection.")
            return
    
    elif choice == "2":
        if not local_datasets:
            print("❌ No local datasets to upload.")
            return
        
        # Suggest repository names
        suggestions = suggest_repository_name(username)
        print(f"\n💡 Suggested repository names:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
        
        repo_name = input(f"\nEnter repository name (or choose 1-{len(suggestions)}): ").strip()
        
        # Handle numeric choice
        try:
            if repo_name.isdigit():
                repo_choice = int(repo_name) - 1
                repo_name = suggestions[repo_choice]
        except (ValueError, IndexError):
            pass
        
        # Ensure proper format
        if '/' not in repo_name:
            repo_name = f"{username}/{repo_name}"
        
        # Select local dataset
        if len(local_datasets) == 1:
            selected_dataset = local_datasets[0]['path']
        else:
            print(f"\nSelect local dataset:")
            for i, dataset in enumerate(local_datasets, 1):
                print(f"{i}. {dataset['path']} ({dataset['train_size_mb']:.1f}MB)")
            
            try:
                dataset_choice = int(input(f"Select dataset (1-{len(local_datasets)}): ")) - 1
                selected_dataset = local_datasets[dataset_choice]['path']
            except (ValueError, IndexError):
                print("❌ Invalid selection.")
                return
        
        print(f"\n📤 Ready to upload:")
        print(f"  Local dataset: {selected_dataset}")
        print(f"  HuggingFace repo: {repo_name}")
        
        confirm = input("Proceed with upload? (y/N): ").strip().lower()
        if confirm == 'y':
            print(f"\n🚀 Uploading dataset...")
            print(f"Run this command:")
            print(f"python scripts/upload_dataset_to_hf.py {selected_dataset} {repo_name}")
            
            # Update configs
            updated_files = update_config_files(repo_name)
            print(f"\n✅ Updated config files: {', '.join(updated_files)}")
            print(f"🎯 Repository configured: {repo_name}")
        else:
            print("❌ Upload cancelled.")
    
    elif choice == "3":
        print(f"\n💡 Manual upload instructions:")
        if local_datasets:
            dataset_path = local_datasets[0]['path']
            suggested_repo = suggest_repository_name(username)[0]
            print(f"1. Upload dataset:")
            print(f"   python scripts/upload_dataset_to_hf.py {dataset_path} {suggested_repo}")
            print(f"2. Update configs with repo_id: {suggested_repo}")
        print("3. Test the setup:")
        print("   python scripts/test_dataset_download.py")
    
    else:
        print("❌ Invalid choice.")
    
    print(f"\n🎉 Setup complete!")
    print(f"Test your configuration:")
    print(f"  python scripts/test_dataset_download.py")


if __name__ == '__main__':
    main()