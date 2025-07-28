"""
Breast Cancer Dataset Loader with Class Balancing
Implements 50/50 class distribution through augmentation and reduction strategies
"""

import os
import random
import platform
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import yaml


class BreastCancerDataset(Dataset):
    """
    Dataset class for breast cancer histopathology images
    Handles patient-level splits and class balancing
    """
    
    def __init__(
        self, 
        data_path: str,
        split: str = 'train',
        config: Dict = None,
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize dataset
        
        Args:
            data_path: Path to BHI dataset
            split: 'train', 'val', or 'test' 
            config: Configuration dictionary
            transform: Image transformations
            target_transform: Label transformations
        """
        self.data_path = Path(data_path)
        self.split = split
        self.config = config or {}
        self.transform = transform
        self.target_transform = target_transform
        
        # Load dataset configuration
        self.data_percentage = self.config.get('data', {}).get('data_percentage', 1.0)
        self.class_balance_config = self.config.get('class_balance', {})
        self.augmentation_config = self.config.get('augmentation', {})
        
        # Debug: print data percentage
        print(f"DEBUG: data_percentage = {self.data_percentage} (from config: {self.config.get('data', {})})")
        
        # Initialize data structures
        self.samples = []
        self.class_counts = {'0': 0, '1': 0}  # cancerous: 0, healthy: 1
        self.patient_data = defaultdict(self._create_patient_dict)
        
        # Load and process dataset
        self._load_dataset()
        self._apply_class_balancing()
    
    def _create_patient_dict(self):
        """Create patient dictionary structure"""
        return {'0': [], '1': []}
        
    def _load_dataset(self):
        """Load images and organize by patient and class"""
        print(f"Loading {self.split} dataset from {self.data_path}")
        
        # Collect all patient data
        patient_folders = [f for f in self.data_path.iterdir() 
                          if f.is_dir() and f.name != "IDC_regular_ps50_idx5"]
        
        # Apply data percentage limit
        total_patient_folders = len(patient_folders)
        print(f"DEBUG: Found {total_patient_folders} patient folders, data_percentage = {self.data_percentage}")
        
        if self.data_percentage < 1.0:
            n_patients = max(1, int(len(patient_folders) * self.data_percentage))  # Ensure at least 1 patient
            if n_patients < len(patient_folders):
                patient_folders = random.sample(patient_folders, n_patients)
                print(f"Using {n_patients}/{total_patient_folders} patients ({self.data_percentage:.1%})")
            else:
                print(f"Using all {total_patient_folders} patients (percentage would select {n_patients})")
        else:
            print(f"Using all {total_patient_folders} patients")
        
        # Load images by patient and class
        for patient_folder in patient_folders:
            patient_id = patient_folder.name
            
            for class_folder in patient_folder.iterdir():
                if class_folder.is_dir() and class_folder.name in ['0', '1']:
                    class_label = class_folder.name
                    
                    for img_file in class_folder.iterdir():
                        if img_file.suffix.lower() == '.png':
                            self.patient_data[patient_id][class_label].append({
                                'path': img_file,
                                'patient_id': patient_id,
                                'label': int(class_label)
                            })
        
        # Split patients into train/val sets to prevent data leakage
        patient_ids = list(self.patient_data.keys())
        
        # Handle train/val split with minimum patient requirements
        if len(patient_ids) < 5:
            # For very small datasets, use simple split without stratification
            print(f"Very small dataset ({len(patient_ids)} patients), using simple 80/20 split")
            split_idx = max(1, int(len(patient_ids) * 0.8))
            random.shuffle(patient_ids)
            train_patients = patient_ids[:split_idx]
            val_patients = patient_ids[split_idx:] if split_idx < len(patient_ids) else [patient_ids[-1]]
        else:
            # Try stratified split, fall back to random split if classes are too small
            try:
                stratify_labels = self._get_patient_stratification_labels(patient_ids)
                train_patients, val_patients = train_test_split(
                    patient_ids, 
                    test_size=0.2, 
                    random_state=42,
                    stratify=stratify_labels
                )
            except ValueError as e:
                print(f"Warning: Cannot stratify split ({e}), using random split")
                train_patients, val_patients = train_test_split(
                    patient_ids, 
                    test_size=0.2, 
                    random_state=42
                )
        
        # Select patients based on split
        if self.split == 'train':
            selected_patients = train_patients
        elif self.split == 'val':
            selected_patients = val_patients
        else:  # test - use validation for now
            selected_patients = val_patients
            
        # Collect samples from selected patients
        for patient_id in selected_patients:
            for class_label in ['0', '1']:
                for sample in self.patient_data[patient_id][class_label]:
                    self.samples.append(sample)
                    self.class_counts[class_label] += 1
                    
        print(f"Loaded {len(self.samples)} images: "
              f"Cancerous: {self.class_counts['0']}, Healthy: {self.class_counts['1']}")
        print(f"Selected {len(selected_patients)} patients for {self.split} split")
        
        # Debug: show patient class distribution
        patient_class_summary = {'cancer_only': 0, 'healthy_only': 0, 'both': 0}
        for patient_id in selected_patients:
            cancer_count = len(self.patient_data[patient_id]['0'])
            healthy_count = len(self.patient_data[patient_id]['1'])
            
            if cancer_count > 0 and healthy_count > 0:
                patient_class_summary['both'] += 1
            elif cancer_count > 0:
                patient_class_summary['cancer_only'] += 1
            else:
                patient_class_summary['healthy_only'] += 1
        
        print(f"Patient distribution: {patient_class_summary}")
        
    def _get_patient_stratification_labels(self, patient_ids: List[str]) -> List[str]:
        """Generate stratification labels based on patient class distribution"""
        labels = []
        for patient_id in patient_ids:
            cancer_count = len(self.patient_data[patient_id]['0'])
            healthy_count = len(self.patient_data[patient_id]['1'])
            
            if cancer_count > healthy_count:
                labels.append('cancer_dominant')
            elif healthy_count > cancer_count:
                labels.append('healthy_dominant')
            else:
                labels.append('balanced')
        return labels
    
    def _apply_class_balancing(self):
        """Apply class balancing strategy (50/50 split)"""
        if not self.class_balance_config.get('enabled', False) or self.split != 'train':
            return
            
        print("Applying class balancing...")
        
        # Current class distribution
        cancer_samples = [s for s in self.samples if s['label'] == 0]
        healthy_samples = [s for s in self.samples if s['label'] == 1]
        
        print(f"Before balancing - Cancer: {len(cancer_samples)}, Healthy: {len(healthy_samples)}")
        
        # Determine target count (50/50 split)
        total_target = min(len(cancer_samples), len(healthy_samples)) * 2
        target_per_class = total_target // 2
        
        # Apply balancing strategy
        strategy = self.class_balance_config.get('strategy', 'hybrid')
        augmentation_ratio = self.class_balance_config.get('augmentation_ratio', 0.5)
        
        balanced_samples = []
        
        if strategy == 'hybrid':
            # 50% augmentation + 50% reduction
            for class_samples, class_name in [(cancer_samples, 'cancer'), (healthy_samples, 'healthy')]:
                if len(class_samples) < target_per_class:
                    # Augment minority class
                    augment_count = int((target_per_class - len(class_samples)) * augmentation_ratio)
                    reduce_count = target_per_class - augment_count
                    
                    # Add original samples (reduced)
                    balanced_samples.extend(random.sample(class_samples, reduce_count))
                    
                    # Add augmented samples
                    augmented_samples = self._augment_samples(class_samples, augment_count)
                    balanced_samples.extend(augmented_samples)
                    
                else:
                    # Reduce majority class
                    balanced_samples.extend(random.sample(class_samples, target_per_class))
                    
        elif strategy == 'augmentation_only':
            # Pure augmentation approach
            max_class_size = max(len(cancer_samples), len(healthy_samples))
            for class_samples in [cancer_samples, healthy_samples]:
                balanced_samples.extend(class_samples)  # Add all original
                if len(class_samples) < max_class_size:
                    # Augment to match larger class
                    needed = max_class_size - len(class_samples)
                    augmented = self._augment_samples(class_samples, needed)
                    balanced_samples.extend(augmented)
                    
        elif strategy == 'reduction_only':
            # Pure reduction approach
            min_class_size = min(len(cancer_samples), len(healthy_samples))
            for class_samples in [cancer_samples, healthy_samples]:
                balanced_samples.extend(random.sample(class_samples, min_class_size))
        
        self.samples = balanced_samples
        
        # Update class counts
        self.class_counts = {'0': 0, '1': 0}
        for sample in self.samples:
            self.class_counts[str(sample['label'])] += 1
            
        print(f"After balancing - Cancer: {self.class_counts['0']}, Healthy: {self.class_counts['1']}")
    
    def _augment_samples(self, samples: List[Dict], count: int) -> List[Dict]:
        """Generate augmented samples"""
        if not samples:
            return []
            
        augmented = []
        for i in range(count):
            # Select random sample to augment
            original_sample = random.choice(samples)
            
            # Create augmented sample entry
            augmented_sample = original_sample.copy()
            augmented_sample['augmented'] = True
            augmented_sample['augment_id'] = i
            
            augmented.append(augmented_sample)
            
        return augmented
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index"""
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['path']).convert('RGB')
        label = sample['label']
        
        # Apply transformations
        if self.transform:
            # Check if this is an augmented sample
            if sample.get('augmented', False):
                # Apply stronger augmentations for augmented samples
                image = self._apply_augmentation(image)
            else:
                image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label
    
    def _apply_augmentation(self, image: Image.Image) -> torch.Tensor:
        """Apply augmentation to image"""
        augment_transform = self._get_augmentation_transform()
        return augment_transform(image)
    
    def _get_augmentation_transform(self) -> transforms.Compose:
        """Get augmentation transform based on config"""
        aug_config = self.augmentation_config
        
        augment_list = [
            transforms.Resize((50, 50)),
        ]
        
        if aug_config.get('rotation_range', 0) > 0:
            augment_list.append(
                transforms.RandomRotation(aug_config['rotation_range'])
            )
            
        if aug_config.get('horizontal_flip', False):
            augment_list.append(transforms.RandomHorizontalFlip(p=0.5))
            
        if aug_config.get('vertical_flip', False):
            augment_list.append(transforms.RandomVerticalFlip(p=0.5))
            
        brightness_range = aug_config.get('brightness_range', [1.0, 1.0])
        contrast_range = aug_config.get('contrast_range', [1.0, 1.0])
        if brightness_range != [1.0, 1.0] or contrast_range != [1.0, 1.0]:
            augment_list.append(
                transforms.ColorJitter(
                    brightness=brightness_range,
                    contrast=contrast_range
                )
            )
        
        augment_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transforms.Compose(augment_list)
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for loss function"""
        total_samples = len(self.samples)
        class_0_count = self.class_counts['0']
        class_1_count = self.class_counts['1']
        
        # Inverse frequency weighting
        weight_0 = total_samples / (2 * class_0_count) if class_0_count > 0 else 1.0
        weight_1 = total_samples / (2 * class_1_count) if class_1_count > 0 else 1.0
        
        return torch.FloatTensor([weight_0, weight_1])
    
    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics"""
        return {
            'total_samples': len(self.samples),
            'class_distribution': self.class_counts.copy(),
            'class_ratio': self.class_counts['0'] / max(self.class_counts['1'], 1),
            'patients_count': len(set(s['patient_id'] for s in self.samples)),
            'augmented_samples': len([s for s in self.samples if s.get('augmented', False)])
        }


def get_transforms(split: str, config: Dict) -> transforms.Compose:
    """Get transforms for different splits"""
    
    if split == 'train':
        transform_list = [
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]
    else:
        # Validation/test transforms (no augmentation)
        transform_list = [
            transforms.Resize((50, 50)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ]
    
    return transforms.Compose(transform_list)


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    data_config = config.get('data', {})
    data_path = data_config.get('data_path', 'data/BHI')
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 4)
    
    # M4 Pro optimized data loading configuration
    num_workers = _get_optimized_num_workers(num_workers)
    
    print("Creating datasets and dataloaders...")
    
    # Create datasets
    train_transform = get_transforms('train', config)
    val_transform = get_transforms('val', config)
    
    train_dataset = BreastCancerDataset(
        data_path=data_path,
        split='train',
        config=config,
        transform=train_transform
    )
    
    val_dataset = BreastCancerDataset(
        data_path=data_path,
        split='val', 
        config=config,
        transform=val_transform
    )
    
    # Create dataloaders with M4 Pro optimizations
    is_m4_pro = _detect_m4_pro()
    pin_memory = is_m4_pro and torch.backends.mps.is_available()  # Enable pin_memory for M4 Pro MPS
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0 and is_m4_pro,  # Keep workers alive on M4 Pro
        prefetch_factor=4 if is_m4_pro else 2  # Optimized prefetch for M4 Pro
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0 and is_m4_pro,
        prefetch_factor=4 if is_m4_pro else 2
    )
    
    # Print dataset statistics
    print("\n=== Dataset Statistics ===")
    train_stats = train_dataset.get_dataset_stats()
    val_stats = val_dataset.get_dataset_stats()
    
    print(f"Train - Samples: {train_stats['total_samples']}, "
          f"Cancer: {train_stats['class_distribution']['0']}, "
          f"Healthy: {train_stats['class_distribution']['1']}, "
          f"Patients: {train_stats['patients_count']}")
          
    print(f"Val - Samples: {val_stats['total_samples']}, "
          f"Cancer: {val_stats['class_distribution']['0']}, "
          f"Healthy: {val_stats['class_distribution']['1']}, "
          f"Patients: {val_stats['patients_count']}")
    
    if train_stats['augmented_samples'] > 0:
        print(f"Augmented samples in training: {train_stats['augmented_samples']}")
    
    return train_loader, val_loader


def _detect_m4_pro() -> bool:
    """Detect if running on Apple M4 Pro"""
    try:
        if platform.system() == 'Darwin':
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True)
            cpu_brand = result.stdout.strip()
            return 'Apple M4 Pro' in cpu_brand or 'M4 Pro' in cpu_brand
    except:
        pass
    return torch.backends.mps.is_available()  # Fallback to MPS availability


def _get_optimized_num_workers(default_num_workers: int) -> int:
    """Get optimized number of workers for M4 Pro 14-core architecture"""
    is_m4_pro = _detect_m4_pro()
    
    if not is_m4_pro:
        # Conservative approach for non-M4 Pro systems
        return min(default_num_workers, 2) if default_num_workers > 0 else 0
    
    # M4 Pro has 14 cores: 10 performance + 4 efficiency
    # Optimal configuration balances CPU usage with memory bandwidth
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=False)  # Physical cores
        
        # M4 Pro optimization: Use 8-10 workers for optimal performance
        # This leaves headroom for system processes and the main training thread
        if cpu_count >= 14:  # M4 Pro
            optimal_workers = 8  # Sweet spot for data loading
        elif cpu_count >= 10:  # M4 or similar
            optimal_workers = 6
        else:
            optimal_workers = min(4, cpu_count - 2)
        
        return min(optimal_workers, max(default_num_workers, 8))
    except:
        # Fallback if psutil unavailable
        return 8 if is_m4_pro else min(default_num_workers, 2)


def test_dataloader():
    """Test the dataloader implementation"""
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Test loading a batch
    print("\n=== Testing Dataloader ===")
    
    train_batch = next(iter(train_loader))
    images, labels = train_batch
    
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label distribution in batch: {torch.bincount(labels)}")
    print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Test class weights
    train_dataset = train_loader.dataset
    class_weights = train_dataset.get_class_weights()
    print(f"Class weights: {class_weights}")
    
    print("✅ Dataloader test completed successfully!")


if __name__ == "__main__":
    test_dataloader()