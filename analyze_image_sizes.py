#!/usr/bin/env python3
"""
Image Size Analysis Script for BHI_uncleaned Dataset
Creates a visual comparison of properly sized vs wrongly sized images for presentation
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from pathlib import Path
from collections import defaultdict, Counter

# Add src to path
sys.path.append('src')

def analyze_image_dimensions(data_path, target_size=(50, 50)):
    """
    Analyze image dimensions in the dataset
    
    Args:
        data_path: Path to BHI_uncleaned dataset
        target_size: Target size tuple (width, height)
    
    Returns:
        Dictionary with properly sized and wrongly sized image paths
    """
    properly_sized = []
    wrongly_sized = []
    dimension_counts = Counter()
    
    print(f"Analyzing images in {data_path}...")
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset path {data_path} not found!")
        return None, None, None
    
    # Walk through all subdirectories
    total_images = 0
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                total_images += 1
                
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        dimension_counts[(width, height)] += 1
                        
                        if (width, height) == target_size:
                            properly_sized.append(image_path)
                        else:
                            wrongly_sized.append(image_path)
                            
                except Exception as e:
                    print(f"⚠️ Could not read {image_path}: {e}")
                    continue
    
    print(f"📊 Total images analyzed: {total_images}")
    print(f"✅ Properly sized ({target_size[0]}x{target_size[1]}): {len(properly_sized)}")
    print(f"❌ Wrongly sized: {len(wrongly_sized)}")
    
    return properly_sized, wrongly_sized, dimension_counts

def select_sample_images(properly_sized, wrongly_sized, n_samples=4):
    """
    Select sample images for comparison
    
    Args:
        properly_sized: List of properly sized image paths
        wrongly_sized: List of wrongly sized image paths
        n_samples: Number of samples to select from each category
    
    Returns:
        Tuple of (proper_samples, wrong_samples)
    """
    # Randomly select samples
    proper_samples = random.sample(properly_sized, min(n_samples, len(properly_sized)))
    wrong_samples = random.sample(wrongly_sized, min(n_samples, len(wrongly_sized)))
    
    return proper_samples, wrong_samples

def create_comparison_visualization(proper_samples, wrong_samples, dimension_counts, output_path="image_size_comparison.png"):
    """
    Create a visual comparison of properly sized vs wrongly sized images
    
    Args:
        proper_samples: List of properly sized image paths
        wrong_samples: List of wrongly sized image paths  
        dimension_counts: Counter of image dimensions
        output_path: Output path for the comparison image
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Top section: Properly sized images (50x50)
    print("📸 Loading properly sized images...")
    # Add subtitle for proper images at top
    fig.text(0.5, 0.90, f'Properly Sized Images (50x50 pixels) - Sample of {len(proper_samples)}', 
             ha='center', fontsize=14, fontweight='bold', color='green')
    
    # Create subplot grid for proper images
    proper_images = []
    proper_titles = []
    for i, img_path in enumerate(proper_samples):
        try:
            img = Image.open(img_path)
            proper_images.append(np.array(img))
            
            # Extract info from path for title
            path_parts = img_path.split(os.sep)
            patient_id = path_parts[-3] if len(path_parts) >= 3 else "unknown"
            class_label = "Cancer" if path_parts[-2] == "0" else "Healthy"
            proper_titles.append(f'Patient {patient_id}\n{class_label}\n{img.size[0]}x{img.size[1]}')
        except Exception as e:
            print(f"⚠️ Could not load {img_path}: {e}")
            continue
    
    # Display proper images
    for i, (img, title) in enumerate(zip(proper_images, proper_titles)):
        ax = plt.subplot(2, 4, i + 1)
        ax.set_position([0.1 + i * 0.2, 0.55, 0.18, 0.3])  # [left, bottom, width, height]
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        plt.title(title, fontsize=10, fontweight='bold')
        plt.axis('off')
    
    # Middle section: Wrongly sized images
    print("📸 Loading wrongly sized images...")
    wrong_images = []
    wrong_titles = []
    wrong_sizes = []
    
    for i, img_path in enumerate(wrong_samples):
        try:
            img = Image.open(img_path)
            wrong_images.append(np.array(img))
            wrong_sizes.append(img.size)
            
            # Extract info from path for title
            path_parts = img_path.split(os.sep)
            patient_id = path_parts[-3] if len(path_parts) >= 3 else "unknown"
            class_label = "Cancer" if path_parts[-2] == "0" else "Healthy"
            wrong_titles.append(f'Patient {patient_id}\n{class_label}\n{img.size[0]}x{img.size[1]}')
        except Exception as e:
            print(f"⚠️ Could not load {img_path}: {e}")
            continue
    
    # Add subtitle for wrong images at middle
    fig.text(0.5, 0.45, f'Improperly Sized Images - Sample of {len(wrong_samples)}', 
             ha='center', fontsize=14, fontweight='bold', color='red')
    
    # Display wrong images
    for i, (img, title) in enumerate(zip(wrong_images, wrong_titles)):
        ax = plt.subplot(2, 4, 4 + i + 1)
        ax.set_position([0.1 + i * 0.2, 0.15, 0.18, 0.25])  # [left, bottom, width, height]
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        plt.title(title, fontsize=10, fontweight='bold')
        plt.axis('off')
    
    # Add simple statistics text at the bottom
    total_images = sum(dimension_counts.values())
    proper_count = dimension_counts.get((50, 50), 0)
    wrong_count = total_images - proper_count
    proper_percentage = (proper_count / total_images * 100) if total_images > 0 else 0
    wrong_percentage = (wrong_count / total_images * 100) if total_images > 0 else 0
    
    # Add text at bottom of figure
    fig.text(0.5, 0.02, f'Total Images Analyzed: {total_images:,}  |  Properly Sized: {proper_count:,} ({proper_percentage:.1f}%)  |  Improperly Sized: {wrong_count:,} ({wrong_percentage:.1f}%)', 
             ha='center', fontsize=12, fontweight='bold')
    
    # No tight_layout since we're manually positioning elements
    
    # Save the comparison image
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Comparison visualization saved to: {output_path}")
    
    return fig

def main():
    """Main function to run the image size analysis"""
    print("🔍 Starting Image Size Analysis for BHI_uncleaned Dataset")
    print("=" * 60)
    
    # Configuration
    data_path = "data/BHI_uncleaned"
    target_size = (50, 50)
    n_samples = 4
    output_path = "image_size_comparison.png"
    
    # Check if dataset exists
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at {data_path}")
        print("Please ensure the BHI_uncleaned dataset is placed in the data/ directory")
        return
    
    # Analyze image dimensions
    properly_sized, wrongly_sized, dimension_counts = analyze_image_dimensions(data_path, target_size)
    
    if properly_sized is None:
        print("❌ Failed to analyze dataset")
        return
    
    # Check if we have enough samples
    if len(properly_sized) < n_samples:
        print(f"⚠️ Only {len(properly_sized)} properly sized images found, using all available")
        n_samples = len(properly_sized)
    
    if len(wrongly_sized) < n_samples:
        print(f"⚠️ Only {len(wrongly_sized)} wrongly sized images found, using all available")
    
    if len(properly_sized) == 0:
        print("❌ No properly sized images found!")
        return
    
    if len(wrongly_sized) == 0:
        print("✅ All images are properly sized! No comparison needed.")
        return
    
    # Select sample images
    print(f"\n🎯 Selecting {n_samples} samples from each category...")
    proper_samples, wrong_samples = select_sample_images(properly_sized, wrongly_sized, n_samples)
    
    print(f"Selected properly sized samples:")
    for i, path in enumerate(proper_samples, 1):
        print(f"  {i}. {path}")
    
    print(f"Selected wrongly sized samples:")
    for i, path in enumerate(wrong_samples, 1):
        with Image.open(path) as img:
            print(f"  {i}. {path} ({img.size[0]}x{img.size[1]})")
    
    # Create visualization
    print(f"\n🎨 Creating comparison visualization...")
    try:
        fig = create_comparison_visualization(proper_samples, wrong_samples, dimension_counts, output_path)
        
        print(f"\n✅ Analysis Complete!")
        print(f"📊 Visualization saved as: {output_path}")
        print(f"🎯 Ready for presentation use!")
        
        # Show additional statistics
        total_images = sum(dimension_counts.values())
        proper_count = dimension_counts.get((50, 50), 0)
        print(f"\n📈 Summary Statistics:")
        print(f"   Total images: {total_images:,}")
        print(f"   Properly sized: {proper_count:,} ({proper_count/total_images*100:.1f}%)")
        print(f"   Need resizing: {total_images-proper_count:,} ({(total_images-proper_count)/total_images*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Failed to create visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set random seed for reproducible sample selection
    random.seed(42)
    main()