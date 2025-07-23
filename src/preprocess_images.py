#!/usr/bin/env python3
"""
Image preprocessing script for breast cancer detection project.
Detects images that aren't 50x50 pixels and resizes them while maintaining aspect ratio.
"""

import os
import sys
from PIL import Image
import random
from pathlib import Path


def check_image_dimensions(data_path, sample_size=20):
    """
    Check dimensions of a random sample of images to understand current sizes.
    
    Args:
        data_path (str): Path to the data directory
        sample_size (int): Number of random images to sample
    """
    print("=== CHECKING SAMPLE IMAGE DIMENSIONS ===")
    
    sample_images = []
    data_dir = Path(data_path)
    
    # Collect sample images from different patient folders
    for patient_folder in data_dir.iterdir():
        if patient_folder.is_dir() and patient_folder.name != "IDC_regular_ps50_idx5":
            for class_folder in patient_folder.iterdir():
                if class_folder.is_dir() and class_folder.name in ['0', '1']:
                    images = [f for f in class_folder.iterdir() if f.suffix.lower() == '.png']
                    sample_images.extend(images[:2])  # Take 2 from each class folder
                    
                    if len(sample_images) >= sample_size:
                        break
            if len(sample_images) >= sample_size:
                break
    
    # Randomly sample from collected images
    sample_images = random.sample(sample_images, min(sample_size, len(sample_images)))
    
    dimension_counts = {}
    
    for img_path in sample_images:
        try:
            with Image.open(img_path) as img:
                size = img.size
                dimension_counts[size] = dimension_counts.get(size, 0) + 1
                print(f"{img_path.name}: {size}")
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
    
    print(f"\nDimension summary from {len(sample_images)} sample images:")
    for size, count in sorted(dimension_counts.items()):
        print(f"  {size}: {count} images")
    
    return dimension_counts


def find_non_standard_images(data_path, target_size=(50, 50)):
    """
    Find all images that don't match the target size.
    
    Args:
        data_path (str): Path to the data directory
        target_size (tuple): Target image size (width, height)
    
    Returns:
        list: List of paths to images that need resizing
    """
    print(f"\n=== SCANNING FOR NON-{target_size[0]}x{target_size[1]} IMAGES ===")
    
    non_standard_images = []
    data_dir = Path(data_path)
    total_images = 0
    
    for patient_folder in data_dir.iterdir():
        if patient_folder.is_dir() and patient_folder.name != "IDC_regular_ps50_idx5":
            print(f"Scanning patient folder: {patient_folder.name}")
            
            for class_folder in patient_folder.iterdir():
                if class_folder.is_dir() and class_folder.name in ['0', '1']:
                    for img_file in class_folder.iterdir():
                        if img_file.suffix.lower() == '.png':
                            total_images += 1
                            try:
                                with Image.open(img_file) as img:
                                    if img.size != target_size:
                                        non_standard_images.append(img_file)
                                        
                                        # Print first few non-standard images as examples
                                        if len(non_standard_images) <= 10:
                                            print(f"  Non-standard: {img_file.name} -> {img.size}")
                                            
                            except Exception as e:
                                print(f"  Error reading {img_file}: {e}")
    
    print(f"\nScan complete!")
    print(f"Total images scanned: {total_images}")
    print(f"Images needing resize: {len(non_standard_images)}")
    
    return non_standard_images


def resize_image(image_path, target_size=(50, 50), method='resize'):
    """
    Resize an image to target size.
    
    Args:
        image_path (Path): Path to the image file
        target_size (tuple): Target size (width, height)
        method (str): Resize method ('resize' or 'crop')
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            if method == 'resize':
                # Simple resize - may distort aspect ratio
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
            elif method == 'crop':
                # Crop from center to maintain aspect ratio
                # This may lose some image data
                width, height = img.size
                left = (width - target_size[0]) / 2
                top = (height - target_size[1]) / 2
                right = left + target_size[0]
                bottom = top + target_size[1]
                
                resized_img = img.crop((left, top, right, bottom))
            
            # Save the resized image, overwriting the original
            resized_img.save(image_path)
            return True
            
    except Exception as e:
        print(f"Error resizing {image_path}: {e}")
        return False


def resize_non_standard_images(non_standard_images, target_size=(50, 50), method='resize'):
    """
    Resize all non-standard images to target size.
    
    Args:
        non_standard_images (list): List of image paths to resize
        target_size (tuple): Target size (width, height)  
        method (str): Resize method ('resize' or 'crop')
    """
    print(f"\n=== RESIZING {len(non_standard_images)} IMAGES TO {target_size[0]}x{target_size[1]} ===")
    print(f"Using method: {method}")
    
    success_count = 0
    failure_count = 0
    
    for i, img_path in enumerate(non_standard_images, 1):
        if resize_image(img_path, target_size, method):
            success_count += 1
            if i % 100 == 0 or i <= 10:  # Progress updates
                print(f"  Resized {i}/{len(non_standard_images)}: {img_path.name}")
        else:
            failure_count += 1
            print(f"  Failed to resize: {img_path}")
    
    print(f"\nResize operation complete!")
    print(f"Successfully resized: {success_count} images")
    print(f"Failed to resize: {failure_count} images")


def main():
    """Main function to run the image preprocessing pipeline."""
    data_path = "data/BHI"
    target_size = (50, 50)
    
    print("Breast Cancer Image Preprocessing Tool")
    print("=" * 50)
    
    # Check if data directory exists
    if not os.path.exists(data_path):
        print(f"Error: Data directory '{data_path}' not found!")
        sys.exit(1)
    
    # Step 1: Check sample image dimensions
    dimension_counts = check_image_dimensions(data_path, sample_size=20)
    
    # Step 2: Find all non-standard images
    non_standard_images = find_non_standard_images(data_path, target_size)
    
    if not non_standard_images:
        print(f"All images are already {target_size[0]}x{target_size[1]}! No preprocessing needed.")
        return
    
    # Step 3: Ask user for confirmation
    print(f"\nFound {len(non_standard_images)} images that need resizing.")
    response = input("Do you want to resize these images? (y/n): ").lower().strip()
    
    if response == 'y':
        # Choose resize method
        print("\nResize methods:")
        print("1. resize - Simple resize (may distort)")
        print("2. crop - Center crop (may lose data)")
        
        method_choice = input("Choose method (1 or 2): ").strip()
        method = 'resize' if method_choice == '1' else 'crop'
        
        # Step 4: Resize images
        resize_non_standard_images(non_standard_images, target_size, method)
        
        # Step 5: Verify results
        print("\n=== VERIFYING RESIZE RESULTS ===")
        remaining_non_standard = find_non_standard_images(data_path, target_size)
        
        if not remaining_non_standard:
            print("✅ All images are now 50x50 pixels!")
        else:
            print(f"⚠️ {len(remaining_non_standard)} images still need attention.")
    else:
        print("Preprocessing cancelled.")


if __name__ == "__main__":
    main()