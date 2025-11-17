# cleanup_images.py
import cv2
import numpy as np
import os
from utils import create_folder_structure

def clean_text_simple(image):
    """
    Simple and effective: Replace only the top text region with background color.
    This avoids blurring the hand.
    """
    result = image.copy()
    
    # Sample background color from a clean area (middle-right side, away from hand and text)
    # Sample from multiple areas and take median to get clean background
    sample_areas = [
        image[150:170, 400:450],  # Right side
        image[250:270, 400:450],  # Lower right
    ]
    
    # Get median color from samples
    samples = np.vstack([area.reshape(-1, 3) for area in sample_areas])
    bg_color = np.median(samples, axis=0)
    
    # Simply paint over the text region with the background color
    # Cover the entire top area where text appears (0-115 pixels from top)
    result[0:115, :] = bg_color
    
    return result

def clean_text_crop(image):
    """
    Alternative: Just crop out the text region entirely.
    This is the safest method - no image processing artifacts.
    """
    # Crop out top 115 pixels where text appears
    return image[115:, :]

def clean_text_blur_then_replace(image):
    """
    Hybrid approach: Blur the text area to get smooth background, then use that.
    """
    result = image.copy()
    
    # Extract the text region
    text_region = image[0:115, :].copy()
    
    # Apply heavy Gaussian blur to remove text but keep color variations
    blurred = cv2.GaussianBlur(text_region, (51, 51), 0)
    
    # Replace the top region with the blurred version
    result[0:115, :] = blurred
    
    return result

def preview_on_sample(sample_path):
    """
    Preview the cleaning on a single sample image before processing all.
    """
    img = cv2.imread(sample_path)
    if img is None:
        print(f"Could not read {sample_path}")
        return None
    
    print("\nProcessing sample image...")
    
    # Apply all methods
    cleaned_simple = clean_text_simple(img)
    cleaned_crop = clean_text_crop(img)
    cleaned_blur = clean_text_blur_then_replace(img)
    
    # Resize crop to match original height for comparison
    if cleaned_crop.shape[0] != img.shape[0]:
        pad_height = img.shape[0] - cleaned_crop.shape[0]
        bg_color = np.median(cleaned_crop[0:20, :], axis=(0, 1))
        padding = np.full((pad_height, cleaned_crop.shape[1], 3), bg_color, dtype=np.uint8)
        cleaned_crop = np.vstack([padding, cleaned_crop])
    
    # Create comparison
    row1 = np.hstack([img, cleaned_simple])
    row2 = np.hstack([cleaned_blur, cleaned_crop])
    comparison = np.vstack([row1, row2])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = img.shape[:2]
    cv2.putText(comparison, "ORIGINAL", (10, 30), font, 0.7, (0, 255, 255), 2)
    cv2.putText(comparison, "SIMPLE (Recommended)", (w + 10, 30), font, 0.7, (0, 255, 255), 2)
    cv2.putText(comparison, "BLUR", (10, h + 30), font, 0.7, (0, 255, 255), 2)
    cv2.putText(comparison, "CROP", (w + 10, h + 30), font, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Preview - Choose Best Method", comparison)
    print("\nPress any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return True

def process_directory(base_path, method='simple'):
    """
    Process all images in the dataset directory structure.
    
    Args:
        base_path: Base directory path
        method: Cleaning method - 'simple' (recommended), 'crop', or 'blur'
    """
    modes = ['train', 'test', 'val']
    letters = 'abcdefghijklmnopqrstuvwxyz'
    
    # Select cleaning function
    clean_func = {
        'simple': clean_text_simple,
        'crop': clean_text_crop,
        'blur': clean_text_blur_then_replace
    }.get(method, clean_text_simple)
    
    total_processed = 0
    errors = 0
    
    print(f"\nUsing cleaning method: {method.upper()}")
    print("-" * 60)
    
    for mode in modes:
        mode_path = os.path.join(base_path, mode)
        if not os.path.exists(mode_path):
            continue
            
        mode_count = 0
        for letter in letters:
            letter_path = os.path.join(mode_path, letter)
            if not os.path.exists(letter_path):
                continue
                
            # Process all jpg files in the letter directory
            for filename in os.listdir(letter_path):
                if not filename.endswith('.jpg'):
                    continue
                    
                file_path = os.path.join(letter_path, filename)
                
                try:
                    # Read the image
                    img = cv2.imread(file_path)
                    if img is None:
                        print(f"⚠ Warning: Could not read {file_path}")
                        errors += 1
                        continue
                    
                    # Clean the image using selected method
                    cleaned = clean_func(img)
                    
                    # Save back to the same location
                    cv2.imwrite(file_path, cleaned)
                    total_processed += 1
                    mode_count += 1
                    
                except Exception as e:
                    print(f"⚠ Error processing {file_path}: {e}")
                    errors += 1
        
        if mode_count > 0:
            print(f"✓ {mode.upper()}: {mode_count} images processed")
    
    return total_processed, errors

if __name__ == "__main__":
    # Get the base path from the folder structure
    base_path = create_folder_structure()
    
    print("\n" + "="*60)
    print("SIMPLE & EFFECTIVE IMAGE CLEANING")
    print("="*60)
    
    # Find a sample image to preview
    sample_found = False
    for mode in ['test', 'train', 'val']:
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            sample_path = os.path.join(base_path, mode, letter)
            if os.path.exists(sample_path):
                files = [f for f in os.listdir(sample_path) if f.endswith('.jpg')]
                if files:
                    sample_path = os.path.join(sample_path, files[0])
                    sample_found = True
                    break
        if sample_found:
            break
    
    # Show preview if sample found
    if sample_found:
        print(f"\nFound sample: {sample_path}")
        preview_choice = input("Preview methods on this sample? (y/n): ").strip().lower()
        if preview_choice == 'y':
            preview_on_sample(sample_path)
    
    # Select method
    print("\nAvailable cleaning methods:")
    print("  1. 'simple' - Replace text area with background color (RECOMMENDED)")
    print("  2. 'blur' - Blur the text area to smooth it out")
    print("  3. 'crop' - Remove top 115px entirely (changes image size)")
    
    choice = input("\nSelect method (1/2/3) or press Enter for simple: ").strip()
    method_map = {'1': 'simple', '2': 'blur', '3': 'crop', '': 'simple'}
    method = method_map.get(choice, 'simple')
    
    # Confirm before processing
    confirm = input(f"\nProcess ALL images using '{method}' method? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Cancelled.")
        exit()
    
    print("\nProcessing images...")
    print("="*60)
    
    # Process all images
    total, errors = process_directory(base_path, method=method)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"✓ Successfully processed: {total} images")
    if errors > 0:
        print(f"⚠ Errors: {errors}")
    print(f"Method used: {method.upper()}")
    print("="*60)