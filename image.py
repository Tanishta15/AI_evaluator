import pandas as pd
import json
from PIL import Image
import pytesseract
import argparse
from pathlib import Path

def extract_text_from_keyword_image(parquet_path, keyword, image_base_dir='.', save_changes=True):
    """
    Search for a keyword in the 'content' or 'title' columns of a Parquet file,
    find the first image path from 'extracted_images', extract text from that image,
    and optionally save the extracted text back to the parquet file.
    """
    df = pd.read_parquet(parquet_path)
    # Search for the keyword in 'content' or 'title'
    mask = df['content'].str.contains(keyword, case=False, na=False) | df['title'].str.contains(keyword, case=False, na=False)
    matches = df[mask]
    if matches.empty:
        print(f"No slide found with keyword '{keyword}'.")
        return None

    # Get the first match and its index
    matched_idx = matches.index[0]
    row = matches.iloc[0]
    
    # Check if extracted_images column exists and has data
    if 'extracted_images' not in df.columns or pd.isna(row['extracted_images']) or not row['extracted_images']:
        print("No images found for this slide.")
        return None

    # Get the image path (it's stored directly as a string, not JSON)
    image_path = row['extracted_images']
    
    # The image path is stored relative to the project root
    # If base_dir is '.', use the path as-is
    # Otherwise, check if the path is already absolute or properly relative
    if image_base_dir == '.' or Path(image_path).is_absolute():
        full_path = str(image_path)
    else:
        # Combine with base directory
        full_path = str(Path(image_base_dir) / image_path)
    
    print(f"Found image: {full_path}")

    # Extract text from the image
    try:
        img = Image.open(full_path)
        text = pytesseract.image_to_string(img)
        print(f"\nExtracted text from image:\n{'-' * 40}\n{text}\n{'-' * 40}")
        
        if save_changes:
            # Create or update the image_extracted column
            if 'image_extracted' not in df.columns:
                df['image_extracted'] = None
            
            # Update the text for the matched slide
            df.at[matched_idx, 'image_extracted'] = text
            
            # Save the updated dataframe back to parquet
            df.to_parquet(parquet_path, index=False)
            print(f"\n✅ Updated parquet file with extracted text at: {parquet_path}")
        
        return text
    except Exception as e:
        print(f"Error reading image or extracting text: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Search for keywords in a Parquet file and extract text from associated images.')
    parser.add_argument('--parquet', type=str, required=True, help='Path to the Parquet file')
    parser.add_argument('--keyword', type=str, required=True, help='Keyword to search for in content/title')
    parser.add_argument('--base-dir', type=str, default='.', help='Base directory for image paths (default: current directory)')
    parser.add_argument('--no-save', action='store_true', help='Do not save extracted text back to the parquet file')
    
    args = parser.parse_args()
    
    extract_text_from_keyword_image(args.parquet, args.keyword, args.base_dir, save_changes=not args.no_save)

if __name__ == "__main__":
    main()