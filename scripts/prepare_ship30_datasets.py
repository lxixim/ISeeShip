#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset preparation script: Sample different numbers of examples from source data (1-shot, 2-shot, 8-shot, 16-shot)
"""
import os
import random
import shutil
from pathlib import Path
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage
from PIL import Image
import argparse

# Set random seed for reproducibility
random.seed(42)

# Source data path
SOURCE_DATA_PATH = "/data/ljx/dataset/boat-29/images"

# Output base path
OUTPUT_BASE_PATH = "/data/ljx/visualRft/share_data"

# Ship categories (30 categories)
CATEGORIES = [
    'aircraft_carrier', 'bitumen', 'bulk_carrier', 'catamaran_yacht',
    'chemical_tanker', 'container_ship', 'crude_oil_tanker', 'cruise',
    'destroyer', 'firefighting', 'fishing_boat', 'frigate', 'fso',
    'general_cargo_ship', 'heavy_load_carrier', 'kayak', 'LNG_tanker',
    'LPG_tanker', 'monohull_sailboat', 'monohull_yacht', 'oil_products_tanker',
    'passenger_cargo_ship', 'passenger_ro-ro_ship', 'passenger_ship',
    'reefer', 'sailing_catamaran', 'sailing_trimaran', 'submarine',
    'tugboat', 'vehicles_carrier'
]

# Problem template for classification task
PROBLEM_TEMPLATE = """This is an image containing a ship. Please identify the model of the ship based on the image.
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
The output answer format should be as follows:
<think> ... </think> <answer>species name</answer>
Please strictly follow the format."""

def get_solution_template(category_name):
    """Generate solution template"""
    return f"<think>Based on the ship's characteristics visible in the image, I can identify this as a {category_name}.</think> <answer>{category_name}</answer>"


def collect_images_from_category(category_path, num_shots):
    """Randomly sample specified number of images from a category"""
    image_files = []
    category_dir = Path(category_path)
    
    if not category_dir.exists():
        print(f"Warning: Category directory does not exist {category_path}")
        return []
    
    # Collect all image files
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(category_dir.glob(ext)))
    
    # If not enough images, return all images
    if len(image_files) < num_shots:
        print(f"Warning: {category_dir.name} category only has {len(image_files)} images, less than required {num_shots}")
        return image_files
    
    # Random sampling
    selected_images = random.sample(image_files, num_shots)
    return selected_images


def create_dataset(num_shots, output_name):
    """Create dataset with specified number of shots"""
    print(f"\n{'='*60}")
    print(f"Starting to create {num_shots}-shot dataset: {output_name}")
    print(f"{'='*60}")
    
    dataset_samples = []
    
    # Iterate through all categories
    for category in CATEGORIES:
        category_path = os.path.join(SOURCE_DATA_PATH, category)
        
        # Sample images from this category
        selected_images = collect_images_from_category(category_path, num_shots)
        
        # Create data sample for each image
        for img_path in selected_images:
            try:
                # Verify image can be opened
                with Image.open(img_path) as img:
                    img.verify()
                
                sample = {
                    'image': str(img_path),
                    'problem': PROBLEM_TEMPLATE,
                    'solution': get_solution_template(category)
                }
                dataset_samples.append(sample)
                
            except Exception as e:
                print(f"Error: Cannot process image {img_path}: {e}")
                continue
    
    print(f"Total collected {len(dataset_samples)} samples")
    print(f"Expected samples: {len(CATEGORIES)} categories x {num_shots} shots = {len(CATEGORIES) * num_shots}")
    
    # Create HuggingFace Dataset
    features = Features({
        'image': HFImage(),
        'problem': Value('string'),
        'solution': Value('string')
    })
    
    dataset = Dataset.from_dict(
        {
            'image': [s['image'] for s in dataset_samples],
            'problem': [s['problem'] for s in dataset_samples],
            'solution': [s['solution'] for s in dataset_samples]
        },
        features=features
    )
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': dataset
    })
    
    # Save dataset
    output_path = os.path.join(OUTPUT_BASE_PATH, output_name)
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Saving dataset to: {output_path}")
    dataset_dict.save_to_disk(output_path)
    print(f"Dataset saved successfully!")
    
    # Print dataset info
    print(f"\nDataset info:")
    print(f"  - Training samples: {len(dataset)}")
    print(f"  - Samples per category: {num_shots}")
    print(f"  - Total categories: {len(CATEGORIES)}")
    
    return dataset_dict


def main():
    parser = argparse.ArgumentParser(description='Prepare ship30 datasets')
    parser.add_argument('--shots', type=str, default='all', 
                       help='Number of shots to create, can be 1,2,4,8,16 or all (create all)')
    args = parser.parse_args()
    
    # Define dataset configurations to create
    dataset_configs = {
        1: 'ViRFT_CLS_ship30_1shot',
        2: 'ViRFT_CLS_ship30_2shot',
        4: 'ViRFT_CLS_ship30_4shot',
        8: 'ViRFT_CLS_ship30_8shot',
        16: 'ViRFT_CLS_ship30_16shot'
    }
    
    # Determine which datasets to create
    if args.shots == 'all':
        shots_to_create = list(dataset_configs.keys())
    else:
        shots_to_create = [int(s.strip()) for s in args.shots.split(',')]
    
    print(f"Will create datasets with following shots: {shots_to_create}")
    print(f"Source data path: {SOURCE_DATA_PATH}")
    print(f"Output path: {OUTPUT_BASE_PATH}")
    
    # Verify source data path
    if not os.path.exists(SOURCE_DATA_PATH):
        print(f"Error: Source data path does not exist: {SOURCE_DATA_PATH}")
        return
    
    # Create each dataset
    for num_shots in shots_to_create:
        if num_shots not in dataset_configs:
            print(f"Warning: Skipping unsupported shot number {num_shots}")
            continue
        
        output_name = dataset_configs[num_shots]
        try:
            create_dataset(num_shots, output_name)
        except Exception as e:
            print(f"Error: Failed to create {num_shots}-shot dataset: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("All datasets created successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

