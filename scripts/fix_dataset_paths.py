#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix dataset paths and category names without regenerating
"""
import os
import sys
from datasets import load_from_disk, Dataset, DatasetDict, Features, Value, Image as HFImage

def fix_dataset(dataset_path):
    """Fix paths and category names in a single dataset"""
    print(f"\n{'='*70}")
    print(f"Processing: {dataset_path}")
    print(f"{'='*70}")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset does not exist, skipping")
        return False
    
    try:
        # Load dataset
        print("Loading dataset...")
        dataset_dict = load_from_disk(dataset_path)
        train_data = dataset_dict['train']
        
        print(f"   Samples: {len(train_data)}")
        
        # Check if fixes are needed
        need_fix = False
        old_path = "/data/ccy/ccy-factory/datasets/boat-29"
        new_path = "/data/ljx/dataset/boat-29"
        
        # Fix data
        print("Fixing paths and category names...")
        fixed_data = []
        path_fixed_count = 0
        fishing_fixed_count = 0
        
        for i, sample in enumerate(train_data):
            new_sample = {
                'image': sample['image'],
                'problem': sample['problem'],
                'solution': sample['solution']
            }
            
            # Fix fishing_ships in solution
            if 'fishing_ships' in new_sample['solution']:
                new_sample['solution'] = new_sample['solution'].replace('fishing_ships', 'fishing_boat')
                fishing_fixed_count += 1
                need_fix = True
            
            fixed_data.append(new_sample)
        
        if not need_fix and fishing_fixed_count == 0:
            print("Dataset is already up to date, no changes needed")
            return True
        
        # Create new dataset
        print("Saving modified dataset...")
        
        features = Features({
            'image': HFImage(),
            'problem': Value('string'),
            'solution': Value('string')
        })
        
        new_train_data = Dataset.from_dict(
            {
                'image': [s['image'] for s in fixed_data],
                'problem': [s['problem'] for s in fixed_data],
                'solution': [s['solution'] for s in fixed_data]
            },
            features=features
        )
        
        new_dataset_dict = DatasetDict({
            'train': new_train_data
        })
        
        # Save (overwrite original dataset)
        new_dataset_dict.save_to_disk(dataset_path)
        
        print(f"Done!")
        print(f"   Fixed fishing_ships: {fishing_fixed_count} samples")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("Batch fix ship30 CLS dataset paths and category names")
    print("="*70)
    print()
    print("Changes:")
    print("  1. Path: /data/ccy/ccy-factory/datasets/boat-29/images")
    print("        -> /data/ljx/dataset/boat-29/images")
    print("  2. Category: fishing_ships -> fishing_boat")
    print()
    
    base_path = "/data/ljx/visualRft/share_data"
    
    # Only fix CLS related datasets
    datasets = [
        'ViRFT_CLS_ship30_1shot',
        'ViRFT_CLS_ship30_2shot',
        'ViRFT_CLS_ship30_4shot',
        'ViRFT_CLS_ship30_4shot_14',
        'ViRFT_CLS_ship30_8shot',
        'ViRFT_CLS_ship30_16shot',
        'ViRFT_CLS_ship30_OOD_4shot',
        'ViRFT_CLS_ship30_OOD_4shot_balanced',
        'ViRFT_CLS_ship30_OOD_16shot',
        'ViRFT_CLS_ship30_OOD_16shot_balanced'
    ]
    
    success_count = 0
    for dataset_name in datasets:
        dataset_path = os.path.join(base_path, dataset_name)
        if fix_dataset(dataset_path):
            success_count += 1
    
    print("\n" + "="*70)
    print(f"Complete! Successfully processed {success_count}/{len(datasets)} datasets")
    print("="*70)


if __name__ == "__main__":
    main()
