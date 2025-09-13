import os
import torch
import random
from PIL import Image
import my_prompt4 as my_prompt
from file_managing import (
    load_selected_samples,
    get_actual_path,
    get_gt_path,
)
from config import AGD20K_PATH, model_name
from VLM_model_dot import QwenVLModel, MetricsTracker
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ENABLE_SDPA"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor


def get_all_ego_images():
    """
    Get all egocentric images from the AGD20K dataset
    Returns:
        list: List of dictionaries containing image info with keys:
              - image_path: Path to the ego image
              - gt_path: Path to the corresponding GT file
              - action: Action name extracted from path
              - object: Object name extracted from path
    """
    all_images = []
    
    # Construct the path to egocentric images
    ego_base_path = os.path.join(AGD20K_PATH, "Seen", "testset", "egocentric")
    
    if not os.path.exists(ego_base_path):
        print(f"⚠️ Egocentric base path does not exist: {ego_base_path}")
        return all_images
    
    # Walk through all action directories
    for action in os.listdir(ego_base_path):
        action_path = os.path.join(ego_base_path, action)
        if not os.path.isdir(action_path):
            continue
            
        # Walk through all object directories within each action
        for object_name in os.listdir(action_path):
            object_path = os.path.join(action_path, object_name)
            if not os.path.isdir(object_path):
                continue
                
            # Get all image files in the object directory
            for image_file in os.listdir(object_path):
                if image_file.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(object_path, image_file)
                    gt_path = get_gt_path(image_path)
                    
                    image_info = {
                        'image_path': image_path,
                        'gt_path': gt_path,
                        'action': action,
                        'object': object_name
                    }
                    
                    all_images.append(image_info)
    
    # Sort by action, then by object, then by image name for consistent ordering
    all_images.sort(key=lambda x: (x['action'], x['object'], x['image_path']))
    
    print(f"📊 Found {len(all_images)} egocentric images")
    return all_images


def affordance_grounding(model, action, object_name, image_path, gt_path, exo_path=None, exo_type=None, failed_heatmap_path=None, validation_reason=None):
    """
    Process each image using Qwen VL model
    """
    # print(f"Processing image: Action: {action}, Object: {object_name}, Image path: {image_path.split('/')[-1]}, GT path: {gt_path.split('/')[-1]}, Image exists: {os.path.exists(image_path)}, GT exists: {os.path.exists(gt_path)}")
    

    if exo_path is None:
        prompt = my_prompt.process_image_ego_prompt(action, object_name)
               
        results = model.process_image_ego(image_path, prompt, gt_path, action, exo_type)

        
    else:
        if failed_heatmap_path is not None:
            # When we have a failed heatmap, include it in the prompt for better context
            
            prompt = my_prompt.process_image_exo_with_heatmap_prompt(action, object_name, validation_reason)
        
            results = model.process_image_exo_with_heatmap(image_path, prompt, gt_path, exo_path, failed_heatmap_path, action, exo_type)
        else:
            prompt = my_prompt.process_image_exo_prompt(action, object_name)
            results = model.process_image_exo(image_path, prompt, gt_path, exo_path, action, exo_type)

    return results

    # Save results


    # Save results

def get_random_exo_path(ego_path, AGD20K_reference_PATH=AGD20K_PATH):
    """
    Get a random exocentric image path based on the egocentric image path
    Args:
        ego_path (str): Path to the egocentric image
    Returns:
        str: Path to a random exocentric image, or None if no exo images found
    """
    try:
        # Extract action and object from ego path
        # Example ego path: .../Seen/testset/egocentric/wash/cup/cup_003621.jpg
        parts = ego_path.split('/')
        action_idx = parts.index('egocentric') + 1
        action = parts[action_idx]
        object_name = parts[action_idx + 1]
        
        # Construct exo directory path
        # Change 'testset/egocentric' to 'trainset/exocentric'
        exo_dir = os.path.join(
            AGD20K_reference_PATH,
            'Seen',
            'trainset',
            'exocentric',
            action,
            object_name
        )
        
        # Check if directory exists
        if not os.path.exists(exo_dir):
            print(f"⚠️ No exocentric directory found: {exo_dir}")
            return None
            
        # Get all jpg files in the directory
        exo_files = [f for f in os.listdir(exo_dir) if f.endswith('.jpg')]
        
        if not exo_files:
            print(f"⚠️ No exocentric images found in: {exo_dir}")
            return None
            
        # Select a random file
        random_exo = random.choice(exo_files)
        exo_path = os.path.join(exo_dir, random_exo)
        
        print(f"Selected exo image: {exo_path}")
        return exo_path
        
    except Exception as e:
        print(f"⚠️ Error finding exocentric image: {str(e)}")
        return None

def get_selected_exo_path(sample_info, data, best_exo_data=None):
    """
    Get exocentric image path from selected_samples.json or best exo data
    Args:
        sample_info (dict): Sample information from selected_samples.json
        data (dict): Full data from selected_samples.json
        best_exo_data (dict): Best exo data from top_image_per_action_object.json
    Returns:
        str: Path to the selected exocentric image, or None if not found
    """
    try:
       
        # Use best exo data if available
        if best_exo_data:
            action = sample_info["action"]
            object_name = sample_info["object"]
            action_object_key = f"{action}/{object_name}"
            best_exo_path = best_exo_data[action_object_key]["image"]
            print(f"Using best exo image from top_image_per_action_object.json: {best_exo_path}")
            print(f"Score: {best_exo_data[action_object_key]['score']:.4f}")
            return best_exo_path


    except Exception as e:
        print(f"⚠️ Error getting selected exocentric image: {str(e)}")
        # Fallback to random selection
        ego_path = get_actual_path(sample_info['image_path'])
        return get_random_exo_path(ego_path, AGD20K_reference_PATH = AGD20K_PATH)

def main():
    # Initialize Qwen VL model
    model = QwenVLModel(model_name = model_name)
    metrics_tracker_ego = MetricsTracker(name="only_ego")


    # # Get all ego images
    # print("🔍 Scanning for all egocentric images...")
    # all_images = get_all_ego_images()
    # total_samples = len(all_images)
    # print(f"📊 Found {total_samples} images to process")
    # print("=" * 100)

    # validation_result_list = []
    # missing_gt = 0
    # processed_count = 0

    # # Get total number of samples
    # total_samples = len(all_images)
    
    # # Process each sample
    # print(f"Processing {total_samples} samples...")
    # print("=" * 50)

    # for idx, image_info in enumerate(all_images):
    #     print(f"--- Start [{idx+1}/{total_samples}] ", "-"*120) 
    #     action = image_info["action"]
    #     object_name = image_info["object"]
    #     image_path = image_info["image_path"]
    #     gt_path = image_info["gt_path"]
    json_path = os.path.join("selected_samples.json")
    data = load_selected_samples(json_path)
    missing_gt = 0
    processed_count = 0

    # Get total number of samples
    total_samples = len(data['selected_samples'])
    
    # Process each sample
    print(f"Processing {total_samples} samples...")
    print("=" * 50)    
    for pair_key, sample_info in data["selected_samples"].items():
        print("--- Start  ", "-"*80) 
        
        action = sample_info["action"]
        object_name = sample_info["object"]

        image_path = get_actual_path(sample_info["image_path"])
        gt_path = get_gt_path(image_path)   
        print(f"Action : {action}, Object : {object_name} image_name : {image_path.split('/')[-1]}")
        # Process the image
        prompt = my_prompt.ask_image_ego_prompt(action, object_name)
        ask_results = model.ask_with_image(prompt,image_path)
        print(f"ANSWER : {ask_results}")

    # Print final summary
    print("=" * 50)
    print(f"Total number of action-object pairs processed: {total_samples}")
    print(f"Number of missing GT files: {missing_gt}")
    print(f"All images successfully processed!")

if __name__ == "__main__":
    main() 