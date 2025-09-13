# nohup python -u ego_only.py > GPT5_relative_coord.log 2>&1 & tail -f GPT5_relative_coord.log
import os
import torch
import random
from PIL import Image
# import my_prompt4_gpt as my_prompt
import my_prompt4_relative_coordi as my_prompt
from file_managing import (
    load_selected_samples,
    get_actual_path,
    get_gt_path,
)
from config import AGD20K_PATH, model_name
# from VLM_model_dot_gpt import QwenVLModel, MetricsTracker
from VLM_model_dot import QwenVLModel, MetricsTracker

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ENABLE_SDPA"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor

model_root = "qihoo360/fg-clip-base"
image_size = 224
fg_clip_model = AutoModelForCausalLM.from_pretrained(model_root, trust_remote_code=True).cuda()
fg_clip_model.eval()  # Set to evaluation mode

fg_clip_tokenizer = AutoTokenizer.from_pretrained(model_root)
fg_clip_image_processor = AutoImageProcessor.from_pretrained(model_root)




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



def affordance_grounding_validation_with_new_points(model, action: str, object_name: str, 
                                                  image_path: str, gt_path, dot_image_path: str, 
                                                  exo_path: str, exo_type: str):
    """
    Perform affordance grounding validation with new point generation using QwenVLModel
    Args:
        model (QwenVLModel): The Qwen VL model instance
        action (str): Action name
        object_name (str): Object name
        image_path (str): Path to ego image
        dot_image_path (str): Path to dot image (ego image with dots marked)
        exo_path (str): Path to exo image
        exp_name (str): Experiment name
    Returns:
        dict: Validation results including validation result and new points
    """
    prompt = my_prompt.validation_and_process_again_prompt(action, object_name)
    

    # Use the model's validation_and_process_again method with dot image instead of heatmap
    results = model.validation_and_process_again(image_path, prompt, gt_path, exo_path, dot_image_path, action, exo_type)  
    return results



        



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
    # return {
    #     'text_result': result.strip(),
    #     'bboxes': bboxes,
    #     'bbox_image_path': bbox_image_path,
    #     'heatmap_tensor': heatmap_tensor,
    #     'metrics': metrics
    # }


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

def get_fgclip_aligned_exo_paths_topK(ego_path, AGD20K_reference_PATH=AGD20K_PATH, top_k=2):
    """
    Get top-k exocentric image paths most similar to the given egocentric image using FG-CLIP.
    
    Args:
        ego_path (str): Path to the egocentric image
        AGD20K_reference_PATH (str): Root path of the dataset
        top_k (int): Number of top similar exocentric images to return

    Returns:
        List[str]: List of top-k exocentric image paths (sorted by similarity), or empty list if none found
    """
    try:
        # Extract action and object
        parts = ego_path.split('/')
        action_idx = parts.index('egocentric') + 1
        action = parts[action_idx]
        object_name = parts[action_idx + 1]

        # Construct exo dir path
        exo_dir = os.path.join(
            AGD20K_reference_PATH,
            'Seen',
            'trainset',
            'exocentric',
            action,
            object_name
        )
        if not os.path.exists(exo_dir):
            print(f"⚠️ Exo directory not found: {exo_dir}")
            return []

        exo_image_list = [
            os.path.join(exo_dir, fname)
            for fname in os.listdir(exo_dir)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if len(exo_image_list) == 0:
            print("⚠️ No exocentric images found.")
            return []

        # Load ego feature
        ego_image = Image.open(ego_path).convert("RGB")
        ego_inputs = fg_clip_image_processor(images=ego_image, return_tensors="pt").to(fg_clip_model.device)
        with torch.no_grad():
            ego_feat = fg_clip_model.get_image_features(**ego_inputs)
            ego_feat = ego_feat / ego_feat.norm(dim=-1, keepdim=True)

        # Compute similarity for all exo images
        similarity_list = []
        for exo_path in exo_image_list:
            try:
                exo_image = Image.open(exo_path).convert("RGB")
                exo_inputs = fg_clip_image_processor(images=exo_image, return_tensors="pt").to(fg_clip_model.device)
                with torch.no_grad():
                    exo_feat = fg_clip_model.get_image_features(**exo_inputs)
                    exo_feat = exo_feat / exo_feat.norm(dim=-1, keepdim=True)
                sim = torch.matmul(ego_feat, exo_feat.T).item()
                similarity_list.append((exo_path, sim))
            except Exception as ie:
                print(f"⚠️ Skipped invalid image {exo_path}: {ie}")
                continue

        # Sort and select top-k
        similarity_list.sort(key=lambda x: x[1], reverse=True)
        top_k_paths = [path for path, _ in similarity_list[:top_k]]

        return top_k_paths

    except Exception as e:
        print(f"⚠️ Error finding exocentric images: {str(e)}")
        return []


def get_fgclip_aligned_exo_path(ego_path, AGD20K_reference_PATH=AGD20K_PATH):
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
        
        if not os.path.exists(exo_dir):
            print(f"⚠️ Exo directory not found: {exo_dir}")
            return None

        exo_image_list = [
            os.path.join(exo_dir, fname)
            for fname in os.listdir(exo_dir)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if len(exo_image_list) == 0:
            print("⚠️ No exocentric images found.")
            return None

        # Step 3: Process ego image
        ego_image = Image.open(ego_path).convert("RGB")
        ego_inputs = fg_clip_image_processor(images=ego_image, return_tensors="pt").to(fg_clip_model.device)

        with torch.no_grad():
            ego_feat = fg_clip_model.get_image_features(**ego_inputs)
            ego_feat = ego_feat / ego_feat.norm(dim=-1, keepdim=True)

        # Step 4: Loop through exo images and compute similarity
        best_path = None
        best_score = -1

        for exo_path in exo_image_list:
            try:
                exo_image = Image.open(exo_path).convert("RGB")
                exo_inputs = fg_clip_image_processor(images=exo_image, return_tensors="pt").to(fg_clip_model.device)

                with torch.no_grad():
                    exo_feat = fg_clip_model.get_image_features(**exo_inputs)
                    exo_feat = exo_feat / exo_feat.norm(dim=-1, keepdim=True)

                sim = torch.matmul(ego_feat, exo_feat.T).item()
                if sim > best_score:
                    best_score = sim
                    best_path = exo_path
            except Exception as ie:
                print(f"⚠️ Skipped invalid image {exo_path}: {ie}")
                continue

        return best_path


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


def affordance_grounding_validation(model, action, object_name, image_path, first_res_image ,validation_exo_path, exo_type=None):
    """
    Validate the affordance grounding results using VLM with new-exocentric image   
    """
    prompt = my_prompt.validaton_prompt(action, object_name)

    results, reason = model.validate_image(image_path, prompt,first_res_image, validation_exo_path, action, exo_type)
    return results, reason




def selecting_best_af(model, ego_image_path, action, object_name, dot_image_paths):
    """
    Selects the best affordance dot image from multiple candidates.
    Args:
        model (QwenVLModel): The Qwen VL model instance.
        ego_image_path (str): Path to the egocentric image.
        action (str): The action name.
        object_name (str): The object name.
        dot_image_paths (list): A list of paths to candidate dot images.
    Returns:
        str: The path to the best dot image selected by the VLM.
    """
    print("\n=== Selecting Best Dot Image ===")
    
    # Filter out None or non-existent paths
    valid_dot_paths = [p for p in dot_image_paths if p and os.path.exists(p)]
    
    if not valid_dot_paths:
        print("⚠️ No valid dot image paths provided for selection.")
        return None
    
    if len(valid_dot_paths) == 1:
        print("Only one valid dot image. Selecting it by default.")
        return valid_dot_paths[0]
        
    best_dot_path, best_index = model.select_best_dot_image(
        ego_image_path,
        action,
        object_name,
        valid_dot_paths
    )
    
    return best_dot_path, best_index


def main():
    # Initialize Qwen VL model
    model = QwenVLModel(model_name = model_name)
    metrics_tracker_ego = MetricsTracker(name="only_ego")

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
        processed_count += 1
        print(f"--- Start  {processed_count}  / {total_samples}", "-"*80) 
        
        action = sample_info["action"]
        object_name = sample_info["object"]

        image_path = get_actual_path(sample_info["image_path"])
        gt_path = get_gt_path(image_path)   
        print(f"Action : {action}, Object : {object_name} image_name : {image_path.split('/')[-1]}")
        # Process the image
        results_ego = affordance_grounding(model, action, object_name, image_path, gt_path)
        metrics_ego = results_ego['metrics']
        if metrics_ego:
            # Update and print metrics
            metrics_tracker_ego.update(metrics_ego)
            metrics_tracker_ego.print_metrics(metrics_ego, image_path.split('/')[-1])
                    
        # Count missing GT files
        if not os.path.exists(gt_path):
            missing_gt += 1
        
        print("*** End  ", "*"*150)
        print("\n\n")

    # Print final summary
    print("=" * 50)
    print(f"Total number of action-object pairs processed: {total_samples}")
    print(f"Number of missing GT files: {missing_gt}")
    print(f"All images successfully processed!")

if __name__ == "__main__":
    main() 