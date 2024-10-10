import json
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import time
import cv2
import copy
from itertools import groupby

def load_yaml(file_path):
    with open(file_path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def create_directories(directory_paths):
    """
    Create directories specified in the directory_paths list.
    If a directory path is a dictionary, it recursively creates
    directories for its values.
    """
    for directory in directory_paths:
        if isinstance(directory, Path):
            directory.mkdir(parents=True, exist_ok=True)
        elif isinstance(directory, dict):
            create_directories(directory.values())

def initialize_paths(results_directory):
    """
    Initialize and create directory paths for storing results.
    """
    base_path = Path(results_directory)
    paths = {
        "images": base_path / "img",
        "img_unoccluded_objects_only": base_path / "img_unoccluded_objects_only",
        "segmentation_maps": base_path / "seg_maps",
        "other_segmentation_maps": base_path / "other_seg_maps",
        "zoomed_out_segmentation_maps": base_path / "zoomed_out_seg_maps",
        "yolo_labels": {
            "hbb_all_objs": base_path / "yolo_labels" / "hbb_all_objects",
            "obb_all_objs": base_path / "yolo_labels" / "obb_all_objects",
            "hbb_unoccluded_objects_only": base_path / "yolo_labels" / "hbb_unoccluded_objects_only",
            "obb_unoccluded_objects_only": base_path / "yolo_labels" / "obb_unoccluded_objects_only",
        },
        "visualization": {
            "hbb_all_objs": base_path / "visualization" / "hbb_all_objects",
            "obb_all_objs": base_path / "visualization" / "obb_all_objects",
            "hbb_unoccluded_objects_only": base_path / "visualization" / "hbb_unoccluded_objects_only",
            "obb_unoccluded_objects_only": base_path / "visualization" / "obb_unoccluded_objects_only",
        }
    }
    create_directories(paths.values())
    return paths

def inst_is_visible(instance_id, occlusion_aware_segmentation_map, clear_segmentation_map, visibility_threshold):
    """
    Determine if an object is considered visible based on the visibility threshold.

    Args:
        instance_id (int): The ID of the object to check.
        occlusion_aware_segmentation_map (np.ndarray): Segmentation map accounting for occlusions.
        clear_segmentation_map (np.ndarray): Segmentation map without occlusions.
        visibility_threshold (float): The minimum ratio of visible area to total area for an object to be considered visible.

    Returns:
        bool: True if the object is considered visible, False otherwise.
    """
    try:
        # Calculate the ratio of visible pixels to total pixels for the object
        visible_ratio = np.count_nonzero(occlusion_aware_segmentation_map == instance_id) / np.count_nonzero(clear_segmentation_map == instance_id)
    except ZeroDivisionError:
        # If there are no pixels for this object in the clear map, set visibility ratio to 0
        visible_ratio = 0
    
    # Return True if the object meets the visibility threshold, otherwise False
    return visible_ratio >= visibility_threshold

def inst_too_small(instance_id, segmentation_map, min_pixel_count):
    """
    Check if an instance has fewer pixels than a given threshold.

    Returns:
        bool: True if the instance is too small, False otherwise.
    """
    if min_pixel_count <= 0:
        return False
    object_size = np.count_nonzero(segmentation_map == instance_id)
    return object_size < min_pixel_count

def binary_mask_to_rle(binary_mask):
    """
    Convert a binary mask to run-length encoding (RLE).

    Args:
        binary_mask (np.ndarray): A 2D binary mask where 1 indicates the presence of an object.

    Returns:
        dict: Dictionary containing 'counts' (RLE counts) and 'size' (shape of the mask).
    """
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Check if two lines intersect.

    Args:
        x1, y1 (float): Start coordinates of the first line.
        x2, y2 (float): End coordinates of the first line.
        x3, y3 (float): Start coordinates of the second line.
        x4, y4 (float): End coordinates of the second line.

    Returns:
        bool: True if the lines intersect, False otherwise.
    """
    def is_counter_clockwise(x1, y1, x2, y2, x3, y3):
        return (y3 - y1) * (x2 - x1) > (y2 - y1) * (x3 - x1)

    return is_counter_clockwise(x1, y1, x3, y3, x4, y4) != is_counter_clockwise(x2, y2, x3, y3, x4, y4) and \
           is_counter_clockwise(x1, y1, x2, y2, x3, y3) != is_counter_clockwise(x1, y1, x2, y2, x4, y4)

def calculate_overlap_ratio(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Calculate the overlap ratio between two bounding boxes.

    Args:
        x1, y1, x2, y2 (float): Coordinates of the first bounding box.
        x3, y3, x4, y4 (float): Coordinates of the second bounding box.

    Returns:
        float: The overlap ratio, ranging from 0.0 (no overlap) to 1.0 (complete overlap).
    """
    overlap_x1 = max(min(x1, x2), min(x3, x4))
    overlap_y1 = max(min(y1, y2), min(y3, y4))
    overlap_x2 = min(max(x1, x2), max(x3, x4))
    overlap_y2 = min(max(y1, y2), max(y3, y4))

    overlap_width = max(0, overlap_x2 - overlap_x1)
    overlap_height = max(0, overlap_y2 - overlap_y1)

    box1_area = abs(x2 - x1) * abs(y2 - y1)
    box2_area = abs(x4 - x3) * abs(y4 - y3)

    overlap_area = overlap_width * overlap_height
    total_area = box1_area + box2_area - overlap_area
    overlap_ratio = overlap_area / total_area if total_area > 0 else 0.0
    
    return overlap_ratio

def inst_on_edge(x_bounding_box, y_bounding_box, bounding_box_width, bounding_box_height, image):
    """
    Check if a bounding box lies on the edge of an image.

    Args:
        x_bounding_box (int): X-coordinate of the top-left corner of the bounding box.
        y_bounding_box (int): Y-coordinate of the top-left corner of the bounding box.
        bounding_box_width (int): Width of the bounding box.
        bounding_box_height (int): Height of the bounding box.
        image (np.ndarray): The image array.

    Returns:
        bool: True if the bounding box lies on the edge, False otherwise.
    """
    return (x_bounding_box == 0 or 
            y_bounding_box == 0 or 
            x_bounding_box + bounding_box_width == image.shape[1] or 
            y_bounding_box + bounding_box_height == image.shape[0])

def setup_categories(class_count, class_names):
    """
    Set up category information for annotations.

    Returns:
        list: List of dictionaries, each containing an 'id' and 'name' for a category.
    """
    return [{"id": (i + 1), "name": class_names[i]} for i in range(class_count)]

def annotate_yolo_labels(yolo_label_path, image_path, output_path):
    """
    Annotate YOLO labels on the given image and save the annotated image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise cv2.error(f"Error: Unable to load image from the specified path: {image_path}")

    try:
        with open(yolo_label_path, 'r') as label_file:
            label_lines = label_file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: YOLO label file not found at path: {yolo_label_path}")

    expected_num_values = -1
    for label_line in label_lines:
        label_values = list(map(float, label_line.strip().split()))

        if expected_num_values == -1:
            expected_num_values = len(label_values)
        
        if len(label_values) != expected_num_values:
            raise ValueError(f"Inconsistent number of components in '{yolo_label_path}'.")

        if len(label_values) == 5:  # Regular bounding box
            class_id, x_center_norm, y_center_norm, width_norm, height_norm = label_values
            image_height, image_width = image.shape[:2]
            x_center = int(x_center_norm * image_width)
            y_center = int(y_center_norm * image_height)
            box_width = int(width_norm * image_width)
            box_height = int(height_norm * image_height)

            x_min = int(x_center - box_width / 2)
            y_min = int(y_center - box_height / 2)
            x_max = int(x_center + box_width / 2)
            y_max = int(y_center + box_height / 2)

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            cv2.putText(image, str(int(class_id)), (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        elif len(label_values) == 9:  # Oriented bounding box
            class_id = int(label_values[0])
            obb_points = [(label_values[i], label_values[i + 1]) for i in range(1, 9, 2)]
            obb_points = np.array(obb_points, dtype=np.int32)
            
            cv2.polylines(image, [obb_points], isClosed=True, color=(0, 255, 0), thickness=1)
            cv2.putText(image, str(class_id), (obb_points[0][0], obb_points[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        else:
            raise ValueError(f"Unexpected number of values in '{yolo_label_path}'.")

    success = cv2.imwrite(str(output_path), image)
    if not success:
        print(f"Error: Could not save the annotated image to the specified path: {output_path}")

def read_image(image_path):
    return cv2.imread(str(image_path), -1)

def append_bbox_data(bbox_dict, category_id, x_bbox, bbox_width, bbox_height, obb_points, image_width):
    bbox_dict["category_id"].append(category_id) 
    bbox_dict["center_x"].append((x_bbox + bbox_width / 2) / image_width)
    bbox_dict["center_y"].append((y_bbox + bbox_height / 2) / image_height)
    bbox_dict["width"].append(bbox_width / image_width)
    bbox_dict["height"].append(bbox_height / image_height)

    for j in range(4):
        bbox_dict[f'obb{j + 1}_x'].append(obb_points[j][0])
        bbox_dict[f'obb{j + 1}_y'].append(obb_points[j][1])

## Main processing code

# Load configuration files for rendering parameters and models
render_parameters = load_yaml("./config/render_parameters.yaml")
model_parameters = load_yaml("./config/models.yaml")

# Print loaded configuration and model parameters for verification
for key, value in {**render_parameters, **model_parameters}.items():
    print(f"{key}: {value}")

# Extract specific configuration settings for annotations and file paths
show_annotations = render_parameters["view_annotations"]
visibility_threshold = render_parameters["visibility_thresh"]
min_pixel_count = render_parameters["min_pixels"]
results_directory = model_parameters["render_to"]

# Get the class names from the model configuration
class_names = [Path(class_path).stem for class_path in model_parameters["classes"]]
class_count = len(model_parameters["classes"])

# Initialize directory paths for storing results and set up categories for annotation
paths = initialize_paths(results_directory)
coco_annotations = {"images": [], "categories": setup_categories(class_count, class_names), "annotations": []}
 
# Start timer to measure the total processing time
start_time = time.time()

# Initialize variables for processing images and tracking annotation IDs
annotation_id = 1
filled_bboxes = set()

# Process each image in the "images" directory
for image_id, image_filepath in enumerate(paths["images"].glob('*'), start=1):
    # Extract image metadata and add to the COCO annotations
    image_filename = image_filepath.name
    image = read_image(image_filepath)
    image_height, image_width = image.shape[:2]
    image_info = {
        "id": image_id,
        "file_name": image_filename,
        "height": image_height,
        "width": image_width,
    }
    coco_annotations["images"].append(image_info)

    # Prepare for handling annotations and segmentation maps
    annotation_all_objects = []
    annotation_after_occlusion_removal = []
    overlapping_instances = set()  # Contains instances that overlap

    occlusion_aware_segmentation_map = read_image(paths["segmentation_maps"] / image_filename)
    occlusion_ignore_segmentation_map = read_image(paths["other_segmentation_maps"] / image_filename)
    zoomed_out_segmentation_map = read_image(paths["zoomed_out_segmentation_maps"] / image_filename)

    # Initialize dictionaries to store bounding box annotations
    bbox_annotations_all_objs = {
        "category_id": [], "center_x": [], "center_y": [], "width": [], "height": [],
        "obb1_x": [], "obb1_y": [], "obb2_x": [], "obb2_y": [],
        "obb3_x": [], "obb3_y": [], "obb4_x": [], "obb4_y": []
    }
    bbox_annotations_after_occlusion_removal = copy.deepcopy(bbox_annotations_all_objs)

    # Process each instance in the segmentation map
    instances = np.unique(occlusion_aware_segmentation_map)
    instances = instances[instances != 0]

    detections = []
    for instance_id in instances:
        inst_is_occluded = False
        category_id = int(instance_id // 1000) + 1

        segmentation_map = occlusion_ignore_segmentation_map

        # Get coordinates of the object from the segmentation map
        points = cv2.findNonZero((segmentation_map == instance_id).astype(int))
        if points is None:
            continue

        # Calculate the bounding boxes (HBB and OBB) for the object
        x_bbox, y_bbox, bbox_width, bbox_height = cv2.boundingRect(points)
        obb = cv2.minAreaRect(points)
        obb_points = cv2.boxPoints(obb).astype(int)  # 4 x 2 --> 4 points for bounding box

        # Check visibility, size, and potential occlusion of the object
        is_too_small = inst_too_small(instance_id, segmentation_map, min_pixel_count)
        # # Check for "invisible" where too much is occluded
        is_invisible = not inst_is_visible(instance_id, occlusion_aware_segmentation_map, occlusion_ignore_segmentation_map, visibility_threshold)

        if is_too_small or is_invisible:
            # Mark the object as occluded by filling the polygon with black color
            image = cv2.fillPoly(image, [obb_points], color=(0, 0, 0))
            inst_is_occluded = True
        elif tuple(map(tuple, obb_points)) in filled_bboxes:
            inst_is_occluded = True
        else:
            # Check if there are intersections with other bounding boxes
            for i in range(obb_points.shape[0]):
                x1, y1, x2, y2 = (obb_points[i][0], obb_points[i][1], 
                                obb_points[(i + 1) % obb_points.shape[0]][0], 
                                obb_points[(i + 1) % obb_points.shape[0]][1])
                
                for bbox_index in range(len(bbox_annotations_all_objs["center_x"])):
                    obb_points2 = np.array([
                        [bbox_annotations_all_objs["obb1_x"][bbox_index], bbox_annotations_all_objs["obb1_y"][bbox_index]],
                        [bbox_annotations_all_objs["obb2_x"][bbox_index], bbox_annotations_all_objs["obb2_y"][bbox_index]],
                        [bbox_annotations_all_objs["obb3_x"][bbox_index], bbox_annotations_all_objs["obb3_y"][bbox_index]],
                        [bbox_annotations_all_objs["obb4_x"][bbox_index], bbox_annotations_all_objs["obb4_y"][bbox_index]]
                    ])

                    width2, height2 = round(bbox_annotations_all_objs["width"][bbox_index] * image_width), round(bbox_annotations_all_objs["height"][bbox_index] * image_height)
                    x_bbox2, y_bbox2 = round((bbox_annotations_all_objs["center_x"][bbox_index] * image_width) - width2 / 2), round((bbox_annotations_all_objs["center_y"][bbox_index] * image_height) - height2 / 2)

                    # Check for line intersections and overlap
                    for k in range(obb_points2.shape[0]):
                        x3, y3, x4, y4 = (obb_points2[k][0], obb_points2[k][1], 
                                        obb_points2[(k + 1) % obb_points2.shape[0]][0], 
                                        obb_points2[(k + 1) % obb_points2.shape[0]][1])
                        
                        if lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4) and (calculate_overlap_ratio(x1, y1, x2, y2, x3, y3, x4, y4) > (1.0 - visibility_threshold)):
                            image = cv2.fillPoly(image, [obb_points], color=(0, 0, 0))
                            image = cv2.fillPoly(image, [obb_points2], color=(0, 0, 0))
                            filled_bboxes.add(tuple(map(tuple, obb_points)))
                            filled_bboxes.add(tuple(map(tuple, obb_points2)))

                            inst_is_occluded = True

        # Store bounding box data for both occluded and unoccluded cases
        append_bbox_data(bbox_annotations_all_objs, category_id, x_bbox, bbox_width, bbox_height, obb_points, image_width)

        inst_annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x_bbox, y_bbox, bbox_width, bbox_height],
            "segmentation": binary_mask_to_rle((segmentation_map == instance_id).astype('uint8')),
            "area": bbox_width * bbox_height,
            "iscrowd": 0
        }

        # Add annotations
        annotation_all_objects.append(inst_annotation)
        if not inst_is_occluded:
            append_bbox_data(bbox_annotations_after_occlusion_removal, category_id, x_bbox, bbox_width, bbox_height, obb_points, image_width)
            annotation_after_occlusion_removal.append(inst_annotation)

        annotation_id += 1

    # Convert bounding box data to DataFrames for COCO annotation
    df_all_objs = pd.DataFrame.from_dict(bbox_annotations_all_objs)
    df_after_occlusion_removal = pd.DataFrame.from_dict(bbox_annotations_after_occlusion_removal)

    # Ensure each image has corresponding labels, even if empty
    label_filename = image_filepath.with_suffix('.txt').name
    for label_folder in paths["yolo_labels"].values():
        label_path = label_folder / label_filename
        with open(label_path, 'w') as file:
            pass

    # Save annotations for all objects and unoccluded objects only
    np.savetxt(paths["yolo_labels"]["hbb_all_objs"] / (label_filename), df_all_objs[["category_id", "center_x", "center_y", "width", "height"]], delimiter=' ', fmt=['%d', '%.4f', '%.4f', '%.4f', '%.4f'])
    np.savetxt(paths["yolo_labels"]["hbb_unoccluded_objects_only"] / (label_filename), df_after_occlusion_removal[["category_id", "center_x", "center_y", "width", "height"]], delimiter=' ', fmt=['%d', '%.4f', '%.4f', '%.4f', '%.4f'])
    np.savetxt(paths["yolo_labels"]["obb_all_objs"] / (label_filename), df_all_objs[["category_id", "obb1_x", "obb1_y", "obb2_x", "obb2_y", "obb3_x", "obb3_y", "obb4_x", "obb4_y"]], delimiter=' ', fmt=['%d'] + ['%d'] * 8)
    np.savetxt(paths["yolo_labels"]["obb_unoccluded_objects_only"] / (label_filename), df_after_occlusion_removal[["category_id", "obb1_x", "obb1_y", "obb2_x", "obb2_y", "obb3_x", "obb3_y", "obb4_x", "obb4_y"]], delimiter=' ', fmt=['%d'] + ['%d'] * 8)

    # Update COCO annotations
    coco_annotations_after_occlusion_removal = copy.deepcopy(coco_annotations)
    coco_annotations["annotations"].extend(annotation_all_objects)
    coco_annotations_after_occlusion_removal["annotations"].extend(annotation_after_occlusion_removal)

    # Save modified image with only unoccluded objects
    image_unoccluded_objects_only_filepath = paths["img_unoccluded_objects_only"] / image_filename
    cv2.imwrite(str(image_unoccluded_objects_only_filepath), image)

    # Create visualizations for the image
    for key, output_vis_folder in paths["visualization"].items():
        output_vis_filepath = output_vis_folder / image_filepath.name
        label_filepath = Path(str(output_vis_folder).replace("visualization", "yolo_labels")) / label_filename
        if "unoccluded_objects_only" in key:
            annotate_yolo_labels(label_filepath, image_unoccluded_objects_only_filepath, output_vis_filepath)
        else:
            annotate_yolo_labels(label_filepath, image_filepath, output_vis_filepath)

    # Print progress for every 50 images processed
    if image_id % 50 == 0:
        print(f"=== Processed {image_id} images")

# Save COCO-style annotations to JSON files
with open(str(Path(results_directory) / "coco_annotations.json"), "w") as f:
    json.dump(coco_annotations, f)

with open(str(Path(results_directory) / "coco_annotations_after_occlusion_removal.json"), "w") as f:
    json.dump(coco_annotations_after_occlusion_removal, f)

# Calculate and display the total processing time
end_time = time.time()
total_time = end_time - start_time
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Time taken for to create labels: {int(hours)}h{int(minutes)}m{seconds:.2f}s")
print("\nCompleted!")
