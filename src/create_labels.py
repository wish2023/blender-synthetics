from itertools import groupby
import json
import os
import yaml

import numpy as np
import pandas as pd
import random

import cv2


with open("./config/render_parameters.yaml") as file:
    config_info = yaml.load(file, Loader=yaml.FullLoader)

with open("./config/models.yaml") as file:
    models_info = yaml.load(file, Loader=yaml.FullLoader)

for key, value in models_info.items():
    print(f"{key}: {value}")

for key, value in config_info.items():
    print(f"{key}: {value}")

view_annotations = config_info["view_annotations"]
occlusion_aware = config_info["occlusion_aware"]
visibility_thresh = config_info["visibility_thresh"]
component_visibility_thresh = config_info["component_visibility_thresh"]
results_dir = models_info["render_to"]
classes = [os.path.basename(os.path.normpath(class_path)) for class_path in models_info["classes"]]
num_classes = len(models_info["classes"])

img_path = os.path.join(results_dir, "img")
occ_aware_seg_path = os.path.join(results_dir, "seg_maps")
occ_ignore_seg_path = os.path.join(results_dir, "other_seg_maps")
zoomed_out_seg_path = os.path.join(results_dir, "zoomed_out_seg_maps")
yolo_annotated_path = os.path.join(results_dir, "yolo_annotated")
obb_annotated_path = os.path.join(results_dir, "obb_annotated")
yolo_labels_path = os.path.join(results_dir, "yolo_labels")
obb_labels_path = os.path.join(results_dir, "obb_labels")


if not os.path.isdir(yolo_annotated_path):
    os.mkdir(yolo_annotated_path)
if not os.path.isdir(obb_annotated_path):
    os.mkdir(obb_annotated_path)
if not os.path.isdir(yolo_labels_path):
    os.mkdir(yolo_labels_path)
if not os.path.isdir(obb_labels_path):
    os.mkdir(obb_labels_path)

coco_ann = {"images": [], "categories": [], "annotations": []}

colors = {}
for i in range(num_classes):
    random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    colors[i] = random_color


def is_inst_visible(inst, occ_aware_seg_map, clear_seg_map, thresh):
    """
    Args:
        inst: Instance ID of object
        occ_aware_seg_map: Original segmentation map
        clear_seg_map: Segmentation map with fully visible objects
        thresh: portion of obj that needs to be visible

    Returns:
        True if obj is considered visible to human
    """

    visibility = np.count_nonzero(occ_aware_seg_map == inst) / np.count_nonzero(clear_seg_map == inst)
    if visibility >= thresh:
        return True
    else:
        return False


def remove_small_components(inst, occ_aware_seg_map, occ_ignore_seg_map, thresh):
    """
    Remove components of objects that aren't visible to human

    Args:
        inst: Instance ID of object
        occ_aware_seg_map: Original segmentation map
        occ_ignore_seg_map: Segmentation map with obstacles removed
        thresh: portion of obj that needs to be visible
    """

    inst_size = np.count_nonzero(occ_ignore_seg_map == inst)
    seg_map_copy = np.copy(occ_aware_seg_map)
    seg_map_copy[occ_aware_seg_map != inst] = 0
    seg_map_copy[occ_aware_seg_map == inst] = 1
    seg_map_copy = seg_map_copy.astype('uint8')

    num_components, components = cv2.connectedComponents(seg_map_copy)

    for i in range(1, num_components):
        component_size = np.count_nonzero(components == i)
        if component_size / inst_size < thresh:
            occ_aware_seg_map[components == i] = 0 # delete component


def binary_mask_to_rle(binary_mask):
    """
    Convert binary mask to run length encoding.
    """

    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Args:
        x1: Line 1 x coordinate of start point
        y1: Line 1 y coordinate of start point
        x2: Line 1 x coordinate of end point
        y2: Line 1 y coordinate of end point
        x3: Line 2 x coordinate of start point
        y3: Line 2 y coordinate of start point
        x4: Line 2 x coordinate of end point
        y4: Line 2 y coordinate of end point

    Returns:
        True if lines intersect
    """
    def ccw(x1, y1, x2, y2, x3, y3):
        return (y3-y1) * (x2-x1) > (y2-y1) * (x3-x1)

    return ccw(x1, y1 ,x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and \
        ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4)


def is_inst_on_edge(x_bb, y_bb, w, h, img):
    """

    Args:
        x_bb: Bounding box top left x coordinate
        y_bb: Bounding box top left y coordinate
        w: Bounding box width
        h: Bounding box height
        img: np array of image

    Returns:
        True if bounding box lies on edge of image
    """
    return x_bb == 0 or y_bb == 0 or x_bb+w == img.shape[1] or y_bb+h == img.shape[0]


for i in range(num_classes):
    cat_info = {
                "id":i,
                "name":classes[i],
                }
    coco_ann["categories"].append(cat_info)


for img_id, img_filename in enumerate(os.listdir(img_path), start=1):

    overlapping = set() # instances which overlap

    bb_ann = {"cat_id": [], "xc": [], "yc": [], "w": [], "h": [], 
        "obb1x": [], "obb1y": [], "obb2x": [], "obb2y": [], 
        "obb3x": [], "obb3y": [], "obb4x": [], "obb4y": []}

    occ_aware_seg_map = cv2.imread(os.path.join(occ_aware_seg_path, img_filename), -1)
    occ_ignore_seg_map = cv2.imread(os.path.join(occ_ignore_seg_path, img_filename), -1)
    zoomed_out_seg_map = cv2.imread(os.path.join(zoomed_out_seg_path, img_filename), -1)
    img = cv2.imread(os.path.join(img_path, img_filename))

    img_h, img_w = occ_aware_seg_map.shape[:2]

    img_info = {
                "id":img_id,
                "file_name":img_filename,
                "height":img_h,
                "width":img_w,
                }
    coco_ann["images"].append(img_info)
    img_ann_info = []

    instances = np.unique(occ_aware_seg_map)
    instances = instances[instances != 0]

    if view_annotations:
        img_bb, img_obb, img_seg = img.copy(), img.copy(), img.copy()

    detections = []
    for inst in instances:
        cat_id = int(inst // 1000)

        if not is_inst_visible(inst, occ_aware_seg_map, occ_ignore_seg_map, visibility_thresh):
            continue

        if occlusion_aware:
            remove_small_components(inst, occ_aware_seg_map, occ_ignore_seg_map, component_visibility_thresh)
            seg_map = occ_aware_seg_map
        else:
            seg_map = occ_ignore_seg_map
        
        points = cv2.findNonZero((seg_map == inst).astype(int))
        if points is None:
            continue
        x_bb, y_bb, w, h = cv2.boundingRect(points)
        obb = cv2.minAreaRect(points)
        obb_points = cv2.boxPoints(obb).astype(int) # 4 x 2 --> 4 points for bounding box

        if occlusion_aware and is_inst_on_edge(x_bb, y_bb, w, h, img):
            if not is_inst_visible(inst, occ_aware_seg_map, zoomed_out_seg_map, visibility_thresh):
                continue
        elif not occlusion_aware and is_inst_on_edge(x_bb, y_bb, w, h, img):
            img = cv2.fillPoly(img, [obb_points], color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            continue

 

        for i in range(obb_points.shape[0]):
            # iterate through line segments
            x1, y1, x2, y2 = obb_points[i][0], obb_points[i][1], \
                                obb_points[(i+1) % obb_points.shape[0]][0], \
                                obb_points[(i+1) % obb_points.shape[0]][1]
            
            for bb_ind in range(len(bb_ann["xc"])): # number of boxes in img so far
                obb_points2 = np.array([[bb_ann["obb1x"][bb_ind], bb_ann["obb1y"][bb_ind]], 
                                        [bb_ann["obb2x"][bb_ind], bb_ann["obb2y"][bb_ind]], 
                                        [bb_ann["obb3x"][bb_ind], bb_ann["obb3y"][bb_ind]], 
                                        [bb_ann["obb4x"][bb_ind], bb_ann["obb4y"][bb_ind]]])

                
                w2, h2 = round(bb_ann["w"][bb_ind]*img_w), round(bb_ann["h"][bb_ind]*img_h)
                x_bb2, y_bb2 = round((bb_ann["xc"][bb_ind] * img_w) - w2/2), round((bb_ann["yc"][bb_ind] * img_h) - h2/2)


                for k in range(obb_points2.shape[0]):
                    # iterate through second set of line segments
                    x3, y3, x4, y4 = obb_points2[k][0], obb_points2[k][1], \
                                obb_points2[(k+1) % obb_points2.shape[0]][0], \
                                obb_points2[(k+1) % obb_points2.shape[0]][1]

                    
                    if intersect(x1, y1, x2, y2, x3, y3, x4, y4):
                        overlapping.add(bb_ind)
                        overlapping.add(len(bb_ann["xc"]))

                        if view_annotations:
                            img_obb = cv2.polylines(img_obb, [obb_points], isClosed=True, color=(0,0,255), thickness=3)
                            img_obb = cv2.polylines(img_obb, [obb_points2], isClosed=True, color=(0,0,255), thickness=3)
                            img_bb = cv2.rectangle(img_bb, (x_bb, y_bb), (x_bb+w, y_bb+h), color=(0,0,255), thickness=3)
                            img_bb = cv2.rectangle(img_bb, (x_bb2, y_bb2), (x_bb2+w2, y_bb2+h2), color=(0,0,255), thickness=3)
                        img = cv2.fillPoly(img, [obb_points], color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                        img = cv2.fillPoly(img, [obb_points2], color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))




        if view_annotations and len(bb_ann["xc"]) not in overlapping: # if current instance isn't overlapping
            img_bb = cv2.rectangle(img_bb, (x_bb, y_bb), (x_bb+w, y_bb+h), color=colors[cat_id], thickness=2)
            img_obb = cv2.polylines(img_obb, [obb_points], isClosed=True, color=colors[cat_id], thickness=2)

        bb_ann["cat_id"].append(cat_id) 
        bb_ann["xc"].append((x_bb + w/2) / img_w)
        bb_ann["yc"].append((y_bb + h/2) / img_h)
        bb_ann["w"].append(w / img_w)
        bb_ann["h"].append(h / img_h)

        bb_ann["obb1x"].append(obb_points[0][0])
        bb_ann["obb1y"].append(obb_points[0][1])
        bb_ann["obb2x"].append(obb_points[1][0])
        bb_ann["obb2y"].append(obb_points[1][1])
        bb_ann["obb3x"].append(obb_points[2][0])
        bb_ann["obb3y"].append(obb_points[2][1])
        bb_ann["obb4x"].append(obb_points[3][0])
        bb_ann["obb4y"].append(obb_points[3][1])

        ann_info = {
                    "id": random.randrange(1,10000), # TODO: find robust way for this
                    "image_id":img_id,
                    "category_id": cat_id,
                    "bbox":[
                        x_bb,
                        y_bb,
                        w,
                        h
                    ],
                    "segmentation": binary_mask_to_rle((seg_map == inst).astype('uint8')),
                    "area": w*h,
                    "iscrowd": 0
                    }

        img_ann_info.append(ann_info)



    df = pd.DataFrame.from_dict(bb_ann)
    df = df.drop(list(overlapping)) # get rid of overlapping labels
    img_ann_info = [i for j, i in enumerate(img_ann_info) if j not in overlapping]
    yolo_col = ["cat_id", "xc", "yc", "w", "h"]
    np.savetxt(os.path.join(yolo_labels_path, img_filename[:-4] + ".txt"), df[yolo_col], delimiter=' ', fmt=['%d', '%.4f', '%.4f', '%.4f', '%.4f'])

    obb_col = ["cat_id", "obb1x", "obb1y", "obb2x", "obb2y", "obb3x", "obb3y", "obb4x", "obb4y"]
    np.savetxt(os.path.join(obb_labels_path, img_filename[:-4] + ".txt"), df[obb_col], delimiter=' ', fmt=['%d', '%d', '%d', '%d', '%d', '%d', '%d', '%d', '%d'])

    coco_ann["annotations"].extend(img_ann_info)
    
    cv2.imwrite(os.path.join(img_path, img_filename), img) # overwrite overlapping regions in image

    if view_annotations:
        cv2.imwrite(os.path.join(yolo_annotated_path, img_filename), img_bb)
        cv2.imwrite(os.path.join(obb_annotated_path, img_filename), img_obb)


with open(os.path.join(results_dir, "coco_annotations.json"), "w") as f:
    json.dump(coco_ann, f)