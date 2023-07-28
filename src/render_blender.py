import bpy
from bpy import context, ops

import bpycv
import cv2

import math
import numpy as np
import random

import os
import yaml
from glob import glob


def create_plane(plane_size=500, scenes_list=None):
    """
    Create surface to place objects on

    Args:
        plane_size: Length of plane side
        scenes_list: Directories containing custom textures
    """

    scene = random.choice(scenes_list) if scenes_list else None

    subdivide_count = 100
    ops.mesh.primitive_plane_add(size=plane_size, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    ops.object.editmode_toggle()
    ops.mesh.subdivide(number_cuts=subdivide_count)
    ops.object.editmode_toggle()

    if scene:
        generate_texture(scene)
    else:
        generate_random_background()

   
def generate_texture(texture_path):
    """
    Create blender nodes for imported texture
    """

    img_tex = glob(os.path.join(texture_path, "*_diff_*"))[0]
    img_rough = glob(os.path.join(texture_path, "*_rough_*"))[0]
    img_norm = glob(os.path.join(texture_path, "*_nor_gl_*"))[0]
    img_dis = glob(os.path.join(texture_path, "*_disp_*"))[0]

    material_basic = bpy.data.materials.new(name="Basic")
    material_basic.use_nodes = True
    context.object.active_material = material_basic
    nodes = material_basic.node_tree.nodes

    principled_node = nodes.get("Principled BSDF")
    node_out = nodes.get("Material Output")

    node_tex = nodes.new('ShaderNodeTexImage')
    node_tex.image = bpy.data.images.load(img_tex)
    node_tex.location = (-700, 800)

    node_rough = nodes.new('ShaderNodeTexImage')
    node_rough.image = bpy.data.images.load(img_rough)
    node_rough.location = (-700, 500)

    node_norm = nodes.new('ShaderNodeTexImage')
    node_norm.image = bpy.data.images.load(img_norm)
    node_norm.location = (-700, 200)

    node_dis = nodes.new('ShaderNodeTexImage')
    node_dis.image = bpy.data.images.load(img_dis)
    node_dis.location = (-700, -100)

    norm_map = nodes.new('ShaderNodeNormalMap')
    norm_map.location = (-250, 0)

    node_disp = nodes.new('ShaderNodeDisplacement')
    node_disp.location = (0, -450)

    node_tex_coor = nodes.new('ShaderNodeTexCoord')
    node_tex_coor.location = (-1400, 500)

    node_map = nodes.new('ShaderNodeMapping')
    node_map.location = (-1200, 500)

    link = material_basic.node_tree.links.new
    link(node_tex.outputs["Color"], principled_node.inputs["Base Color"])
    link(node_rough.outputs["Color"], principled_node.inputs["Roughness"])
    link(node_norm.outputs["Color"], norm_map.inputs["Color"])
    link(node_dis.outputs["Color"], node_disp.inputs["Height"])

    link(node_tex_coor.outputs["UV"], node_map.inputs["Vector"])
    link(node_map.outputs["Vector"], node_tex.inputs["Vector"])
    link(node_map.outputs["Vector"], node_rough.inputs["Vector"])
    link(node_map.outputs["Vector"], node_norm.inputs["Vector"])
    link(node_map.outputs["Vector"], node_dis.inputs["Vector"])

    link(norm_map.outputs["Normal"], principled_node.inputs["Normal"])
    link(node_disp.outputs["Displacement"], node_out.inputs["Displacement"])


def generate_random_background():
    """
    Create blender nodes for random colour pattern
    """

    material_basic = bpy.data.materials.new(name="Basic")
    material_basic.use_nodes = True
    context.object.active_material = material_basic
    nodes = material_basic.node_tree.nodes

    principled_node = nodes.get("Principled BSDF")
    colorramp_node = nodes.new("ShaderNodeValToRGB")
    voronoi_node = nodes.new("ShaderNodeTexVoronoi")

    voronoi_node.location = (-500, 0)
    colorramp_node.location = (-280,0)

    dimensions = ['2D', '3D']
    features = ['F1', 'F2', 'SMOOTH_F1']
    distances = ['EUCLIDEAN', 'MANHATTAN', 'CHEBYCHEV', 'MINKOWSKI']

    voronoi_node.voronoi_dimensions = random.choice(dimensions)
    voronoi_node.distance = random.choice(distances)
    voronoi_node.feature = random.choice(features)
    voronoi_node.inputs[2].default_value = random.uniform(2, 10) # scale


    link = material_basic.node_tree.links.new
    link(colorramp_node.outputs[0], principled_node.inputs[0])
    voronoi_output = random.randint(0,2)
    link(voronoi_node.outputs[voronoi_output], colorramp_node.inputs[0])

    num_elements = random.randint(5, 15)

    for i in range(num_elements - 2):
        colorramp_node.color_ramp.elements.new(0.1 * (i+1))

    for i in range(num_elements):     
        colorramp_node.color_ramp.elements[i].position = i * (1 / (num_elements-1))
        colorramp_node.color_ramp.elements[i].color = (random.random(), random.random(), random.random(),1)

def add_sky():
    """
    Create sky background
    """

    sky_texture = bpy.context.scene.world.node_tree.nodes.new("ShaderNodeTexSky")
    bg = bpy.context.scene.world.node_tree.nodes["Background"]
    bpy.context.scene.world.node_tree.links.new(bg.inputs["Color"], sky_texture.outputs["Color"])

    sky_texture.sky_type = 'HOSEK_WILKIE' # or 'PREETHAM'
    sky_texture.sun_intensity = 0.0

def add_sun(min_sun_energy, max_sun_energy, max_sun_tilt):
    """
    Create light source with random intensity and ray angles

    Args:
        min_sun_energy: Minimum power of sun
        max_sun_energy: Maximum power of sun
        max_sun_tilt: Maximum angle of sun's rays
    """

    ops.object.light_add(type='SUN', radius=10, align='WORLD', location=(0,0,0), scale=(10, 10, 1))
    context.scene.objects["Sun"].data.energy = random.randrange(min_sun_energy, max_sun_energy)
    context.scene.objects["Sun"].rotation_euler[0] = random.uniform(0, math.radians(max_sun_tilt))
    context.scene.objects["Sun"].rotation_euler[1] = random.uniform(0, math.radians(max_sun_tilt))
    context.scene.objects["Sun"].rotation_euler[2] = random.uniform(0, 2*math.pi)
    
    
def add_camera(min_camera_height, max_camera_height, max_camera_tilt):
    """
    Create camera with random height and viewing angles

    Args:
        min_camera_height: Minimum height of camera
        max_camera_height: Maximum height of camera
        max_camera_tilt: Maximum viewing angle
    """

    z = random.randrange(min_camera_height, max_camera_height)
    ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0,0,z), rotation=(0, 0, 0), scale=(1, 1, 1))
    context.scene.camera = context.object

    ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))

    ops.object.select_all(action='DESELECT')
    context.scene.objects["Camera"].select_set(True)
    context.scene.objects["Empty"].select_set(True)
    ops.object.parent_set(type='OBJECT', keep_transform=False)
    ops.object.select_all(action='DESELECT')

    context.scene.objects["Empty"].rotation_euler[0] = random.uniform(0, math.radians(max_camera_tilt))
    context.scene.objects["Empty"].rotation_euler[2] = random.uniform(0, 2*math.pi)
    

def print_inputs():
    pass

def import_from_path(class_path, class_name=None):
    """
    Import 3D models into scene given directory

    Args:
        class_path: Directory containing 3D objects
        class_name: Object class. Defaults to None if object type is irrelevant.
    """

    for filename in os.listdir(class_path):
        filepath = os.path.join(class_path, filename)
        obj_name = os.path.splitext(filepath)[0].split("/")[-1]
        ext = os.path.splitext(filepath)[1]

        if ext == ".fbx":
            ops.import_scene.fbx(filepath=filepath)
        elif ext == ".obj":
            ops.import_scene.obj(filepath=filepath)
        elif ext == ".blend":
            blender_path = os.path.join("Object", obj_name)
            ops.wm.append(
                filepath=os.path.join(filepath, blender_path),
                directory=os.path.join(filepath, "Object"),
                filename=obj_name)
        else:
            continue
        
        if class_name:
            parent_class[obj_name] = class_name

        ops.object.select_all(action='DESELECT') # May be redundant
        
        object = bpy.data.objects[obj_name]
        object.hide_render = True
        object.rotation_euler.y += math.radians(90)
        
        for coll in object.users_collection:
            coll.objects.unlink(object)
        context.scene.collection.children.get("Models").objects.link(object)


def import_objects():
    """
    Import all objects into scene
    """

    if obstacles_path: import_from_path(obstacles_path)
    for i, class_path in enumerate(classes_list):
        class_name = os.path.basename(os.path.normpath(class_path))
        objects_dict[class_name] = [os.path.splitext(os.path.normpath(obj))[0] for obj in os.listdir(class_path)]
        class_ids[class_name] = i
        import_from_path(class_path, class_name)


def delete_objects():
    """
    Delete all objects from scene
    """
    ops.object.select_all(action='SELECT')
    ops.object.delete()

def configure_gpu():
    """
    Use GPU if available
    """
    context.scene.render.engine = 'CYCLES'
    context.scene.cycles.samples = 200
    context.scene.cycles.device = 'GPU' if context.preferences.addons["cycles"].preferences.has_active_device() else 'CPU'
    print(f"Using {context.scene.cycles.device}")


def create_collections():
    """
    Create blender collections for all 3D models
    """
    collection = bpy.data.collections.new("Models") # not rendered
    context.scene.collection.children.link(collection)
    collection2 = bpy.data.collections.new("Instances")
    context.scene.collection.children.link(collection2)
    collection3 = bpy.data.collections.new("Obstacles")
    context.scene.collection.children.link(collection3)


def get_cat_id(obj):
    """
    Args:
        obj: Blender object

    Returns:
        Class ID of object
    """

    return class_ids[parent_class[obj.name.split('.')[0]]]

def is_target(obj):
    """
    Args:
        obj: Blender object

    Returns:
        True if object's class is to be annotated
    """

    return obj.name.split('.')[0] in parent_class

def is_obstacle(obj):
    """
    Args:
        obj: Blender object

    Returns:
        True if object is an obstacle
    """

    try:
        return obj.name.split('.')[0] in obstacles_list
    except TypeError as e:
        print("No objects and obstacles found. Unable to generate scene.")
        raise TypeError

def hair_emission(min_obj_count, max_obj_count, scale=1):
    """
    Emit 3D models from plane

    Args:
        min_obj_count: Minimum number of objects in scene
        max_obj_count: Maximum number of objects in scene

    Raises:
        Exception: When emitted object is neither a target nor an obstacle
    """

    objects = bpy.data.objects
    plane = objects["Plane"] 

    context.view_layer.objects.active = plane
    ops.object.particle_system_add()
    
    particle_count = random.randrange(min_obj_count, max_obj_count)
    particle_scale = scale

    ps = plane.modifiers.new("part", 'PARTICLE_SYSTEM')
    psys = plane.particle_systems[ps.name]

    psys.settings.type = "HAIR"
    psys.settings.use_advanced_hair = True

    # EMISSION
    seed = random.randrange(10000)
    psys.settings.count = particle_count
    psys.settings.hair_length = particle_scale
    psys.seed = seed

    # #RENDER
    psys.settings.render_type = "COLLECTION"
    plane.show_instancer_for_render = True
    psys.settings.instance_collection = bpy.data.collections["Models"]
    psys.settings.particle_size = particle_scale
    
    psys.settings.use_scale_instance = True
    psys.settings.use_rotation_instance = True
    psys.settings.use_global_instance = True
    
    # # ROTATION
    psys.settings.use_rotations = True
    psys.settings.rotation_mode = "NOR" # "GLOB_Z"
    psys.settings.phase_factor_random = 2.0 # change to random num (0 to 2.0)
    psys.settings.child_type = "NONE"
        
    plane.select_set(True)
    ops.object.duplicates_make_real()
    plane.modifiers.remove(ps)
    
    objs = context.selected_objects
    coll_target = context.scene.collection.children.get("Instances")
    coll_obstacles = context.scene.collection.children.get("Obstacles")
    for i, obj in enumerate(objs):
        for coll in obj.users_collection:
            coll.objects.unlink(obj)
        
        obj_copy = obj
        obj_copy.data = obj.data.copy()
        obj_copy.hide_render = False

        if is_target(obj_copy):
            coll_target.objects.link(obj_copy)
            inst_id = get_cat_id(obj_copy) * 1000 + i + 1 # cannot have inst_id = 0
            obj_copy["inst_id"] = inst_id # for bpycv
        elif is_obstacle(obj_copy):
            coll_obstacles.objects.link(obj_copy)
        else:
            raise Exception(obj_copy.name, "is neither an obstacle nor a target")


def blender_setup():
    """
    Initial blender setup
    """
    delete_objects()
    create_collections()
    configure_gpu()


def render(render_path, render_name="synthetics.png"):
    """
    Render scene

    Args:
        render_path: Directory to save render to
        render_name: Filename of render to be saved
    """
    img_path = os.path.join(render_path, "img")
    occ_aware_seg_path = os.path.join(render_path, "seg_maps")
    occ_ignore_seg_path = os.path.join(render_path, "other_seg_maps")
    zoomed_out_seg_path = os.path.join(render_path, "zoomed_out_seg_maps")
    
    if not os.path.isdir(render_path):
        os.mkdir(render_path)
    if not os.path.isdir(img_path):
        os.mkdir(img_path)
    if not os.path.isdir(occ_aware_seg_path):
        os.mkdir(occ_aware_seg_path)
    if not os.path.isdir(occ_ignore_seg_path):
        os.mkdir(occ_ignore_seg_path)
    if not os.path.isdir(zoomed_out_seg_path):
        os.mkdir(zoomed_out_seg_path)

    result = bpycv.render_data()
    for obj in bpy.data.collections['Obstacles'].all_objects:
        obj.hide_render = True
    hidden_obstacles_result = bpycv.render_data(render_image=False)
    bpy.data.objects["Empty"].scale = (1.05, 1.05, 1.05)
    zoomed_out_result = bpycv.render_data(render_image=False)
    bpy.data.objects["Empty"].scale = (1, 1, 1)

    cv2.imwrite(os.path.join(img_path, render_name), result["image"][..., ::-1])
    cv2.imwrite(os.path.join(occ_aware_seg_path, render_name), np.uint16(result["inst"]))
    cv2.imwrite(os.path.join(occ_ignore_seg_path, render_name), np.uint16(hidden_obstacles_result["inst"]))
    cv2.imwrite(os.path.join(zoomed_out_seg_path, render_name), np.uint16(zoomed_out_result["inst"]))


if __name__ == "__main__":

    with open("./config/models.yaml") as file:
        models_info = yaml.load(file, Loader=yaml.FullLoader)
    with open("./config/render_parameters.yaml") as file:
        config_info = yaml.load(file, Loader=yaml.FullLoader)

    for key, value in models_info.items():
        print(f"{key}: {value}")

    for key, value in config_info.items():
        print(f"{key}: {value}")

    classes_list = models_info["classes"]
    scenes_list = [os.path.join(models_info["scenes"], s) for s in os.listdir(models_info["scenes"])] if "scenes" in models_info else None
    obstacles_path = models_info["obstacles_path"] if "obstacles_path" in models_info else None
    obstacles_list =  [os.path.splitext(os.path.normpath(obj))[0] for obj in os.listdir(obstacles_path)] if obstacles_path else None
    render_path = models_info["render_to"]
    min_camera_height = config_info["min_camera_height"]
    max_camera_height = config_info["max_camera_height"]
    max_camera_tilt = config_info["max_camera_tilt"]
    min_sun_energy = config_info["min_sun_energy"]
    max_sun_energy = config_info["max_sun_energy"]
    max_sun_tilt = config_info["max_sun_tilt"]
    num_img = config_info["num_img"]
    plane_size = config_info["plane_size"]
    min_obj_count = config_info["min_obj_count"]
    max_obj_count = config_info["max_obj_count"]
    create_sky = config_info["create_sky"]

    objects_dict = {} # objects_dict[class_name] = objects_names_list
    class_ids = {} # class_ids[class_name] = i
    parent_class = {} # parent_class[obj_name] = class_name

    print_inputs()
    blender_setup()

    for i in range(num_img):
        render_name = f"synthetics{i}.png"
        delete_objects()
        import_objects()

        print("---------------------------------------")
        print("Objects imported")
        print("---------------------------------------")
        
        create_plane(plane_size, scenes_list=scenes_list)
        if create_sky: add_sky()
        add_sun(min_sun_energy, max_sun_energy, max_sun_tilt)

        add_camera(min_camera_height, max_camera_height, max_camera_tilt)
        hair_emission(min_obj_count, max_obj_count)
        render(render_path, render_name)

        print("---------------------------------------")
        print(f"Image {i+1} of {num_img} complete")
        print("---------------------------------------")
    