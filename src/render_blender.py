import bpy
from bpy import context, ops

import bpycv
import cv2

import math
import numpy as np
import random

import yaml
from glob import glob
import time

from pathlib import Path

IMG_SUFFIXES = [".png", ".jpg", ".jpeg"]

def calculate_plane_size(camera_height, camera_tilt, camera_fov, aspect_ratio=16/9):
    """
    Calculate the minimum plane size needed to capture the whole plane in the camera view.

    Args:
        camera_height (float): The height of the camera above the plane.
        camera_tilt (float): The tilt angle of the camera in degrees.
        camera_fov (float): The field of view of the camera in degrees.
        aspect_ratio (float): The aspect ratio of the camera (width/height). Default is 16/9.

    Returns:
        (float, float): The width and height of the plane.
    """
    # Convert camera tilt and FOV from degrees to radians
    tilt_radians = math.radians(camera_tilt)
    fov_radians = math.radians(camera_fov)

    # Calculate the distance from the camera to the plane (along the line of sight)
    distance_to_plane = camera_height / math.cos(tilt_radians)

    # Calculate the viewable width and height on the plane
    viewable_height = 2 * (distance_to_plane * math.tan(fov_radians / 2))
    viewable_width = viewable_height * aspect_ratio

    return max(viewable_width, viewable_height)

def create_plane(plane_size=250, scenes_list=None): # Change to 750
    """
    Create surface to place objects on

    Args:
        plane_size: Length of plane side
        scenes_list: Directories containing custom textures

    Returns:
        Type of background (str), Name of scene (str)
    """

    attempt_count = 10
    if scenes_list:
        for _ in range(attempt_count):
            scene = random.choice(scenes_list)
            # Image as plane or as texture
            if Path(scene).suffix in IMG_SUFFIXES and import_image_as_plane(scene, plane_size):
                return "Image", Path(scene).stem
            
            else:
                print(f"Using plane size of {plane_size}")
                subdivide_count = 100
                ops.mesh.primitive_plane_add(size=plane_size, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
                ops.object.editmode_toggle()
                ops.mesh.subdivide(number_cuts=subdivide_count)
                ops.object.editmode_toggle()
                if generate_texture(Path(scene)):
                    return "Plane", Path(scene).stem

            print(f"WARNING: {scene} invalid. Unable to generate texture. Trying again...")
        else:
            print(f"Unable to find a suitable scene within {attempt_count} attempts")

    # Generate colormap
    generate_random_background()
    return "Colormap", "colormap"

def import_image_as_plane(image_path, plane_size):
    addon_name = "io_import_images_as_planes"  # Replace with the name of the add-on you want to enable

    if addon_name not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module=addon_name)
        print(f"The add-on '{addon_name}' has been enabled.")
    else:
        print(f"The add-on '{addon_name}' is already enabled.")

    if Path(image_path).exists():
        bpy.ops.import_image.to_plane(
            files=[{'name':str(image_path)}],
            location=(0, 0, 0),
            align='WORLD', 
            align_axis='Z+',
            size_mode='ABSOLUTE',
            height=plane_size,
            blend_method='OPAQUE',
            extension='REPEAT',
            shader='PRINCIPLED'
            )
        bpy.context.view_layer.update()
        bpy.context.view_layer.objects.active = None

    else:
        print("Image path doesn't exist")
        return False
    
    return True
   
def generate_texture(texture_path):
    """
    Create blender nodes for imported texture
    
    Returns:
        True if texture has been succesfully generated, False otherwise
    """

    texture_files = list(texture_path.glob("**/*"))
    img_tex = next((file for file in texture_files if "_diff_" in file.name), None)
    img_rough = next((file for file in texture_files if "_rough_" in file.name), None)
    img_norm = next((file for file in texture_files if "_nor_gl_" in file.name), None)
    img_dis = next((file for file in texture_files if "_disp_" in file.name), None)

    # Check if all four texture files were found
    if all([img_tex, img_rough, img_norm, img_dis]):
        img_tex = str(img_tex)  # Convert Path object to string if needed
        img_rough = str(img_rough)
        img_norm = str(img_norm)
        img_dis = str(img_dis)
    else:
        print(f"One or more texture files not found for texture path {str(texture_path)}.")
        return False

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

    return True


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
    
    
def add_camera(camera_height, camera_tilt, camera_horizontal_fov):
    """
    Create camera with random height and viewing angles

    Args:
        camera_height_range: Tuple consisting of the minimum and maximum height of camera
        camera_tilt_range: Tuple consisting of minimum and maximum viewing angle
    
    Returns:
        Details of generated camera (str)
    """
    
    # min_camera_tilt, max_camera_tilt = camera_tilt_range
    # min_camera_height, max_camera_height = camera_height_range

    # z = random.randrange(min_camera_height, max_camera_height)
    x_loc = random.randint(-15, 15) # TODO: Values currently hard-coded
    y_loc = random.randint(-15, 15) # TODO: Values currently hard-coded
    ops.object.camera_add(enter_editmode=False, align='VIEW', location=(x_loc,y_loc,camera_height), rotation=(0, 0, 0), scale=(1, 1, 1))
    context.scene.camera = context.object

    # TODO: HERE
    # Set clipping distances
    camera = context.scene.camera.data
    camera.clip_start = camera_height / 2.0
    camera.clip_end = camera_height * 1000

    camera.angle = math.radians(camera_horizontal_fov)

    ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))

    ops.object.select_all(action='DESELECT')
    context.scene.objects["Camera"].select_set(True)
    context.scene.objects["Empty"].select_set(True)
    ops.object.parent_set(type='OBJECT', keep_transform=False)
    ops.object.select_all(action='DESELECT')

    # camera_tilt = random.randint(min_camera_tilt, max_camera_tilt)
    context.scene.objects["Empty"].rotation_euler[0] = math.radians(camera_tilt)
    context.scene.objects["Empty"].rotation_euler[2] = random.uniform(0, 2*math.pi)
    
    return f"{camera_height}m_{camera_tilt}deg"
    

def print_inputs():
    pass

def import_from_path(class_path, class_name=None):
    """
    Import 3D models into scene given directory

    Args:
        class_path: Directory containing 3D objects
        class_name: Object class. Defaults to None if object type is irrelevant.
    """

    for filename in Path(class_path).iterdir():
        filepath = class_path / filename
        obj_name = filepath.stem
        ext = filepath.suffix

        if ext == ".fbx":
            ops.import_scene.fbx(filepath=str(filepath))
        elif ext == ".obj":
            ops.import_scene.obj(filepath=str(filepath))
        elif ext == ".blend":
            blender_path = "Object" / obj_name
            ops.wm.append(
                filepath=str(filepath / blender_path),
                directory=str(filepath / "Object"),
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
        class_name = str(Path(class_path).resolve().name)
        objects_dict[class_name] = [obj.stem for obj in Path(class_path).iterdir() if obj.is_file()]
        class_ids[class_name] = i
        import_from_path(class_path, class_name)


def delete_objects():
    """
    Delete all objects from scene
    """
    ops.object.select_all(action='SELECT')
    ops.object.delete()

    ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

    bpy.data.orphans_purge()

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

    # RENDER
    psys.settings.render_type = "COLLECTION"
    plane.show_instancer_for_render = True
    psys.settings.instance_collection = bpy.data.collections["Models"]
    psys.settings.particle_size = particle_scale
    
    psys.settings.use_scale_instance = True
    psys.settings.use_rotation_instance = True
    psys.settings.use_global_instance = True
    
    # ROTATION
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
    # Define subdirectories
    img_path = render_path / "img"
    occ_aware_seg_path = render_path / "seg_maps"
    occ_ignore_seg_path = render_path / "other_seg_maps"
    zoomed_out_seg_path = render_path / "zoomed_out_seg_maps"

    # Create directories if they don't exist
    render_path.mkdir(parents=True, exist_ok=True)
    img_path.mkdir(parents=True, exist_ok=True)
    occ_aware_seg_path.mkdir(parents=True, exist_ok=True)
    occ_ignore_seg_path.mkdir(parents=True, exist_ok=True)
    zoomed_out_seg_path.mkdir(parents=True, exist_ok=True)

    result = bpycv.render_data()
    for obj in bpy.data.collections['Obstacles'].all_objects:
        obj.hide_render = True
    hidden_obstacles_result = bpycv.render_data(render_image=False)
    bpy.data.objects["Empty"].scale = (1.05, 1.05, 1.05)
    zoomed_out_result = bpycv.render_data(render_image=False)
    bpy.data.objects["Empty"].scale = (1, 1, 1)

    # Write the images using cv2
    print(f"Writing image to {str(img_path / render_name)}")
    cv2.imwrite(str(img_path / render_name), result["image"][..., ::-1])
    cv2.imwrite(str(occ_aware_seg_path / render_name), np.uint16(result["inst"]))
    cv2.imwrite(str(occ_ignore_seg_path / render_name), np.uint16(hidden_obstacles_result["inst"]))
    cv2.imwrite(str(zoomed_out_seg_path / render_name), np.uint16(zoomed_out_result["inst"]))


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

    if "scenes" in models_info:
        scenes_path = Path(models_info["scenes"])
        scenes_list = [
            str(scene_path) 
            for scene_path in scenes_path.iterdir() 
            if scene_path.is_dir() or (scene_path.is_file() and scene_path.suffix in IMG_SUFFIXES)
        ]
    else:
        scenes_list = None

    obstacles_path = models_info["obstacles_path"] if "obstacles_path" in models_info else ""
    obstacles_list = [obj.stem for obj in Path(obstacles_path).glob("*") if obj.is_file()]
    render_path = models_info["render_to"]
    min_camera_height = config_info["min_camera_height"]
    max_camera_height = config_info["max_camera_height"]
    min_camera_tilt = config_info["min_camera_tilt"]
    max_camera_tilt = config_info["max_camera_tilt"]
    min_sun_energy = config_info["min_sun_energy"]
    max_sun_energy = config_info["max_sun_energy"]
    max_sun_tilt = config_info["max_sun_tilt"]
    num_img = config_info["num_img"]
    camera_horizontal_fov = config_info["camera_horizontal_fov"]
    plane_size = config_info["plane_size"]
    min_obj_count = config_info["min_obj_count"]
    max_obj_count = config_info["max_obj_count"]
    create_sky = config_info["create_sky"]

    objects_dict = {} # objects_dict[class_name] = objects_names_list
    class_ids = {} # class_ids[class_name] = i
    parent_class = {} # parent_class[obj_name] = class_name

    print_inputs()
    blender_setup()

    start_time = time.time()
    for i in range(num_img):
        delete_objects()
        import_objects()

        print("---------------------------------------")
        print("Objects imported")
        print("---------------------------------------")

        # Randomizing parameters
        camera_height = random.randrange(min_camera_height, max_camera_height)
        camera_tilt = random.randint(min_camera_tilt, max_camera_tilt)
        
        # plane_size = calculate_plane_size(camera_height, camera_tilt, camera_horizontal_fov)
        scene_type, scene_name = create_plane(plane_size, scenes_list=scenes_list)
        if create_sky: add_sky()        
        add_sun(min_sun_energy, max_sun_energy, max_sun_tilt)

        camera_details = add_camera(camera_height, camera_tilt, camera_horizontal_fov)
        if scene_type == "Image":
            hair_emission(min_obj_count, max_obj_count, scene_name)
        elif scene_type == "Colormap" or scene_type == "Plane":
            hair_emission(min_obj_count, max_obj_count, "Plane")
        else:
            raise Exception(f"Scene type '{scene_type}' doesn't fit any of the acceptable types: Image, Plane or Colormap")

        # idx_scene-name_cam-height_cam-tilt
        render_name = f"{i}_{scene_name}_{camera_details}.png"
        render(Path(render_path), render_name)

        print("---------------------------------------")
        print(f"Image {i+1} of {num_img} complete")
        print("---------------------------------------")

    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time taken for {num_img} images: {int(hours)}h{int(minutes)}m{seconds:.2f}s")
    print(f"Avg time taken for each image: {total_time / num_img:.2f}s")
    