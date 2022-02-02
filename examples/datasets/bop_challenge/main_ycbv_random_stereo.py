import blenderproc as bproc
import bpy
import argparse
import os
from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.utility import CollisionUtility
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', help="Path to the bop datasets parent directory")
parser.add_argument('cc_textures_path', default="resources/cctextures", help="Path to downloaded cc textures")
parser.add_argument('output_dir', help="Path to where the final files will be saved ")
parser.add_argument('--num_scenes', type=int, default=2000, help="How many scenes with 25 images each to generate")
args = parser.parse_args()

bproc.init()

# load bop objects into the scene
target_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'ycbv'), mm2m = True, model_type='fine')

# load distractor bop objects
tless_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'tless'), model_type = 'cad', mm2m = True)
hb_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'hb'), mm2m = True)
tyol_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'tyol'), mm2m = True)

# load BOP datset intrinsics
print('load intrinsics')
bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_parent_path, 'ycbv'))

# set shading and hide objects
for obj in (target_bop_objs + tless_dist_bop_objs + hb_dist_bop_objs + tyol_dist_bop_objs):
# for obj in (target_bop_objs):
    obj.set_shading_mode('auto')
    obj.hide(True)
    
# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(200)

# load cc_textures
print('load textures')
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)

print('sample poses')
# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())
    
# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=True)

for i in range(args.num_scenes):

    # Sample bop objects for a scene
    sampled_target_bop_objs = list(np.random.choice(target_bop_objs, size=21, replace=False))
    # sampled_target_bop_objs = [target_bop_objs[0]]
    sampled_distractor_bop_objs = list(np.random.choice(tless_dist_bop_objs, size=2, replace=False))
    sampled_distractor_bop_objs += list(np.random.choice(hb_dist_bop_objs, size=2, replace=False))
    sampled_distractor_bop_objs += list(np.random.choice(tyol_dist_bop_objs, size=2, replace=False))

    # Randomize materials and set physics
    for obj in (sampled_target_bop_objs + sampled_distractor_bop_objs):        
    # for obj in (sampled_target_bop_objs):        
        mat = obj.get_materials()[0]
        if obj.get_cp("bop_dataset_name") in ['itodd', 'tless']:
            grey_col = np.random.uniform(0.1, 0.9)   
            mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])        
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
        obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        obj.hide(False)
    
    # Sample two light sources
    light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                    emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                            elevation_min = 5, elevation_max = 89)
    light_point.set_location(location)

    # sample CC Texture and assign to room planes
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)


    # Sample object poses and check collisions 
    bproc.object.sample_poses(objects_to_sample = sampled_target_bop_objs + sampled_distractor_bop_objs,
    # bproc.object.sample_poses(objects_to_sample = sampled_target_bop_objs,
                            sample_pose_func = sample_pose_func, 
                            max_tries = 1000)
            
    # Physics Positioning
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                    max_simulation_time=10,
                                                    check_object_interval=1,
                                                    substeps_per_frame = 20,
                                                    solver_iters=25)
    
    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs + sampled_distractor_bop_objs)
    # bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs)

    cam_poses = 0 
    while cam_poses < 25: # NOTE if not debugging should be 25
        # Sample location
        location = bproc.sampler.shell(center = [0, 0, 0],
                                radius_min = 0.61,
                                radius_max = 1.24,
                                elevation_min = 5,
                                elevation_max = 89)
        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=15, replace=False))
        # poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=1, replace=False))
        # poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs + list(np.random.choice(sampled_distractor_bop_objs, size=14)), size=15, replace=False))
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159))
        # Add homog cam pose based on location an rotation
        baseline = np.array([0.05, 0, 0])
        cam2world_matrix_center = bproc.math.build_transformation_mat(location, rotation_matrix)
        cam2world_matrix_left = bproc.math.build_transformation_mat(location-baseline/2, rotation_matrix)
        cam2world_matrix_right = bproc.math.build_transformation_mat(location+baseline/2, rotation_matrix)
        
        visible_objs = bproc.camera.visible_objects(cam2world_matrix_center)
        if sampled_target_bop_objs[0] in visible_objs:
            print('target object not in view')
            continue

        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix_center, {"min": 0.3}, bop_bvh_tree):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix_left, frame=cam_poses*2)
            bproc.camera.add_camera_pose(cam2world_matrix_right, frame=cam_poses*2+1)
            cam_poses += 1

    # enable stereo
    # bproc.renderer.toggle_stereo(True)
    # bproc.camera.set_stereo_parameters(interocular_distance=0.05, convergence_mode="PARALLEL", convergence_distance=0.00001)

    # render the whole pipeline
    data = bproc.renderer.render()
    print('len_data', len(data))

    colors = data['colors']
    depths = data['depth']
    print('colors', np.array(colors).shape, type(colors), len(colors))
    print('depths', np.array(depths).shape, type(depths), len(depths))

    colors_left = colors[1::2]
    depth_left = depths[1::2]
    colors_right = colors[0::2]
    depth_right = depths[0::2]

    # halt number of frames
    bpy.context.scene.frame_end = len(colors)//2

    # Write data in bop format (left)
    bproc.writer.write_bop(os.path.join(args.output_dir, 'render'),
                           target_objects = sampled_target_bop_objs,
                           dataset = 'ycb-stereo-left',
                           depth_scale = 0.1,
                           depths = depth_left,
                           colors = colors_left,
                           color_file_format = "JPEG",
                           ignore_dist_thres = 10)
    # Write data in bop format (right)
    bproc.writer.write_bop(os.path.join(args.output_dir, 'render'),
                           target_objects = sampled_target_bop_objs,
                           dataset = 'ycb-stereo-right',
                           depth_scale = 0.1,
                           depths = depth_right,
                           colors = colors_right,
                           color_file_format = "JPEG",
                           ignore_dist_thres = 10)
    
    for obj in (sampled_target_bop_objs + sampled_distractor_bop_objs):      
    # for obj in (sampled_target_bop_objs):      
        obj.disable_rigidbody()
        obj.hide(True)
