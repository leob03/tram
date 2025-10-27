import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import numpy as np
import torch
from glob import glob
from lib.models.smpl import SMPL
from lib.vis.traj import traj_filter
from scipy.spatial.transform import Rotation as R


def generate_blender_script(camera_data, track_data, img_shape, output_path):
    """
    Generate a Blender Python script that creates the full scene with animated characters and camera
    """

    world_cam_R = camera_data['world_cam_R']
    world_cam_T = camera_data['world_cam_T']
    img_focal = camera_data['img_focal']
    img_width, img_height = img_shape

    # Start building the Blender script
    script = """import bpy
import math
from mathutils import Vector, Euler, Matrix

# Clear existing scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Scene setup
scene = bpy.context.scene
scene.render.fps = 30
scene.frame_start = 1
scene.frame_end = {total_frames}

print("Setting up TRAM scene with {num_people} characters...")

""".format(total_frames=len(world_cam_T), num_people=len(track_data))

    # Add each person's mesh with animation
    for person_idx, person_data in enumerate(track_data):
        frames = person_data['frames']
        vertices_per_frame = person_data['vertices']
        faces = person_data['faces']

        # Create base mesh for this person
        script += f"""
# Create Character {person_idx}
verts_base = {vertices_per_frame[0].tolist()}
faces = {faces.tolist()}

mesh = bpy.data.meshes.new(name='Character_{person_idx}_Mesh')
mesh.from_pydata(verts_base, [], faces)
mesh.update()

obj = bpy.data.objects.new('Character_{person_idx}', mesh)
bpy.context.collection.objects.link(obj)

# Add material
mat = bpy.data.materials.new(name='Character_{person_idx}_Mat')
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs['Base Color'].default_value = {tuple(np.random.rand(3).tolist() + [1.0])}
bsdf.inputs['Roughness'].default_value = 0.8
obj.data.materials.append(mat)

# Enable shape keys for animation
obj.shape_key_add(name='Basis')

"""

        # Add shape keys for each frame
        script += f"# Add shape keys for Character {person_idx}\n"
        script += f"frame_to_shapekey = {{}}\n"

        for frame_num, verts in zip(frames, vertices_per_frame):
            script += f"""
# Frame {frame_num}
sk = obj.shape_key_add(name='Frame_{frame_num}')
for v_idx, co in enumerate({verts.tolist()}):
    sk.data[v_idx].co = co
frame_to_shapekey[{frame_num}] = sk
"""

        # Animate shape keys
        script += f"""
# Animate shape keys for Character {person_idx}
for frame_num, sk in frame_to_shapekey.items():
    # Set all shape keys to 0
    for other_sk in obj.data.shape_keys.key_blocks[1:]:
        other_sk.value = 0.0
        other_sk.keyframe_insert(data_path='value', frame=frame_num)

    # Set this shape key to 1
    sk.value = 1.0
    sk.keyframe_insert(data_path='value', frame=frame_num)

"""

    # Convert camera using the same method as test_camera_blender.py
    # This ensures camera trajectory and rotation match exactly

    # Step 1: Apply coordinate transformation rotations (Rx(+90°) then Rz(+180°))
    Rx_90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    Rz_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    combined_rotation = Rz_180 @ Rx_90

    world_cam_T_rotated = np.array([combined_rotation @ pos for pos in world_cam_T])
    world_cam_R_rotated = np.array([combined_rotation @ rot @ combined_rotation.T for rot in world_cam_R])

    # Step 2: Convert to view camera (like visualization.py lines 85-87)
    view_cam_R = np.transpose(world_cam_R_rotated, (0, 2, 1))
    view_cam_T = -np.einsum('bij,bj->bi', view_cam_R, world_cam_T_rotated)

    # Step 3: Apply trajectory fix
    view_cam_T[:, 2] = -view_cam_T[:, 2]

    # Step 4: Extract Euler angles and apply corrections
    cam_positions_blender = []
    cam_rotations_blender = []

    cam_euler_angles = []
    for cam_rot in view_cam_R:
        r = R.from_matrix(cam_rot)
        euler = r.as_euler('XYZ', degrees=False)
        cam_euler_angles.append(euler)

    cam_euler_angles = np.array(cam_euler_angles)

    # Apply static rotation correction
    cam_euler_corrected = cam_euler_angles.copy()
    cam_euler_corrected[:, 0] += np.radians(-90)

    # Apply angular velocity fix
    cam_euler_velocity = np.diff(cam_euler_corrected, axis=0, prepend=cam_euler_corrected[[0]])
    cam_euler_velocity[:, 2] = -cam_euler_velocity[:, 2]

    # Integrate back
    cam_euler_modified = np.cumsum(cam_euler_velocity, axis=0)
    offset_euler = cam_euler_corrected[0] - cam_euler_modified[0]
    cam_euler_modified += offset_euler

    # Store final positions and rotations
    cam_positions_blender = view_cam_T.tolist()
    cam_rotations_blender = cam_euler_modified.tolist()

    # Calculate focal length properly using FOV-based approach
    # Apply the -100 adjustment that visualization.py uses
    focal_pixels = img_focal - 100
    sensor_width_mm = 36.0

    # Calculate horizontal FOV from TRAM's intrinsics
    import math
    hfov_rad = 2 * math.atan(img_width / (2 * focal_pixels))

    # Convert FOV to focal length in mm for Blender
    focal_mm = sensor_width_mm / (2 * math.tan(hfov_rad / 2))

    # Determine sensor fit based on image orientation
    is_portrait = img_height > img_width
    sensor_fit = 'HORIZONTAL' if is_portrait else 'AUTO'

    script += f"""
# Create Camera with proper focal length and settings
cam_data = bpy.data.cameras.new(name='TRAM_Camera')
cam_data.lens = {focal_mm:.2f}  # focal length in mm (converted from {img_focal} pixels)
cam_data.sensor_width = {sensor_width_mm}
cam_data.sensor_fit = '{sensor_fit}'

# Principal point is centered, so no lens shift needed
cam_data.shift_x = 0.0
cam_data.shift_y = 0.0

cam_obj = bpy.data.objects.new('Camera', cam_data)
bpy.context.collection.objects.link(cam_obj)
scene.camera = cam_obj

# Set render resolution to match image dimensions
scene.render.resolution_x = {img_width}
scene.render.resolution_y = {img_height}
scene.render.resolution_percentage = 100

# Animate camera position and rotation
cam_positions = {cam_positions_blender}
cam_rotations = {cam_rotations_blender}

for frame_idx, (pos, rot) in enumerate(zip(cam_positions, cam_rotations), start=1):
    cam_obj.location = Vector(pos)
    cam_obj.rotation_euler = Euler(rot, 'XYZ')

    cam_obj.keyframe_insert(data_path='location', frame=frame_idx)
    cam_obj.keyframe_insert(data_path='rotation_euler', frame=frame_idx)

print(f"Camera settings: {{cam_data.lens:.2f}}mm focal length, {{'{sensor_fit}'}} sensor fit")
"""

    # Add ground plane
    locations = []
    for person_data in track_data:
        locations.extend([v.mean(axis=0) for v in person_data['vertices']])

    locations = np.array(locations)
    cx, cy = (locations.max(axis=0) + locations.min(axis=0))[[0, 1]] / 2.0
    sx, sy = (locations.max(axis=0) - locations.min(axis=0))[[0, 1]]
    scale = max(sx, sy) * 3

    # Calculate tile width in Blender units (matching TRAM's 0.5 unit tiles)
    tile_width = 0.5

    script += f"""
# Create ground plane (Z=0 in Blender, which is the ground after coordinate transform)
bpy.ops.mesh.primitive_plane_add(size={scale}, location=({cx}, {cy}, 0))
ground = bpy.context.active_object
ground.name = 'Ground'

# Ground material with checker texture
ground_mat = bpy.data.materials.new(name='Ground_Mat')
ground_mat.use_nodes = True
nodes = ground_mat.node_tree.nodes
links = ground_mat.node_tree.links

# Clear default nodes
nodes.clear()

# Add shader nodes
tex_coord = nodes.new(type='ShaderNodeTexCoord')
mapping = nodes.new(type='ShaderNodeMapping')
checker = nodes.new(type='ShaderNodeTexChecker')
bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
output = nodes.new(type='ShaderNodeOutputMaterial')

# Position nodes for clarity
tex_coord.location = (-800, 0)
mapping.location = (-600, 0)
checker.location = (-400, 0)
bsdf.location = (-200, 0)
output.location = (0, 0)

# Configure checker texture to match TRAM's pattern
checker.inputs['Scale'].default_value = {1.0 / tile_width}  # Scale to match 0.5 unit tiles
checker.inputs['Color1'].default_value = (0.8, 0.9, 0.9, 1.0)  # Light cyan-ish
checker.inputs['Color2'].default_value = (0.6, 0.7, 0.7, 1.0)  # Darker cyan-ish

# Configure mapping to center the pattern at ({cx}, {cy})
mapping.inputs['Location'].default_value = ({-cx}, {-cy}, 0.0)

# Configure material properties
bsdf.inputs['Roughness'].default_value = 0.9
bsdf.inputs['Specular IOR Level'].default_value = 0.0

# Link nodes
links.new(tex_coord.outputs['Object'], mapping.inputs['Vector'])
links.new(mapping.outputs['Vector'], checker.inputs['Vector'])
links.new(checker.outputs['Color'], bsdf.inputs['Base Color'])
links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

ground.data.materials.append(ground_mat)

# Add lighting
bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
sun = bpy.context.active_object
sun.data.energy = 2.0

print("✅ Scene setup complete! Press spacebar to play animation.")
"""

    # Write script to file
    with open(output_path, 'w') as f:
        f.write(script)

    print(f"✓ Generated Blender script: {output_path}")


def export_to_blender(seq_folder, output_folder='blender_export'):
    """
    Export SMPL meshes and camera trajectory as a Blender Python script
    """

    os.makedirs(f'{seq_folder}/{output_folder}', exist_ok=True)

    img_folder = f'{seq_folder}/images'
    hps_folder = f'{seq_folder}/hps'
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    hps_files = sorted(glob(f'{hps_folder}/*.npy'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smpl = SMPL().to(device)

    max_track = len(hps_files)

    ##### Load camera data #####
    pred_cam = np.load(f'{seq_folder}/camera.npy', allow_pickle=True).item()
    world_cam_R = torch.tensor(pred_cam['world_cam_R']).to(device)
    world_cam_T = torch.tensor(pred_cam['world_cam_T']).to(device)
    img_focal = pred_cam['img_focal'].item()

    # Get image dimensions
    import cv2
    img = cv2.imread(imgfiles[0])
    img_shape = (img.shape[1], img.shape[0])  # width, height

    ##### Load and process SMPL data #####
    print("Loading SMPL data...")
    track_data = []
    lowest = []

    for i in range(max_track):
        hps_file = hps_files[i]

        pred_smpl = np.load(hps_file, allow_pickle=True).item()
        pred_rotmat = pred_smpl['pred_rotmat'].to(device)
        pred_shape = pred_smpl['pred_shape'].to(device)
        pred_trans = pred_smpl['pred_trans'].to(device)
        frame = pred_smpl['frame']

        # Use mean shape across all frames for consistency
        mean_shape = pred_shape.mean(dim=0, keepdim=True)
        pred_shape = mean_shape.repeat(len(pred_shape), 1)

        # Generate SMPL mesh
        pred = smpl(body_pose=pred_rotmat[:, 1:],
                    global_orient=pred_rotmat[:, [0]],
                    betas=pred_shape,
                    transl=pred_trans.squeeze(),
                    pose2rot=False,
                    default_smpl=True)
        pred_vert = pred.vertices
        pred_j3d = pred.joints[:, :24]

        # Transform to world coordinates
        cam_r = world_cam_R[frame]
        cam_t = world_cam_T[frame]

        pred_vert_w = torch.einsum('bij,bnj->bni', cam_r, pred_vert) + cam_t[:, None]
        pred_j3d_w = torch.einsum('bij,bnj->bni', cam_r, pred_j3d) + cam_t[:, None]

        # Apply trajectory filtering for smoothness
        pred_vert_w, pred_j3d_w = traj_filter(pred_vert_w.cpu(), pred_j3d_w.cpu())

        # Calculate lowest point BEFORE coordinate transformation (like visualization.py)
        lowest.append(pred_vert_w[:, :, 1].min())

        # Store person track data (will convert coords later)
        faces = smpl.faces if isinstance(smpl.faces, np.ndarray) else smpl.faces.cpu().numpy()
        track_data.append({
            'frames': frame.tolist(),
            'vertices_tram': [v.numpy() for v in pred_vert_w],
            'joints_tram': [j.numpy() for j in pred_j3d_w],
            'faces': faces
        })

    # Calculate ground offset in TRAM coordinates (Y-axis, like visualization.py line 76)
    offset_y = torch.min(torch.stack(lowest)).item()

    # For characters: use normal offset to move them to ground level
    offset_tram_characters = torch.tensor([0, offset_y, 0])

    # For camera: use inverted offset (like test_camera_blender.py with use_inverted_offset=True)
    # This raises the camera so it's at proper height instead of below ground
    offset_tram_camera = torch.tensor([0, -offset_y, 0])

    # Now convert vertices to Blender coordinates and apply offset
    def to_blender_coords(coords):
        """Convert [X, Y, Z] in TRAM to [X, -Z, Y] in Blender"""
        return np.stack([coords[:, 0], -coords[:, 2], coords[:, 1]], axis=1)

    for person_data in track_data:
        # Apply offset in TRAM coords, then convert to Blender
        verts_offset = [v - offset_tram_characters.numpy() for v in person_data['vertices_tram']]
        person_data['vertices'] = [to_blender_coords(v) for v in verts_offset]

        joints_offset = [j - offset_tram_characters.numpy() for j in person_data['joints_tram']]
        person_data['joints'] = [to_blender_coords(j) for j in joints_offset]

        # Remove TRAM coords
        del person_data['vertices_tram']
        del person_data['joints_tram']

    # Convert camera to Blender coordinates with inverted offset
    world_cam_T_offset = world_cam_T - offset_tram_camera.to(device)

    ##### Generate Blender script #####
    print("Generating Blender Python script...")
    camera_data = {
        'world_cam_R': world_cam_R.cpu().numpy(),
        'world_cam_T': world_cam_T_offset.cpu().numpy(),
        'img_focal': img_focal
    }

    script_path = f'{seq_folder}/{output_folder}/tram_blender_scene.py'
    generate_blender_script(camera_data, track_data, img_shape, script_path)

    print(f"\n✅ Export complete!")
    print(f"Blender script saved to: {script_path}")
    print(f"\nTo use in Blender:")
    print(f"1. Open Blender")
    print(f"2. Go to Scripting tab")
    print(f"3. Open the script: {script_path}")
    print(f"4. Click 'Run Script' or press Alt+P")
    print(f"5. Press spacebar in the viewport to play the animation")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export TRAM results to Blender Python script')
    parser.add_argument('--video', type=str, default='./example_video.mov', help='input video')
    parser.add_argument('--output_folder', type=str, default='blender_export', help='output folder name')
    args = parser.parse_args()

    file = args.video
    seq = os.path.basename(file).split('.')[0]
    seq_folder = f'results/{seq}'

    export_to_blender(seq_folder, args.output_folder)