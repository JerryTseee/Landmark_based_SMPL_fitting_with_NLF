import numpy as np
import torch
import torchvision
import pyrender
import trimesh
import cv2
import smplx
import os
import joblib
from tqdm import tqdm

# --- 1. Setup Paths & Environment ---
INPUT_PKL = "./output/output_smoothed.pkl"
HUMAN_MODEL_FOLDER = 'body_models'
OUTPUT_VIDEO_DIR = './output'
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_VIDEO_DIR, 'output.mp4')

os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
# Headless rendering setup
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# --- 2. Load Data ---
print(f"Loading data from {INPUT_PKL}...")
if not os.path.exists(INPUT_PKL):
    raise FileNotFoundError(f"PKL file not found: {INPUT_PKL}")

data = joblib.load(INPUT_PKL)

# Access Person ID 1 (matches your first script)
person_key = 1
if person_key not in data:
    # Fallback if key 1 isn't found, pick the first available key
    person_key = list(data.keys())[0]

person_data = data[person_key]
frames = person_data["pose"].shape[0]
print(f"Processing {frames} frames for Person {person_key}...")

# --- 3. Initialize SMPL-X Model ---
device = torch.device("cpu") # CPU is fine for geometry generation

try:
    smplx_layer = smplx.create(
        HUMAN_MODEL_FOLDER,
        model_type='smplx',
        gender="NEUTRAL",
        use_pca=False,
        use_face_contour=True,
        batch_size=frames
    ).to(device)
except Exception as e:
    raise FileNotFoundError(f"Error loading SMPL-X model from '{HUMAN_MODEL_FOLDER}'.\nDetails: {e}")

# --- 4. Generate Mesh (Using Original Betas) ---
print("Generating SMPL-X meshes...")

# Load the raw betas from the PKL
# Your first script saves (T, 10), so we use that directly.
custom_betas = person_data["betas"]

# Ensure shape matches the batch size (Frames, 10)
if custom_betas.shape[0] == frames:
    # Case A: We have betas for every frame (Your requirement)
    shape_tensor = torch.from_numpy(custom_betas).float()
else:
    # Case B: Safety fallback (if betas were (1, 10))
    print("Warning: Beta shape mismatch, tiling first frame betas.")
    shape_tensor = torch.from_numpy(custom_betas[0]).float().repeat(frames, 1)

# Run SMPL-X Forward Pass
with torch.no_grad():
    smplx_output = smplx_layer(
        global_orient=torch.from_numpy(person_data["pose"][:, :3]).float(),
        body_pose=torch.from_numpy(person_data["pose"][:, 3:66]).float(),
        betas=shape_tensor # <--- Passing the per-frame betas here
    )

# --- 5. Align/Translate Mesh ---
# Calculate translation to match joints3d (VIBE/ROMP style alignment)
smplx_root_position = smplx_output.joints[:, 0].cpu().numpy()
transl = person_data["joints3d"][:, 0, :].copy()

# Coordinate system adjustments (OpenCV -> OpenGL)
transl[:, 1] = -transl[:, 1]
transl[:, 2] = -transl[:, 2]

transl = (transl - smplx_root_position).reshape(frames, 1, 3)
vertices = smplx_output.vertices.cpu().numpy() + transl

# --- 6. Rendering Setup ---
renderer = pyrender.OffscreenRenderer(1080, 720)
scene = pyrender.Scene()

# Camera Setup
camera = pyrender.PerspectiveCamera(
    yfov=np.arctan(24/100)*2,  # 24mm sensor height approx
    aspectRatio=1080/720
)

camera_pose = np.eye(4)
camera_pose[2, 3] = 10 # Camera 10 units back
scene.add(camera, pose=camera_pose)

# Light
light = pyrender.DirectionalLight()
scene.add(light, pose=np.eye(4))

# Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 30, (1080, 720))

# Object Rotation (Face Front)
R_y_180 = np.array([
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

# --- 7. Render Loop ---
print(f"Rendering to {OUTPUT_VIDEO_PATH}...")
for frame in tqdm(range(frames)):
    # Create Trimesh
    mesh = trimesh.Trimesh(
        vertices=vertices[frame],
        faces=smplx_layer.faces
    )
    
    # Optional: Add color to make it look better
    mesh.visual.vertex_colors = [150, 150, 200, 255]

    # Add to Pyrender Scene
    pr_mesh = pyrender.Mesh.from_trimesh(mesh)
    mesh_node = scene.add(pr_mesh, pose=R_y_180)
    
    # Render
    color, _ = renderer.render(scene)
    video_writer.write(color)
    
    # Clean up for next frame
    scene.remove_node(mesh_node)

video_writer.release()
renderer.delete()
print("Done.")