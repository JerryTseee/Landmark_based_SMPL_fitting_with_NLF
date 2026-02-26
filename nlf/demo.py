import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import urllib.request
import smplx
# from smplx import SMPLX
import trimesh

# --- 1. Setup Devices and Paths ---
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {DEVICE}")

# Create directories
os.makedirs('models', exist_ok=True)
BM_PATH = 'body_models/smplx/SMPLX_NEUTRAL.npz' 

# Resource URLs
NLF_MODEL_URL = 'https://bit.ly/nlf_l_pt'
NLF_MODEL_PATH = 'models/nlf_l_multi.torchscript'
IMAGE_PATH = './input/test5.jpg'

# --- 2. Helper Functions ---
def download_file(url, path):
    if not os.path.exists(path):
        print(f"Downloading {url} to {path}...")
        urllib.request.urlretrieve(url, path)
    else:
        print(f"Found {path}, skipping download.")

def intrinsic_matrix_from_field_of_view(fov_degrees, im_shape, dtype=None):
    """
    Creates a simple intrinsic matrix based on FOV and image shape (H, W).
    Assumes principal point is at the center.
    """
    H, W = im_shape
    focal_length = max(H, W) / (2 * np.tan(np.radians(fov_degrees / 2)))
    # Create tensor with specified dtype or default
    if dtype is None:
        K = torch.tensor([
            [focal_length, 0, W / 2.0],
            [0, focal_length, H / 2.0],
            [0, 0, 1]
        ], device=DEVICE)
    else:
        K = torch.tensor([
            [focal_length, 0, W / 2.0],
            [0, focal_length, H / 2.0],
            [0, 0, 1]
        ], device=DEVICE, dtype=dtype)
    return K.unsqueeze(0) # Batch dim

def project_vertices(coords3d, intrinsic_matrix):
    """
    Projects 3D vertices to 2D using perspective projection.
    coords3d: (B, N, 3)
    intrinsic_matrix: (B, 3, 3)
    """
    if coords3d.dtype != intrinsic_matrix.dtype:
        intrinsic_matrix = intrinsic_matrix.to(coords3d.dtype)

    # Perspective division (x/z, y/z, 1)
    z = coords3d[..., 2:]
    z = torch.maximum(torch.tensor(0.001, device=coords3d.device, dtype=coords3d.dtype), z)
    projected = coords3d / z
    
    # Apply intrinsics: (f*x/z + cx, f*y/z + cy)
    # intrinsic_matrix[..., :2, :] extracts the top 2 rows (for u, v computation)
    return torch.einsum('bnk,bjk->bnj', projected, intrinsic_matrix[..., :2, :])

# --- 3. Execution ---

# Download NLF model and Image
download_file(NLF_MODEL_URL, NLF_MODEL_PATH)

# Check for SMPL-X model file
if not os.path.exists(BM_PATH):
    raise FileNotFoundError(
        f"SMPL-X model not found at {BM_PATH}.\n"
        "Please download 'SMPLX_NEUTRAL.npz' from https://smpl-x.is.tue.mpg.de/ "
        "and place it in the 'body_models/smplx/' directory."
    )

# A. Load NLF Model
print("Loading NLF model...")
model = torch.jit.load(NLF_MODEL_PATH, map_location=DEVICE).eval()

# B. Load SMPL-X Body Model
print("Loading SMPL-X body model...")
# Initialize SMPL-X layer. use_pca=False is crucial as NLF predicts raw rotation matrices/axis-angles
bm = smplx.SMPLX(BM_PATH, use_pca=False).to(DEVICE).eval()

# C. Prepare Input
image = torchvision.io.read_image(IMAGE_PATH)
image_np = image.permute(1, 2, 0).numpy()
frame_batch = image.unsqueeze(0).to(DEVICE)

# D. Run Inference
print("Running inference...")
with torch.inference_mode():
    pred = model.detect_smpl_batched(frame_batch)

    pose = pred['pose'][0]
    betas = pred['betas'][0]
    trans = pred['trans'][0]

    num_people = pose.shape[0]
    if num_people == 0:
        print("No people detected in the image.")
        exit(0)
    print(f"Detected {num_people} person(s).")

    global_orient = pose[:, :3]
    body_pose     = pose[:, 3:66]


    # setting part
    jaw_pose = torch.zeros((num_people, 3), device=DEVICE, dtype=pose.dtype)
    leye_pose = torch.zeros((num_people, 3), device=DEVICE, dtype=pose.dtype)
    reye_pose = torch.zeros((num_people, 3), device=DEVICE, dtype=pose.dtype)
    left_hand_pose = torch.zeros((num_people, 45), device=DEVICE, dtype=pose.dtype)
    right_hand_pose = torch.zeros((num_people, 45), device=DEVICE, dtype=pose.dtype)

    # E. Run SMPL-X Forward Pass
    print("Generating mesh from parameters...")
    smplx_output = bm(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas,
        transl=trans
        # jaw_pose=jaw_pose,
        # leye_pose=leye_pose,
        # reye_pose=reye_pose,
        # left_hand_pose=left_hand_pose,
        # right_hand_pose=right_hand_pose
    )
    vertices_3d = smplx_output.vertices

    faces = bm.faces
    mesh = trimesh.Trimesh(vertices=vertices_3d[0].cpu().numpy(), faces=faces, process=False)
    mesh.export('output_body.obj')
    print("Saved 3D mesh to output_body.obj")

    # F. Project to 2D
    fov = 55  # Degrees
    K = intrinsic_matrix_from_field_of_view(fov, image.shape[1:3], dtype=vertices_3d.dtype)
    vertices_2d = project_vertices(vertices_3d, K).cpu().numpy()
    

# --- 4. Visualization ---
plt.figure(figsize=(10, 10))
plt.imshow(image_np)

for i in range(len(vertices_2d)):
    pts = vertices_2d[i]
    plt.scatter(pts[:, 0], pts[:, 1], s=0.5, c='cyan', alpha=0.5, label='Mesh' if i == 0 else "")

plt.axis('off')
plt.title(f"SMPL-X Body Only ({num_people} detected)")
plt.legend()
plt.savefig('output_body_only.png')
print("Saved to output_body_only.png")
plt.show()