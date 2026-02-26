import torch
import torchvision
import cv2
import numpy as np
import os
import urllib.request
import smplx
import joblib
import argparse
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# --- 1. Setup Devices and Paths ---
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {DEVICE}")

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('output', exist_ok=True)
BM_PATH = 'body_models/smplx/SMPLX_NEUTRAL.npz' 

# Resource URLs
NLF_MODEL_URL = 'https://bit.ly/nlf_l_pt'
NLF_MODEL_PATH = 'models/nlf_l_multi.torchscript'
VIDEO_PATH = './input/test6.mp4'
OUTPUT_PKL = './output/output.pkl'

# custom shape parameters path
CUSTOM_SHAPE_PATH = "./customized_shape/weilinai_zyy_betas.npy"

# --- 2. Helper Functions ---
def download_file(url, path):
    if not os.path.exists(path):
        print(f"Downloading {url} to {path}...")
        urllib.request.urlretrieve(url, path)

def smooth_pkl(input_path, output_path, sigma=2.0):
    # Simple Gaussian smoothing for pose data
    data = joblib.load(input_path)
    data[1]["pose"] = gaussian_filter1d(data[1]["pose"], sigma=sigma, axis=0)
    joblib.dump(data, output_path)

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
custom_betas = np.load(CUSTOM_SHAPE_PATH)
print(f"Custom betas values: {custom_betas}")
custom_betas_tensor = torch.from_numpy(custom_betas).float().to(DEVICE)

# C. Prepare Input
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video not found at {VIDEO_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Processing video: {VIDEO_PATH} ({num_frames} frames)")

# If multiple people exist, this script takes the first detection.
results = {
    "pose": [],
    "betas": [], # store fixed betas for each frame
    "trans": [],
    "joints3d": []
}

# D. process loop
print("Running inference...")
for _ in tqdm(range(num_frames)):
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).to(DEVICE)
    frame_batch = frame_tensor.unsqueeze(0)

    with torch.inference_mode():
        pred = model.detect_smpl_batched(frame_batch)

    pose_raw = pred['pose'][0]

    if pose_raw.shape[0] > 0:
        current_pose = pose_raw[0:1]
        # current_betas = pred['betas'][0][0:1]
        current_betas = custom_betas_tensor
        current_trans = pred['trans'][0][0:1]

        global_orient = current_pose[:, :3]
        body_pose = current_pose[:, 3:66]

        smplx_output = bm(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=current_betas,
            transl=current_trans
            # jaw_pose=jaw_pose,
            # leye_pose=leye_pose,
            # reye_pose=reye_pose,
            # left_hand_pose=left_hand_pose,
            # right_hand_pose=right_hand_pose
        )
        current_joints = smplx_output.joints

        results["pose"].append(current_pose.detach().cpu().numpy())
        results['betas'].append(current_betas.detach().cpu().numpy())
        results['trans'].append(current_trans.detach().cpu().numpy())
        results['joints3d'].append(current_joints.detach().cpu().numpy())
    
    else:
        print("No person detected in frame, skipping...")
        pass

cap.release()

print("Consolidating data ...")

if len(results["pose"]) > 0:
    final_pose = np.concatenate(results["pose"], axis=0)
    final_betas = np.concatenate(results["betas"], axis=0)
    final_trans = np.concatenate(results["trans"], axis=0)
    final_joints = np.concatenate(results["joints3d"], axis=0)

    output_db = {
        1:{
        "pose" : final_pose,
        "betas" : final_betas,
        "trans" : final_trans,
        "joints3d" : final_joints,
        "global_orient" : final_pose[:, :3],
        "body_pose" : final_pose[:, 3:66]
        }
    }

    joblib.dump(output_db, OUTPUT_PKL)
    print(f"Success! Saved sequence ({len(final_pose)} frames) to {OUTPUT_PKL}")

    # smoother version
    smoothed_output = OUTPUT_PKL.replace('.pkl', '_smoothed.pkl')
    smooth_pkl(OUTPUT_PKL, smoothed_output, sigma=2.0)
    print(f"Smoothed version saved to {smoothed_output}")
else:
    print("Failed: No people detected in the video.")
