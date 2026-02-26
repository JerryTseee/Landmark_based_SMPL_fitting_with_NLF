import trimesh
import torch
import numpy as np
import json
from smplx import SMPLX


class ShapeFitter:
    def __init__(self, smpl_model_path, character_landmarks_path, smpl_landmarks_path, smpl_index_path, standard_height=1.6, learning_rate=0.1, max_iter=300):
        # Model paths and parameters
        self.smpl_model_path = smpl_model_path
        self.character_landmarks_path = character_landmarks_path
        self.smpl_landmarks_path = smpl_landmarks_path
        self.smpl_index_path = smpl_index_path
        self.standard_height = standard_height
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
        # Load data
        self.character_landmarks = self.load_landmark_coords(self.character_landmarks_path)
        self.smpl_landmarks = self.load_landmark_coords(self.smpl_landmarks_path)
        self.smpl_index = self.load_landmark_coords(self.smpl_index_path)

        self.common_keys = list(set(self.character_landmarks.keys()) & set(self.smpl_landmarks.keys()))

        # Normalize character landmarks
        self.scale_factor = self.normalize_character_landmarks()

        # SMPL model initialization
        self.smpl_model = SMPLX(model_path=self.smpl_model_path, gender='neutral', batch_size=1)

        # Initialize pose and shape
        self.pose = torch.zeros([1, 63], dtype=torch.float32, requires_grad=True)
        self.shape = torch.zeros([1, 10], dtype=torch.float32, requires_grad=True)

        # Target character landmarks
        self.char_pts = np.array([self.character_landmarks[i] for i in self.common_keys])
        self.char_pts_torch = torch.tensor(self.char_pts, dtype=torch.float32)

        # Get indices from smpl_index.json
        self.smpl_indices = torch.tensor([self.smpl_index[i] for i in self.common_keys], dtype=torch.long)

        # Optimizer setup
        self.optimizer = torch.optim.Adam([self.shape, self.pose], lr=self.learning_rate)

    def load_landmark_coords(self, file_path):
        """Load the landmark coordinates from JSON."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def normalize_character_landmarks(self):
        """Normalize the character landmarks to a certain height."""
        
        top_head = np.array(self.character_landmarks["top_head"])
        left_foot = np.array(self.character_landmarks["left_front_foot"])
        right_foot = np.array(self.character_landmarks["right_front_foot"])

        foot_y = min(left_foot[1], right_foot[1])
        head_y = top_head[1]

        scale_factor = self.standard_height / (head_y - foot_y)

        # apply to all character landmarks
        for i in self.character_landmarks:
            self.character_landmarks[i] = (np.array(self.character_landmarks[i]) * scale_factor).tolist()

        return scale_factor # this must be returned so that for the output result to be scaled back to original size!

    def umeyama_alignment(self, X, Y):
        """Align two 3D point sets using Umeyama's method."""
        mu_X = X.mean(axis=0)
        mu_Y = Y.mean(axis=0)
        X0 = X - mu_X
        Y0 = Y - mu_Y
        U, S, Vt = np.linalg.svd(X0.T @ Y0)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        t = mu_Y - R @ mu_X
        return R, t

    def apply_transform(self, vertices, R, t):
        """Apply the transformation to the SMPL vertices."""
        return (vertices @ R.T) + t

    def fit(self):
        """Run the optimization loop to fit the SMPL model to the character."""
        # Alignment step to get transformation
        char_pts = np.array([self.character_landmarks[i] for i in self.common_keys])
        smpl_pts = np.array([self.smpl_landmarks[i] for i in self.common_keys])
        R, t = self.umeyama_alignment(smpl_pts, char_pts)
        R = torch.tensor(R, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)

        # Optimization loop
        for i in range(self.max_iter):
            smpl_output = self.smpl_model(
                global_orient=self.pose[:, :3].reshape(1, 1, 3),
                body_pose=self.pose.reshape(1, 21, 3),
                betas=self.shape,
                pose2rot=True
            )

            vertices = smpl_output.vertices[0]
            vertices = self.apply_transform(vertices, R, t)

            model_landmarks = vertices[self.smpl_indices]
            loss = torch.mean((model_landmarks - self.char_pts_torch) ** 2)

            # Regularization term
            reg = 0.00008 * torch.mean(self.shape ** 2) + 0.001 * torch.mean(self.pose ** 2)
            loss += reg

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward() # backward propagation
            self.optimizer.step() # update pose and shape parameters

            # Print progress
            if i % 2 == 0:
                print(f"Step {i}, Loss: {loss.item():.5f}")

        # Final output
        # to check the validation of the shape parameters
        print(f"final shape: {self.shape.detach().cpu().numpy()}")
        final_shape = self.shape.detach().cpu().numpy()
        
        # save betas as npy
        np.save('./output_mesh/lupa_betas.npy', final_shape)

        self.finalize(vertices, R, t)



    def finalize(self, vertices, R, t):
        """Finalize the fitting and save the result."""
        # Scale back the vertices
        vertices = vertices.detach().cpu().numpy() * (1.0 / self.scale_factor)

        # Export the mesh
        faces = self.smpl_model.faces
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export('./output_mesh/lupa.obj')
        


# Usage - modify the paths as needed
shape_fitter = ShapeFitter(
    smpl_model_path='./human_model_files/smplx',
    character_landmarks_path='./index/lupa.json', # path to your character landmarks
    smpl_landmarks_path='./index/smpl.json',
    smpl_index_path='./index/smpl_index.json'
)
shape_fitter.fit()
