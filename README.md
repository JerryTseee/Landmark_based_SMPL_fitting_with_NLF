# Landmark_based_SMPL_fitting_with_NLF
Landmark based method for SMPLX shape fitting, and Neural Localizer Fields (NLF) for Continuous 3D Human Pose Estimation.

https://github.com/user-attachments/assets/40c78b38-9f56-49e2-bf5b-efec72ffdcce



## First Part: Marker Labeling Workflow for Shape Fitting
1. Download the SMPLX models from the official website, and put it under the folder (according to the path in code).
2. Open your 3D character OBJ file in Blender and then select the markers on your OBJ mesh (must follow the selection scheme here! Please check "how to label markers" folder).
3. Video tutorial for labeling markeres: https://drive.google.com/file/d/1G04oJmln6iW6_glFhkXBOAe6ezLZ-2QX/view?usp=drive_link.
4. Save it as json file with key-value pair (eg. position: coordinate).
5. Replace the file paths in code.
6. Run python ShapeFitter.py.
7. You can modify the hyperparameters like loop number, regularization numbers, etc.
(If you want, you can decide your own markers, but you also need to change the markers for the SMPLX model)

## Second Part: NLF Workflow for Pose Fitting
1. Change to the NLF directory.
2. Place the SMPLX models and input image/video in the correct folders according to the code.
3. Copy and place your obtained shape npy format files in the correct folders according to the code.
4. Run python demo.py to get the output image and also the obj format smplx model.
5. Or, run python demo_pkl.py and then demo_rendering.py to get the output pkl and animation video of the input video.

## More
1. Original NLF repo: https://github.com/isarandi/nlf.git
2. An automated workflow for composing, rendering, and retargeting MMD assets: https://github.com/AfterJourney00/mmd_to_smpl.git
