# Landmark_based_SMPL_fitting_with_NLF
Landmark based method for SMPLX shape fitting, and Neural Localizer Fields (NLF) for Continuous 3D Human Pose Estimation.

## Marker Labeling Workflow
1. Download the SMPLX models from the official website, and put it under the folder (according to the path in code).
2. Open your 3D character OBJ file in Blender and then select the markers on your OBJ mesh (must follow the selection scheme here! Please check "how to label markers" folder).
3. Video tutorial for labeling markeres: https://drive.google.com/file/d/1G04oJmln6iW6_glFhkXBOAe6ezLZ-2QX/view?usp=drive_link.
4. save it as json file with key-value pair (eg. position: coordinate).
5. replace the file paths in code.
6. run python ShapeFitter.py.
7. You can modify the hyperparameters like loop number, regularization numbers, etc.
(If you want, you can decide your own markers, but you also need to change the markers for the SMPLX model)
