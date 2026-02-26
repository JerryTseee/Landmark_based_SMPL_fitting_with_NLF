import bpy
import bmesh
import json

# Ensure in edit mode with selected vertices
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    # Step 1: Get selected vertex indices in Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    selected_indices = [v.index for v in bm.verts if v.select]

    if not selected_indices:
        print("no selected")
    else:
        # Step 2: Switch to Object Mode to access vertex coordinates
        bpy.ops.object.mode_set(mode='OBJECT')
        mesh = obj.data
        landmark_dict = {}
        index_dict = {}

        for i, index in enumerate(selected_indices):
            coord = list(mesh.vertices[index].co)
            name ="right_back_foot"
            landmark_dict[name] = coord
            index_dict[name] = index

        # Save to JSON
        output_path = "E:/landmark_based_SMPL_fitting/index/temp.json"
        with open(output_path, "w") as f:
            json.dump(landmark_dict, f, indent=4)
            
        index_path = "E:/landmark_based_SMPL_fitting/index/temp_index.json"
        with open(index_path, "w") as f:
            json.dump(index_dict, f, indent=4)

else:
    print("error")
