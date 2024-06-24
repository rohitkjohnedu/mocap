import bpy
import os
import re

# Set the output directory
output_dir = "D:\\Personal\\Projects\\badminton_drone\\drone_localisation\\stereo_triangulation\\images"  # Change this to your desired output directory

# Function to get the highest index for the filenames
def get_next_index(directory):
    files = os.listdir(directory)
    pattern = re.compile(r'(\d+)_([012])\.png')
    indices = []
    for file in files:
        match = pattern.match(file)
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=0) + 1

# Get the next index
next_index = get_next_index(output_dir)

# Function to render and save an image from a camera
def render_from_camera(camera, output_path):
    scene = bpy.context.scene
    scene.camera = camera
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

# Get the cameras
camera_0 = bpy.data.objects.get("Camera_0")  
camera_1 = bpy.data.objects.get("Camera_1")  
camera_2 = bpy.data.objects.get("Camera_2")  

if not camera_0 or not camera_1 or not camera_2:
    print("Error: Cameras not found.")
else:
    # Render and save images
    render_from_camera(camera_0, os.path.join(output_dir, f"{next_index}_0.png"))
    render_from_camera(camera_1, os.path.join(output_dir, f"{next_index}_1.png"))
    render_from_camera(camera_2, os.path.join(output_dir, f"{next_index}_2.png"))

