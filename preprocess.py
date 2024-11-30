
import consts
import subprocess
from utils.xtimer import Timer
import cv2
import os, random
import glob
from PIL import Image



def upsample_and_replace_images(input_folder, scale_factor=2):
    # Define the upsampling methods
    methods = [Image.LANCZOS, Image.BICUBIC]

    # Find all image files in the input folder
    img_files = glob.glob(f"{input_folder}/*.*")  # You can specify particular formats if needed, e.g., '*.jpg'

    for img_file in img_files:
        # Open the image
        with Image.open(img_file) as img:
            # Calculate the new size
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))

            # Randomly select an upsampling method
            method = random.choice(methods)

            # Upsample the image using the selected method
            upsampled_img = img.resize(new_size, method)

            # Save the upsampled image, replacing the original
            upsampled_img.save(img_file)

def run_command(command):
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


def create_video(input_folder, output_file, fps=30):
    def image_files_generator(folder):
        """Generator to yield image file paths."""
        for img_file in sorted(glob.glob(f"{folder}/*.png")):
            yield img_file

    # Initialize VideoWriter with the first frame
    first_frame_path = next(image_files_generator(input_folder), None)
    if first_frame_path is None:
        print("No images found in the folder.")
        return

    frame = cv2.imread(first_frame_path)
    if frame is None:
        print(f"Failed to read the image: {first_frame_path}")
        return

    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Process each image
    for img_file in image_files_generator(input_folder):
        frame = cv2.imread(img_file)
        if frame is not None:
            video.write(frame)
        else:
            print(f"Warning: Failed to read the image: {img_file}")

    video.release()
    print(f"Video saved as {output_file}")


def get_train_cmd(scene, method):


    
    if method == "splatfacto":
        train_cmd = f"ns-train splatfacto --method-name {method} --data datasets/nerfstudio/{scene} --pipeline.datamanager.images-on-gpu True --timestamp latest --pipeline.datamanager.cache-images gpu --viewer.quit-on-train-completion True nerfstudio-data"

    elif "kplanes" in method:

        far_value = method.split("-")[1]
        train_cmd = f"ns-train kplanes --data datasets/nerfstudio/{scene} --pipeline.model.near-plane 0 --pipeline.model.far-plane {far_value} --pipeline.datamanager.images-on-gpu True --timestamp {method} --viewer.quit-on-train-completion True nerfstudio-data"
    
    elif scene == "storefront":
        train_cmd = f"ns-train {method} --method-name {method} --data datasets/nerfstudio/{scene} --pipeline.datamanager.images-on-gpu True --downscale-factor 2 --timestamp latest --viewer.quit-on-train-completion True nerfstudio-data"


    else:
        train_cmd = f"ns-train {method} --method-name {method} --data datasets/nerfstudio/{scene} --pipeline.datamanager.images-on-gpu True --timestamp latest --viewer.quit-on-train-completion True nerfstudio-data"

        
    
    return train_cmd



def get_render_images_cmd(scene, method):
    render_images_cmd = f"ns-render interpolate --load-config outputs/{scene}/{method}/latest/config.yml --output-path renders/{scene}__{method} --frame-rate 30 --image-format png --output-format images"

    if "kplanes" in method:
        core_method = method.split("-")[0]
        render_images_cmd = f"ns-render interpolate --load-config outputs/{scene}/{core_method}/{method}/config.yml --output-path renders/{scene}__{method} --frame-rate 30 --image-format png --output-format images"
    
    elif method == "instant-ngp":
        render_images_cmd = f"ns-render interpolate --load-config outputs/{scene}/{method}/latest/config.yml --output-path renders/{scene}__{method} --frame-rate 30 --image-format png --output-format images"
    
    


    return render_images_cmd


def get_render_mp4_cmd(scene, method):
    render_mp4_cmd = f"ns-render interpolate --load-config outputs/{scene}/{method}/latest/config.yml --frame-rate 30 --output-path renders/{scene}__{method}.mp4"

    if "kplanes" in method:
        core_method = method.split("-")[0]
        render_mp4_cmd = f"ns-render interpolate --load-config outputs/{scene}/{core_method}/{method}/config.yml --frame-rate 30 --output-path renders/{scene}__{method}.mp4"
    return render_mp4_cmd



def train_and_render_nerfs_core(scene, method):
    print(f"{'+'*20} Training and rendering {scene}-{method}... {'+'*20}")
    # Training command
    train_cmd = get_train_cmd(scene, method)
    
    run_command(train_cmd)

    # Rendering commands
    render_mp4_cmd = get_render_mp4_cmd(scene, method)
    render_images_cmd = get_render_images_cmd(scene, method)

    run_command(render_images_cmd)

    if scene == "storefront":
        upsample_and_replace_images(f"renders/{scene}__{method}")

    input_folder = f'renders/{scene}__{method}'  # Replace with your folder path
    output_file = f'renders/{scene}__{method}.mp4'  # Replace with your desired output file name
    create_video(input_folder, output_file)

    # run_command(render_mp4_cmd)




def train_and_render_nerfs():

    nerfstudio_scenes = consts.nerfstudio_scenes
    nerfstudio_methods = consts.nerfstudio_methods

    for s_i, scene in enumerate(nerfstudio_scenes, 1):

        print(f"{'*'*30} Training and rendering {scene} [{s_i}/{len(nerfstudio_scenes)}]... {'*'*30}")
        for method in nerfstudio_methods:


            if (method in ["splatfacto", "nerfacto", "tensorf"]) and (scene in ["aspen", "bww_entrance", "campanile", "desolation", "dozer", "Egypt"]):
                continue
        

            if method == "nerfacto-huge" and scene in ["campanile", "desolation", "dozer"]:
                continue
            

            far_value = int(method.split("-")[1]) if "kplanes" in method else None
            

            train_and_render_nerfs_core(scene, method)

            tim.lap()




    print("Training and rendering completed.")






if __name__ == "__main__":
    tim = Timer()
    tim.start()
    train_and_render_nerfs()

    tim.stop()