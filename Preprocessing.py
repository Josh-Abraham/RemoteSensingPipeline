import argparse
import cv2
import glob
from pipeline import pipeline

PIPELINE_ORDER = {
    1: ["white_balancing", "super_res", "deshadow", "anisotropic"],
    2: ["super_res", "white_balancing"]
}

# Args
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", type=str, default="Images/*.jpg", help="Image folder path")
ap.add_argument("-s", "--scale", type=bool, default=True, help="Resize Images")
ap.add_argument("-o", "--order", type=int, default=1, help="Pipeline Preprocessing Order")

args = vars(ap.parse_args())
path = args["path"]
scale = args["scale"]
preprocess_order = PIPELINE_ORDER[args["order"]]

def load_image(location):
    global scale
    image = cv2.imread(location)
    (H, W) = image.shape[:2]
    if scale:
        scale_factor = 250 / W
        width = int(W * scale_factor)
        height = int(H * scale_factor)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)	
    return image

def fetch_images():
    global path
    for filename in glob.glob(path): #assuming jpg
        image = load_image(filename)
        run_pipeline(image)

def run_pipeline(image):
    global preprocess_order
    preprocessed = image
    for elem in preprocess_order:
        preprocessed = getattr(pipeline, elem)(preprocessed)
    show_image(image, preprocessed)

def show_image(orig_image, preprocess_image):
    cv2.imshow("Original", orig_image)
    cv2.imshow("Pre-processed", preprocess_image)
    cv2.waitKey(0)

fetch_images()

# Shadows
# https://github.com/vinthony/ghost-free-shadow-removal
# https://paperswithcode.com/task/shadow-removal