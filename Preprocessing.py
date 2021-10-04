import argparse
import cv2
import glob
from pipeline import pipeline

PIPELINE_ORDER = {
    1: ["white_balancing", "super_res"],
    2: ["super_res", "white_balancing"]
}

# Args
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", type=str, default="Images/*.jpg", help="Image folder path")
ap.add_argument("-s", "--scale", type=float, default=1, help="Image folder path")
ap.add_argument("-o", "--order", type=int, default=1, help="Pipeline Preprocessing Order")

args = vars(ap.parse_args())
path = args["path"]
scale_factor = args["scale"]
preprocess_order = PIPELINE_ORDER[args["order"]]

def load_image(location):
    global scale_factor
    image = cv2.imread(location)
    (H, W) = image.shape[:2]
    width = int(W * scale_factor)
    height = int(H * scale_factor)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)	
    return [image, dim]

def fetch_images():
    global path
    for filename in glob.glob(path): #assuming jpg
        image, dim = load_image(filename)
        run_pipeline(image)

def run_pipeline(image):
    global preprocess_order
    image2 = image
    for elem in preprocess_order:
        image2 = getattr(pipeline, elem)(image2)

    cv2.imshow("1", image)
    cv2.imshow("2", image2)
    cv2.waitKey(0)
fetch_images()