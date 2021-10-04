import cv2
import numpy as np

class pipeline:
    def white_balancing(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.mean(image[:, :, 1])
        avg_b = np.mean(image[:, :, 2])
        # Adjusting LAB values
        image[:, :, 1] = image[:, :, 1] - ((avg_a - 128) * (image[:, :, 0] / 255.0) * 1.1)
        image[:, :, 2] = image[:, :, 2] - ((avg_b - 128) * (image[:, :, 0] / 255.0) * 1.1)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        return image

    def super_res(image):
        return image