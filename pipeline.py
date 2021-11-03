import cv2
import numpy as np
from ISR.models import RDN, RRDN
from medpy.filter.smoothing import anisotropic_diffusion
from deshadower import *

test_w,test_h = 640,480 
shadow_model = './Models/srdplus-pretrained'
vgg_19_path = './Models/VGG/imagenet-vgg-verydeep-19.mat'
use_gpu = 0
is_hyper = 1
deshadower = Deshadower(shadow_model, vgg_19_path, use_gpu, is_hyper)
isr_model = RDN(weights='noise-cancel')

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
        sr_img = isr_model.predict(np.array(image))
        return sr_img
        
    def anisotropic(image):
        ani_image = anisotropic_diffusion(image)
        return ani_image.astype('uint8')

    def deshadow(image):
        test_w = int(image.shape[1]* test_h/float(image.shape[0]))
        if test_w >0 and test_h > 0:
            image = cv2.resize( np.float32(image), (test_w, test_h), cv2.INTER_CUBIC)
        image = image/255.0  
        image_no_shadow = deshadower.run(image)
        return image_no_shadow