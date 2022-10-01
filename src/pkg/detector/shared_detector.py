import SharedArray as sa
import numpy as np
import cv2
import time
import os
import sys
import subprocess

DETECTOR_DIR = os.path.join(os.environ["HOME"], "Projects/3D_Object_Detection/")
sys.path.append(os.path.join(os.path.join(DETECTOR_DIR, "src/")))

if sys.version.startswith("3"):
    from pkg.utils.utils_python3 import *
    try:
        from pkg.detector.mmdet.apis import init_detector, inference_detector
    except Exception as e:
        print(TextColors.RED.println("[ERROR] Could not import mmdet"))
        raise(e)
elif sys.version.startswith("2"):
    from pkg.utils.utils import *

from pkg.utils.shared_function import shared_fun, CallType, ArgSpec, ResSpec, set_serving, is_serving, serve_forever, SHARED_FUNC_ALL

IMG_URI = "shm://color_img"
MASK_URI = "shm://mask_img"
REQ_URI = "shm://request"
RESP_URI = "shm://response"

# IMG_DIM = (1080, 1920, 3)

FILE_PATH = os.path.join(DETECTOR_DIR, 'src/pkg/detector/shared_detector.py')

def SharedDetectorGenerator(IMG_DIM=(720, 1280, 3)):
    if __name__ != "__main__":
        output = subprocess.Popen(['python3', FILE_PATH, "--dims", str(IMG_DIM)], cwd=os.path.dirname(FILE_PATH))
    class SharedDetector:
        def __init__(self):
            self.initizlied = False

        @shared_fun(CallType.SYNC, "SharedDetector")
        def init(self):
            if not self.initizlied:
                self.initizlied = True
                # Load config, checkpoint file of cascade mask rcnn swin based
                config_file = os.path.join(DETECTOR_DIR,
                                           'model/mmdet/configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py')
                checkpoint_file = os.path.join(DETECTOR_DIR,
                                               'model/mmdet/cascade_mask_rcnn_swin_base_patch4_window7.pth')

                device = 'cuda:0'
                try:
                    self.model = init_detector(config_file, checkpoint_file, device=device)
                except Exception as e:
                    TextColors.RED.println("[ERROR] Could not initialize detector")
                    raise(e)

        @shared_fun(CallType.SYNC, "SharedDetector",
                    ArgSpec("color_img", IMG_DIM, np.uint8),
                    ResSpec(0, (80, IMG_DIM[0],IMG_DIM[1]), np.uint8))
        def inference(self, color_img):
            if self.initizlied:
                result = inference_detector(self.model, color_img)
                boxes, masks = result[0], result[1]

                detect_false = np.zeros((IMG_DIM[0],IMG_DIM[1]), np.uint8)

                return_img = np.zeros((80, IMG_DIM[0],IMG_DIM[1]), np.uint8)

                for idx in range(80):
                    if len(masks[idx]) != 0:
                        mask_res = np.empty((IMG_DIM[0], IMG_DIM[1]), np.uint8)
                        mask_res[:,:] = False
                        mask_idx = 1
                        for i_t in range(len(masks[idx])):
                            mask_temp = masks[idx][i_t]
                            h_pixels, w_pixels = np.where(mask_temp==True)
                            h_mean = np.mean(h_pixels)
                            w_mean = np.mean(w_pixels)
                            if w_mean < IMG_DIM[1]/10 or w_mean > IMG_DIM[1] - IMG_DIM[1]/10:
                                print("Detected masks out of width range")
                            elif h_mean < IMG_DIM[0]/7 or h_mean > IMG_DIM[0] - IMG_DIM[0]/7:
                                print("Detected masks out of height range")
                            else:
                                mask_res[np.where(mask_temp==True)] = int(mask_idx)
                                mask_idx += 1
                        return_img[idx][:,:] = mask_res
                    else:
                        return_img[idx][:,:] = detect_false

                return return_img
            else:
                raise(RuntimeError("[Error] Not initilized"))
    return SharedDetector

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='SharedDetector - usage: python3 shared_detector --dims=(height,width,3)')
    parser.add_argument('--dims', type=str, required=True,
                        help='size of network input in form of (height,width,3)')
    args = parser.parse_args()

    img_dims = tuple(map(int, args.dims[1:-1].split(",")))
    sdet = SharedDetectorGenerator(img_dims)()
    set_serving(True)
    serve_forever("SharedDetector", [sdet.inference, sdet.init], verbose=True)