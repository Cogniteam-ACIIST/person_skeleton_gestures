from typing import List, Tuple

import ament_index_python
package_name = 'skeleton_gesture_detector'
package_share_directory = ament_index_python.get_package_share_directory(package_name)

import sys
sys.path.append('/home/jetson/skeleton_ws/src/skeleton_gesture_detector/skeleton_gesture_detector/yolo7/')
# yolo7
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts, get_person_skeleton
import torch
import cv2
import numpy as np
import time




class YoloSkeletonDetector():
    def __init__(self):        
        super().__init__()
        
        if torch.cuda.is_available():
            print(f"torch.cuda.device_count() {torch.cuda.device_count()}")        
        self.gpu_device = torch.device("cuda:0")#if torch.cuda.is_available() else "cpu")        
        self.model_path = None
        self.yolomodel =  None
        skeleton_model_path = f"{package_share_directory}/resource/yolov7-w6-pose.pt"

        self.set_model(skeleton_model_path)
        
    def set_model(self, model_path: str):
        try:
            self.model_path = model_path
            print(f"Loading the model...{self.model_path}")
            self.model, self.yolomodel = self.load_yolov7_model(yolomodel=self.model_path)
        except Exception as e: 
            print(f"{e}")
            exit(0)
    
    def get_type(self):
        return 'yolo'
    
    def detect_skeletons(self, image) -> List[List[Tuple[int, int]]]:        
        
        skeletons = [] 
        
        # start_time = time.time() * 1000

        confidence=0.6#0.25 
        threshold=0.65
        yolo_output, new_image = self.running_inference(image)
         
        
            
        # end_time = time.time() * 1000
        # elapsed_time = end_time - start_time
        # print("111111111111111 Elapsed time:", elapsed_time, "milliseconds")        
        
        # get yolo skeletons output
        output = non_max_suppression_kpt(
            yolo_output,
            confidence,  # Confidence Threshold
            threshold,  # IoU Threshold
            nc=self.model.yaml['nc'],  # Number of Classes
            nkpt=self.model.yaml['nkpt'],  # Number of Keypoints
            kpt_label=True)

        with torch.no_grad():
            output = output_to_keypoint(output)

        nimg = new_image[0].permute(1, 2, 0) * 255
        nimg = cv2.cvtColor(nimg.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)        

        # calculate hri persons
        num_of_persons = output.shape[0]
        for person_id in range(num_of_persons):           

            skeleton_2d = get_person_skeleton(output[person_id, 7:].T, 3, image,  nimg)
        
            skeletons.append(skeleton_2d)
            
        return skeletons

    def load_yolov7_model(self, yolomodel):        

        #print("Loading model:", yolomodel)
        model = torch.load(yolomodel, map_location=self.gpu_device)['model']
        model.float().eval()

        if torch.cuda.is_available():
            # half() turns predictions into float16 tensors
            # which significantly lowers inference time
            model.half().to(self.gpu_device)

        return model, yolomodel
    
    
    def running_inference(self, image):
    
        image = letterbox(image, 960, 
                        stride=64,
                        auto=True)[0]  # shape: (567, 960, 3)
        image = transforms.ToTensor()(image)  # torch.Size([3, 567, 960])        
        
        if torch.cuda.is_available():
            image = image.half().to(self.gpu_device)

        new_image = image.unsqueeze(0)  # torch.Size([1, 3, 567, 960])

        with torch.no_grad():
            output, _ = self.model(new_image)        
                
        return output, new_image 