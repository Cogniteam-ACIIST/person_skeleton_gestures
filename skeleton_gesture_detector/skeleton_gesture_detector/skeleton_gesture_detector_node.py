#!/usr/bin/env python3

# sysytem
import os
import numpy as np
import sys
import math


import cv2
import matplotlib.pyplot as plt

# ros
import rclpy
from rclpy.time import Time
from rclpy.clock import Clock
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image,CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge, CvBridgeError
import ament_index_python
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default, qos_profile_services_default
import geometry_msgs

from skeleton_gesture_detector.yolo_skeleton_detector import YoloSkeletonDetector 
   

def euclidean_distance_pose(pose1, pose2):
    """Calculate the Euclidean distance between two poses."""
    return math.sqrt((pose1.position.x - pose2.position.x) ** 2 +
                     (pose1.position.y - pose2.position.y) ** 2)
    
def euclidean_distance_2d(point1, point2):
    if len(point1) != 3 or len(point2) != 3:
        raise ValueError("Both points must be 3D points with x, y, and z coordinates.")

    x1, y1, z1 = point1
    x2, y2, z2 = point2

    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 )
    return distance

class SkeletonGestureDetector(Node):

    def __init__(self):
        super().__init__('skeleton_gesture_detector_node')

        
        self.skeleton_detecotor = YoloSkeletonDetector()
        # publishers
        self.gesture_detected_pub = self.create_publisher(String, 
            "/detected_classes", 10)

        self.cv_image = None
        self.br = CvBridge()

        self.img_subscription = self.create_subscription(
           Image, '/camera/image_raw', 
            self.image_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        self.persons_skeletons_img_pub = self.create_publisher(
            CompressedImage, '/persons_skeletons/compressed',
                qos_profile_sensor_data)        
        
        
    def image_callback(self, msg):
        
        try:          
            
            frame = self.br.compressed_imgmsg_to_cv2(msg, 'bgr8')
            debug_image = frame.copy()        
            skeletons = self.skeleton_detecotor.detect_skeletons(frame)
            
            hand_raised = False
            for skeleton in skeletons:                
                
                head = (int(skeleton[0][0]), int(skeleton[0][1]))
                right =  (int(skeleton[10][0]), int(skeleton[10][1]))
                left =(int(skeleton[9][0]), int(skeleton[9][1]))
                
                # Alternatively, you can check all in one condition
                if head == (0, 0) or right == (0, 0) or left == (0, 0):
                   continue
                str_text = ''
                if left[1] < head[1] or right[1] < head[1]:
                    str_text = 'true'
                    hand_raised = True
                else:
                    print(f'{self.count} false')  
                    
                self.draw_skeleton_2d(skeleton, debug_image)

                cv2.putText(debug_image, str_text, (200,200), 3, 3, (0,255,0), 1, lineType=cv2.LINE_AA)

            
            if hand_raised == True:
                msg = String()
                msg.data = 'hand_up'
                self.gesture_detected_pub.publish(msg)
            
            
            img_msg = self.br.cv2_to_compressed_imgmsg(debug_image)           
            compressed_image_msg = CompressedImage()
            compressed_image_msg.format = "jpeg"  # Set compression format
            compressed_image_msg.data = img_msg.data                
            self.persons_skeletons_img_pub.publish(compressed_image_msg)
           
      
    
            
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting compressed image: {e}")
            return
    
    
    
    
    def draw_skeleton_2d(self, skeleton_2d, image):        
        
        #skeleton edges
        edges = [(0, 2),(0, 1),
                (0, 4),(0, 3),
                (0, 6),(0, 5),
                (14, 16),(13, 15),
                (12, 14),(11, 13), (0, 1), 
                (6, 5),(12, 11),(11, 5),(12, 6), 
                (9, 7), (10, 8), (6, 8), (5, 7)]
        
        
        for edge in edges:
            joint_pix_0 = skeleton_2d[edge[0]]
            joint_pix_1 = skeleton_2d[edge[1]]
            
            if joint_pix_0[0] == 0.0 or joint_pix_0[1] == 0.0 or joint_pix_1[0] == 0.0 or joint_pix_1[1] == 0.0:
                continue
            
            cv2.line(image,(int(joint_pix_0[0]), int(joint_pix_0[1])),
                   (int(joint_pix_1[0]), int(joint_pix_1[1])), (255,255,255), 2)            
            
        # head     
        cv2.circle(image, (int(skeleton_2d[0][0]), int(skeleton_2d[0][1])), radius=15, color=(255,0,0), thickness=-1)
        #right
        cv2.circle(image, (int(skeleton_2d[10][0]), int(skeleton_2d[10][1])), radius=15, color=(0,255,0), thickness=-1)
        #left
        cv2.circle(image, (int(skeleton_2d[9][0]), int(skeleton_2d[9][1])), radius=15, color=(0,0,255), thickness=-1)

    def destroy_node(self):
        # Properly release the camera and close OpenCV windows
        super().destroy_node()
                
def main(args=None):
    rclpy.init(args=args)
    skeleton_gesture_detector = SkeletonGestureDetector()
    rclpy.spin(skeleton_gesture_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()