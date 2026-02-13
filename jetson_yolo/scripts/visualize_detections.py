#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge

class EdgeVisualizer:
    def __init__(self):
        self.bridge = CvBridge()
        self.latest_detections = None
        
        # subscribe to the image and the detections
        rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.image_callback)
        rospy.Subscriber("/fruit_detections", Detection2DArray, self.det_callback)
        
        rospy.loginfo("Visualizer online")

    def det_callback(self, msg):
        self.latest_detections = msg

    def image_callback(self, msg):
        # image decompression
        np_arr = np.fromstring(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if self.latest_detections:
            for det in self.latest_detections.detections:
                #bounding corners
                w, h = int(det.bbox.size_x), int(det.bbox.size_y)
                cx, cy = int(det.bbox.center.x), int(det.bbox.center.y)
                x1, y1 = cx - w//2, cy - h//2
                x2, y2 = cx + w//2, cy + h//2

                # ID extraction: first from the 'detections' list
                if len(det.results) > 0:
                    class_id = det.results[0].id
                    score = det.results[0].score
                    # print ID and confidence score
                    label = "ID: {} ({:.2f})".format(class_id, score)
                else:
                    label = "No ID"

                # bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("RB-Kairos Edge Vision", frame)
        cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('laptop_visualizer')
    viz = EdgeVisualizer()
    rospy.spin()
