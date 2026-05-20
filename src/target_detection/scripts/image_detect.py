#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo
import tf
import cv2
from cv_bridge import CvBridge

REFER_TARGET_BOX_PIX_AREA = 3.82e5 * 10 ** 2 # 10m深度目标识别框像素面积

class UltralyticsROS:
    def __init__(self):
        # 初始化参数
        self.drone_position_cb_flag = False
        self.last_target_px = None
        self.last_right_target_px = None
        self.height = None
        ROOT = Path(__file__).resolve().parent
        # weights_path = ROOT / "../ultralytics/runs/detect/train/weights/yolo11n.pt" 
        weights_path = ROOT / "../model/ball.pt"
        self.detection_model = None
        self.last_target_px = None

        self.bridge = CvBridge()
        rospy.Subscriber("/mavros/local_position/odom", Odometry, self.position_callback)
        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        self.target_point_pub = rospy.Publisher("/detection/ultralytics/target_point_pub", PointStamped, queue_size=5)      

        # 订阅参数
        self.detection_model = YOLO(weights_path)
        # YOLO预热
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.detection_model(dummy, imgsz=640, conf=0.5)
        rospy.loginfo("YOLO warm-up done")
        while self.height == None:
            pass

        # 算法参数
        self.max_track_dist = self.height/4
        self.target_pixel_size = None

    def position_callback(self, msg: Odometry):
        self.position = msg.pose.pose.position
        self.orientation = np.array([msg.pose.pose.orientation.x,
                                msg.pose.pose.orientation.y,
                                msg.pose.pose.orientation.z,
                                msg.pose.pose.orientation.w])
        self.euler_angles = tf.transformations.euler_from_quaternion(self.orientation, 'sxyz')
        self.linear_velocity = msg.twist.twist.linear
        self.angular_velocity = msg.twist.twist.angular
        self.drone_position_cb_flag = True
    
    def image_callback(self, msg: Image):
        """Callback function to process image and publish annotated images."""
        self.height = msg.height
        self.width = msg.width
        if self.detection_model != None:
            # 使用ros_numpy获取np格式图像-->便于直接输入YOLO
            array = ros_numpy.numpify(msg)
            if array.ndim == 2:
                h, w = array.shape          # h=行数，w=列数
            # 彩色图（BGR 或 RGB）
            elif array.ndim == 3:
                h, w = array.shape[:2]      # 前两个维度就是高、宽
            # print('size =', self.target_pixel_size)
            # print('width  =', w)

            # 发布离中心点最近的识别框中心
            det_result = self.detection_model(array, imgsz=640, conf=0.85)
            cv2.imshow("YOLO", det_result[0].plot())
            cv2.waitKey(1)
            boxes = det_result[0].boxes.xyxy.cpu().numpy()
            classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
            names = [det_result[0].names[i] for i in classes]
            if 'Balloon' not in names:
                rospy.loginfo("No Balloon class in this frame")
                pub_point = PointStamped()
                pub_point.header.stamp = rospy.Time.now()
                pub_point.header.frame_id = "None"
                self.target_point_pub.publish(pub_point)
                return
            names_dict = det_result[0].names
            reverse_map = {v: k for k, v in names_dict.items()}
            car_cls_id = reverse_map['Balloon']
            car_idx = np.where(classes == car_cls_id)[0]

            # car框中心(N, 2)
            car_boxes = boxes[car_idx]
            centers = np.stack([(car_boxes[:, 0] + car_boxes[:, 2]) / 2,
                    (car_boxes[:, 1] + car_boxes[:, 3]) / 2], axis=1) 
            area = abs(car_boxes[:, 0] - car_boxes[:, 2]) * abs(car_boxes[:, 1] - car_boxes[:, 3])
            # 镜头中心
            img_center = np.array([self.width / 2.0, self.height / 2.0])

            if self.last_target_px is None:
                # 第一次：选离镜头中心最近的框中心
                dist_to_center = np.linalg.norm(centers - img_center, axis=1)
                target_idx = np.argmin(dist_to_center)
                self.target_px = centers[target_idx]
            else:
                # 计算与上次目标点的距离
                dist_to_last = np.linalg.norm(centers - self.last_target_px, axis=1)
                within_range = dist_to_last <= self.max_track_dist

                if np.any(within_range):
                    # 在阈值范围内选最近的
                    target_idx = np.argmin(dist_to_last[within_range])
                    target_idx = np.where(within_range)[0][target_idx]
                    self.target_px = centers[target_idx]
                else:
                    # 全部超出阈值，退回到镜头中心最近
                    dist_to_center = np.linalg.norm(centers - img_center, axis=1)
                    target_idx = np.argmin(dist_to_center)
                    self.target_px = centers[target_idx]
            self.target_pixel_size = float(area[target_idx])
            self.last_target_px = self.target_px.copy()
            
            pub_point = PointStamped()
            pub_point.header.stamp = rospy.Time.now()
            pub_point.header.frame_id = "Camera_Optical_Frame"   # 改成你相机的 frame
            pub_point.point.x = float(self.target_px[0])   # x 像素
            pub_point.point.y = float(self.target_px[1])   # y 像素 
            self.target_point_pub.publish(pub_point)


# 加载训练好的模型，改为自己的路径
if __name__ == "__main__":
    rospy.init_node("ultralytics")
    rospy.loginfo("ultralytics init")
    node = UltralyticsROS()
    rospy.spin()
    # print(model.names)
    # detection_model.predict(source, save=True)

