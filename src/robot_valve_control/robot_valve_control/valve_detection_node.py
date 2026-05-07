#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import torch
import yaml
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# 建议把 msg 文件命名为 ValveCommand.msg，然后这样导入
from valve_interfaces.msg import ValveCommand, ValveVision

# 如果 judge_proper 在 utility.py 里，就改成：from utility import judge_proper
#from utility import judge_proper
from . import dev_angle

class ValveDetectionNode(Node):
    """
    只负责：
    1. 接收 RGB + Depth
    2. YOLO 检测
    3. 计算 3D 坐标
    4. 判断运动类型和旋转校正
    5. 发布 ValveCommand

    不在这里直接控制机械臂。
    """

    def __init__(self):
        super().__init__('valve_detection_node')

        self.declare_parameter('model_path', '/home/jetson/ultralytics_robot/best.engine')
        self.declare_parameter('camera_info_yaml', '/home/jetson/yolov5/ost.yaml')
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('command_topic', '/valve/command')
        self.declare_parameter('vision_topic', '/valve/vision')
        self.declare_parameter('show_image', True)

        model_path = self.get_parameter('model_path').value
        camera_info_yaml = self.get_parameter('camera_info_yaml').value
        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        command_topic = self.get_parameter('command_topic').value
        vision_topic = self.get_parameter('vision_topic').value

        self.model = YOLO(model_path, task='detect')
        self.names = self.model.names
        self.img_size = 640
        self.bridge = CvBridge()

        self.depth_image = None
        self.camera_info = self.load_camera_info(camera_info_yaml)

        self.command_pub = self.create_publisher(ValveCommand, command_topic, 10)
        self.vision_pub = self.create_publisher(ValveVision, vision_topic, 10)

        self.create_subscription(Image, rgb_topic, self.image_callback, 1)
        self.create_subscription(Image, depth_topic, self.depth_callback, 1)

        self.get_logger().info('ValveDetectionNode started. Publishing command and vision topics; robot motion is separated.')

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'深度图解析失败: {e}')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            if self.depth_image is None or self.camera_info is None:
                self.get_logger().warn('等待深度图和相机内参...')
                if self.get_parameter('show_image').value:
                    cv2.imshow('YOLOv11 Detection', frame)
                    cv2.waitKey(1)
                return

            det = self.run_yolo(frame)
            if det is None or len(det) == 0:
                self.publish_vision(frame)
                if self.get_parameter('show_image').value:
                    cv2.imshow('YOLOv11 Detection', frame)
                    cv2.waitKey(1)
                return

            valve_target = self.select_target(det, class_id=0, frame=frame)
            small_target = self.select_target(det, class_id=1, frame=frame)

            command = self.decide_command(frame, valve_target, small_target)
            if command is not None and command.valid:
                self.command_pub.publish(command)

            self.publish_vision(frame)

            if self.get_parameter('show_image').value:
                cv2.imshow('YOLOv11 Detection', frame)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'图像处理出错: {e}')

    def publish_vision(self, processed_bgr_frame):
        if self.depth_image is None:
            return

        try:
            msg = ValveVision()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_color_optical_frame'
            msg.rgb_image = self.bridge.cv2_to_imgmsg(processed_bgr_frame, encoding='bgr8')
            msg.depth_image = self.bridge.cv2_to_imgmsg(self.depth_image)
            msg.rgb_image.header = msg.header
            msg.depth_image.header = msg.header
            self.vision_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'图像消息发布失败: {e}')

    def judge_proper(self, roi):
        """
        判断目标（如阀门）是否对正：通过dev_angle模块计算角度偏差
        参数：
            roi: 目标区域图像（ROI，OpenCV格式）
        返回：
            (is_proper, offset_deg): 元组，is_proper为是否对正（布尔值），offset_deg为角度偏差（度）
        """
        try:
            # 生成目标掩码（如阀门区域掩码）
            mask = dev_angle.valve_mask(roi)
            # 计算目标主方向角度（如阀门十字的角度）
            angle_deg, center = dev_angle.dominant_cross_angle(mask)
            # 计算与正方向的偏差角度
            offset_deg = float(dev_angle.angle_offset_from_upright(angle_deg))
            # 偏差小于10度视为对正
            return (abs(offset_deg) < 10.0), offset_deg
        except Exception as e:
            self.get_logger().warn(f"judge_proper 计算失败: {e}")
            return False, 0.0  # 异常时返回默认值        

    def run_yolo(self, frame):
        results = self.model(frame, imgsz=self.img_size, conf=0.65, iou=0.45, verbose=False)
        if not results:
            return torch.empty((0, 6))

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return torch.empty((0, 6))

        return torch.cat((boxes.xyxy, boxes.conf.view(-1, 1), boxes.cls.view(-1, 1)), dim=1)

    def select_target(self, det, class_id, frame):
        class_det = det[det[:, 5] == class_id]
        if len(class_det) == 0:
            return None

        idx = torch.argmax(class_det[:, 4]).item()
        x1, y1, x2, y2, conf, cls = class_det[idx].tolist()
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cls = int(cls)

        label = f'{self.names[cls]} {conf:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        xyz = self.pixel_to_3d(u, v, self.depth_image, self.camera_info)
        if xyz is None:
            return None

        return {
            'xyz': xyz,              # camera coordinate, meter: X right, Y down, Z forward
            'bbox': (x1, y1, x2, y2),
            'confidence': float(conf),
            'class_id': cls,
            'is_small': cls == 1,
        }

    def decide_command(self, frame, valve_target, small_target):
        """
        这里集中放“识别判断运动部分”。
        输出的是给机械臂节点看的高层运动指令，而不是直接 Move.LOffset。
        """
        if valve_target is not None:
            xyz = valve_target['xyz']
            x_mm = 1000.0 * xyz[0]
            y_mm = 1000.0 * xyz[2]      # 机器人前后方向，沿用你原来 ly=Z_mm 的逻辑
            z_mm = -1000.0 * xyz[1]     # 机器人上下方向，沿用你原来 lz=-Y 的逻辑
            z_depth_mm = 1000.0 * xyz[2]

            if z_depth_mm > 260:
                if abs(z_depth_mm) >= 400.0:
                    return self.build_command(
                        x=x_mm,
                        y=z_depth_mm - 350.0,
                        z=z_mm,
                        is_small=False,
                        motion_type='far_move',
                        need_rotation=False,
                        rotation_deg=0.0,
                        target=valve_target,
                    )

                if abs(x_mm) > 5.0 or abs(y_mm) > 5.0:
                    return self.build_command(
                        x=x_mm,
                        y=0.0,
                        z=z_mm,
                        is_small=False,
                        motion_type='no_ahead_check',
                        need_rotation=False,
                        rotation_deg=0.0,
                        target=valve_target,
                    )

                is_proper, offset_deg = self.estimate_rotation(frame, valve_target['bbox'])
                return self.build_command(
                    x=x_mm,
                    y=z_depth_mm - 250.0,
                    z=z_mm,
                    is_small=False,
                    motion_type='check_and_spin',
                    need_rotation=not is_proper,
                    rotation_deg=float(offset_deg),
                    target=valve_target,
                )

        if small_target is not None:
            xyz = small_target['xyz']
            x_mm = 1000.0 * xyz[0]
            y_mm = 1000.0 * xyz[2]
            z_mm = -1000.0 * xyz[1]
            z_depth_mm = 1000.0 * xyz[2]

            if z_depth_mm < 260:
                # 注意：这里建议用 abs，否则负方向偏差也会误判为满足条件
                if abs(x_mm) < 2.0 and abs(y_mm) < 2.0:
                    motion_type = 'small_move'
                    y_cmd = z_depth_mm
                else:
                    motion_type = 'small_no_head'
                    y_cmd = 0.0

                return self.build_command(
                    x=x_mm,
                    y=y_cmd,
                    z=z_mm,
                    is_small=True,
                    motion_type=motion_type,
                    need_rotation=False,
                    rotation_deg=0.0,
                    target=small_target,
                )

        return None

    def estimate_rotation(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1c, x2c = sorted((max(0, x1), min(w, x2)))
        y1c, y2c = sorted((max(0, y1), min(h, y2)))

        if x2c - x1c <= 1 or y2c - y1c <= 1:
            return True, 0.0

        roi = frame[y1c:y2c, x1c:x2c]
        return dev_angle.judge_proper(roi)

    def build_command(self, x, y, z, is_small, motion_type, need_rotation, rotation_deg, target):
        msg = ValveCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_color_optical_frame'

        msg.x = float(x)
        msg.y = float(y)
        msg.z = float(z)
        msg.valid = True
        msg.is_small = bool(is_small)
        msg.motion_type = motion_type
        msg.need_rotation_correction = bool(need_rotation)
        msg.rotation_correction_deg = float(rotation_deg)
        msg.confidence = float(target['confidence'])
        msg.class_id = int(target['class_id'])
        return msg

    def pixel_to_3d(self, u, v, depth_img, cam_info_dict):
        fx = cam_info_dict['camera_matrix']['data'][0]
        fy = cam_info_dict['camera_matrix']['data'][4]
        cx = cam_info_dict['camera_matrix']['data'][2]
        cy = cam_info_dict['camera_matrix']['data'][5]

        if v >= depth_img.shape[0] or u >= depth_img.shape[1] or v < 0 or u < 0:
            return None

        d = depth_img[int(v), int(u)]
        if d == 0:
            return None

        d = float(d) / 1000.0
        x = (u - cx) * d / fx
        y = (v - cy) * d / fy
        z = d
        return x, y, z

    def load_camera_info(self, yaml_file):
        with open(yaml_file, 'r') as f:
            cam_info = yaml.safe_load(f)
        self.get_logger().info(f'Loaded camera info from: {yaml_file}')
        return cam_info


def main(args=None):
    rclpy.init(args=args)
    node = ValveDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
