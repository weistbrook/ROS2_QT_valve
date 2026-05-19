#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np
import torch
import yaml
from collections import deque
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
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
        self.declare_parameter('point_cloud_topic', '/camera/depth/points')
        self.declare_parameter('command_topic', '/valve/command')
        self.declare_parameter('vision_topic', '/valve/vision')
        self.declare_parameter('show_image', True)
        self.declare_parameter('plane_min_points', 80)
        self.declare_parameter('plane_max_points', 2500)
        self.declare_parameter('plane_ransac_iterations', 80)
        self.declare_parameter('plane_ransac_threshold_m', 0.008)
        self.declare_parameter('plane_depth_window_m', 0.08)
        self.declare_parameter('plane_roi_shrink_ratio', 0.08)
        self.declare_parameter('plane_debug', True)
        self.declare_parameter('plane_debug_interval_sec', 1.0)
        self.declare_parameter('target_depth_roi_ratio', 0.08)
        self.declare_parameter('target_depth_min_roi_px', 9)
        self.declare_parameter('target_depth_min_points', 12)
        self.declare_parameter('target_depth_window_m', 0.06)
        self.declare_parameter('pose_smoothing_window', 5)
        self.declare_parameter('plane_smoothing_window', 5)

        model_path = self.get_parameter('model_path').value
        camera_info_yaml = self.get_parameter('camera_info_yaml').value
        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        point_cloud_topic = self.get_parameter('point_cloud_topic').value
        command_topic = self.get_parameter('command_topic').value
        vision_topic = self.get_parameter('vision_topic').value

        self.model = YOLO(model_path, task='detect')
        self.names = self.model.names
        self.img_size = 640
        self.bridge = CvBridge()

        self.depth_image = None
        self.point_cloud = None
        self.camera_info = self.load_camera_info(camera_info_yaml)
        self.last_plane_debug_log_time = 0.0
        self.position_history = {0: deque(), 1: deque()}
        self.plane_pose_history = deque()

        self.command_pub = self.create_publisher(ValveCommand, command_topic, 10)
        self.vision_pub = self.create_publisher(ValveVision, vision_topic, 10)

        self.create_subscription(Image, rgb_topic, self.image_callback, 1)
        self.create_subscription(Image, depth_topic, self.depth_callback, 1)
        self.create_subscription(PointCloud2, point_cloud_topic, self.point_cloud_callback, 1)

        self.get_logger().info('ValveDetectionNode started. Publishing command and vision topics; robot motion is separated.')

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'深度图解析失败: {e}')

    def point_cloud_callback(self, msg):
        self.point_cloud = msg

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
                self.reset_smoothing()
                self.publish_none_command()
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
            else:
                self.publish_none_command()

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

    def publish_none_command(self):
        self.reset_smoothing()
        msg = ValveCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_color_optical_frame'
        msg.valid = False
        msg.is_small = False
        msg.motion_type = 'none'
        msg.need_rotation_correction = False
        msg.rotation_correction_deg = 0.0
        msg.plane_valid = False
        msg.valve_yaw_deg = 0.0
        msg.valve_pitch_deg = 0.0
        msg.plane_normal_x = 0.0
        msg.plane_normal_y = 0.0
        msg.plane_normal_z = 0.0
        msg.plane_inlier_count = 0
        msg.plane_inlier_ratio = 0.0
        msg.confidence = 0.0
        msg.class_id = -1
        self.command_pub.publish(msg)

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

        xyz = self.bbox_to_3d((x1, y1, x2, y2), frame.shape[:2], self.depth_image, self.camera_info)
        if xyz is None:
            return None
        xyz = self.smooth_xyz(cls, xyz)

        plane_pose = None
        if cls == 0:
            plane_pose = self.estimate_plane_pose(frame, (x1, y1, x2, y2))
            if plane_pose is not None:
                plane_pose = self.smooth_plane_pose(plane_pose)
                cv2.putText(
                    frame,
                    f"yaw {plane_pose['yaw_deg']:.1f} pitch {plane_pose['pitch_deg']:.1f} avg",
                    (x1, min(frame.shape[0] - 8, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

        return {
            'xyz': xyz,              # camera coordinate, meter: X right, Y down, Z forward
            'bbox': (x1, y1, x2, y2),
            'confidence': float(conf),
            'class_id': cls,
            'is_small': cls == 1,
            'plane_pose': plane_pose,
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

    def reset_smoothing(self):
        for history in self.position_history.values():
            history.clear()
        self.plane_pose_history.clear()

    def smooth_xyz(self, class_id, xyz):
        window = max(1, int(self.get_parameter('pose_smoothing_window').value))
        history = self.position_history.setdefault(class_id, deque())
        history.append(np.asarray(xyz, dtype=np.float32))
        while len(history) > window:
            history.popleft()

        smoothed = np.mean(np.stack(list(history), axis=0), axis=0)
        return float(smoothed[0]), float(smoothed[1]), float(smoothed[2])

    def smooth_plane_pose(self, plane_pose):
        window = max(1, int(self.get_parameter('plane_smoothing_window').value))
        self.plane_pose_history.append({
            'normal': np.asarray(plane_pose['normal'], dtype=np.float32),
            'yaw_deg': float(plane_pose['yaw_deg']),
            'pitch_deg': float(plane_pose['pitch_deg']),
            'inlier_count': int(plane_pose['inlier_count']),
            'inlier_ratio': float(plane_pose['inlier_ratio']),
        })
        while len(self.plane_pose_history) > window:
            self.plane_pose_history.popleft()

        poses = list(self.plane_pose_history)
        normal = np.mean(np.stack([pose['normal'] for pose in poses], axis=0), axis=0)
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 1e-6:
            normal = normal / normal_norm
            if normal[2] < 0.0:
                normal = -normal
            yaw_deg, pitch_deg = self.normal_to_yaw_pitch(normal)
        else:
            normal = np.asarray(plane_pose['normal'], dtype=np.float32)
            yaw_deg = float(np.mean([pose['yaw_deg'] for pose in poses]))
            pitch_deg = float(np.mean([pose['pitch_deg'] for pose in poses]))

        return {
            'normal': normal,
            'yaw_deg': float(yaw_deg),
            'pitch_deg': float(pitch_deg),
            'inlier_count': int(round(np.mean([pose['inlier_count'] for pose in poses]))),
            'inlier_ratio': float(np.mean([pose['inlier_ratio'] for pose in poses])),
        }

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
        plane_pose = target.get('plane_pose')
        msg.plane_valid = plane_pose is not None
        if plane_pose is not None:
            normal = plane_pose['normal']
            msg.valve_yaw_deg = float(plane_pose['yaw_deg'])
            msg.valve_pitch_deg = float(plane_pose['pitch_deg'])
            msg.plane_normal_x = float(normal[0])
            msg.plane_normal_y = float(normal[1])
            msg.plane_normal_z = float(normal[2])
            msg.plane_inlier_count = int(plane_pose['inlier_count'])
            msg.plane_inlier_ratio = float(plane_pose['inlier_ratio'])
        else:
            msg.valve_yaw_deg = 0.0
            msg.valve_pitch_deg = 0.0
            msg.plane_normal_x = 0.0
            msg.plane_normal_y = 0.0
            msg.plane_normal_z = 0.0
            msg.plane_inlier_count = 0
            msg.plane_inlier_ratio = 0.0
        msg.confidence = float(target['confidence'])
        msg.class_id = int(target['class_id'])
        return msg

    def log_plane_debug(self, parts):
        if not bool(self.get_parameter('plane_debug').value):
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        interval_sec = max(0.0, float(self.get_parameter('plane_debug_interval_sec').value))
        if interval_sec > 0.0 and now_sec - self.last_plane_debug_log_time < interval_sec:
            return

        self.last_plane_debug_log_time = now_sec
        self.get_logger().info('[PlaneDebug] ' + ' | '.join(str(part) for part in parts))

    def estimate_plane_pose(self, frame, bbox):
        min_points = int(self.get_parameter('plane_min_points').value)
        debug = [f"bbox={tuple(int(v) for v in bbox)}"]

        points = None
        source = 'none'
        if self.point_cloud is not None:
            points, reason = self.points_in_bbox_from_cloud(frame, bbox, self.point_cloud)
            if points is not None and len(points) >= min_points:
                source = 'point_cloud'
                debug.append(f"point_cloud: used raw_points={len(points)}")
            else:
                count = 0 if points is None else len(points)
                debug.append(f"point_cloud: not used ({reason}, raw_points={count}, min_points={min_points})")
        else:
            debug.append("point_cloud: not used (no PointCloud2 message received)")

        if points is None or len(points) < min_points:
            points, reason = self.points_in_bbox_from_depth(frame, bbox, self.depth_image, self.camera_info)
            if points is not None and len(points) >= min_points:
                source = 'depth_image'
                debug.append(f"depth_image: used raw_points={len(points)}")
            else:
                count = 0 if points is None else len(points)
                debug.append(f"depth_image: not used ({reason}, raw_points={count}, min_points={min_points})")

        if points is None or len(points) < min_points:
            self.log_plane_debug(debug + ["result: failed before depth filtering"])
            return None

        raw_count = len(points)
        points = self.filter_points_by_depth(points)
        debug.append(f"depth_filter: {raw_count}->{len(points)} points")
        if len(points) < min_points:
            self.log_plane_debug(debug + ["result: failed after depth filtering"])
            return None

        fit = self.fit_plane_ransac(points)
        if fit is None:
            self.log_plane_debug(debug + ["result: failed in RANSAC plane fitting"])
            return None

        normal, inlier_count, inlier_ratio = fit
        yaw_deg, pitch_deg = self.normal_to_yaw_pitch(normal)

        self.log_plane_debug(debug + [
            f"result: success source={source}",
            f"yaw={yaw_deg:.2f} deg pitch={pitch_deg:.2f} deg",
            f"normal=({normal[0]:.3f},{normal[1]:.3f},{normal[2]:.3f}), "
            f"inliers={inlier_count}/{len(points)} ratio={inlier_ratio:.2f}",
        ])

        return {
            'normal': normal,
            'yaw_deg': yaw_deg,
            'pitch_deg': pitch_deg,
            'inlier_count': inlier_count,
            'inlier_ratio': inlier_ratio,
        }

    def points_in_bbox_from_cloud(self, frame, bbox, cloud_msg):
        if cloud_msg.height <= 1:
            return None, f"unorganized cloud height={cloud_msg.height}"

        x1, y1, x2, y2 = self.clipped_shrunk_bbox(frame.shape[:2], bbox)
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            return None, f"invalid shrunk bbox=({x1},{y1},{x2},{y2})"

        roi = frame[y1:y2, x1:x2]
        pixel_mask = self.valve_roi_mask(roi)
        ys, xs = np.nonzero(pixel_mask)
        if len(xs) == 0:
            return None, "ROI mask has no pixels"

        max_points = int(self.get_parameter('plane_max_points').value)
        mask_count = len(xs)
        if len(xs) > max_points:
            idx = np.linspace(0, len(xs) - 1, max_points, dtype=np.int32)
            xs = xs[idx]
            ys = ys[idx]

        frame_h, frame_w = frame.shape[:2]
        scale_x = float(cloud_msg.width) / float(frame_w)
        scale_y = float(cloud_msg.height) / float(frame_h)
        uvs = []
        for x, y in zip(xs, ys):
            u = int(round(float(x1 + x) * scale_x))
            v = int(round(float(y1 + y) * scale_y))
            if 0 <= u < cloud_msg.width and 0 <= v < cloud_msg.height:
                uvs.append((u, v))
        if not uvs:
            return None, "all ROI pixels mapped outside point cloud"

        try:
            cloud_points = pc2.read_points(
                cloud_msg,
                field_names=('x', 'y', 'z'),
                skip_nans=True,
                uvs=uvs,
            )
            points = self.point_cloud_iter_to_array(cloud_points)
        except Exception as e:
            return None, f"PointCloud2 ROI read failed: {e}"

        if points.size == 0:
            return None, f"read_points returned no finite xyz, mask_pixels={mask_count}, sampled_uvs={len(uvs)}"

        finite = np.isfinite(points).all(axis=1)
        forward = points[:, 2] > 0.0
        points = points[finite & forward]
        if points.size == 0:
            return None, f"no valid forward xyz after filtering, mask_pixels={mask_count}, sampled_uvs={len(uvs)}"
        return points, f"ok mask_pixels={mask_count}, sampled_uvs={len(uvs)}"

    def points_in_bbox_from_depth(self, frame, bbox, depth_img, cam_info_dict):
        if depth_img is None or cam_info_dict is None:
            return None, "missing depth image or camera info"

        x1, y1, x2, y2 = self.clipped_shrunk_bbox(frame.shape[:2], bbox)
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            return None, f"invalid shrunk bbox=({x1},{y1},{x2},{y2})"

        roi = frame[y1:y2, x1:x2]
        pixel_mask = self.valve_roi_mask(roi)
        ys, xs = np.nonzero(pixel_mask)
        if len(xs) == 0:
            return None, "ROI mask has no pixels"

        max_points = int(self.get_parameter('plane_max_points').value)
        mask_count = len(xs)
        if len(xs) > max_points:
            idx = np.linspace(0, len(xs) - 1, max_points, dtype=np.int32)
            xs = xs[idx]
            ys = ys[idx]

        fx = float(cam_info_dict['camera_matrix']['data'][0])
        fy = float(cam_info_dict['camera_matrix']['data'][4])
        cx = float(cam_info_dict['camera_matrix']['data'][2])
        cy = float(cam_info_dict['camera_matrix']['data'][5])

        depth_h, depth_w = depth_img.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        scale_x = float(depth_w) / float(frame_w)
        scale_y = float(depth_h) / float(frame_h)

        points = []
        for x, y in zip(xs, ys):
            u_color = float(x1 + x)
            v_color = float(y1 + y)
            u_depth = int(round(u_color * scale_x))
            v_depth = int(round(v_color * scale_y))

            if u_depth < 0 or u_depth >= depth_w or v_depth < 0 or v_depth >= depth_h:
                continue

            d = self.depth_value_to_meters(depth_img[v_depth, u_depth])
            if d is None:
                continue

            u_cam = u_color if depth_w == frame_w else u_depth
            v_cam = v_color if depth_h == frame_h else v_depth
            x_m = (u_cam - cx) * d / fx
            y_m = (v_cam - cy) * d / fy
            points.append((x_m, y_m, d))

        if not points:
            return None, f"no valid depth samples, mask_pixels={mask_count}, sampled_pixels={len(xs)}"

        points = np.asarray(points, dtype=np.float32)
        finite = np.isfinite(points).all(axis=1)
        forward = points[:, 2] > 0.0
        points = points[finite & forward]
        if points.size == 0:
            return None, f"no valid forward depth points, mask_pixels={mask_count}, sampled_pixels={len(xs)}"
        return points, f"ok mask_pixels={mask_count}, sampled_pixels={len(xs)}"

    def clipped_shrunk_bbox(self, image_shape, bbox):
        h, w = image_shape
        x1, y1, x2, y2 = bbox
        x1c, x2c = sorted((max(0, int(x1)), min(w, int(x2))))
        y1c, y2c = sorted((max(0, int(y1)), min(h, int(y2))))

        shrink_ratio = float(self.get_parameter('plane_roi_shrink_ratio').value)
        shrink_ratio = min(max(shrink_ratio, 0.0), 0.45)
        dx = int((x2c - x1c) * shrink_ratio)
        dy = int((y2c - y1c) * shrink_ratio)
        return x1c + dx, y1c + dy, x2c - dx, y2c - dy

    def valve_roi_mask(self, roi):
        try:
            mask = dev_angle.valve_mask(roi)
            mask = mask > 0
            if np.count_nonzero(mask) >= int(self.get_parameter('plane_min_points').value):
                return mask
        except Exception:
            pass

        h, w = roi.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        rx = max(w * 0.38, 1.0)
        ry = max(h * 0.38, 1.0)
        return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0

    def point_cloud_iter_to_array(self, cloud_points):
        if isinstance(cloud_points, np.ndarray):
            if cloud_points.dtype.names:
                return np.column_stack([
                    cloud_points['x'],
                    cloud_points['y'],
                    cloud_points['z'],
                ]).astype(np.float32, copy=False)
            return np.asarray(cloud_points[:, :3], dtype=np.float32)

        rows = []
        for point in cloud_points:
            rows.append((float(point[0]), float(point[1]), float(point[2])))
        if not rows:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def filter_points_by_depth(self, points):
        z = points[:, 2]
        median_z = float(np.median(z))
        mad = float(np.median(np.abs(z - median_z)))
        configured_window = float(self.get_parameter('plane_depth_window_m').value)
        adaptive_window = max(configured_window, 3.0 * mad)
        keep = np.abs(z - median_z) <= adaptive_window
        return points[keep]

    def fit_plane_ransac(self, points):
        min_points = int(self.get_parameter('plane_min_points').value)
        if len(points) < min_points:
            return None

        iterations = int(self.get_parameter('plane_ransac_iterations').value)
        threshold = float(self.get_parameter('plane_ransac_threshold_m').value)
        rng = np.random.default_rng()
        best_inliers = None
        best_count = 0

        for _ in range(iterations):
            sample_idx = rng.choice(len(points), size=3, replace=False)
            p0, p1, p2 = points[sample_idx]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm
            distances = np.abs((points - p0) @ normal)
            inliers = distances < threshold
            count = int(np.count_nonzero(inliers))
            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_inliers is None or best_count < min_points:
            return None

        inlier_points = points[best_inliers]
        centroid = np.mean(inlier_points, axis=0)
        _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
        normal = vh[-1]
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            return None

        normal = normal / normal_norm
        if normal[2] < 0.0:
            normal = -normal

        return normal, best_count, float(best_count) / float(len(points))

    def normal_to_yaw_pitch(self, normal):
        nx, ny, nz = normal
        yaw_deg = math.degrees(math.atan2(nx, nz))
        pitch_deg = math.degrees(math.atan2(-ny, math.sqrt(nx * nx + nz * nz)))
        return yaw_deg, pitch_deg

    def depth_value_to_meters(self, value):
        if isinstance(value, np.ndarray):
            value = value.item()

        d = float(value)
        if not math.isfinite(d) or d <= 0.0:
            return None

        if np.issubdtype(self.depth_image.dtype, np.floating):
            return d
        return d / 1000.0

    def bbox_to_3d(self, bbox, image_shape, depth_img, cam_info_dict):
        if depth_img is None or cam_info_dict is None:
            return None

        h, w = image_shape
        x1, y1, x2, y2 = bbox
        x1c, x2c = sorted((max(0, int(x1)), min(w, int(x2))))
        y1c, y2c = sorted((max(0, int(y1)), min(h, int(y2))))
        if x2c - x1c <= 1 or y2c - y1c <= 1:
            return None

        cx_color = 0.5 * (x1c + x2c)
        cy_color = 0.5 * (y1c + y2c)
        bbox_size = max(1.0, min(float(x2c - x1c), float(y2c - y1c)))
        ratio = float(self.get_parameter('target_depth_roi_ratio').value)
        min_roi_px = int(self.get_parameter('target_depth_min_roi_px').value)
        roi_size = int(round(max(float(min_roi_px), bbox_size * ratio)))
        roi_size = max(3, roi_size)
        half = roi_size // 2

        rx1 = max(0, int(round(cx_color)) - half)
        rx2 = min(w, int(round(cx_color)) + half + 1)
        ry1 = max(0, int(round(cy_color)) - half)
        ry2 = min(h, int(round(cy_color)) + half + 1)

        fx = float(cam_info_dict['camera_matrix']['data'][0])
        fy = float(cam_info_dict['camera_matrix']['data'][4])
        cam_cx = float(cam_info_dict['camera_matrix']['data'][2])
        cam_cy = float(cam_info_dict['camera_matrix']['data'][5])

        depth_h, depth_w = depth_img.shape[:2]
        scale_x = float(depth_w) / float(w)
        scale_y = float(depth_h) / float(h)

        points = []
        for v_color in range(ry1, ry2):
            for u_color in range(rx1, rx2):
                u_depth = int(round(float(u_color) * scale_x))
                v_depth = int(round(float(v_color) * scale_y))
                if u_depth < 0 or u_depth >= depth_w or v_depth < 0 or v_depth >= depth_h:
                    continue

                d = self.depth_value_to_meters(depth_img[v_depth, u_depth])
                if d is None:
                    continue

                u_cam = float(u_color) if depth_w == w else float(u_depth)
                v_cam = float(v_color) if depth_h == h else float(v_depth)
                x_m = (u_cam - cam_cx) * d / fx
                y_m = (v_cam - cam_cy) * d / fy
                points.append((x_m, y_m, d))

        min_points = int(self.get_parameter('target_depth_min_points').value)
        if len(points) < min_points:
            return self.center_pixel_to_3d(cx_color, cy_color, image_shape, depth_img, cam_info_dict)

        points = np.asarray(points, dtype=np.float32)
        z = points[:, 2]
        median_z = float(np.median(z))
        mad = float(np.median(np.abs(z - median_z)))
        configured_window = float(self.get_parameter('target_depth_window_m').value)
        adaptive_window = max(configured_window, 3.0 * mad)
        keep = np.abs(z - median_z) <= adaptive_window
        points = points[keep]

        if len(points) < min_points:
            return self.center_pixel_to_3d(cx_color, cy_color, image_shape, depth_img, cam_info_dict)

        xyz = np.median(points, axis=0)
        return float(xyz[0]), float(xyz[1]), float(xyz[2])

    def center_pixel_to_3d(self, u_color, v_color, image_shape, depth_img, cam_info_dict):
        h, w = image_shape
        depth_h, depth_w = depth_img.shape[:2]
        scale_x = float(depth_w) / float(w)
        scale_y = float(depth_h) / float(h)
        u_depth = int(round(float(u_color) * scale_x))
        v_depth = int(round(float(v_color) * scale_y))

        if u_depth < 0 or u_depth >= depth_w or v_depth < 0 or v_depth >= depth_h:
            return None

        d = self.depth_value_to_meters(depth_img[v_depth, u_depth])
        if d is None:
            return None

        fx = float(cam_info_dict['camera_matrix']['data'][0])
        fy = float(cam_info_dict['camera_matrix']['data'][4])
        cx = float(cam_info_dict['camera_matrix']['data'][2])
        cy = float(cam_info_dict['camera_matrix']['data'][5])
        u_cam = float(u_color) if depth_w == w else float(u_depth)
        v_cam = float(v_color) if depth_h == h else float(v_depth)
        x = (u_cam - cx) * d / fx
        y = (v_cam - cy) * d / fy
        z = d
        return x, y, z

    def pixel_to_3d(self, u, v, depth_img, cam_info_dict):
        fx = cam_info_dict['camera_matrix']['data'][0]
        fy = cam_info_dict['camera_matrix']['data'][4]
        cx = cam_info_dict['camera_matrix']['data'][2]
        cy = cam_info_dict['camera_matrix']['data'][5]

        if v >= depth_img.shape[0] or u >= depth_img.shape[1] or v < 0 or u < 0:
            return None

        d = self.depth_value_to_meters(depth_img[int(v), int(u)])
        if d is None:
            return None

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
