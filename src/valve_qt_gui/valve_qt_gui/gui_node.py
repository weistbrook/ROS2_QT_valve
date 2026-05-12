import os
import signal
import subprocess
import sys
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from PyQt5.QtCore import QProcess, QTimer, Qt
from PyQt5.QtGui import QFont, QFontDatabase, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from std_msgs.msg import Bool
from valve_interfaces.msg import ValveCommand, ValveVision

from .valve_detect_gui import Ui_MainWindow


class Ros2GuiNode(Node):
    def __init__(self, on_vision_cb, on_command_cb, on_arm_status_cb):
        super().__init__('qt_gui_node')
        self._on_vision_cb = on_vision_cb
        self._on_command_cb = on_command_cb
        self._on_arm_status_cb = on_arm_status_cb
        self._vision_sub = self.create_subscription(
            ValveVision,
            '/valve/vision',
            self._vision_callback,
            10,
        )
        self._command_sub = self.create_subscription(
            ValveCommand,
            '/valve/command',
            self._command_callback,
            10,
        )
        self.arm_motion_enable_pub = self.create_publisher(
            Bool,
            '/valve/arm_motion_enable',
            10,
        )
        self._arm_status_sub = self.create_subscription(
            Bool,
            '/valve/arm_connected',
            self._arm_status_callback,
            10,
        )
        self.get_logger().info('Qt GUI ROS2 node started.')

    def _vision_callback(self, msg):
        self._on_vision_cb(msg)

    def _command_callback(self, msg):
        self._on_command_cb(msg)

    def _arm_status_callback(self, msg):
        self._on_arm_status_cb(msg)


class GUINode(QMainWindow):
    RGB_MODE_INDEX = 0
    DEPTH_MODE_INDEX = 1
    FPS_WINDOW_SECONDS = 2.0
    FPS_WINDOW_MAX_FRAMES = 60

    def __init__(self, ros_node, parent=None):
        super().__init__(parent)
        self.ros_node = ros_node
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.bridge = CvBridge()
        self.last_rgb_frame = None
        self.last_depth_frame = None
        self.received_frame_count = 0
        self.frame_timestamps = deque(maxlen=self.FPS_WINDOW_MAX_FRAMES)
        self._stopping_detector = False
        self._stopping_arm = False
        self.arm_connected = False
        self.arm_motion_enabled = False

        self.detector_process = QProcess(self)
        self.detector_process.setProcessChannelMode(QProcess.MergedChannels)
        self.detector_process.readyReadStandardOutput.connect(self._append_process_log)
        self.detector_process.finished.connect(self._on_detector_finished)
        self.detector_process.errorOccurred.connect(self._on_detector_error)

        self.arm_process = QProcess(self)
        self.arm_process.setProcessChannelMode(QProcess.MergedChannels)
        self.arm_process.readyReadStandardOutput.connect(self._append_arm_process_log)
        self.arm_process.finished.connect(self._on_arm_finished)
        self.arm_process.errorOccurred.connect(self._on_arm_error)

        self.ui.startCameraButton.clicked.connect(self.start_camera)
        self.ui.stopCameraButton.clicked.connect(self.stop_camera)
        self.ui.connectArmButton.clicked.connect(self.connect_arm)
        self.ui.stopArmButton.clicked.connect(self.stop_arm)
        self.ui.rotateValveButton.clicked.connect(self.enable_arm_motion)
        self.ui.imageModeComboBox.currentIndexChanged.connect(self.update_camera_view)

        self._reset_valve_data()
        self._set_current_command('none')
        self._reset_fps()
        self._set_arm_process_running(False)
        self._set_arm_connected(False)
        self._set_arm_motion_enabled(False)
        self._set_camera_running(False)
        self._set_placeholder('等待图像流...')

    def on_vision_message(self, msg):
        try:
            rgb_frame = self.bridge.imgmsg_to_cv2(msg.rgb_image, desired_encoding='bgr8')
            depth_frame = self.bridge.imgmsg_to_cv2(msg.depth_image, desired_encoding='passthrough')
        except Exception as exc:
            self.ui.commandLogEdit.append(f'[Vision] 图像解码失败: {exc}')
            return

        self.last_rgb_frame = rgb_frame.copy()
        self.last_depth_frame = depth_frame.copy()
        self.received_frame_count += 1
        self._update_fps()

        if self.detector_process.state() != QProcess.NotRunning:
            self._set_camera_running(True)
        self.update_camera_view()

    def on_command_message(self, msg):
        motion_type = (msg.motion_type or 'none').strip() or 'none'
        self._set_current_command(motion_type)

        if not msg.valid or motion_type == 'none':
            self._reset_valve_data()
            return

        self.ui.xValueLabel.setText(self._format_mm(msg.x))
        self.ui.yValueLabel.setText(self._format_mm(msg.y))
        self.ui.zValueLabel.setText(self._format_mm(msg.z))
        self.ui.confidenceValueLabel.setText(self._format_confidence(msg.confidence))

    def on_arm_status_message(self, msg):
        if self.arm_process.state() == QProcess.NotRunning:
            return

        connected = bool(msg.data)
        if connected == self.arm_connected:
            return

        self._set_arm_connected(connected)
        self._set_arm_motion_enabled(False)
        if connected:
            self.ui.commandLogEdit.append('[Arm] 机械臂 TCP 连接成功，可以点击“一键旋阀”。')
        else:
            self.ui.commandLogEdit.append('[Arm] 机械臂 TCP 已断开，已暂停一键旋阀。')

    def start_camera(self):
        if self.detector_process.state() != QProcess.NotRunning:
            self.ui.commandLogEdit.append('[Vision] 图像处理节点已在运行。')
            return

        self._stopping_detector = False
        self.last_rgb_frame = None
        self.last_depth_frame = None
        self.received_frame_count = 0
        self._reset_fps()
        self._reset_valve_data()
        self._set_current_command('none')
        self._set_placeholder('正在启动图像处理节点...')

        self.detector_process.start(
            'ros2',
            [
                'run',
                'robot_valve_control',
                'valve_detection_node',
                '--ros-args',
                '-p',
                'show_image:=false',
            ],
        )
        started = self.detector_process.waitForStarted(3000)
        if started:
            self._set_camera_running(True)
            self.ui.commandLogEdit.append('[Vision] 已启动图像处理节点，等待 /valve/vision 图像流。')
        else:
            self._set_camera_running(False)
            self._set_placeholder('图像处理节点启动失败')
            self.ui.commandLogEdit.append('[Vision] 启动图像处理节点失败，请确认 ROS2 环境已 source。')

    def connect_arm(self):
        if self.arm_process.state() != QProcess.NotRunning:
            self.ui.commandLogEdit.append('[Arm] 机械臂控制节点已在运行。')
            return

        host = self.ui.armIpLineEdit.text().strip() or '192.168.0.200'
        try:
            port = int(self.ui.armPortLineEdit.text().strip())
        except ValueError:
            self.ui.commandLogEdit.append('[Arm] 端口号无效，请输入整数。')
            return

        self._stopping_arm = False
        self._set_arm_process_running(False)
        self._set_arm_connected(False)
        self._set_arm_motion_enabled(False)
        self.ui.commandLogEdit.append(f'[Arm] 正在启动机械臂控制节点: {host}:{port}')

        self.arm_process.start(
            'ros2',
            [
                'run',
                'robot_valve_control',
                'arm_controller_node',
                '--ros-args',
                '-p',
                f'host:={host}',
                '-p',
                f'port:={port}',
                '-p',
                'require_ui_enable:=true',
            ],
        )
        started = self.arm_process.waitForStarted(3000)
        if started:
            self._set_arm_process_running(True)
            self._set_arm_connected(False)
            self._set_arm_motion_enabled(False)
            self.ui.commandLogEdit.append('[Arm] 已启动机械臂控制节点，等待 TCP 连接成功。')
        else:
            self._set_arm_process_running(False)
            self._set_arm_connected(False)
            self.ui.commandLogEdit.append('[Arm] 启动机械臂控制节点失败，请确认 ROS2 环境已 source。')

    def enable_arm_motion(self):
        if self.arm_process.state() == QProcess.NotRunning:
            self.ui.commandLogEdit.append('[Arm] 请先连接机械臂。')
            return
        if not self.arm_connected:
            self.ui.commandLogEdit.append('[Arm] 机械臂 TCP 尚未连接成功，不能使能运动。')
            return

        msg = Bool()
        msg.data = True
        if self.ros_node is not None:
            self.ros_node.arm_motion_enable_pub.publish(msg)
        self._set_arm_motion_enabled(True)
        self.ui.commandLogEdit.append('[Arm] 已使能一键旋阀，机械臂控制节点将接受后续 ValveCommand 指令。')

    def stop_arm(self):
        self._set_arm_motion_enabled(False)
        if self.ros_node is not None:
            msg = Bool()
            msg.data = False
            self.ros_node.arm_motion_enable_pub.publish(msg)

        if self.arm_process.state() == QProcess.NotRunning:
            self._set_arm_process_running(False)
            self._set_arm_connected(False)
            return

        self._stopping_arm = True
        self._terminate_process_tree(self.arm_process)
        self._set_arm_process_running(False)
        self._set_arm_connected(False)
        self.ui.commandLogEdit.append('[Arm] 已停止机械臂控制节点。')

    def stop_camera(self):
        self.last_rgb_frame = None
        self.last_depth_frame = None
        self.received_frame_count = 0
        self._reset_fps()
        self._reset_valve_data()
        self._set_current_command('none')

        if self.detector_process.state() == QProcess.NotRunning:
            self._set_camera_running(False)
            self._set_placeholder('相机未启动')
            return

        self._stopping_detector = True
        self._set_placeholder('正在停止图像处理节点...')
        self._terminate_process_tree(self.detector_process)
        self._set_camera_running(False)
        self._set_placeholder('相机已停止')
        self.ui.commandLogEdit.append('[Vision] 已停止图像处理节点。')

    def _terminate_process_tree(self, process):
        pid = int(process.processId() or 0)
        if pid <= 0:
            process.kill()
            process.waitForFinished(1000)
            return

        if sys.platform.startswith('win'):
            subprocess.run(
                ['taskkill', '/PID', str(pid), '/T', '/F'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            process.waitForFinished(3000)
            return

        child_pids = self._child_pids(pid)
        for child_pid in child_pids:
            self._send_signal(child_pid, signal.SIGINT)
        self._send_signal(pid, signal.SIGINT)

        if process.waitForFinished(3000):
            return

        child_pids = self._child_pids(pid)
        for child_pid in child_pids:
            self._send_signal(child_pid, signal.SIGTERM)
        self._send_signal(pid, signal.SIGTERM)
        process.waitForFinished(2000)

        if process.state() != QProcess.NotRunning:
            process.kill()
            process.waitForFinished(1000)

    def _child_pids(self, pid):
        try:
            result = subprocess.run(
                ['pgrep', '-P', str(pid)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except Exception:
            return []

        return [int(line) for line in result.stdout.splitlines() if line.strip().isdigit()]

    def _send_signal(self, pid, sig):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except Exception:
            pass

    def _on_detector_finished(self):
        self._set_camera_running(False)
        self._reset_fps()
        if not self._stopping_detector:
            self._set_placeholder('图像处理节点已退出')
            self.ui.commandLogEdit.append('[Vision] 图像处理节点已退出。')

    def _on_detector_error(self, error):
        self._set_camera_running(False)
        self._reset_fps()
        self.ui.commandLogEdit.append(f'[Vision] 图像处理节点进程错误: {error}')

    def _append_process_log(self):
        output = bytes(self.detector_process.readAllStandardOutput()).decode('utf-8', errors='ignore').strip()
        if output:
            self.ui.commandLogEdit.append(output)

    def _append_arm_process_log(self):
        output = bytes(self.arm_process.readAllStandardOutput()).decode('utf-8', errors='ignore').strip()
        if output:
            self.ui.commandLogEdit.append(output)

    def _on_arm_finished(self):
        self._set_arm_process_running(False)
        self._set_arm_connected(False)
        self._set_arm_motion_enabled(False)
        if not self._stopping_arm:
            self.ui.commandLogEdit.append('[Arm] 机械臂控制节点已退出。')

    def _on_arm_error(self, error):
        self._set_arm_process_running(False)
        self._set_arm_connected(False)
        self._set_arm_motion_enabled(False)
        self.ui.commandLogEdit.append(f'[Arm] 机械臂控制节点进程错误: {error}')

    def _set_camera_running(self, running):
        self.ui.startCameraButton.setEnabled(not running)
        self.ui.stopCameraButton.setEnabled(running)
        if running:
            self.ui.cameraStatusValueLabel.setText('运行中')
            self.ui.systemStatusLabel.setText('● 系统状态：运行中')
        else:
            self.ui.cameraStatusValueLabel.setText('未启动')
            self.ui.systemStatusLabel.setText('● 系统状态：待连接')

    def _set_arm_process_running(self, running):
        self.ui.connectArmButton.setEnabled(not running)
        self.ui.stopArmButton.setEnabled(running)
        self.ui.armIpLineEdit.setEnabled(not running)
        self.ui.armPortLineEdit.setEnabled(not running)
        if running:
            self.ui.connectArmButton.setText('控制节点启动中')
        else:
            self.ui.connectArmButton.setText('连接机械臂')

    def _set_arm_connected(self, connected):
        self.arm_connected = bool(connected)
        if connected:
            self.ui.connectArmButton.setText('机械臂已连接')
        else:
            if self.arm_process.state() != QProcess.NotRunning:
                self.ui.connectArmButton.setText('控制节点启动中')
            else:
                self.ui.connectArmButton.setText('连接机械臂')

    def _set_arm_motion_enabled(self, enabled):
        self.arm_motion_enabled = bool(enabled)
        self.ui.rotateValveButton.setEnabled(
            self.arm_process.state() != QProcess.NotRunning
            and self.arm_connected
            and not enabled
        )
        if enabled:
            self.ui.rotateValveButton.setText('一键旋阀已使能')
        else:
            self.ui.rotateValveButton.setText('一键旋阀')

    def _reset_valve_data(self):
        self.ui.xValueLabel.setText('-- mm')
        self.ui.yValueLabel.setText('-- mm')
        self.ui.zValueLabel.setText('-- mm')
        self.ui.confidenceValueLabel.setText('--')

    def _set_current_command(self, motion_type):
        command_text = self._motion_type_text(motion_type)
        self.ui.currentCommandLabel.setText(command_text)

    def _motion_type_text(self, motion_type):
        command_names = {
            'far_move': 'far_move：远距离靠近',
            'no_ahead_check': 'no_ahead_check：平面校正',
            'check_and_spin': 'check_and_spin：检测并旋转',
            'small_move': 'small_move：小阀门靠近',
            'small_no_head': 'small_no_head：小阀门校正',
            'none': 'none：暂无执行命令',
        }
        return command_names.get(motion_type, f'{motion_type}：未知命令')

    def _format_mm(self, value):
        return f'{float(value):.1f} mm'

    def _format_confidence(self, confidence):
        confidence = float(confidence)
        if confidence <= 1.0:
            return f'{confidence * 100.0:.1f}%'
        return f'{confidence:.1f}%'

    def _set_placeholder(self, text):
        self.ui.cameraViewLabel.setPixmap(QPixmap())
        self.ui.cameraViewLabel.setText(text)

    def _reset_fps(self):
        self.frame_timestamps.clear()
        self.ui.fpsValueLabel.setText('-- FPS')

    def _update_fps(self):
        now = time.monotonic()
        self.frame_timestamps.append(now)
        while (
            len(self.frame_timestamps) > 1
            and now - self.frame_timestamps[0] > self.FPS_WINDOW_SECONDS
        ):
            self.frame_timestamps.popleft()

        if len(self.frame_timestamps) < 2:
            self.ui.fpsValueLabel.setText('-- FPS')
            return

        elapsed = self.frame_timestamps[-1] - self.frame_timestamps[0]
        if elapsed <= 0.0:
            self.ui.fpsValueLabel.setText('-- FPS')
            return

        fps = (len(self.frame_timestamps) - 1) / elapsed
        self.ui.fpsValueLabel.setText(f'{fps:.1f} FPS')

    def _show_bgr_frame(self, frame):
        if frame is None or frame.size == 0:
            self._set_placeholder('收到空图像')
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        image = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image)
        target_size = self.ui.cameraViewLabel.size()
        if pixmap.size() == target_size:
            scaled = pixmap
        else:
            scaled = pixmap.scaled(
                target_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        self.ui.cameraViewLabel.setText('')
        self.ui.cameraViewLabel.setPixmap(scaled)

    def _show_depth_frame(self, depth):
        if depth is None or depth.size == 0:
            self._set_placeholder('收到空深度图')
            return

        depth_float = np.asarray(depth, dtype=np.float32)
        finite = np.isfinite(depth_float)
        valid = finite & (depth_float > 0)
        if not np.any(valid):
            self._set_placeholder('深度图暂无有效像素')
            return

        near = float(np.percentile(depth_float[valid], 1))
        far = float(np.percentile(depth_float[valid], 99))
        if far <= near:
            far = near + 1.0

        depth_norm = np.clip((depth_float - near) * 255.0 / (far - near), 0, 255).astype(np.uint8)
        depth_norm[~valid] = 0
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        self._show_bgr_frame(depth_color)

    def update_camera_view(self):
        mode_index = self.ui.imageModeComboBox.currentIndex()

        if mode_index == self.RGB_MODE_INDEX:
            if self.last_rgb_frame is None:
                self._set_placeholder('等待处理后的 RGB 图像...')
                return
            self._show_bgr_frame(self.last_rgb_frame)
            return

        if mode_index == self.DEPTH_MODE_INDEX:
            if self.last_depth_frame is None:
                self._set_placeholder('等待深度图像...')
                return
            self._show_depth_frame(self.last_depth_frame)
            return

        self._set_placeholder('未知图像模式')

    def closeEvent(self, event):
        self.stop_camera()
        self.stop_arm()
        super().closeEvent(event)


def configure_chinese_font(app):
    preferred_families = (
        'Noto Sans CJK SC',
        'Noto Sans CJK',
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'Source Han Sans SC',
        'Microsoft YaHei',
        'SimHei',
    )
    installed_families = set(QFontDatabase().families())
    for family in preferred_families:
        if family in installed_families:
            app.setFont(QFont(family, 10))
            return

    QFont.insertSubstitution('Sans Serif', 'WenQuanYi Micro Hei')
    QFont.insertSubstitution('Arial', 'WenQuanYi Micro Hei')
    app.setFont(QFont('Sans Serif', 10))


def main(args=None):
    ros_args = args if args is not None else sys.argv
    rclpy.init(args=ros_args)

    qt_args = remove_ros_args(args=ros_args)
    app = QApplication(qt_args)
    configure_chinese_font(app)

    window = GUINode(ros_node=None)
    ros_node = Ros2GuiNode(
        window.on_vision_message,
        window.on_command_message,
        window.on_arm_status_message,
    )
    window.ros_node = ros_node
    window.show()

    spin_timer = QTimer()
    spin_timer.timeout.connect(lambda: rclpy.spin_once(ros_node, timeout_sec=0.0))
    spin_timer.start(10)

    exit_code = 0
    try:
        exit_code = app.exec_()
    finally:
        spin_timer.stop()
        window.stop_camera()
        window.stop_arm()
        ros_node.destroy_node()
        rclpy.shutdown()

    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
