import os
import signal
import subprocess
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from PyQt5.QtCore import QProcess, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from valve_interfaces.msg import ValveVision

from .valve_detect_gui import Ui_MainWindow


class Ros2GuiNode(Node):
    def __init__(self, on_vision_cb):
        super().__init__('qt_gui_node')
        self._on_vision_cb = on_vision_cb
        self._vision_sub = self.create_subscription(
            ValveVision,
            '/valve/vision',
            self._vision_callback,
            10,
        )
        self.get_logger().info('Qt GUI ROS2 node started.')

    def _vision_callback(self, msg):
        self._on_vision_cb(msg)


class GUINode(QMainWindow):
    RGB_MODE_INDEX = 0
    DEPTH_MODE_INDEX = 1

    def __init__(self, ros_node, parent=None):
        super().__init__(parent)
        self.ros_node = ros_node
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.bridge = CvBridge()
        self.last_rgb_frame = None
        self.last_depth_frame = None
        self.received_frame_count = 0
        self._stopping_detector = False

        self.detector_process = QProcess(self)
        self.detector_process.setProcessChannelMode(QProcess.MergedChannels)
        self.detector_process.readyReadStandardOutput.connect(self._append_process_log)
        self.detector_process.finished.connect(self._on_detector_finished)
        self.detector_process.errorOccurred.connect(self._on_detector_error)

        self.ui.startCameraButton.clicked.connect(self.start_camera)
        self.ui.stopCameraButton.clicked.connect(self.stop_camera)
        self.ui.imageModeComboBox.currentIndexChanged.connect(self.update_camera_view)

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

        if self.detector_process.state() != QProcess.NotRunning:
            self._set_camera_running(True)
        self.update_camera_view()

    def start_camera(self):
        if self.detector_process.state() != QProcess.NotRunning:
            self.ui.commandLogEdit.append('[Vision] 图像处理节点已在运行。')
            return

        self._stopping_detector = False
        self.last_rgb_frame = None
        self.last_depth_frame = None
        self.received_frame_count = 0
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

    def stop_camera(self):
        self.last_rgb_frame = None
        self.last_depth_frame = None
        self.received_frame_count = 0

        if self.detector_process.state() == QProcess.NotRunning:
            self._set_camera_running(False)
            self._set_placeholder('相机未启动')
            return

        self._stopping_detector = True
        self._set_placeholder('正在停止图像处理节点...')
        self._terminate_detector_tree()
        self._set_camera_running(False)
        self._set_placeholder('相机已停止')
        self.ui.commandLogEdit.append('[Vision] 已停止图像处理节点。')

    def _terminate_detector_tree(self):
        pid = int(self.detector_process.processId() or 0)
        if pid <= 0:
            self.detector_process.kill()
            self.detector_process.waitForFinished(1000)
            return

        if sys.platform.startswith('win'):
            subprocess.run(
                ['taskkill', '/PID', str(pid), '/T', '/F'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            self.detector_process.waitForFinished(3000)
            return

        child_pids = self._child_pids(pid)
        for child_pid in child_pids:
            self._send_signal(child_pid, signal.SIGINT)
        self._send_signal(pid, signal.SIGINT)

        if self.detector_process.waitForFinished(3000):
            return

        child_pids = self._child_pids(pid)
        for child_pid in child_pids:
            self._send_signal(child_pid, signal.SIGTERM)
        self._send_signal(pid, signal.SIGTERM)
        self.detector_process.waitForFinished(2000)

        if self.detector_process.state() != QProcess.NotRunning:
            self.detector_process.kill()
            self.detector_process.waitForFinished(1000)

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
        if not self._stopping_detector:
            self._set_placeholder('图像处理节点已退出')
            self.ui.commandLogEdit.append('[Vision] 图像处理节点已退出。')

    def _on_detector_error(self, error):
        self._set_camera_running(False)
        self.ui.commandLogEdit.append(f'[Vision] 图像处理节点进程错误: {error}')

    def _append_process_log(self):
        output = bytes(self.detector_process.readAllStandardOutput()).decode('utf-8', errors='ignore').strip()
        if output:
            self.ui.commandLogEdit.append(output)

    def _set_camera_running(self, running):
        self.ui.startCameraButton.setEnabled(not running)
        self.ui.stopCameraButton.setEnabled(running)
        if running:
            self.ui.cameraStatusValueLabel.setText('运行中')
            self.ui.systemStatusLabel.setText('● 系统状态：运行中')
        else:
            self.ui.cameraStatusValueLabel.setText('未启动')
            self.ui.systemStatusLabel.setText('● 系统状态：待连接')

    def _set_placeholder(self, text):
        self.ui.cameraViewLabel.setPixmap(QPixmap())
        self.ui.cameraViewLabel.setText(text)

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
        scaled = pixmap.scaled(
            self.ui.cameraViewLabel.size(),
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
        super().closeEvent(event)


def main(args=None):
    ros_args = args if args is not None else sys.argv
    rclpy.init(args=ros_args)

    qt_args = remove_ros_args(args=ros_args)
    app = QApplication(qt_args)

    window = GUINode(ros_node=None)
    ros_node = Ros2GuiNode(window.on_vision_message)
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
        ros_node.destroy_node()
        rclpy.shutdown()

    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
