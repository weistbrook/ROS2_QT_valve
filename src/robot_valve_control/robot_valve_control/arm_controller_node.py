#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import socket
import threading
import queue
import time

from std_msgs.msg import Bool
from valve_interfaces.msg import ValveCommand


class ArmControllerNode(Node):
    def __init__(self, name='arm_controller_node'):
        super().__init__(name)

        self.declare_parameter('host', '192.168.0.200')
        self.declare_parameter('port', 2090)
        self.declare_parameter('require_ui_enable', False)
        self.declare_parameter('plane_alignment_sleep_sec', 3.0)

        self.host = self.get_parameter('host').value
        self.port = int(self.get_parameter('port').value)
        self.require_ui_enable = bool(self.get_parameter('require_ui_enable').value)
        self.ui_motion_enabled = not self.require_ui_enable
        self.plane_alignment_sleep_sec = float(self.get_parameter('plane_alignment_sleep_sec').value)
        self.plane_alignment_done = False

        self.client_socket = None
        self.connected = False
        self.response_queue = queue.Queue()

        self.command_queue = queue.Queue(maxsize=1)
        self.last_command_time = 0.0
        self.command_interval = 5.0

        self.connected_pub = self.create_publisher(
            Bool,
            '/valve/arm_connected',
            10
        )
        self.subscription = self.create_subscription(
            ValveCommand,
            '/valve/command',
            self.command_callback,
            10
        )
        self.enable_subscription = self.create_subscription(
            Bool,
            '/valve/arm_motion_enable',
            self.enable_callback,
            10
        )
        self.status_timer = self.create_timer(0.5, self.publish_connected_status)

        self._connect()

        self.worker_thread = threading.Thread(
            target=self.command_worker,
            daemon=True
        )
        self.worker_thread.start()

        if self.require_ui_enable:
            self.get_logger().info('Arm Controller Node has been started in UI-gated mode.')
        else:
            self.get_logger().info('Arm Controller Node has been started in direct-control mode.')

    def _connect(self):
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.host, self.port))
            self.connected = True

            recv_thread = threading.Thread(
                target=self._receive_messages,
                daemon=True
            )
            recv_thread.start()

            self.get_logger().info(f"已连接机械臂: {self.host}:{self.port}")
            return True

        except Exception as e:
            self.get_logger().error(f"连接机械臂失败: {e}")
            self.connected = False
            return False

    def _receive_messages(self):
        while self.connected:
            try:
                response = self.client_socket.recv(8192)
                if response:
                    response_str = response.decode('utf-8').strip()
                    self.response_queue.put(response_str)
                else:
                    self.get_logger().warn("机器人已断开连接")
                    break
            except Exception as e:
                self.get_logger().error(f"接收消息时出错: {e}")
                break

        self.connected = False
        if self.require_ui_enable:
            self.ui_motion_enabled = False
        try:
            self.client_socket.close()
        except Exception:
            pass

    def send_command(self, command, timeout=5):
        if not self.connected:
            if not self._connect():
                return None

        try:
            while not self.response_queue.empty():
                self.response_queue.get()

            formatted_message = f"[1#{command}]"
            self.client_socket.sendall(formatted_message.encode('utf-8'))

            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = self.response_queue.get(timeout=0.1)
                    return response
                except queue.Empty:
                    continue

            return None

        except Exception as e:
            self.get_logger().error(f"发送指令失败: {e}")
            self.connected = False
            if self.require_ui_enable:
                self.ui_motion_enabled = False
            return None

    def command_callback(self, msg):
        if not msg.valid:
            return

        if self.require_ui_enable and not self.ui_motion_enabled:
            self.get_logger().debug('等待 UI 一键旋阀使能，暂不执行运动指令。')
            return

        if self.command_queue.full():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                pass

        self.command_queue.put_nowait(msg)

    def enable_callback(self, msg):
        if not self.require_ui_enable:
            return

        self.ui_motion_enabled = bool(msg.data)
        if self.ui_motion_enabled:
            self.plane_alignment_done = False
            self.get_logger().info('已收到 UI 一键旋阀使能，开始接受运动指令。')
        else:
            self.get_logger().info('UI 已关闭运动使能，暂停接受运动指令。')

    def publish_connected_status(self):
        msg = Bool()
        msg.data = bool(self.connected)
        self.connected_pub.publish(msg)

    def command_worker(self):
        while rclpy.ok():
            try:
                msg = self.command_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            elapsed = time.time() - self.last_command_time
            if elapsed < self.command_interval:
                time.sleep(self.command_interval - elapsed)

            self.execute_valve_command(msg)
            self.last_command_time = time.time()

    def execute_valve_command(self, msg):
        self.get_logger().info(
            f"收到运动指令: type={msg.motion_type}, "
            f"x={msg.x:.2f}, y={msg.y:.2f}, z={msg.z:.2f}, "
            f"rotate={msg.need_rotation_correction}, "
            f"angle={msg.rotation_correction_deg:.2f}, "
            f"plane_valid={msg.plane_valid}, "
            f"yaw={msg.valve_yaw_deg:.2f}, pitch={msg.valve_pitch_deg:.2f}"
        )

        if not self.execute_plane_alignment_once(msg):
            return

        if msg.need_rotation_correction:
            rotate_cmd = f"Move.Axis 6,{msg.rotation_correction_deg:.3f}"
            resp = self.send_command(rotate_cmd)
            self.get_logger().info(f"旋转校正: {rotate_cmd} -> {resp}")
            time.sleep(5.0)

        if msg.motion_type == "far_move":
            move_cmd = f"Move.LOffset {{{msg.x:.3f},{msg.y:.3f},{-msg.z:.3f},0,0,0}}"

        elif msg.motion_type == "no_ahead_check":
            move_cmd = f"Move.LOffset {{{msg.x:.3f},0.000,{-msg.z:.3f},0,0,0}}"

        elif msg.motion_type == "check_and_spin":
            move_cmd = f"Move.LOffset {{{msg.x:.3f},{msg.y:.3f},{-msg.z:.3f},0,0,0}}"

        elif msg.motion_type == "small_move":
            move_cmd = f"Move.LOffset {{{msg.x:.3f},{msg.y:.3f},{-msg.z:.3f},0,0,0}}"

        elif msg.motion_type == "small_no_head":
            move_cmd = f"Move.LOffset {{{msg.x:.3f},0.000,{-msg.z:.3f},0,0,0}}"

        else:
            self.get_logger().warn(f"未知运动类型: {msg.motion_type}")
            return

        resp = self.send_command(move_cmd)
        self.get_logger().info(f"执行移动: {move_cmd} -> {resp}")

    def execute_plane_alignment_once(self, msg):
        if self.plane_alignment_done:
            return True

        if not msg.plane_valid:
            self.get_logger().warn(
                "Valve plane alignment skipped for this command: plane_valid=False. "
                "Movement is skipped and will retry on the next valid plane command."
            )
            return False

        pitch_cmd_deg = -float(msg.valve_pitch_deg)
        yaw_cmd_deg = -float(msg.valve_yaw_deg)

        pitch_cmd = f"Move.Axis 4,{pitch_cmd_deg:.3f}"
        pitch_resp = self.send_command(pitch_cmd)
        self.get_logger().info(
            f"Plane pitch alignment: valve_pitch={msg.valve_pitch_deg:.3f}, "
            f"axis_cmd={pitch_cmd} -> {pitch_resp}"
        )
        time.sleep(self.plane_alignment_sleep_sec)

        yaw_cmd = f"Move.Axis 5,{yaw_cmd_deg:.3f}"
        yaw_resp = self.send_command(yaw_cmd)
        self.get_logger().info(
            f"Plane yaw alignment: valve_yaw={msg.valve_yaw_deg:.3f}, "
            f"axis_cmd={yaw_cmd} -> {yaw_resp}"
        )
        time.sleep(self.plane_alignment_sleep_sec)

        self.plane_alignment_done = True
        self.get_logger().info("Valve plane alignment finished once; later commands will not repeat axis 4/5 alignment.")
        return True

    def close(self):
        self.connected = False
        try:
            if self.client_socket:
                self.client_socket.close()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)

    node = ArmControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
