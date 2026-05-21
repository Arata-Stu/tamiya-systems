#!/usr/bin/env python3
from __future__ import annotations

import math
import queue
import select
import sys
import termios
import threading
import tty
from dataclasses import dataclass
from pathlib import Path

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from vslam_map_tools.config_io import load_alignment_config, save_alignment_config


def wrap_angle(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad <= -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


@dataclass
class AlignmentState:
    x: float
    y: float
    z: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float

    def copy(self) -> "AlignmentState":
        return AlignmentState(
            x=self.x,
            y=self.y,
            z=self.z,
            roll_rad=self.roll_rad,
            pitch_rad=self.pitch_rad,
            yaw_rad=self.yaw_rad,
        )


class KeyboardReader(threading.Thread):
    def __init__(self, output_queue: "queue.Queue[str]", stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self.output_queue = output_queue
        self.stop_event = stop_event

    def run(self) -> None:
        if not sys.stdin.isatty():
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self.stop_event.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                chunk = sys.stdin.read(1)
                if chunk == "\x1b":
                    seq = chunk
                    for _ in range(2):
                        ready, _, _ = select.select([sys.stdin], [], [], 0.01)
                        if not ready:
                            break
                        seq += sys.stdin.read(1)
                    self.output_queue.put(seq)
                else:
                    self.output_queue.put(chunk)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class ManualTfAlignmentNode(Node):
    def __init__(self) -> None:
        super().__init__("manual_tf_alignment_node")

        self.declare_parameter("config_path", "")
        config_path = str(self.get_parameter("config_path").value).strip()
        config_defaults = load_alignment_config(config_path) if config_path else {}

        self.parent_frame = str(
            self.declare_parameter("parent_frame", config_defaults.get("parent_frame", "map")).value
        )
        self.child_frame = str(
            self.declare_parameter("child_frame", config_defaults.get("child_frame", "vslam_map")).value
        )
        self.publish_rate_hz = max(1.0, float(self.declare_parameter("publish_rate_hz", 20.0).value))
        self.translation_step_m = max(1.0e-4, float(self.declare_parameter("translation_step_m", 0.02).value))
        self.rotation_step_deg = max(0.01, float(self.declare_parameter("rotation_step_deg", 1.0).value))
        self.enable_keyboard = bool(self.declare_parameter("enable_keyboard", True).value)
        self.auto_save_on_update = bool(self.declare_parameter("auto_save_on_update", False).value)

        self.state = AlignmentState(
            x=float(self.declare_parameter("x", float(config_defaults.get("x", 0.0))).value),
            y=float(self.declare_parameter("y", float(config_defaults.get("y", 0.0))).value),
            z=float(self.declare_parameter("z", float(config_defaults.get("z", 0.0))).value),
            roll_rad=float(
                self.declare_parameter("roll_rad", float(config_defaults.get("roll_rad", 0.0))).value
            ),
            pitch_rad=float(
                self.declare_parameter("pitch_rad", float(config_defaults.get("pitch_rad", 0.0))).value
            ),
            yaw_rad=float(
                self.declare_parameter("yaw_rad", float(config_defaults.get("yaw_rad", 0.0))).value
            ),
        )
        self.initial_state = self.state.copy()
        self.config_path = Path(config_path).expanduser().resolve() if config_path else None

        self.tf_broadcaster = TransformBroadcaster(self)
        self.publish_timer = self.create_timer(1.0 / self.publish_rate_hz, self.publish_transform)

        self.stop_event = threading.Event()
        self.key_queue: "queue.Queue[str]" = queue.Queue()
        self.keyboard_reader: KeyboardReader | None = None
        self.keyboard_timer = None

        if self.enable_keyboard and sys.stdin.isatty():
            self.keyboard_reader = KeyboardReader(self.key_queue, self.stop_event)
            self.keyboard_reader.start()
            self.keyboard_timer = self.create_timer(0.05, self.process_keyboard_input)
        elif self.enable_keyboard:
            self.get_logger().warn("Keyboard control requested, but stdin is not a TTY. Running publish-only mode.")

        self.print_help()
        self.print_state("Initial alignment")

    def destroy_node(self) -> bool:
        self.stop_event.set()
        if self.keyboard_reader is not None:
            self.keyboard_reader.join(timeout=0.5)
        return super().destroy_node()

    def current_rotation_step_rad(self) -> float:
        return math.radians(self.rotation_step_deg)

    def publish_transform(self) -> None:
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.parent_frame
        msg.child_frame_id = self.child_frame
        msg.transform.translation.x = self.state.x
        msg.transform.translation.y = self.state.y
        msg.transform.translation.z = self.state.z
        qx, qy, qz, qw = quaternion_from_euler(
            self.state.roll_rad, self.state.pitch_rad, self.state.yaw_rad
        )
        msg.transform.rotation.x = qx
        msg.transform.rotation.y = qy
        msg.transform.rotation.z = qz
        msg.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(msg)

    def print_help(self) -> None:
        print("")
        print("[manual_tf_alignment_node]")
        print("  w/s : +/- x, a/d : +/- y, r/f : +/- z")
        print("  q/e : +/- yaw, t/g : +/- roll, y/h : +/- pitch")
        print("  Arrow keys : x/y translation shortcuts")
        print("  Uppercase keys apply 10x step")
        print("  [ / ] : translation step /2, x2")
        print("  - / = : rotation step /2, x2")
        print("  0 : reset to startup values")
        print("  p : save current alignment to config_path")
        print("  i or ? : print help")
        print("")

    def print_state(self, prefix: str) -> None:
        print(
            f"{prefix}: parent={self.parent_frame} child={self.child_frame} "
            f"xyz=({self.state.x:.4f}, {self.state.y:.4f}, {self.state.z:.4f}) "
            f"rpy_deg=({math.degrees(self.state.roll_rad):.2f}, "
            f"{math.degrees(self.state.pitch_rad):.2f}, {math.degrees(self.state.yaw_rad):.2f}) "
            f"step_m={self.translation_step_m:.4f} step_deg={self.rotation_step_deg:.3f}"
        )

    def save_current_alignment(self) -> None:
        if self.config_path is None:
            self.get_logger().warn("config_path is empty. Nothing was saved.")
            return
        save_alignment_config(
            self.config_path,
            {
                "parent_frame": self.parent_frame,
                "child_frame": self.child_frame,
                "x": self.state.x,
                "y": self.state.y,
                "z": self.state.z,
                "roll_rad": self.state.roll_rad,
                "pitch_rad": self.state.pitch_rad,
                "yaw_rad": self.state.yaw_rad,
            },
        )
        self.get_logger().info(f"Saved alignment config: {self.config_path}")

    def maybe_auto_save(self) -> None:
        if self.auto_save_on_update:
            self.save_current_alignment()

    def adjust_translation(self, axis: str, delta: float) -> None:
        if axis == "x":
            self.state.x += delta
        elif axis == "y":
            self.state.y += delta
        elif axis == "z":
            self.state.z += delta

    def adjust_rotation(self, axis: str, delta_rad: float) -> None:
        if axis == "roll":
            self.state.roll_rad = wrap_angle(self.state.roll_rad + delta_rad)
        elif axis == "pitch":
            self.state.pitch_rad = wrap_angle(self.state.pitch_rad + delta_rad)
        elif axis == "yaw":
            self.state.yaw_rad = wrap_angle(self.state.yaw_rad + delta_rad)

    def process_keyboard_input(self) -> None:
        changed = False

        while not self.key_queue.empty():
            key = self.key_queue.get_nowait()
            if key in ("\x1b[A",):
                self.adjust_translation("x", self.translation_step_m)
                changed = True
                continue
            if key in ("\x1b[B",):
                self.adjust_translation("x", -self.translation_step_m)
                changed = True
                continue
            if key in ("\x1b[C",):
                self.adjust_translation("y", -self.translation_step_m)
                changed = True
                continue
            if key in ("\x1b[D",):
                self.adjust_translation("y", self.translation_step_m)
                changed = True
                continue

            if key in ("i", "?"):
                self.print_help()
                continue
            if key == "0":
                self.state = self.initial_state.copy()
                changed = True
                continue
            if key == "p":
                self.save_current_alignment()
                continue
            if key == "[":
                self.translation_step_m = max(1.0e-4, self.translation_step_m / 2.0)
                self.print_state("Updated step size")
                continue
            if key == "]":
                self.translation_step_m = min(10.0, self.translation_step_m * 2.0)
                self.print_state("Updated step size")
                continue
            if key == "-":
                self.rotation_step_deg = max(0.01, self.rotation_step_deg / 2.0)
                self.print_state("Updated step size")
                continue
            if key == "=":
                self.rotation_step_deg = min(180.0, self.rotation_step_deg * 2.0)
                self.print_state("Updated step size")
                continue

            scale = 10.0 if len(key) == 1 and key.isalpha() and key.isupper() else 1.0
            key_lower = key.lower()
            translation_delta = self.translation_step_m * scale
            rotation_delta = self.current_rotation_step_rad() * scale

            if key_lower == "w":
                self.adjust_translation("x", translation_delta)
                changed = True
            elif key_lower == "s":
                self.adjust_translation("x", -translation_delta)
                changed = True
            elif key_lower == "a":
                self.adjust_translation("y", translation_delta)
                changed = True
            elif key_lower == "d":
                self.adjust_translation("y", -translation_delta)
                changed = True
            elif key_lower == "r":
                self.adjust_translation("z", translation_delta)
                changed = True
            elif key_lower == "f":
                self.adjust_translation("z", -translation_delta)
                changed = True
            elif key_lower == "q":
                self.adjust_rotation("yaw", rotation_delta)
                changed = True
            elif key_lower == "e":
                self.adjust_rotation("yaw", -rotation_delta)
                changed = True
            elif key_lower == "t":
                self.adjust_rotation("roll", rotation_delta)
                changed = True
            elif key_lower == "g":
                self.adjust_rotation("roll", -rotation_delta)
                changed = True
            elif key_lower == "y":
                self.adjust_rotation("pitch", rotation_delta)
                changed = True
            elif key_lower == "h":
                self.adjust_rotation("pitch", -rotation_delta)
                changed = True

        if changed:
            self.print_state("Updated alignment")
            self.maybe_auto_save()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ManualTfAlignmentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
