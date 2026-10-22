#!/usr/bin/env python3
"""SITL 转角/巡航计时标定（位置设定点 + 到位判据）。

设计要点：用"同长度、不同转角数"的路径对，可直接分离巡航项与转角附加项：
    T = L/v_eff + A * t_r + D + c
其中 L 为航迹总长、A 为 90° 等效转角数、D 为悬停等待时长、c 为起飞/收敛常数。
"""
import argparse
import csv
import json
import math
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import BatteryState
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

QOS = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT)


class Flight(Node):
    def __init__(self, cfg):
        super().__init__('sitl_turns')
        self.cfg = cfg
        self.pose = None
        self.state = None
        self.batt = None
        self.rows = []
        self.phase = 'INIT'
        self.wall0 = time.time()
        self.pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.on_pose, QOS)
        self.create_subscription(BatteryState, '/mavros/battery', self.on_batt, QOS)
        self.create_subscription(State, '/mavros/state', self.on_state, QOS)
        self.arm_cli = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_cli = self.create_client(SetMode, '/mavros/set_mode')

    def on_pose(self, m):
        self.pose = m
        p = m.pose.position
        q = m.pose.orientation
        self.rows.append(dict(wall=round(time.time() - self.wall0, 4), phase=self.phase,
                              x=round(p.x, 4), y=round(p.y, 4), z=round(p.z, 4),
                              yaw=round(2 * math.atan2(q.z, q.w), 4),
                              volt=round(self.batt.voltage, 3) if self.batt else '',
                              pct=round(self.batt.percentage, 4) if self.batt else '',
                              mode=self.state.mode if self.state else ''))

    def on_batt(self, m):
        self.batt = m

    def on_state(self, m):
        self.state = m

    def pos(self):
        p = self.pose.pose.position
        return (p.x, p.y, p.z)

    def call(self, cli, req, to=6.0):
        if not cli.wait_for_service(timeout_sec=3.0):
            return False
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=to)
        return fut.done()

    def set_mode(self, m):
        r = SetMode.Request()
        r.custom_mode = m
        return self.call(self.mode_cli, r)

    def arm(self, v=True):
        r = CommandBool.Request()
        r.value = v
        return self.call(self.arm_cli, r)

    def sp(self, x, y, z, yaw):
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), float(z)
        m.pose.orientation.z = math.sin(yaw / 2.0)
        m.pose.orientation.w = math.cos(yaw / 2.0)
        self.pub.publish(m)

    def hold(self, sec, sp=None):
        t0 = time.time()
        while time.time() - t0 < sec:
            if sp:
                self.sp(*sp)
            rclpy.spin_once(self, timeout_sec=0.01)

    def goto(self, x, y, z, yaw, tol=0.45, timeout=45.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.sp(x, y, z, yaw)
            rclpy.spin_once(self, timeout_sec=0.01)
            px, py, pz = self.pos()
            if math.hypot(px - x, py - y) < tol and abs(pz - z) < 0.35:
                return time.time() - t0, True
            if math.hypot(px - self.x0, py - self.y0) > 90.0:
                return time.time() - t0, False
        return time.time() - t0, False

    def run(self):
        cfg = self.cfg
        alt = cfg['alt']
        t0 = time.time()
        while self.pose is None and time.time() - t0 < 20:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.pose is None:
            raise RuntimeError('no pose')
        self.x0, self.y0, z0 = self.pos()

        self.phase = 'PRESTREAM'
        self.hold(3.0, (self.x0, self.y0, z0, 0.0))
        self.set_mode('OFFBOARD')
        self.hold(0.6, (self.x0, self.y0, z0, 0.0))
        self.arm(True)
        self.hold(0.6, (self.x0, self.y0, z0, 0.0))

        self.phase = 'TAKEOFF'
        ta = time.time()
        while time.time() - ta < 6.5:
            f = min(1.0, (time.time() - ta) / 4.5)
            self.sp(self.x0, self.y0, z0 + (alt - z0) * f, 0.0)
            rclpy.spin_once(self, timeout_sec=0.01)
        self.hold(1.5, (self.x0, self.y0, alt, 0.0))

        marks = []
        t_path0 = time.time()
        prev = (self.x0, self.y0)
        total = 0.0
        for i, (wx, wy) in enumerate(cfg['waypoints']):
            tx, ty = self.x0 + wx, self.y0 + wy
            seg = math.hypot(tx - prev[0], ty - prev[1])
            total += seg
            self.phase = 'SEG%d' % i
            yaw = math.atan2(ty - prev[1], tx - prev[0])
            dt, ok = self.goto(tx, ty, alt, yaw)
            marks.append(dict(seg=i, len=round(seg, 2), dt=round(dt, 3), reached=ok))
            print('SEG%d L=%5.1f dt=%6.2f reached=%s' % (i, seg, dt, ok), flush=True)
            prev = (tx, ty)
        t_path1 = time.time()

        hover = cfg.get('hover', 0.0)
        if hover > 0:
            self.phase = 'HOVER'
            self.hold(hover, (prev[0], prev[1], alt, 0.0))

        self.phase = 'LAND'
        self.set_mode('AUTO.LAND')
        self.hold(11.0)
        self.arm(False)
        for _ in range(120):
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.state and not self.state.armed:
                break

        turns = cfg['turns']
        return dict(marks=marks, L=round(total, 2), A=turns, D=hover,
                    path_wall=round(t_path1 - t_path0, 3),
                    origin=[round(self.x0, 3), round(self.y0, 3)], alt=alt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', default='S20')
    ap.add_argument('--alt', type=float, default=5.0)
    ap.add_argument('--hover', type=float, default=0.0)
    ap.add_argument('--tag', default='cal')
    ap.add_argument('--outdir', default=os.path.expanduser('~/paper_sitl/runs2'))
    args = ap.parse_args()

    # (航点列表, 90°等效转角数)
    P = {
        'S10': ([(10.0, 0.0)], 0),                                   # L=10  A=0
        'S20': ([(20.0, 0.0)], 0),                                   # L=20  A=0
        'S40': ([(40.0, 0.0)], 0),                                   # L=40  A=0
        'Q20': ([(5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)], 3),  # L=20 A=3
        'Q40': ([(20.0, 0.0), (20.0, 10.0), (0.0, 10.0)], 2),        # L=40+10+20? -> 见下
        'L60': ([(30.0, 0.0), (30.0, 10.0), (0.0, 10.0)], 2),        # L=30+10+30=70 待核
    }
    wps, turns = P[args.path]
    if args.path == 'Q40':
        wps, turns = ([(20.0, 0.0), (20.0, 10.0), (0.0, 10.0)], 2)   # L=20+10+20=50, A=2
    if args.path == 'L60':
        wps, turns = ([(40.0, 0.0), (40.0, 10.0), (0.0, 10.0)], 2)   # L=40+10+40=90, A=2

    os.makedirs(args.outdir, exist_ok=True)
    rclpy.init()
    node = Flight(dict(waypoints=wps, turns=turns, alt=args.alt, hover=args.hover))
    summary = {}
    try:
        summary = node.run()
    except Exception as exc:
        print('ERR', repr(exc), flush=True)
        try:
            node.set_mode('AUTO.LAND')
            node.arm(False)
        except Exception:
            pass
    finally:
        tag = '%s_%s_h%.0f' % (args.tag, args.path, args.hover)
        with open(os.path.join(args.outdir, tag + '.csv'), 'w', newline='') as f:
            keys = ['wall', 'phase', 'x', 'y', 'z', 'yaw', 'volt', 'pct', 'mode']
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in node.rows:
                w.writerow(r)
        with open(os.path.join(args.outdir, tag + '.json'), 'w') as f:
            json.dump(summary, f, indent=1)
        print('wrote', tag, len(node.rows), flush=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
