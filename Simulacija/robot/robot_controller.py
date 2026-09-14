"""
robot/robot_controller.py
Thin, safety-conscious wrapper around ur_rtde for the UR3: motion, active
TCP configuration (derived from config.TOOL_TCP_OFFSET), and suction
control via a digital output. Adjust SUCTION_DO_INDEX / VACUUM_SENSOR_DI_INDEX
to match whatever you wire on the control box's I/O tab.
"""

import time
import rtde_control
import rtde_receive
import rtde_io
import config


SUCTION_DO_INDEX = 0           # digital output driving the suction valve/ejector
VACUUM_SENSOR_DI_INDEX = None  # digital input from a vacuum switch; set to an
                                 # int once wired, e.g. 0. None = no sensor yet.


class RobotController:
    def __init__(self):
        self.rtde_c = rtde_control.RTDEControlInterface(config.ROBOT_IP)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(config.ROBOT_IP)
        self.rtde_io = rtde_io.RTDEIOInterface(config.ROBOT_IP)
        self.set_tool_tcp()

    def set_tool_tcp(self):
        """Reloads the active TCP offset from config.TOOL_TCP_OFFSET so every
        downstream pose already accounts for the suction cup's length."""
        self.rtde_c.setTcp(config.TOOL_TCP_OFFSET)

    def go_home(self):
        self.rtde_c.moveJ(config.HOME_JOINTS,
                           config.DEFAULT_JOINT_SPEED,
                           config.DEFAULT_JOINT_ACCEL)

    def move_l(self, pose, speed=None, accel=None):
        self.rtde_c.moveL(pose,
                           speed or config.DEFAULT_LIN_SPEED,
                           accel or config.DEFAULT_LIN_ACCEL)

    def move_l_slow(self, pose):
        self.move_l(pose, config.APPROACH_LIN_SPEED, config.APPROACH_LIN_ACCEL)

    def suction_on(self):
        self.rtde_io.setStandardDigitalOut(SUCTION_DO_INDEX, True)

    def suction_off(self):
        self.rtde_io.setStandardDigitalOut(SUCTION_DO_INDEX, False)

    def suction_engaged(self, timeout_s=0.5):
        """Returns True once the vacuum sensor confirms a seal. If no
        sensor is wired yet, falls back to 'assume success' after a short
        settle time -- replace with real feedback as soon as you have one,
        since this fallback can't actually detect a failed pick."""
        if VACUUM_SENSOR_DI_INDEX is None:
            time.sleep(timeout_s)
            return True
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.rtde_r.getDigitalInState(VACUUM_SENSOR_DI_INDEX):
                return True
            time.sleep(0.02)
        return False

    def current_pose(self):
        return self.rtde_r.getActualTCPPose()

    def shutdown(self):
        try:
            self.suction_off()
        finally:
            self.rtde_c.stopScript()
