
from pid import PID
import rospy
from lowpass import LowPassFilter
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


"""
the implementation closely replicates that of Aaron Brown:
https://github.com/awbrown90/CarND-Capstone/blob/master/ros/src/twist_controller/twist_controller.py
"""

class Controller(object):
    def __init__(self, vehicle):

        assert isinstance(vehicle, Vehicle)
        self.vehicle = vehicle

        self.accel_pid = PID(kp=.4, ki=.1, kd=0., mn=0., mx=1.)

        self.speed_pid = PID(kp=2., ki=0., kd=0.,
                             mn=self.vehicle.deceleration_limit,
                             mx=self.vehicle.acceleration_limit)

        self.last_velocity = 0.

        self.curr_fuel = LowPassFilter(tau=60.0, ts=0.1)
        self.lpf_accel = LowPassFilter(tau=.5, ts=0.02)

        self.yaw_control = YawController(wheel_base=self.vehicle.wheel_base,
                                         steer_ratio=self.vehicle.steer_ratio,
                                         min_speed=4. * ONE_MPH,
                                         max_lat_accel=self.vehicle.max_lat_acceleration,
                                         max_steer_angle=self.vehicle.max_steer_angle)
        self.last_ts = None

    def control(self, linear_velocity, angular_velocity, current_velocity):
        time_elapsed = self.time_elapsed()
        if time_elapsed > 1. / 5 or time_elapsed < 1e-4:
            self.reset_pids()
            self.last_velocity = current_velocity
            return 0., 0., 0.

        vehicle_mass = self.vehicle.gross_mass(curr_fuel=self.curr_fuel.get(), fuel_density=GAS_DENSITY)
        vel_error = linear_velocity - current_velocity

        if abs(linear_velocity) < ONE_MPH:
            self.speed_pid.reset()

        accel_cmd = self.speed_pid.step(vel_error, time_elapsed)

        min_speed = ONE_MPH * 5.
        if linear_velocity < .01:
            accel_cmd = min(accel_cmd, -530. / vehicle_mass / self.vehicle.wheel_radius)
        elif linear_velocity < min_speed:
            angular_velocity *= min_speed / linear_velocity
            linear_velocity = min_speed

        accel = (current_velocity - self.last_velocity) / time_elapsed

        _ = self.lpf_accel.filt(accel)
        self.last_velocity = current_velocity

        throttle, brake = 0., 0.

        if accel_cmd >= 0:
            throttle = self.accel_pid.step(accel_cmd - self.lpf_accel.get(),
                                           time_elapsed)
        else:
            self.accel_pid.reset()

        if (accel_cmd < -self.vehicle.brake_deadband) or (linear_velocity < min_speed):
            brake = -accel_cmd * vehicle_mass * self.vehicle.wheel_radius

        steering = self.yaw_control.get_steering(linear_velocity, angular_velocity, current_velocity)

        return throttle, brake, steering

    def reset_pids(self):
        self.accel_pid.reset()
        self.speed_pid.reset()

    def time_elapsed(self):
        now = rospy.get_time()
        if not self.last_ts:
            self.last_ts = now

        elapsed, self.last_ts = now - self.last_ts, now

        return elapsed


class Vehicle(object):
    def __init__(self):
        self.mass = rospy.get_param('~vehicle_mass', 1736.35)
        self.fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        self.brake_deadband = rospy.get_param('~brake_deadband', .1)
        self.deceleration_limit = rospy.get_param('~decel_limit', -5)
        self.acceleration_limit = rospy.get_param('~accel_limit', 1.)
        self.wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        self.wheel_base = rospy.get_param('~wheel_base', 2.8498)
        self.steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        self.max_lat_acceleration = rospy.get_param('~max_lat_accel', 3.)
        self.max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

    def gross_mass(self, curr_fuel, fuel_density=2.858):
        return self.mass + curr_fuel / 100.0 * self.fuel_capacity * fuel_density