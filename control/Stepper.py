import RPi.GPIO as GPIO
import time
import math

class Motor:
    def __init__(self, pins: list, max_step=1000):
        self.pins = list(pins)
        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        self.counter = 0
        self.position = 0
        self.max_step = max_step
        self.sequence = [[1,0,0,1],
                         [1,0,0,0],
                         [1,1,0,0],
                         [0,1,0,0],
                         [0,1,1,0],
                         [0,0,1,0],
                         [0,0,1,1],
                         [0,0,0,1]]

    def cleanup(self):
        for pin in self.pins:
            GPIO.output(pin, GPIO.LOW)

    def step_once(self, direction, delay=1e-3):
        if (self.position >= self.max_step and direction > 0) or \
           (self.position <= -self.max_step and direction < 0):
            return  # 제한 초과 → 무시
        else:        
            for i, pin in enumerate(self.pins):
                GPIO.output(pin, self.sequence[self.counter][i])
            if direction > 0:
                self.counter = (self.counter - 1) % 8
            else:
                self.counter = (self.counter + 1) % 8
            time.sleep(delay)

            self.position += direction

        # self.cleanup()


class MotorPair:
    def __init__(self, motors: list):
        self.motor0 = motors[0]
        self.motor1 = motors[1]
        self.pins0 = self.motor0.pins
        self.pins1 = self.motor1.pins

    def tilt(self, steps):
        for _ in range(abs(steps)):
                dir0 = 1 if steps > 0 else -1
                dir1 = -dir0
                self.motor0.step_once(dir0)
                self.motor1.step_once(dir1)

    def cleanup(self):
        self.motor0.cleanup()
        self.motor1.cleanup()
        
class Platform:
    def __init__(self, motors: list):
        self.motor_x0 = motors[0]
        self.motor_x1 = motors[1]
        self.motor_y0 = motors[2]
        self.motor_y1 = motors[3]
        self.toggle_flag_x = True
        self.toggle_flag_y = True

    def tilt(self, x_axis, y_axis, toggle_mode_x=False, toggle_mode_y=False):
        steps = max(abs(x_axis), abs(y_axis))

        dir_x0 = 1 if x_axis > 0 else -1
        dir_x1 = -dir_x0
        dir_y0 = 1 if y_axis > 0 else -1
        dir_y1 = -dir_y0

        for i in range(steps):
            delay = 1e-3*(1 - math.pow(10,-i)) 
            if i < abs(x_axis):
                if not toggle_mode_x:
                    self.motor_x0.step_once(dir_x0, delay)
                    self.motor_x1.step_once(dir_x1, delay)
                else:
                    if self.toggle_flag_x:
                        self.motor_x0.step_once(dir_x0, delay)
                    else:
                        self.motor_x1.step_once(dir_x1, delay)
                    self.toggle_flag_x = not self.toggle_flag_x

            if i < abs(y_axis):
                if not toggle_mode_y:
                    self.motor_y0.step_once(dir_y0, delay)
                    self.motor_y1.step_once(dir_y1, delay)
                else:
                    if self.toggle_flag_y:
                        self.motor_y0.step_once(dir_y0, delay)
                    else:
                        self.motor_y1.step_once(dir_y1, delay)
                    self.toggle_flag_y = not self.toggle_flag_y


    def cleanup(self):
        self.motor_x0.cleanup()
        self.motor_x1.cleanup()
        self.motor_y0.cleanup()
        self.motor_y1.cleanup()