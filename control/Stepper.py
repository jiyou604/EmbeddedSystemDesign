import RPi.GPIO as GPIO
import time
class Motor:
    def __init__(self, pins: list):
        self.pins = list(pins)
        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        self.counter = 0
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

    def step_once(self, direction):
        for i, pin in enumerate(self.pins):
            GPIO.output(pin, self.sequence[self.counter][i])
        if direction > 0:
            self.counter = (self.counter - 1) % 8
        else:
            self.counter = (self.counter + 1) % 8
        time.sleep(1e-3)

        self.cleanup()


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

class Platform:
    def __init__(self, motors: list):
        self.motor_x0 = motors[0]
        self.motor_x1 = motors[1]
        self.motor_y0 = motors[2]
        self.motor_y1 = motors[3]

    def tilt(self, x_axis, y_axis):
        if abs(x_axis) >= abs(y_axis):
            for i in range(abs(x_axis)):
                dir_x0 = 1 if x_axis > 0 else -1
                dir_x1 = -dir_x0
                self.motor_x0.step_once(dir_x0)
                self.motor_x1.step_once(dir_x1)

                if i > abs(y_axis):
                    continue

                dir_y0 = 1 if y_axis > 0 else -1
                dir_y1 = -dir_y0
                self.motor_y0.step_once(dir_y0)
                self.motor_y1.step_once(dir_y1)

        if abs(x_axis) < abs(y_axis):
            for i in range(abs(y_axis)):
                dir_y0 = 1 if y_axis > 0 else -1
                dir_y1 = -dir_y0
                self.motor_y0.step_once(dir_y0)
                self.motor_y1.step_once(dir_y1)

                if i > abs(x_axis):
                    continue

                dir_x0 = 1 if x_axis > 0 else -1
                dir_x1 = -dir_x0
                self.motor_x0.step_once(dir_x0)
                self.motor_x1.step_once(dir_x1)

        



        