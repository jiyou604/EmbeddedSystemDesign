import Stepper
import RPi.GPIO as GPIO

# initialization
GPIO.setmode(GPIO.BCM)
pins_x0 = [14, 15, 17, 18]
pins_x1 = [27, 22, 23, 24]
pins_y0 = [10, 9, 25, 11]
pins_y1 = [16, 26, 20, 21]

motor_x0 = Stepper.Motor(pins_x0)
motor_x1 = Stepper.Motor(pins_x1)
motor_y0 = Stepper.Motor(pins_y0)
motor_y1 = Stepper.Motor(pins_y1)

platform = Stepper.Platform([motor_x0, motor_x1, motor_y0, motor_y1])

x_axis = Stepper.MotorPair([motor_x0, motor_x1])
y_axis = Stepper.MotorPair([motor_y0, motor_y1])

platform.tilt(0, 100)

# for i in range(100):
#     platform.tilt(100, 0)
#     platform.tilt(0, 100)
#     platform.tilt(-100, 0)
#     platform.tilt(0, -100)
    