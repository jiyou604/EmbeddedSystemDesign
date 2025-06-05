import Stepper
import RPi.GPIO as GPIO

# initialization
GPIO.setmode(GPIO.BCM)

pins_x0 = [17, 18, 27, 22]
# pins_x1 = [17, 18, 27, 22]
# pins_y0 = [17, 18, 27, 22]
# pins_y1 = [17, 18, 27, 22]

motor_x0 = Stepper.motor(pins_x0)
# motor_x1 = Stepper.motor(pins_x0)
# motor_y0 = Stepper.motor(pins_x0)
# motor_y1 = Stepper.motor(pins_x0)

motor_x0.rotate(4096, 6e-4)
motor_x0.rotate(-4096, 7e-4)

motor_x0.rotate(1024, 1e-3)
motor_x0.rotate(-4096, 7e-4)

motor_x0.cleanup()
