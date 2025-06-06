import Stepper # jiyou code
import time

# 4pin stepper, pin setting
pins_x = [17, 18, 27, 22]
pins_y = [23, 24, 25, 4]
motor_x = Stepper.motor(pins_x)
motor_y = Stepper.motor(pins_y)

# option 1. from pipe.py, we can read astar_path.txt or .npz(np list)
# option 2. from pipe.py, we directly dataload 'path = a_star(~)'

from pipe.py import path 

# for saving previous location
prev_y, prev_x = path[0]

for y, x in path[1:]:
    dx = x - prev_x
    dy = y - prev_y
  
  # each pixel changes = # of step on stepper 1
  # 1 pixel =/ 1 step
  # need correction
    step_x = dx  # now, 1 pixel = 1 step
    step_y = dy

    if step_x != 0:
        motor_x.rotate(step_x, 1e-3)
# step x, y !=0(is free), motor rotate >> move
    if step_y != 0:
        motor_y.rotate(step_y, 1e-3)

    prev_y, prev_x = y, x

    time.sleep(0.01)
# after total move, cleanup
motor_x.cleanup()
motor_y.cleanup()
