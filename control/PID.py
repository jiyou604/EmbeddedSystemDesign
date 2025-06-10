import time

class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.last_error = 0
        self.integral = 0
        self.derivative = 0
        self.dt = 0
        self.last_time = 0

    def compute(self, error):
        current_time = time.time()
        self.dt = current_time - self.last_time
        self.last_time = current_time

        if error*self.last_error < 0:
            self.integral = 0

        self.integral += error
        if self.integral != 0:
            self.integral = max(abs(self.integral), 2000) * (abs(self.integral) // self.integral)
        self.derivative = (error - self.last_error)//self.dt

        output = self.kp*error + self.ki*self.integral + self.kd*self.derivative
        self.last_error = error

        return output
