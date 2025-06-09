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

        self.integral += error
        self.derivative = (error - self.last_error)//self.dt

        # if abs(error) < 50:
        #     kp = 0.6*self.kp
        # elif abs(error) < 20:
        #     kp = 0.3*self.kp
        # elif abs(error) <7:
        #     kp = 0
        # else:
        #     kp = self.kp

        # if abs(self.derivative) > 600:
        #     kd = 1e-2*self.kd
        # elif abs(self.derivative) > 300:
        #     kd = 0.3*self.kd
        # elif abs(self.derivative) > 150:
        #     kd = 0.75*self.kd
        # elif abs(self.derivative) < 30:
        #     kd = 0
        # else:
        #     kd = self.kd

        output = self.kp*error + self.ki*self.integral + self.kd*self.derivative
        self.last_error = error
        
        # if abs(error) < 50:
        #     output *= 0.7
        # elif abs(error) < 20:
        #     output *= 0.3
        # elif abs(error) <7:
        #     output = 0
        # elif abs(error) > 250:
        #     output *=1.2
        return output
