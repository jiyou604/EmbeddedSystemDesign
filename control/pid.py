class PID:
    def __init__(self, kp, ki, kd, setpoint=0, integral_limit=10, threshold=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.last_error = 0
        self.integral = 0
        self.derivative = 0
        self.integral_limit = integral_limit
        self.threshold = threshold
        self.just_crossed = False

    def compute(self, measurement):
        error = self.setpoint - measurement

        if self.last_error * error < 0:
            self.just_crossed = True
        else:
            self.just_crossed = False

        if abs(error) < self.threshold: # Deadband
            return 0

        else:
            self.integral += error
            self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
            self.derivative = error - self.last_error
            output = self.kp*error + self.ki*self.integral + self.kd*self.derivative
            if self.just_crossed:
                output = 0.2*output
                self.integral = 0
            self.last_error = error
            return output
