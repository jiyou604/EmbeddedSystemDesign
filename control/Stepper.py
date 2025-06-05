import RPi.GPIO as GPIO
import time

class motor():
    def __init__(self, pins: list):
        self.pins = list(pins)  # 집합(set) → 리스트로 변경 (순서 보장)
        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        self.counter = 0

    def cleanup(self):
        for pin in self.pins:
            GPIO.output(pin, GPIO.LOW)

    def rotate(self, steps, delay):
        sequence = [[1,0,0,1],
                    [1,0,0,0],
                    [1,1,0,0],
                    [0,1,0,0],
                    [0,1,1,0],
                    [0,0,1,0],
                    [0,0,1,1],
                    [0,0,0,1]]

        if delay < 6e-4:
            delay = 6e-4

        for i in range(abs(steps)):
            for idx, pin in enumerate(self.pins):  # enumerate 사용
                GPIO.output(pin, sequence[self.counter][idx])
            if steps > 0:
                self.counter = (self.counter - 1) % 8
            else:
                self.counter = (self.counter + 1) % 8
            time.sleep(delay)  # self.delay로 고정
