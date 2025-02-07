import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import RPi.GPIO as GPIO  # For Raspberry Pi GPIO control
import time

class SmallRobotCar:
    def __init__(self):
        # GPIO Setup (adjust pin numbers as needed for your car)
        self.MOTOR_LEFT_FWD = 17
        self.MOTOR_LEFT_BWD = 18
        self.MOTOR_RIGHT_FWD = 22
        self.MOTOR_RIGHT_BWD = 23
        self.setup_gpio()

        # Camera setup
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Create and compile the model
        self.model = self.build_simple_model()

    def build_simple_model(self):
        model = Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=(120, 160, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')  # [left, forward, right]
        ])
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.MOTOR_LEFT_FWD, GPIO.OUT)
        GPIO.setup(self.MOTOR_LEFT_BWD, GPIO.OUT)
        GPIO.setup(self.MOTOR_RIGHT_FWD, GPIO.OUT)
        GPIO.setup(self.MOTOR_RIGHT_BWD, GPIO.OUT)

    def move_forward(self):
        GPIO.output(self.MOTOR_LEFT_FWD, GPIO.HIGH)
        GPIO.output(self.MOTOR_RIGHT_FWD, GPIO.HIGH)
        GPIO.output(self.MOTOR_LEFT_BWD, GPIO.LOW)
        GPIO.output(self.MOTOR_RIGHT_BWD, GPIO.LOW)

    def turn_left(self):
        GPIO.output(self.MOTOR_LEFT_FWD, GPIO.LOW)
        GPIO.output(self.MOTOR_RIGHT_FWD, GPIO.HIGH)
        GPIO.output(self.MOTOR_LEFT_BWD, GPIO.HIGH)
        GPIO.output(self.MOTOR_RIGHT_BWD, GPIO.LOW)

    def turn_right(self):
        GPIO.output(self.MOTOR_LEFT_FWD, GPIO.HIGH)
        GPIO.output(self.MOTOR_RIGHT_FWD, GPIO.LOW)
        GPIO.output(self.MOTOR_LEFT_BWD, GPIO.LOW)
        GPIO.output(self.MOTOR_RIGHT_BWD, GPIO.HIGH)

    def stop(self):
        GPIO.output(self.MOTOR_LEFT_FWD, GPIO.LOW)
        GPIO.output(self.MOTOR_RIGHT_FWD, GPIO.LOW)
        GPIO.output(self.MOTOR_LEFT_BWD, GPIO.LOW)
        GPIO.output(self.MOTOR_RIGHT_BWD, GPIO.LOW)

    def process_frame(self, frame):
        # Preprocess image
        resized = cv2.resize(frame, (160, 120))
        normalized = resized / 255.0

        # Get prediction
        prediction = self.model.predict(np.expand_dims(normalized, axis=0))[0]

        # Simple obstacle detection using color thresholding
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, 0, 0])  # Adjust these values based on your obstacles
        upper_bound = np.array([180, 255, 50])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Check if obstacle is too close (in the bottom third of the image)
        bottom_region = mask[80:120, :]
        obstacle_detected = np.sum(bottom_region) > 10000  # Adjust threshold as needed

        return prediction, obstacle_detected

    def train_model(self, training_data, labels, epochs=10):
        """
        Training function - you'll need to create a dataset of images and corresponding
        direction labels (left/forward/right)
        """
        self.model.fit(training_data, labels, epochs=epochs, validation_split=0.2)

    def run(self):
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    continue

                prediction, obstacle_detected = self.process_frame(frame)

                if obstacle_detected:
                    self.stop()
                    time.sleep(0.5)
                    self.turn_right()  # Or implement more sophisticated avoidance
                    time.sleep(0.5)
                else:
                    # Get direction with highest probability
                    direction = np.argmax(prediction)

                    if direction == 0:  # Left
                        self.turn_left()
                    elif direction == 1:  # Forward
                        self.move_forward()
                    else:  # Right
                        self.turn_right()

                time.sleep(0.1)  # Control loop rate

        except KeyboardInterrupt:
            self.stop()
            GPIO.cleanup()
            self.camera.release()

if __name__ == "__main__":
    car = SmallRobotCar()
    # First collect training data and train the model
    # Then run the car
    car.run()
