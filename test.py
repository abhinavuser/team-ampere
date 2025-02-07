import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import time

class BasicObjectDetection:
    def __init__(self):
        # Initialize camera parameters
        self.focal_length = 1000  # Need to calibrate for specific camera
        self.known_width = 2.0    # Average car width in meters

        # Create a simple CNN model for object detection
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(4, activation='sigmoid')  # [x, y, width, height]
        ])

    def estimate_distance(self, perceived_width):
        """
        Estimate distance using the principle of similar triangles
        Distance = (known_width * focal_length) / perceived_width
        """
        if perceived_width > 0:
            distance = (self.known_width * self.focal_length) / perceived_width
            return distance
        return None

    def process_frame(self, frame):
        """
        Process a single frame from the camera
        """
        # Preprocess the frame
        resized = cv2.resize(frame, (224, 224))
        normalized = resized / 255.0

        # Get object detection prediction
        prediction = self.model.predict(np.expand_dims(normalized, axis=0))[0]
        x, y, w, h = prediction

        # Convert predictions to pixel coordinates
        frame_h, frame_w = frame.shape[:2]
        box_x = int(x * frame_w)
        box_y = int(y * frame_h)
        box_w = int(w * frame_w)
        box_h = int(h * frame_h)

        # Estimate distance
        distance = self.estimate_distance(box_w)

        # Draw bounding box and distance
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
        if distance:
            cv2.putText(frame, f"Distance: {distance:.2f}m",
                       (box_x, box_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 2)

        return frame, distance

    def run_detection(self):
        """
        Main loop for continuous detection
        """
        cap = cv2.VideoCapture(0)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, distance = self.process_frame(frame)

                # Display the processed frame
                cv2.imshow('Object Detection', processed_frame)

                # Basic safety check - if object is too close
                if distance and distance < 2.0:  # 2 meters safety threshold
                    print("WARNING: Object too close!")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(0.1)  # Basic rate limiting

        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = BasicObjectDetection()
    detector.run_detection()
