import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('digit_model.h5')

cap = cv2.VideoCapture(0)

while True:
    loaded, frame = cap.read()
    if not loaded: break

    height, width, stuff = frame.shape
    x1, y1 = int(width/2 - 100), int(height/2 - 100)
    x2, y2 = x1 + 200, y1 + 200
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # cut out frame outside box
    area = frame[y1:y2, x1:x2]

    # grayscale
    gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)

    # removes gray shades for pure black and white. Inverts colors.
    # numbers > 128 are white at 255, otherwise black at 0
    # val is not needed for later
    val, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Downscale 200x200 to 28x28
    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 1 image that is 28x28, divided for 0-1 brightness
    imgInput = resized.reshape(1, 28, 28) / 255.0

    prediction = model.predict(imgInput, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    if confidence > 0.9:
        cv2.putText(frame, "Seeing: " + str(digit), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("AI View", thresh)
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()