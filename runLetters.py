import cv2
import numpy as np
import tensorflow as tf
import string

model = tf.keras.models.load_model('letter_model.h5')
alphabet = list(string.ascii_uppercase)

cap = cv2.VideoCapture(0)

sentence = ""
current_buffer = ""
buffer_count = 0
CONFIRMATION_THRESHOLD = 20

while True:
    loaded, frame = cap.read()
    if not loaded: break

    height, width, _ = frame.shape
    x1, y1 = int(width/2 - 100), int(height/2 - 100)
    cv2.rectangle(frame, (x1, y1), (x1+200, y1+200), (255, 0, 255), 2)

    area = frame[y1:y1+200, x1:x1+200]
    gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
    img_input = resized.reshape(1, 28, 28) / 255.0

    prediction = model.predict(img_input, verbose=0)
    letter_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    if confidence > 0.7:
        detected_letter = alphabet[letter_idx]
        
        #Only count if it's the SAME letter as the last frame
        if detected_letter == current_buffer:
            buffer_count += 1
        else:
            current_buffer = detected_letter
            buffer_count = 0
            
        # Add if > 20 frames
        if buffer_count == CONFIRMATION_THRESHOLD:
            sentence += detected_letter
            buffer_count = 0 # Reset
    else:
        buffer_count = 0 # Reset if AI sees nothing

    cv2.putText(frame, "Sentence: " + sentence, (50, height - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.putText(frame, f"Seeing: {current_buffer} ({buffer_count})", 
                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    cv2.imshow("Camera", frame)
    cv2.imshow("AI View", thresh)

    # --- CONTROLS ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'): # Clear sentence
        sentence = ""
    elif key == ord(' '): # Add a space
        sentence += " "

cap.release()
cv2.destroyAllWindows()