import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Load your 12-class model
model_path = 'waste_classifier_12classes.keras'
print("Loading model...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# 2. Dictionary of all 12 specific classes
CLASS_NAMES = {
    0: 'battery', 
    1: 'biological', 
    2: 'brown-glass', 
    3: 'cardboard', 
    4: 'clothes', 
    5: 'green-glass', 
    6: 'metal', 
    7: 'paper', 
    8: 'plastic', 
    9: 'shoes', 
    10: 'trash', 
    11: 'white-glass'
}

# 3. Setup the Broad Category Logic
# If the prediction is in this list, it is Hazardous. Otherwise, Non-Hazardous.
HAZARDOUS_ITEMS = ['battery'] 

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' in the video window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the camera for intuitive movement
    frame = cv2.flip(frame, 1)

    # Define the Targeting Box
    h, w, _ = frame.shape
    box_size = 224 
    start_x = (w - box_size) // 2
    start_y = (h - box_size) // 2
    end_x = start_x + box_size
    end_y = start_y + box_size

    # Draw the targeting rectangle
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
    cv2.putText(frame, "Place Waste Here", (start_x, start_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Crop and Preprocess
    roi = frame[start_y:end_y, start_x:end_x]
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(rgb_roi, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    class_idx = np.argmax(predictions)
    
    # Get the specific label and confidence
    specific_label = CLASS_NAMES[class_idx]
    confidence = predictions[class_idx]

    # Map to the broad category
    if specific_label in HAZARDOUS_ITEMS:
        broad_category = "Hazardous"
        color = (0, 0, 255) # Red text
    else:
        broad_category = "Non-Hazardous"
        color = (0, 200, 0) # Green text

    # 4. Visual Overlay (Stacked Text like the Kaggle output)
    text_line1 = f"Pred: {broad_category}"
    text_line2 = f"Type: {specific_label} ({confidence*100:.1f}%)"

    # Draw the text at the top left of the screen
    cv2.putText(frame, text_line1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(frame, text_line2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # Show the video feed
    cv2.imshow('Live Waste Classifier', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
