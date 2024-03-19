import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('number_recog_model.keras')
#model.summary()

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    # Get the size of the video frames
    height, width = frame.shape[:2]

    # Calculate the coordinates of the box
    top_left = (width // 2 - 50, height // 2 - 50)
    bottom_right = (width // 2 + 50, height // 2 + 50)

    # Draw the box
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # Get the region of interest
    roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    gray_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, black_white_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
    resized_img = cv2.resize(black_white_img, (28, 28)).astype('float32')

    resized_img = np.expand_dims(resized_img, axis=0)
    resized_img = np.expand_dims(resized_img, axis=3)
    resized_img = resized_img / 255.0

    result = model.predict(resized_img)
    predicted_label = np.argmax(result)
    percentages = result[0] * 100

    # Display the predictions and percentages on the image
    label_text = f"Label: {predicted_label}"
    cv2.putText(frame, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for i, percentage in enumerate(percentages):
        percentage_text = f"{i}: {percentage:.2f}%"
        cv2.putText(frame, percentage_text, (50, 100 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()