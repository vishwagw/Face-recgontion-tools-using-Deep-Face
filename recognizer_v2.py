import cv2
from deepface import DeepFace

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Preload a reference image for comparison
reference_img = cv2.imread('reference.jpg')  # Replace with the known person's image

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Perform face recognition
        result = DeepFace.verify(frame, reference_img, enforce_detection=False)

        # Display result on the frame
        if result['verified']:
            cv2.putText(frame, "Match Found!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Match", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    except Exception as e:
        print(f"Error: {e}")

    # Show the frame
    cv2.imshow('Real-Time Face Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
