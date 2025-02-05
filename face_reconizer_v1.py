import cv2
from deepface import DeepFace

# open webcam:
cap = cv2.VideoCapture(0)

# while loop:
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # usng the Deep face lib:
    try:
        result = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'])

        # Display the results on the frame
        age = result['age']
        gender = result['gender']
        race = result['dominant_race']
        emotion = result['dominant_emotion']

        text = f"Age: {age}, Gender: {gender}, Race: {race}, Emotion: {emotion}"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"Error: {e}")

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing the programs:
cap.release()
cv2.destroyAllWindows()

