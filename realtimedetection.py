import cv2
import numpy as np
from keras.models import model_from_json


def load_emotion_model():
    try:
        from keras.models import load_model
        return load_model("emotiondetector.h5", compile=False)
    except Exception:
        json_file = open("emotiondetector.json", "r")
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
    
        model.load_weights("emotiondetector.h5")
        return model

model = load_emotion_model()

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

def open_webcam_with_fallback():
    # Try common indices
    for index in [0, 1, 2, 3]:
        cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cam.isOpened():
            return cam
        cam.release()
    # Last resort without CAP_DSHOW
    cam = cv2.VideoCapture(0)
    if cam.isOpened():
        return cam
    return cam

webcam = open_webcam_with_fallback()
if not webcam.isOpened():
    raise RuntimeError("Could not open any webcam. Ensure camera permissions and that no other app is using it.")

labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}

while True:
    success, im = webcam.read()
    if not success or im is None:
        cv2.waitKey(10)
        continue

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    try:
        for (p,q,r,s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p,q), (p+r, q+s), (255,0,0), 2)
            image = cv2.resize(image, (48,48))
            img = extract_features(image)
            pred = model.predict(img, verbose=0)
            prediction_label = labels[int(np.argmax(pred))]
            cv2.putText(im, '% s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255))
        cv2.imshow("Output", im)
        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break
    except cv2.error:
        pass

webcam.release()
cv2.destroyAllWindows()
