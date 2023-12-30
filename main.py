import cv2
import copy

# Wymiary znaczników
CIRCLE_DIAMETER_DIVIDER = 2
CIRCLE_THICKNESS = 2
VIEWFINDER_LENGTH = 20
VIEWFINDER_THICKNESS = 2

#Strojenie Face Cascade
SCALE_FACTOR = 1.5
MIN_NEIGHBOURS = 6

# Inicjalizacja przechwytywania wideo z pierwszego urządzenia kamery
cap = cv2.VideoCapture(0)

# Załaduj klasyfikator Haar do wykrywania twarzy
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

prev_faces = []


def is_face_moved(current_faces, prev_faces, threshold=10):
    if len(prev_faces) != len(current_faces):
        return True  # Liczba twarzy się zmieniła

    for prev_face in prev_faces:
        px, py, pw, ph = prev_face
        moved = True  # Zakładamy, że twarz się poruszyła
        for curr_face in current_faces:
            x, y, w, h = curr_face
            if abs(x - px) < threshold and abs(y - py) < threshold:
                moved = False  # Znaleziono twarz w podobnym położeniu
                break
        if moved:
            return True  # Jeśli któraś twarz się poruszyła, zwróć True

    return False  # Żadna twarz się nie poruszyła



while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konwersja obrazu do skali szarości
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wykrywanie twarzy
    faces = face_cascade.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBOURS)

    if is_face_moved(faces, prev_faces):
        color = (0, 0, 255)  # Czerwony dla ruchomej twarzy
    else:
        color = (0, 255, 0)  # Zielony dla nieruchomej twarzy

    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        center_x, center_y = x + w // 2, y + h // 2
        cv2.circle(frame, center, w // CIRCLE_DIAMETER_DIVIDER, color, CIRCLE_THICKNESS) # Okrąg
        cv2.line(frame, (center_x, center_y - VIEWFINDER_LENGTH), (center_x, center_y + VIEWFINDER_LENGTH), color, VIEWFINDER_THICKNESS)  # Pionowa linia
        cv2.line(frame, (center_x - VIEWFINDER_LENGTH, center_y), (center_x + VIEWFINDER_LENGTH, center_y), color, VIEWFINDER_THICKNESS)  # Pozioma linia

    cv2.imshow('Rozpoznawanie twarzy i ruchu', frame)

    prev_faces = copy.deepcopy(faces)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
