import cv2

# Inicjalizacja przechwytywania wideo z pierwszego urządzenia kamery
cap = cv2.VideoCapture(0)

# Załaduj klasyfikator Haar do wykrywania twarzy
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konwersja obrazu do skali szarości
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wykrywanie twarzy
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Rysowanie prostokątów wokół twarzy
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Wyświetlanie wyniku
    cv2.imshow('Rozpoznawanie twarzy', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

