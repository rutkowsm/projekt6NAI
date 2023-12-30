
import cv2

# Inicjalizacja przechwytywania wideo z pierwszego urządzenia kamery
cap = cv2.VideoCapture(0)

while True:
    # Przechwytuje klatka po klatce
    ret, frame = cap.read()

    # Jeśli klatka jest poprawnie przechwycona, ret jest True
    if not ret:
        print("Nie można przechwycić obrazu z kamery. Sprawdź podłączenie kamery.")
        break

    # Wyświetlanie klatki
    cv2.imshow('Obraz z kamery', frame)

    # Wyjdź z pętli, jeśli użytkownik naciśnie 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Gdy wszystko zostało zakończone, zwolnij uchwyt przechwytywania
cap.release()
cv2.destroyAllWindows()
