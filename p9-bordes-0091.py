import cv2
import os

# ian gutierrez NC 0091
# Cargar imagen

imagen = cv2.imread("camello 0091.jpg")

# Comprobar imagen
if imagen is None:
    print("Error: no se pudo cargar la imagen.")
    print("Revisa que 'camello 0091.jpg' esté en la misma carpeta que este programa.")
    exit()

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Convertir a imagen binaria mediante umbral
_, binaria = cv2.threshold(
    gris,
    127,
    255,
    cv2.THRESH_BINARY
)

# Detectar contornos
contornos, jerarquia = cv2.findContours(
    binaria,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Dibujar los contornos
resultado = imagen.copy()

cv2.drawContours(
    resultado,
    contornos,
    -1,
    (0, 255, 0),
    2
)

# Mostrar resultados
cv2.imshow("Imagen original 0091", imagen)
cv2.imshow("Imagen binaria 0091", binaria)
cv2.imshow("Contornos detectados 0091", resultado)

# Crear carpeta resultados2 si no existe
os.makedirs("resultados2", exist_ok=True)

# Guardar resultado
cv2.imwrite(
    "resultados2/camello 0091.jpg",
    resultado
)

print("Cantidad de contornos encontrados:", len(contornos))
print("Resultado guardado en resultados2/camello 0091.jpg")

# Esperar
cv2.waitKey(0)

# Cerrar ventanas
cv2.destroyAllWindows()

print("programa realizado por ian gutierrez NC = 0091")