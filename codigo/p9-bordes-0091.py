import cv2
import os

# ==============================================================================
# EJEMPLO 2 — Detección de contornos
# Camello
# Ian Gutierrez NC 0091
# ==============================================================================

# Obtener la carpeta donde está el código
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))

# Buscar la imagen en la carpeta del código
ruta_imagen_local = os.path.join(
    DIRECTORIO_ACTUAL,
    "camello 0091.jpg"
)

# Buscar la imagen una carpeta arriba
ruta_imagen_raiz = os.path.join(
    DIRECTORIO_ACTUAL,
    "..",
    "camello 0091.jpg"
)

# Elegir la ruta donde se encuentre la imagen
if os.path.exists(ruta_imagen_local):
    ruta_imagen = ruta_imagen_local
elif os.path.exists(ruta_imagen_raiz):
    ruta_imagen = ruta_imagen_raiz
else:
    ruta_imagen = "camello 0091.jpg"

# Crear carpeta de resultados
carpeta_resultados = os.path.join(
    DIRECTORIO_ACTUAL,
    "..",
    "resultados"
)

os.makedirs(carpeta_resultados, exist_ok=True)

# Cargar imagen
imagen = cv2.imread(ruta_imagen)

# Comprobar imagen
if imagen is None:
    print("Error: no se pudo cargar la imagen.")
    exit()

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Convertir a imagen binaria
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

# Guardar resultado
ruta_guardado = os.path.join(
    carpeta_resultados,
    "camello_contornos_0091.jpg"
)

cv2.imwrite(ruta_guardado, resultado)

# Mostrar información
print("--- PROCESAMIENTO EXITOSO ---")
print("Cantidad de contornos encontrados:", len(contornos))
print("Resultado guardado en:", ruta_guardado)
print("Programa realizado por Ian Gutierrez NC = 0091")

# Mostrar imágenes
cv2.imshow("Imagen original 0091", imagen)
cv2.imshow("Imagen binaria 0091", binaria)
cv2.imshow("Contornos detectados 0091", resultado)

cv2.waitKey(0)
cv2.destroyAllWindows()
