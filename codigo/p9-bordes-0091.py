import cv2
import os
import numpy as np

# ==============================================================================
# EJEMPLO 2 — Detección de contornos
# Camello
# Ian Gutierrez NC 0091
# ==============================================================================

# Carpeta donde está el código
carpeta_codigo = os.path.dirname(os.path.abspath(__file__))

# Ruta de la imagen
ruta_imagen = os.path.join(
    carpeta_codigo,
    "..",
    "imagenes",
    "camello 0091.jpg"
)

# Carpeta de resultados
carpeta_resultados = os.path.join(
    carpeta_codigo,
    "..",
    "resultados2"
)

os.makedirs(carpeta_resultados, exist_ok=True)

# Cargar imagen
imagen = cv2.imread(ruta_imagen)

# Comprobar imagen
if imagen is None:
    print("ERROR: No se encontró la imagen")
    print(ruta_imagen)
    input("Presiona ENTER para cerrar...")
    exit()

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Crear imagen binaria
_, binaria = cv2.threshold(
    gris,
    127,
    255,
    cv2.THRESH_BINARY
)

# Detectar bordes
bordes = cv2.Canny(
    gris,
    100,
    200
)

# Copiar imagen original
resultado = imagen.copy()

# Obtener puntos de los bordes
puntos = np.argwhere(bordes > 0)

# Dibujar puntitos verdes
for punto in puntos[::5]:
    y, x = punto

    cv2.circle(
        resultado,
        (x, y),
        2,
        (0, 255, 0),
        -1
    )

# ==============================================================================
# GUARDAR LAS 3 IMÁGENES
# ==============================================================================

cv2.imwrite(
    os.path.join(
        carpeta_resultados,
        "camello_original_0091.jpg"
    ),
    imagen
)

cv2.imwrite(
    os.path.join(
        carpeta_resultados,
        "camello_binaria_0091.jpg"
    ),
    binaria
)

cv2.imwrite(
    os.path.join(
        carpeta_resultados,
        "camello_contornos_destacados_0091.jpg"
    ),
    resultado
)

# ==============================================================================
# MOSTRAR LAS 3 IMÁGENES
# ==============================================================================

cv2.imshow(
    "1 - Camello original camello 0091",
    imagen
)

cv2.imshow(
    "2 - Imagen binaria camello 0091",
    binaria
)

cv2.imshow(
    "3 - contornos destacados camello 0091",
    resultado
)

print("----------------------------------------")
print("PROCESAMIENTO TERMINADO")
print("----------------------------------------")
print("Las 3 imágenes fueron guardadas en resultados2.")

cv2.waitKey(0)
cv2.destroyAllWindows()


print("programa ralizado por Ian Gutierrez NC 0091")