import cv2

# Función para binarizar las imagenes, retornando la imagen binaria
def binarizar_imagen(ruta_imagen):
    imagen = cv2.imread(ruta_imagen)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen_suavizada = cv2.GaussianBlur(gris, (5, 5), 0)
    _, imagen_binaria = cv2.threshold(imagen_suavizada, 100, 255, cv2.THRESH_BINARY)
    return imagen_binaria

# Función para calcular los primeros cuatro momentos de Hu
def calcular_momentos_hu(imagen):
    momentos = cv2.moments(imagen)
    momentos_hu = cv2.HuMoments(momentos)
    return momentos_hu[:4]

# Se binarizan las imagenes
img_cercana = binarizar_imagen('cerca.jpg')
img_lejana = binarizar_imagen('lejos.jpg')

# Se calculan los momentos de Hu para las imagenes
momentos_hu1 = calcular_momentos_hu(img_cercana)
momentos_hu2 = calcular_momentos_hu(img_lejana)

# Representación de la formulas
formulas = [
    "I1 = η20 + η02",
    "I2 = (η20 - η02)^2 + 4η11^2",
    "I3 = (η30 - 3η12)^2 + (3η21 - η03)^2",
    "I4 = (η30 + η12)^2 + (η21 + η03)^2"
]

# Se crea una tabla con los valores de los momentos de Hu
print(f"{'Fórmula':<40} {'Imagen 1':<25} {'Imagen 2':<20}")
print('-' * 90)

# Se recorren los array con los valores para imprimirse
for i, formula in enumerate(formulas):
    print(f"{formula:<40} {momentos_hu1[i][0]:<25} {momentos_hu2[i][0]:<20}")

# Se muestran las dos imagenes de la escala
cv2.imshow('Cerca', img_cercana)
cv2.imshow('Lejos', img_lejana)
cv2.waitKey(0)
cv2.destroyAllWindows()