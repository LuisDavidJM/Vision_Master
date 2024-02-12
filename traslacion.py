import cv2
import numpy as np

#-------------PROCESO DE DETECCIÓN DE OBJETO--------------------
# Se carga la imagen original y la lee en escala de grises
ruta_imagen = 'lejos.jpg'
imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

# Se aplica un filtro de suavizado para reducir el ruido
imagen_suavizada = cv2.GaussianBlur(imagen, (5, 5), 0)

# Se binariza la imagen para obtener una imagen binaria
_, imagen_binaria = cv2.threshold(imagen_suavizada, 120, 255, cv2.THRESH_BINARY)

# Se detectan los bordes del objeto
bordes = cv2.Canny(imagen_binaria, threshold1=100, threshold2=200)

# Se encontran los contornos del objeto a partir de los bordes detectados
contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Se selecciona el contorno mas grande, que corresponde al objeto
contorno_objeto = max(contornos, key=cv2.contourArea)

#------------PREPARACIÓN PARA EL TRASLADO DEL OBJETO--------------
# Se crea una máscara en blanco del mismo tamaño que la imagen original
mascara = np.zeros_like(imagen)

# Se rellena el contorno del objeto en la máscara en blanco
cv2.fillPoly(mascara, [contorno_objeto], (255, 255, 255))

# Se crea una imagen solo del objeto usando la máscara
objeto = cv2.bitwise_and(imagen, mascara)

# Se dibuja el contorno del objeto para su visualización
imagen_contorno = cv2.drawContours(imagen.copy(), [contorno_objeto], -1, (255, 0, 0), 3)

#--------------------TRASLADO DEL OBJETO------------------------
# Se define la matriz de traslación con los valores de X y Y correcpondientes
tx, ty = 200, 50 
matriz_traslacion = np.float32([[1, 0, tx], [0, 1, ty]])

# Se traslada la máscara y el objeto en función de la matriz
mascara_trasladada = cv2.warpAffine(mascara, matriz_traslacion, (imagen.shape[1], imagen.shape[0]))
objeto_trasladado = cv2.warpAffine(objeto, matriz_traslacion, (imagen.shape[1], imagen.shape[0]))

#---------------CALCULO DE LOS MOMENTOS CENTRALES--------------------
# Se calculan los momentos del objeto antes de la traslación
momentos_antes = cv2.moments(contorno_objeto)
cx_antes = int(momentos_antes['m10']/momentos_antes['m00'])
cy_antes = int(momentos_antes['m01']/momentos_antes['m00'])

# Se encontran los contornos del objeto trasladado
contornos_trasladados, _ = cv2.findContours(mascara_trasladada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Se selecciona el contorno mas grande, que corresponde al objeto trasladado
objeto_trasladado_contorno = max(contornos_trasladados, key=cv2.contourArea)

# Se calculan los momentos del objeto despues de la traslación
momentos_trasladados = cv2.moments(objeto_trasladado_contorno)
cx_despues = int(momentos_trasladados['m10']/momentos_trasladados['m00'])
cy_despues = int(momentos_trasladados['m01']/momentos_trasladados['m00'])

print(f"Momento central antes: ({cx_antes}, {cy_antes})")
print(f"Momento central después: ({cx_despues}, {cy_despues})")

#Se muestra el objeto original y el objeto trasladado
cv2.imshow('Objeto original', imagen_contorno)
cv2.imshow('Objeto trasladado', objeto_trasladado)
cv2.waitKey(0)
cv2.destroyAllWindows()