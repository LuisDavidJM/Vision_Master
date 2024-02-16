import cv2

# Se carga la imagen original y la lee en escala de grises
ruta_imagen = 'Imagen1.jpg'
imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

# Se binariza la imagen para obtener una imagen binaria
_, imagen_binaria = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)

# Se revierte la imagen binaria para que se use el fondo negro
imagen_binaria_invertida = cv2.bitwise_not(imagen_binaria)

# Se encuentran los componentes conectados para la vecindad 4 utilizando la imagen binaria invertida
num_etiquetas_v4, _, _, _ = cv2.connectedComponentsWithStats(imagen_binaria_invertida, connectivity=4)

# Se encuentran los componentes conectados para la vecindad 8 utilizando la imagen binaria invertida
num_etiquetas_v8, _, _, _ = cv2.connectedComponentsWithStats(imagen_binaria_invertida, connectivity=8)

# Se le resta 1 para excluir el fondo del conteo
componentes_conectados_v4 = num_etiquetas_v4 - 1
componentes_conectados_v8 = num_etiquetas_v8 - 1

# Se mmprime el número de componentes conectados
print("Número de componentes conectados en vedindad 4: ", componentes_conectados_v4)
print("Número de componentes conectados en vecindad 8: ", componentes_conectados_v8)
