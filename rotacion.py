import cv2
import matplotlib.pyplot as plt

# Función para binarizar imagene original, retornando la imagen binaria
def binarizar_imagen(ruta_imagen):
    imagen = cv2.imread(ruta_imagen)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen_suavizada = cv2.GaussianBlur(gris, (5, 5), 0)
    _, imagen_binaria = cv2.threshold(imagen_suavizada, 100, 255, cv2.THRESH_BINARY)
    return imagen_binaria

#Función para rotar la imagen en los angulos seleccionados
def rotar_imagen(imagen, angulo):
    altura, anchura = imagen.shape[:2]
    punto_rotacion = (anchura // 2, altura // 2)
    matriz_rotacion = cv2.getRotationMatrix2D(punto_rotacion, angulo, 1.0)
    imagen_rotada = cv2.warpAffine(imagen, matriz_rotacion, (anchura, altura))
    return imagen_rotada

# Función para calcular los primeros cuatro momentos de Hu
def calcular_momentos_hu(imagen):
    momentos = cv2.moments(imagen)
    momentos_hu = cv2.HuMoments(momentos)
    return momentos_hu[:3]

# Se binariza la imagen original
imagen_binaria = binarizar_imagen('cerca.jpg')

# Se designan los angulos de rotación en un array
angulos = [15, 45, 90, 180]

# Se crea la figura y ejes en Matplotlib
fig, axs = plt.subplots(1, len(angulos) + 1, figsize=(13, 5))

# Se muestra la imagen original al inicio
axs[0].imshow(imagen_binaria, cmap='gray')
axs[0].set_title("Original")
axs[0].axis('off')

# Se crea una tabla con los valores de los momentos de Hu
print(f"{'Ángulo':<10} {'Hu1':<23} {'Hu2':<23} {'Hu3':<23}")
print('-' * 80)

# Se recorren el array con los angulos y se usan en la funcion para rotar
# También se muestran las imagenes rotadas y los valores de los momentos de Hu
for i, angulo in enumerate(angulos):
    imagen_rotada = rotar_imagen(imagen_binaria, angulo)
    axs[i + 1].imshow(imagen_rotada, cmap='gray')
    axs[i + 1].set_title(f"Rotada {angulo}°")
    axs[i + 1].axis('off')
    momentos_hu = calcular_momentos_hu(imagen_rotada)
    print(f"{angulo:<10} {momentos_hu[0][0]:<23} {momentos_hu[1][0]:<23} {momentos_hu[2][0]:<23}")

#Se muestran todas las imagenes
plt.show()