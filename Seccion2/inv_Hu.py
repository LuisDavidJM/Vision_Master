import numpy as np
import cv2
from matplotlib import pyplot as plt

def rotar_imagen(imagen, angulo, cx, cy):
    # Se construye la matriz de transformación para la rotación
    M = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)  # El factor de escala es 1.0 porque solo queremos rotar

    # Se rota la imagen
    h, w = imagen.shape[:2]
    imagen_rotada = cv2.warpAffine(imagen, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
    return imagen_rotada


def trasladar_imagen(imagen_a_trasladar, dx, dy):
    # Se crea una nueva matriz llena de ceros (fondo)
    imagen_trasladada = np.zeros_like(imagen_a_trasladar)
    # Se obtienen los indices de los píxeles '1' en la imagen binaria
    y_indices, x_indices = np.nonzero(imagen_a_trasladar)
    # Se aplica la traslación a las posiciones de los píxeles '1', verificando limites de imagen
    translated_y_indices = np.clip(y_indices + dy, 0, imagen_a_trasladar.shape[0] - 1)
    translated_x_indices = np.clip(x_indices + dx, 0, imagen_a_trasladar.shape[1] - 1)
    # Se establece las posiciones de píxeles trasladadas a '1'
    imagen_trasladada[translated_y_indices, translated_x_indices] = 1
    return imagen_trasladada


# Función para calcular los momentos de Hu
def calcular_momentos_hu(imagen_binaria):
    momentos = cv2.moments(imagen_binaria)
    hu_momentos = cv2.HuMoments(momentos).flatten()
    return hu_momentos[:7]


def escalar_y_obtener_momentos_hu_img(ruta_archivo, factor_escala):
    # Se lee la imagen binaria desde un archivo de texto
    with open(ruta_archivo, 'r') as archivo:
        imagen_binaria = np.array([[int(char) for char in line.strip()] for line in archivo.readlines()],
                                  dtype=np.uint8)

    # Se calcula el centro de masa de la imagen
    M = cv2.moments(imagen_binaria.astype(np.float32))
    if M["m00"] == 0:
        print("Error: La imagen no puede estar vacía.")
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Se construye la matriz de transformación para la escala alrededor del centro de masa
    M = cv2.getRotationMatrix2D((cx, cy), 0, factor_escala)
    # Se escala la imagen
    h, w = imagen_binaria.shape[:2]
    imagen_escalada = cv2.warpAffine(imagen_binaria, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
    # Se calculan los momentos de Hu para la imagen escalada
    momentos = cv2.moments(imagen_escalada.astype(np.float32))
    hu_momentos = cv2.HuMoments(momentos).flatten()

    # Se mostran las imagenes
    mostrar_imagenes_con_matplotlib(
        [imagen_binaria, imagen_escalada],
        ['Imagen Binaria', 'Imagen Binaria Escalada']
    )
    # Se devolver los invariantes de Hu
    return hu_momentos[:7]


# Función para mostrar de mejor manera las imagenes
def mostrar_imagenes_con_matplotlib(imagenes, titulos_imagenes):
    plt.figure(figsize=(15, 5))  # Se establece el tamaño de la figura
    for i, (image, title) in enumerate(zip(imagenes, titulos_imagenes), start=1):
        plt.subplot(1, len(imagenes), i)
        plt.imshow(image, cmap='gray')  # Se muestra la imagen en escala de grises
        plt.title(title) 
        plt.axis('on')
    plt.show()


# Se convierte la imagen a un archivo txt con 0 y 1
def crear_archivo_txt_de_la_imagen(imagen_cargada, ruta_salida):
    renglones, columnas = imagen_cargada.shape[:2]
    with open(ruta_salida, 'w') as archivo_salida:
        for i in range(renglones):
            for j in range(columnas):
                valor_pixel = imagen_cargada[i, j]
                archivo_salida.write('0' if valor_pixel > 125 else '1')
            archivo_salida.write('\n')


# Se lee la imagen desde el archivo de texto
def procesar_imagen_desde_archivo(ruta_imagen, ruta_archivo_txt):
    imagen_cargada = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    crear_archivo_txt_de_la_imagen(imagen_cargada, ruta_archivo_txt)

if __name__ == "__main__":
    ruta_imagen = "planta.jpg"
    ruta_archivo_txt = 'salida_img.txt'
    procesar_imagen_desde_archivo(ruta_imagen, ruta_archivo_txt)


with open(ruta_archivo_txt, 'r') as file:
    imagen_binaria = np.array([[int(char) for char in line.strip()] for line in file.readlines()], dtype=np.uint8)

# Se traslada la imagene en las coordenadas especificadas
dx, dy = 300, 300
imagen_binaria_trasladada = trasladar_imagen(imagen_binaria, dx, dy)

# Se calculan los momentos de Hu para ambas imágenes
momentos_hu_originales = calcular_momentos_hu(imagen_binaria)
momentos_hu_trasladados = calcular_momentos_hu(imagen_binaria_trasladada)

factor_escala = 0.7  # Se cambia el valor de la escala
momentos_hu_img_escalada = escalar_y_obtener_momentos_hu_img(ruta_archivo_txt, factor_escala)

print("Momentos Hu Originales:", momentos_hu_originales[:4])
print(f"Momentos Hu Imagen Escalada:{momentos_hu_img_escalada[:4]}")

# Se calcula el centro de masa de la imagen original
centro_masa = cv2.moments(imagen_binaria.astype(np.float32))
cx = int(centro_masa["m10"] / centro_masa["m00"])
cy = int(centro_masa["m01"] / centro_masa["m00"])

# Se calcula el centro de masa de la imagen trasladada
centro_masa_trl = cv2.moments(imagen_binaria_trasladada.astype(np.float32))
cx_despues = int(centro_masa_trl["m10"] / centro_masa_trl["m00"])
cy_despues = int(centro_masa_trl["m01"] / centro_masa_trl["m00"])

print(f"Momento central imagen original: ({cx}, {cy})")
print(f"Momento central imagen trasladada: ({cx_despues}, {cy_despues})")


angulos = [15, 45, 90, 180]

# Se mostra la imagen original y trasladada
mostrar_imagenes_con_matplotlib(
    [imagen_binaria, imagen_binaria_trasladada],
    ['Imagen Binaria', 'Imagen Binaria Trasladada']
)

# Se rota la imagen en cada uno de los angulos y se imprimen sus valores
for angulo in angulos:
    imagen_binaria_rotada = rotar_imagen(imagen_binaria, angulo, cx, cy)
    momentos_hu_img_rotada = calcular_momentos_hu(imagen_binaria_rotada)
    # Se mostran los invariantes de Hu de la imagen rotada
    print(f"Invariantes de Hu de la imagen rotada {angulo}° {momentos_hu_img_rotada[:3]}:")
    mostrar_imagenes_con_matplotlib(
        [imagen_binaria_rotada],
        [f'Imagen Rotada {angulo}°']
    )
