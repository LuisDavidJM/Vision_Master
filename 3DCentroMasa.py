import numpy as np
import os
import binvox_rw

# Función para calcular el centro de masa de un modelo 3D voxelizado
def calcular_centro_de_masa(ruta_binvox):
    # Se abre el archivo .binvox para lectura en modo binario
    with open(ruta_binvox, 'rb') as f:
        # Se usa la función de binvox para leer el archivo como una matriz 3D
        modelo = binvox_rw.read_as_3d_array(f)
    # Se buscan las coordenadas de todos los voxels
    voxeles = np.argwhere(modelo.data)
    # Se calcula el promedio de las coordenadas de cada voxel para obtener el centro de masa
    centro_de_masa = np.round(voxeles.mean(axis=0), 6)
    return centro_de_masa

# Se recorre la lista todos los archivos '.binvox'
ruta_archivos = 'Objetos_3D/Archivos_BINVOX'
nombres_archivos = [f for f in os.listdir(ruta_archivos) if f.endswith('.binvox')]

# Se crea una lista para almacenar los centros de masa
resultados = []

# Ciclo que se encarga de reccorer todos los archivos y calcular el centro de masa con la fucnión
for nombre_archivo in nombres_archivos:
    ruta = os.path.join(ruta_archivos, nombre_archivo)
    centro_de_masa = calcular_centro_de_masa(ruta)
    # Se agrega el nombre del archivo y su centro de masa a la lista
    resultados.append([nombre_archivo, centro_de_masa])

# Se imprimen todos los centros de masa
print(f"{'Nombre del Archivo':<25} {'Centro de Masa':>20}")
for resultado in resultados:
    nombre_archivo, centro_de_masa = resultado
    # Se convierte el centro de masa a una cadena redondeando a 6 decimales
    centro_de_masa_r = ", ".join(f"{x:.6f}" for x in centro_de_masa)
    print(f"{nombre_archivo:<25} {centro_de_masa_r:>30}")
