import numpy as np
import binvox_rw
import os

#Función para encontrar los voxeles de la superficie y calcular el área
def encontrar_voxeles_superficie(ruta_binvox):
    # Se abre el archivo .binvox para lectura en modo binario y se lee la matriz 3d
    with open(ruta_binvox, 'rb') as f:
        modelo = binvox_rw.read_as_3d_array(f)

    area_superficie = 0
    superficie = np.zeros(modelo.data.shape, dtype=bool)
    # Se recorre cada voxel en la matriz 3D
    for x in range(modelo.data.shape[0]):
        for y in range(modelo.data.shape[1]):
            for z in range(modelo.data.shape[2]):
                # Se aplica si el voxel esta activo
                if modelo.data[x, y, z]: 
                    caras_expuestas = 0
                    # Se verifica cada una de las seis direcciones alrededor del voxel
                    for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
                        # Se calcula la posición del voxel adyacente
                        nx, ny, nz = x + dx, y + dy, z + dz
                        # Se verifica si el voxel adyacente esta fuera de los limites o no esta activo
                        if nx < 0 or nx >= modelo.data.shape[0] or ny < 0 or ny >= modelo.data.shape[1] or nz < 0 or nz >= modelo.data.shape[2] or not modelo.data[nx, ny, nz]:
                            caras_expuestas += 1
                            superficie[x, y, z] = True
                    # Se suma al area de la superficie el numero de voxeles de la superficie
                    area_superficie += caras_expuestas

    return superficie, area_superficie

# Función para guardar los archivos binarios
def guardar_como_binario(nombre_archivo, datos, carpeta_destino):
    # Se comprueba que la carpeta de destino existe
    os.makedirs(carpeta_destino, exist_ok=True)
    # Se crea la ruta completa para guardar el archivo en .txt
    ruta = os.path.join(carpeta_destino, nombre_archivo + '.txt')
    
    # Se verifica si los datos son un arreglo de NumPy
    if isinstance(datos, np.ndarray):
        # Se convierte el arreglo al tipo adecuado para el archivo binario
        datos.astype(np.uint8).tofile(ruta)
    # Se verifica si los datos son un entero
    elif isinstance(datos, int):
        # Se convierte el entero a bytes para escribir la información
        with open(ruta, 'wb') as f:
            f.write(datos.to_bytes(4, byteorder='little', signed=True))

# Se recorre la lista todos los archivos '.binvox'
ruta_archivos = 'Objetos_3D/Archivos_BINVOX'
nombres_archivos = [f for f in os.listdir(ruta_archivos) if f.endswith('.binvox')]

# Ruta donde se van a guardar los archivos binarios
carpeta_destino = 'Objetos_3D/BIN_Superficie'

# Ciclo que se encarga de recorrer todos los archivos e imprimir el area de la superficie
for nombre_archivo in nombres_archivos:
    # Se crea la ruta completa del archivo
    path_completo = os.path.join(ruta_archivos, nombre_archivo)
    superficie, area_superficie = encontrar_voxeles_superficie(path_completo)
    # Se genera un nuevo nombre para el archivo contiene datos de la superficie
    nombre_archivo_s = os.path.splitext(nombre_archivo)[0]
    # Se guardan los datos de los voxeles de la superficie en el nuevo archivo
    guardar_como_binario(nombre_archivo_s + '_superficie', superficie, carpeta_destino)
    print(f"Área de la superficie de {nombre_archivo}: {area_superficie}")