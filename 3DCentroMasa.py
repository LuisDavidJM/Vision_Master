import numpy as np
import os
import binvox_rw

def calcular_centro_de_masa(binvox_path):
    with open(binvox_path, 'rb') as f:
        modelo = binvox_rw.read_as_3d_array(f)
    voxels_activos = np.argwhere(modelo.data)
    centro_de_masa = np.round(voxels_activos.mean(axis=0), 6)
    return centro_de_masa

# Asumiendo que tienes una carpeta llamada 'modelos_binvox' con tus archivos .binvox
directorio_modelos = 'Objetos_3D/Archivos_BINVOX'
nombres_archivos = [f for f in os.listdir(directorio_modelos) if f.endswith('.binvox')]

# Lista para almacenar los resultados
resultados = []

# Procesar cada archivo .binvox
for nombre_archivo in nombres_archivos:
    path_completo = os.path.join(directorio_modelos, nombre_archivo)
    centro_de_masa = calcular_centro_de_masa(path_completo)
    resultados.append([nombre_archivo, centro_de_masa])

# Imprimir los resultados como una tabla
print(f"{'Nombre del Archivo':<20} {'Centro de Masa':>20}")
for resultado in resultados:
    nombre_archivo, centro_de_masa = resultado
    # Convertir el centro de masa a una cadena con cada valor redondeado a 6 decimales
    centro_de_masa_str = ", ".join(f"{x:.4f}" for x in centro_de_masa)
    print(f"{nombre_archivo:<20} {centro_de_masa_str:>30}")

