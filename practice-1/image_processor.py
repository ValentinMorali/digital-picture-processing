import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    print("Procesamiento Digital de Imágenes")
    
    # 1. Seleccionar tres imágenes en color (jpg, png, bmp)
    jpg_image_path = "schumacher.jpg"  
    png_image_path = "mclaren.png"  
    bmp_image_path = "bmp_13.bmp"  
    
    # 2 y 3. Abrir cada imagen y levantar información
    print("\n--- Información de la imagen JPG ---")
    jpg_img = cv2.imread(jpg_image_path)
    analyze_image(jpg_img, jpg_image_path)
    
    print("\n--- Información de la imagen PNG ---")
    png_img = cv2.imread(png_image_path)
    analyze_image(png_img, png_image_path)
    
    print("\n--- Información de la imagen BMP ---")
    bmp_img = cv2.imread(bmp_image_path)
    analyze_image(bmp_img, bmp_image_path)
    
    # 4. Elegir una de las tres imágenes (elijo a schumacher para este ejemplo)
    selected_img = jpg_img
    selected_img_path = jpg_image_path
    print(f"\nImagen seleccionada: {selected_img_path}")
    
    # Cerrar las otras imágenes
    print("Cerrando las otras imágenes")
    
    # 4.a. Transformar a escala de grises
    gray_img = cv2.cvtColor(selected_img, cv2.COLOR_BGR2GRAY)
    gray_img_name = f"gris01{os.path.basename(selected_img_path)}"
    cv2.imwrite(gray_img_name, gray_img)
    print(f"Imagen en escala de grises guardada como: {gray_img_name}")
    
    # 4.b. Encontrar valor máximo y modificar filas
    max_val = np.max(gray_img)
    max_val_positions = np.where(gray_img == max_val)
    first_max_row = max_val_positions[0][0]  # Primera fila con valor máximo
    
    # Crear una copia para modificar
    modified_img = gray_img.copy()
    # Reemplazar el valor de gris en esa fila y dos filas contiguas
    half_max_val = max_val // 2
    
    # Asegurarse de que estamos dentro de los límites de la imagen
    rows_to_modify = [first_max_row]
    if first_max_row > 0:
        rows_to_modify.append(first_max_row - 1)
    if first_max_row < modified_img.shape[0] - 1:
        rows_to_modify.append(first_max_row + 1)
    
    for row in rows_to_modify:
        modified_img[row, :] = half_max_val
    
    gray_img_name2 = f"gris02{os.path.basename(selected_img_path)}"
    cv2.imwrite(gray_img_name2, modified_img)
    print(f"Imagen modificada guardada como: {gray_img_name2}")
    print(f"Valor máximo encontrado: {max_val}")
    print(f"Posición del primer píxel con valor máximo: fila {first_max_row}")
    
    # 4.c. Duplicar valores de gris01
    doubled_img = np.clip(gray_img.astype(np.int16) * 2, 0, 255).astype(np.uint8)
    max_doubled_val = np.max(doubled_img)
    gray_img_name3 = f"gris03{os.path.basename(selected_img_path)}"
    cv2.imwrite(gray_img_name3, doubled_img)
    print(f"Imagen con valores duplicados guardada como: {gray_img_name3}")
    print(f"Valor máximo en la imagen duplicada: {max_doubled_val}")
    
    # 5. Visualizar las cuatro imágenes
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(selected_img, cv2.COLOR_BGR2RGB))
    plt.title("Imagen Original")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(gray_img, cmap='gray')
    plt.title("Imagen en Escala de Grises (gris01)")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(modified_img, cmap='gray')
    plt.title("Imagen Modificada (gris02)")
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(doubled_img, cmap='gray')
    plt.title("Imagen con Valores Duplicados (gris03)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("comparacion_imagenes.png")

    # Mostrar sin bloquear, pausar y luego cerrar
    plt.show(block=False)
    plt.pause(5)  # Pausar durante 5 segundos
    plt.close('all')  # Cerrar después de la pausa
 
    
    # 6. Cerrar todas las imágenes
    print("\nCerrando todas las imágenes")
    selected_img = None
    gray_img = None
    modified_img = None
    doubled_img = None
    

def analyze_image(img, img_path):
    if img is None:
        print(f"Error: No se pudo abrir la imagen {img_path}")
        return
    
    # Obtener información de la imagen
    height, width = img.shape[:2]
    channels = 1 if len(img.shape) == 2 else img.shape[2]
    dtype = img.dtype
    
    print(f"Tamaño de la imagen: {width}x{height} píxeles")
    print(f"Número de canales: {channels}")
    print(f"Tipo de datos: {dtype}")
    
    # Si la imagen es a color, obtenemos el valor máximo y mínimo en cada canal
    if channels > 1:
        for i in range(channels):
            channel = img[:,:,i]
            min_val = np.min(channel)
            max_val = np.max(channel)
            print(f"Canal {i}: Valor mínimo = {min_val}, Valor máximo = {max_val}")
    else:
        min_val = np.min(img)
        max_val = np.max(img)
        print(f"Valor mínimo = {min_val}, Valor máximo = {max_val}")

if __name__ == "__main__":
    main()