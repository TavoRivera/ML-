import os
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,  classification_report, confusion_matrix
import joblib
def load_images_from_folder(folder_path):
    """ preprocesamiento de las imagenes """
    images = []
    labels = []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                try:

                    img = preprocess_image(img_path)


                   # img = Image.open(img_path).convert('L')  # Convertimos a escala de grises
                    img = img.resize((64, 64))  # Redimensionamos las imágenes para que tengan el mismo tamaño
                    img_array = np.array(img).flatten()  # Aplanamos la imagen
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Error al procesar la imagen {img_path}: {e}")
    return np.array(images), np.array(labels)

def images_to_csv(images, labels, output_csv):
    """ convertir a matriz csv """
    df = pd.DataFrame(images)
    df['label'] = labels
    df.to_csv(output_csv, index=False)

def preprocess_image(image_path):
    """ Aplicacion de filtros a imagen """
    # Abre la imagen usando Pillow y conviértela a un formato que OpenCV puede usar
    original_image = Image.open(image_path)
    image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

    # Convertir a escala de grises
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro de mediana para reducir el ruido
    median_filtered_image = cv2.medianBlur(gray_image, 1)

    # Aplicar filtro Gaussian blur para suavizar la imagen
    gaussian_blur_image = cv2.GaussianBlur(median_filtered_image, (1, 1), 1)

    # Aplicar filtro de Sobel para detectar bordes
    sobelx = cv2.Sobel(gaussian_blur_image, cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(gaussian_blur_image, cv2.CV_64F, 0, 1, ksize=1)
    sobel_combined = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 2, cv2.convertScaleAbs(sobely), 2, 0)

    # Aplicar umbralización para segmentación de objetos
    ret, threshold_image = cv2.threshold(sobel_combined, 127, 255, cv2.THRESH_BINARY)

    # Convierte la imagen procesada de OpenCV a Pillow y guárdala si es necesario
    final_image = Image.fromarray(threshold_image)
    # if save_path:
    #     final_image.save(save_path)
    #
    # Muestra la imagen procesada
    # final_image.show(title="Processed Image")

    return final_image

def predict_new_image(image_path):
    clf = joblib.load('modelo_entrenado.pkl')
    image = preprocess_image(image_path)
    image = image.resize((64, 64))
    image_array = np.array(image).flatten().reshape(1, -1)
    prediction = clf.predict(image_array)
    return prediction[0]

def main():

    folder_path = 'images'
    images, labels = load_images_from_folder(folder_path)

    output_csv = 'imagenes.csv'
    images_to_csv(images, labels, output_csv)

    # Cargar el CSV
    data = pd.read_csv(output_csv)
    X = data.drop('label', axis=1)
    y = data['label']

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar un clasificador
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    joblib.dump(clf, 'modelo_entrenado.pkl')
    # # Hacer predicciones
    # y_pred = clf.predict(X_test)
    #
    # # Evaluar el modelo
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f'Precisión del modelo: {accuracy}')
    #
    # # Generar el reporte de clasificación
    # report = classification_report(y_test, y_pred)
    # print('Reporte de Clasificación:')
    # print(report)
    #
    # # Generar la matriz de confusión
    # conf_matrix = confusion_matrix(y_test, y_pred)
    # print('Matriz de Confusión:')
    # print(conf_matrix)

if __name__ == '__main__':
    main()
    nueva_imagen_path = '3.jpg'
    prediccion = predict_new_image(nueva_imagen_path)
    print(f'La predicción para la nueva imagen es: {prediccion}')