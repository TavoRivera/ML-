import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
                    img = Image.open(img_path).convert('L')  # Convertimos a escala de grises
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

    # Hacer predicciones
    y_pred = clf.predict(X_test)

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precisión del modelo: {accuracy}')

if __name__ == '__main__':
    main()