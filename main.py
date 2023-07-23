import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tkinter import ttk

# Charger le modèle pré-entraîné
modele = tf.keras.applications.MobileNetV2(weights='imagenet')
target_size = (224, 224)  # Taille d'image attendue pour le modèle

# Fonction de détection des objets
def detecter_objets(image_path):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normaliser les valeurs de pixel entre 0 et 1
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension pour correspondre au format d'entrée du modèle
    predictions = modele.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)
    return decoded_predictions[0]

# Fonction pour charger une image et afficher les objets détectés
def charger_image_et_detecter():
    chemin_image = filedialog.askopenfilename()
    if chemin_image:
        objets_detectes = detecter_objets(chemin_image)
        result_label.config(text="Objets détectés :")
        liste_objets.delete(0, tk.END)  # Effacer la liste précédente des objets
        for objet in objets_detectes:
            nom_objet, categorie, confiance = objet
            label = f"{categorie} ({confiance:.2f})"
            liste_objets.insert(tk.END, label)

# Interface utilisateur Tkinter
racine = tk.Tk()
racine.title('Détection d\'objets avec TensorFlow')
racine.geometry('500x400')
racine.configure(background='white')

frame = ttk.Frame(racine)
frame.pack(pady=20)

btn_charger = ttk.Button(frame, text='Charger une image et détecter', command=charger_image_et_detecter)
btn_charger.pack(pady=10)

result_label = ttk.Label(racine, text="", font=('Helvetica', 12, 'bold'), background='white')
result_label.pack()

liste_objets = tk.Listbox(racine, width=50, height=10, font=('Helvetica', 10), selectbackground='#90CAF9')
liste_objets.pack(pady=10)

racine.mainloop()
