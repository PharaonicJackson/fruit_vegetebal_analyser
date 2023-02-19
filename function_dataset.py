import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelBinarizer
import imageio.v2 as io
import math
import pandas as pd
from matplotlib.ticker import MultipleLocator


'''
Differentes fonctions pour travailler avec un dataset:

display_image : affiche les images type array passe en argumant
read_csv_dataset : lit les image d'un repository et les transforme en matrice
get_data : Prepare les donnees en filtrant la quantite d'un dataset a lire et en separant les donnees d'entrainement des donnees de validation
'''

def display_image(images, im_ligne=5):
    '''
    Affiche les images en noir et blanc et les images en couleurs
    sur la base de array passes en argument.
    Toutes les matrice d'images doivent etre de memes dimensions.

    :param images: liste de tableau 3D * [RGB] de l'image ou 2D si noir et blanc
    :param im_ligne: nbr d'image affiche par ligne
    :return: None
    '''


    # Si image en RGB
    if len(images[-1].shape) == 3:
        m, n, o = images[-1].shape
        _ = []
        # Toutes les images doivent avoir la meme dimension
        [_.append() for each_image in images if each_image.shape != (m, n, o)]
        if len(_) > 0:
            raise(ValueError('Images de tailles non identiques'))

        # Creation de la matrice nulle pour completer les images manquantes de la derniere ligne
        # les 1 de np.ones sont pour afficher des pixels blancs.
        mat_nul = np.ones(shape=(m, n, o))
        mat_finale = np.ones(shape=(1, im_ligne * n, 3))

    # Si image en N et B
    if len(images[-1].shape) == 2:
        m, n = images[-1].shape
        _ = []
        # Toutes les images doivent avoir la meme dimension
        [_.append() for each_image in images if each_image.shape != (m, n)]
        if len(_) > 0:
            raise (ValueError('Images de tailles non identiques'))
        # Creation de la matrice nulle pour completer les images manquantes de la derniere ligne
        # les 1 de np.ones sont pour afficher des pixels blancs.
        mat_nul = np.ones(shape=(m, n))

        mat_finale = np.zeros(shape=(1, im_ligne * n))  # initialise la matrice NB mais 1 seul ligne, puis je concatene sur cette matrice

    # Determination du nombre de ligne dimage
    nbr_image = len(images)
    nbr_ligne = nbr_image // im_ligne  # Nombre de ligne de im_ligne photos
    nbr_photo_der_ligne = nbr_image % im_ligne  # Nombre de photo sur la derniere ligne

    num_image = 0
    # affichage des photos par groupe de im_ligne
    for l in range(nbr_ligne):
        # Initialisation de la premiere photo de chaque ligne
        a = images[num_image]
        num_image += 1
        # Affichage des photos le long de la ligne
        for i in range(im_ligne - 1):
            a = np.concatenate((a, images[num_image]), axis=1) # Photos concatenees horizontalement
            num_image += 1
        mat_finale = np.concatenate((mat_finale, a), axis=0)  # En fin de ligne, Photos concatenees verticalement
    nbr_mat_null = im_ligne - nbr_photo_der_ligne

    # Affichage de la derniere ligne
    if nbr_photo_der_ligne > 0: # Si il y a des photo en fin de ligne
        mat_der_ligne = images[num_image]
        num_image += 1
        # Affichage des photos de la derniere ligne
        for i in range(nbr_photo_der_ligne - 1):
            mat_der_ligne = np.concatenate((mat_der_ligne, images[num_image]), axis=1)
            num_image += 1
        # Je complete par des images blanches pour finir la ligne
        for i in range(nbr_mat_null):
            mat_der_ligne = np.concatenate((mat_der_ligne, mat_nul), axis=1)
        mat_finale = np.concatenate((mat_finale, mat_der_ligne), axis=0)

    # Affichage des images
    if len(images[-1].shape) == 3:
        # La matrice est compose de valeur entiere entre 0 et 255
        # le dtype des valeurs donc etre uint8
        # Si compose de Reel la valeur devrait etre entre 0 et 1
        plt.imshow(mat_finale.astype('uint8'))
    else:
        plt.imshow(mat_finale.astype('uint8'), cmap='binary')
    plt.show()

def read_csv_dataset(dataset_path: str):

    '''
    Lit les photos situees a l'adresse en argument.
    et transforme les photos en matrice.
    Si les labels sont des noms, je les transforme en entier pour etre lisible par numpy
    :param dataset_path:
    :return: tuple(np.array, np.array) data image et label
    '''

    dataset_path = os.path.join(dataset_path)
    out_image = list() # list final qui contient les images

    # Lecture du fichier csv si il existe
    csv_in_folder = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and 'csv' in f]
    if len(csv_in_folder) == 1: # Si il existe un et un seul fichier csv
        name_csv_dataset = csv_in_folder[0]
    else: raise (FileNotFoundError("Verifier qu'il n'y ai qu'un seul fichier csv dans le dataset ou quil ne soit pas deja ouvert."))

    path_csv = os.path.join(dataset_path, name_csv_dataset)

    df = pd.read_csv(path_csv) # Lecture nom des images: Name, label: ClassId
    df = df.sample(frac=1)


    label_vecteur = create_dict_label(df.ClassId.to_list(), name_csv='dictionnaire_label.csv') # Transforme les nom des labels en int et cree un fichier csv pour faire le lien entre int et nom du label
    out_target = label_vecteur

    list_image = df['Name'].to_list()

    for i, name_image in enumerate(list_image):
        image = io.imread(os.path.join(dataset_path, name_image))
        out_image.append(image)

    # format np.array pour etre compatible avec keras
    out_image = np.asarray(out_image)

    return out_image, out_target


def create_dict_label(name_label, name_csv=None) -> list:
    '''
    Transforme le nom des labels en valeur pour etre integre dans mon reseau de neuronnes
    Retourne name_label mappe en int
    Cree un dictionnaire enregistre format csv pour etre reutilise separement de l'entrainement
    :param name_label: numpy.array (n,) des label a transformer
    :return: list des noms de labels encodes
    '''

    ord_enc = LabelBinarizer()
    dict_label = {}
    ord_enc = ord_enc.fit(name_label)
    key_label_label = np.sort(ord_enc.classes_) # recupere le nom des label

    # Creer le dictionnaire de label a enregistrer
    # Creation du df pour enregistrer le dictionnaire de correspondance entre nom de la clef et sa valeur
    df = pd.DataFrame(columns=['name_label', 'value_label'])
    for each_label in key_label_label:
        dict_label[each_label] = ord_enc.transform([each_label])
        df = pd.concat([df, pd.DataFrame([(each_label, dict_label[each_label])], columns=['name_label', 'value_label'])]) # Concatenation de chaque ligne correspondant a un label.

    # Transformation des valeurs en imput
    ordinal_label = ord_enc.transform(name_label)

    df.to_csv(name_csv) # Enregistrement du dictionnaire

    return ordinal_label



def get_data(dataset_dir, pourc_dataset=1) -> tuple:
    '''
    Lit un dossier d'image.
    Le dossier doit etre compose dans sous dossier test et validation.
    Un fichier csv avec le nom de l'image (Name) et le label (ClassId) doit etre inclus dans chaque sous dossier
    Ne lit nqu'un pourcentage fixe des dossiers.
    Transforme les images en matrice.


    :param dataset_dir: Chemin absolue du dossier d'images muni du fichier csv correspondant avec les labels
    :param pourc_dataset: Pourcentage du dataset a lire.
    :return: array image_train, array image_val, label_train, array label_val
    '''

    # Verification
    if not 0 < pourc_dataset <= 1:
        raise ValueError(f"pourc_dataset doit etre compris entre 0 et 1, pourc_dataset = {pourc_dataset}")

    # Importation des datas
    x_train, y_train = read_csv_dataset(os.path.join(dataset_dir, 'train'))
    x_val, y_val = read_csv_dataset(os.path.join(dataset_dir, 'test'))


    # Reduction de la taille du dataset si necessaire
    x_train = x_train[:math.ceil(len(x_train) * pourc_dataset)]
    y_train = y_train[:math.ceil(len(y_train) * pourc_dataset)]
    x_val = x_val[:math.ceil(len(x_val) * pourc_dataset)]
    y_val = y_val[:math.ceil(len(y_val) * pourc_dataset)]

    return x_train, x_val, y_train, y_val



if __name__ == '__main__':

    x_train, x_val, y_train, y_val,  = get_data(dataset_dir='./Dataset_full_227_227_RGB',
                                                pourc_dataset=1)
    # Affichage de l'ensemble des echantillion fruit et legume (1 par type).
    echantillion = []
    label = []
    # for i, image in enumerate(x_train):
    #     echantillion.append(image)
    #     # if y_train[i] not in label:
    #     #     label.append(y_train[i])
    #     #     echantillion.append(image)
    #     # if len(label) >= 36:
    #     #     break
    #     display_image(echantillion[40:100], im_ligne=7)

    df_tr = pd.DataFrame(y_train, columns=['name'])
    df_val = pd.DataFrame(y_val, columns=['name'])
    # print(df)
    # print(df.groupby(df['name']).value_counts())
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.hist([df_tr['name'].to_list(), df_val['name'].to_list()], label=['Data test', 'Data validation'], bins=36)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlabel('Label')
    ax.set_ylabel('Number of images')
    ax.set_title('Repartition of the image among the 36 labels.')
    ax.legend()
    plt.show()


