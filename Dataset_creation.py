'''

Objectif :

Creer un dataset dans un fichier :
    - les images
    - fichier csv avec le nom de l'image et son label

Chaque photo est rangee dans un dossier comportant le label de la photo

1 Creation du dictionnaire de tout les label

'''
import datetime
import os, shutil
from datetime import datetime

import numpy as np
import pandas as pd
from skimage import io, color, exposure, transform
import matplotlib.pyplot as plt


class NameError(Exception):
    def __init__(self, key_word):
        message = f"{key_word} n'est pas un mot clef compatible. Veuillew choisir parmi les suivants RGB | RGB-HE | L | L-HE | L-LHE | L-CLAHE "
        super().__init__(message)


def update_dict(dict_name: dict, clef: str, value: int) -> None:
    '''
    Mise a jour du dictionnaire
    :param dict_name: non ductionnaire
    :param clef: clef de la valeur a qjouter
    :param value: nouvelle valeur a ajouter
    :return:
    '''
    dict_name[clef] = value
    return dict_name


def create_fichier_csv_dataset(folder_path: str):
    '''
    Creation d'un repository pour stocker l'ensemble des photos.
    Chaque repository de l'adresse folder_path est ouvert, et les fichiers a l'interieur sont analyses.
    En fonction du nom du repository qui contient ces fichier (le nom = la class) , les fichier sont ajoutes au documnt.csv avec le label correspondant

    Un nouveau folder est cree avec les images renommee ainsi que le fichier csv correspondant

    :param folder_path str:  chemin du dossier comportant les directorys de labels depuis ce fichier py
    :return: chemin du dossier cree, chemin du fichier csv cree.
    '''

    # Adresse du repository cree compose de la date et l'heure
    # : enleve pour etre un nom de dossier correcte

    dir_name = f"New Dataset_{str(datetime.now().today()).replace(':', '.')}"
    dir_save = os.path.join(os.getcwd(), dir_name)
    # Nom du fichier csv

    dir_path = os.path.join(os.getcwd(), folder_path)  # chemin du dossier contenant les folder d'images
    dir_folder_name = os.path.basename(dir_path)  # nom du folder entree en argument (si chemin)
    csv_file_path = os.path.join(dir_save, f"{dir_folder_name}_fichier.csv")

    # Creation du directory seulement si le nom de ce directory n'existe pas
    folders = [f for f in os.listdir(os.getcwd()) if not os.path.isfile(f)]
    df_out = pd.DataFrame(columns=['Name', 'ClassId'])
    if dir_name not in folders:
        os.mkdir(dir_save)  # Creation du dossier de sauvegarde

    folders = [f for f in os.listdir(dir_path) if not os.path.isfile(f)]
    for folder in folders:
        sub_folder_path = os.path.join(os.getcwd(), folder_path, folder)  # Adresse du folder de chaque label
        images = [f for f in os.listdir(sub_folder_path)]  # Liste d'images

        for i, image in enumerate(images):  # Parcours des images
            image_path = os.path.join(sub_folder_path, image)  # Adresse de l'image

            new_name = f"image_{dir_folder_name}_{folder}_{i}.png"  # definition du nouveau nom des images
            new_name_path = os.path.join(sub_folder_path, new_name)
            os.rename(image_path, new_name_path)  # Modification du nom de l'image

            image_path = os.path.join(sub_folder_path, new_name_path)
            label = folder

            df_out = pd.concat([df_out, pd.DataFrame([(new_name, label)], columns=['Name', 'ClassId'])])

            shutil.move(image_path, dir_save)  # Je deplace la photo dans le nouveau repertoire

    df_out = df_out.sample(frac=1)  # Melange les lignes
    df_out.to_csv(csv_file_path, index=False)  # Creation du fichier csv vide

    return dir_save, csv_file_path


def analyse_data(dataset_path: str, csv_path: str):
    '''
    Affichage des caracteristiques des images dans le dataset
    :param dataset_path: Chemin absolu du dataset
    :param csv_path: Chemin absolu du du fichier csv
    :return: None
    '''
    csv_path = os.path.join(csv_path)
    dataset_path = os.path.join(dataset_path)

    df_data_src = pd.read_csv(csv_path).sample(frac=1)  # csv data images sources + melange

    # Extraction des donnees
    filenames = df_data_src['Name'].to_list()  # Recuperation du nom de l'image
    y = df_data_src['ClassId'].to_list()  # Recuperation des classes des images
    list_dataset_pixel = list()
    list_ratio_taille = list()
    for i, nom_image in enumerate(filenames):
        image = io.imread(os.path.join(dataset_path, nom_image))
        list_dataset_pixel.append(image.shape[0] * image.shape[1])
        list_ratio_taille.append(image.shape[0] / image.shape[1])
        # if i > 50:
        #     break

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
    ax1.hist([list_dataset_pixel], color='green', bins=200, label=['Initial Dataset'])
    ax1.grid(axis='y')
    ax1.set_xlabel('Number of pixels')
    ax1.set_ylabel("Number of images")
    ax1.set_title("Size of images available")
    ax1.set_xlim(0, 0.6 * 10 ** 7)
    ax1.legend()

    ax2.hist([list_ratio_taille], color='red', bins=200, label=['Initial Dataset'])
    ax2.grid(axis='y')
    ax2.set_xlabel('Ratio height / width')
    ax2.set_ylabel("Number of images")
    ax2.set_title("Dimension of the images available")
    ax2.legend()

    plt.show()


def analyse_data2(dataset_path: str, csv_path: str):
    '''
    Affichage des caracteristiques des images dans le dataset avec des scatters.
    :param dataset_path: Chemin absolu du dataset
    :param csv_path: Chemin absolu du du fichier csv
    :return: None
    '''
    print('ok')
    csv_path = os.path.join(csv_path)
    dataset_path = os.path.join(dataset_path)

    df_data_src = pd.read_csv(csv_path).sample(frac=1)  # csv data images sources + melange

    # Extraction des donnees
    filenames = df_data_src['Name'].to_list()  # Recuperation du nom de l'image

    list_width = list()
    list_height = list()

    mat_display = np.zeros(shape=(len(list_width), len(list_height)))
    for i, nom_image in enumerate(filenames):
        image = io.imread(os.path.join(dataset_path, nom_image))
        height_im = image.shape[1]
        width_im = image.shape[0]
        list_width.append(width_im)
        list_height.append(height_im)


        # # Je classe les image par taille et je compte leur occurence
        # for each_width in list_width:
        #     for each_height in list_height:
        #         if each_height - 12 < height_im < each_height + 12 and \
        #                 each_width - 12 < width_im < each_width + 12:
        #             mat_display[list_height.index(each_height), list_width.index(each_width)] += 1

    fig, ax = plt.subplots(figsize=(15, 10))


    ax.scatter(list_width, list_height, c = 'red', marker='3')
    ax.set_title('Dimension of the images')
    ax.set_xlabel('Width (pixel)')
    ax.set_ylabel('Height (pixel)')


    plt.legend()
    plt.show()




if __name__ == "__main__":
    # create_fichier_csv_dataset('mais')

    analyse_data2(dataset_path='./Dataset_full_from_kaggle',
                 csv_path='./Dataset_full_from_kaggle/fruit_dataset.csv')
