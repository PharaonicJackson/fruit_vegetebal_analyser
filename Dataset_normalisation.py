'''
A partir d'un dataset initiale normalize le dataset
 en en creant un nouveau avec :

    - taille normalisee
    - couleur normalisee

'''

import os, shutil
import pandas as pd
from function_dataset import display_image
import function_dataset
from fonctions_utiles import combinaison
from skimage import io, color, exposure, transform
import numpy as np

def create_uniformized_dataset(dataset_path: str, csv_path: str, traitement: list,ratio_train: int , size: list(), r=1, header=0, ):
    '''
    Cree differents datasets composes des combinaison des parametres en argument.
    Les images sont melangees dans le datasource car elles peuvent etre triee par classe
    Les images et le fichier csv sont enregistre dans le meme dossier.

    :param ratio_test: Pourcentage de donnees dans le repository train, les reste dans le repository validation
    :param dataset_path: Chemin du datasets source depuis ce fichier.py.
    :param csv_path: Chemin du fichier csv en relation avec les images depuis ce fichier.py.
    :param r: % des images importes
    :param header: Presence de header dans le dataset -> 1 sinon 0
    :param traitement: Liste des traitements a apporter au dataset, si 2 elements, creera 2 datasets differents
    liste traitement possible :['RGB', 'RGB-HE', 'L', 'L-HE',  'L-CLAHE']
    :param size: Liste des tailles (largeur, hauteur) : tuple. Si 2 elements, creera 2 datasets differents
    :return: tuple (dossier des images standardisees, csv du dossier correspondant)
    '''

    # Verification des entrees
    if type(traitement) != list:
        raise TypeError(f"traitement doit etre une liste au lieu de {type(traitement)}")
    if type(size) != list:
        raise TypeError(f"size doit etre une liste au lieu de {type(size)}")
    if not 0 < r <= 1:
        raise ValueError(f"r doit etre compris entre 0 et 1, r = {r}")
    if not 0 < ratio_train <= 1:
        raise ValueError(f"r doit etre compris entre 0 et 1, r = {ratio_train}")
    for each_size in size:
        if type(each_size) != tuple:
            raise TypeError(f"size doit etre une liste de tuples au lieu de {type(each_size)}")
        if len(each_size) != 2:
            raise TypeError(f"size doit etre une liste de tuples de dimension 2 au lieu de dimension {len(each_size)}")



    dataset_dir = os.path.join(os.getcwd(), dataset_path)
    dataset_dir_name = os.path.basename(dataset_dir)  # nom du folder entree en argument (si chemin)
    csv_path = os.path.join(os.getcwd(), csv_path)
    traitement_possible = ['RGB', 'RGB-HE', 'L', 'L-HE', 'L-CLAHE']

    # Recuperation des parametres uniques venant des listes de parametres
    for each_comb in combinaison(dataset_dir=[dataset_dir], csv_path=[csv_path], size=size, r=[r],
                                 traitement=traitement, header=[header]):  # Creation de chaque dataset unique

        print(f'Nouvelle combinaison : {each_comb.items()}')
        dataset_src_path = each_comb['dataset_dir']
        csv_src_path = each_comb['csv_path']
        dim_x_y = each_comb['size']
        this_traitement = each_comb['traitement']
        r = each_comb['r']
        header = each_comb['header']

        if this_traitement not in traitement_possible:  # Verification que le traitement demande est disponible.
            raise NameError(this_traitement)
        width, height = dim_x_y  # caracteristique de l'image
        dir_name_dataset_save = f"{dataset_dir_name}_{width}_{height}_{this_traitement}"  # Nom du dossier de sauvegarde des nouvelles photos.
        csv_name_save_train = f"{dataset_dir_name}_{width}_{height}_{this_traitement}_train_data.csv"  # Nom du nouveau fichier CSV
        csv_name_save_test = f"{dataset_dir_name}_{width}_{height}_{this_traitement}_test_data.csv"  # Nom du nouveau fichier CSV
        df_data_src = pd.read_csv(csv_src_path, header=header).sample(frac=1)  # csv data images sources + melange


        # Creation du dossier de sauvegarde du dataset
        [shutil.rmtree(f) for f in os.listdir(os.getcwd()) if
         f == dir_name_dataset_save]  # Si le dossier existe deja, je le supprime.
        os.mkdir(dir_name_dataset_save)  # Dossier de sauvegarde du nouveau dataset.

        #Creation des dossier de sauvegarder train et validation
        path_dataset_train = os.path.join(os.getcwd(), dir_name_dataset_save, 'train')
        path_dataset_test = os.path.join(os.getcwd(), dir_name_dataset_save, 'test')

        os.mkdir(path_dataset_train) # Creation du sous folder test
        os.mkdir(path_dataset_test)  # Creation du sous folder test

        # Gestion du nombre de photos total a recuperer
        nbr_row = df_data_src.iloc[:, 0].count()  # Nombre de donnees total.
        nbr_row = int(r * nbr_row) # reduction du nombre de dopnnees extraites.
        df_data_src = df_data_src.head(nbr_row)

        # Extraction des donnees
        filenames = df_data_src['Name'].to_list()  # Recuperation du nom de l'image
        y = df_data_src['ClassId'].to_list()  # Recuperation des classes des images

        df_new_dataset_train = pd.DataFrame(columns=['Name', 'ClassId']) # Creation du dataframe des nouvelles donnees generees
        df_new_dataset_test = pd.DataFrame(columns=['Name', 'ClassId']) # Creation du dataframe des nouvelles donnees generees
        # Modification taille et traitement de l'image
        for i, image in enumerate(filenames):
            image = io.imread(os.path.join(dataset_src_path, image))
            image = image_augmentation(image, width=width, height=height, mode=this_traitement) # Traitement de l'image

            # Image float [0 - 1] -> int [0 - 255] pour enlever le warning car float non pris en charge
            image = (image * 255).astype(np.uint8)
            name_image = f"{this_traitement}_{width}_{height}_image_{i}.png" # Creation du nouveau nom de l'image.

            if i < ratio_train * nbr_row: # Les ratio_test % premieres images appartiennent au dataset test
                io.imsave(fname=os.path.join(path_dataset_train, name_image),
                          arr =image)
                # Maj du fichier csv
                df_new_dataset_train = pd.concat([df_new_dataset_train, pd.DataFrame([(name_image, y[i])], columns=['Name', 'ClassId'])])
            else: # Sinon elles appartienneent au dataset validation.
                io.imsave(fname=os.path.join(path_dataset_test, name_image),
                          arr=image)
                # Maj du fichier csv
                df_new_dataset_test = pd.concat(
                    [df_new_dataset_test, pd.DataFrame([(name_image, y[i])], columns=['Name', 'ClassId'])])

        # Enregistrement du fichier csv dans le dataset test et validation
        new_csv_path_train = os.path.join(path_dataset_train, csv_name_save_train)
        df_new_dataset_train.to_csv(new_csv_path_train)

        new_csv_path_test = os.path.join(path_dataset_test, csv_name_save_test)
        df_new_dataset_test.to_csv(new_csv_path_test)

        return path_dataset_train, new_csv_path_train


def image_augmentation(image, width=25, height=25, mode='RGB'):
    '''
    image : natrice
    modifie la taille selon valeur width et height
    Les images d'entrees sont soit NB soit RGB soit RGBA.
    Si RGBA, l'image est redimensionne en 3 dim comme RGB

    args:
        images :         liste d'images
        width,height :   nouvelle taille des images
        mode :           RGB | RGB-HE | L | L-LHE | L-CLAHE
    return:
        numpy array des images modifiees
    '''
    # mode = {'RGB': 3, 'RGB-HE': 3, 'L': 1, 'L-HE': 1, 'L-CLAHE': 1}

    if image.shape[2] == 4:  # Si RGBA, conversion en RGB
        image = color.rgba2rgb(image)

    image = transform.resize(image, (width, height))  # Modification de la taille

    if mode == 'RGB-HE':  # RGB + histogram Egalisation
        hsv = color.rgb2hsv(image.reshape(width, height, 3))
        hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
        image = color.hsv2rgb(hsv)

    if mode == 'L':  # Niveau de gris
        image = color.rgb2gray(image)

    if mode == 'L-HE':  # Niveau de gris + histogram Egalisation => davantage de contraste
        image = color.rgb2gray(image)
        image = exposure.equalize_hist(image)

    if mode == 'L-CLAHE':  # Niveau de gris (CLAHE)
        image = color.rgb2gray(image)
        image = exposure.equalize_adapthist(image)

    return image


if __name__ == '__main__':
    create_uniformized_dataset(dataset_path='New Dataset_2023-01-29 11.14.22.132872',
                               csv_path='New Dataset_2023-01-29 11.14.22.132872/mais_fichier.csv',
                               traitement=['RGB'],
                               size=[(227, 227)],
                               r=1,
                               ratio_train=0.9,
                               header=0)

