import os, shutil
from Dataset_creation import create_fichier_csv_dataset
from Dataset_normalisation import create_uniformized_dataset
from tensorflow import keras
import imageio.v2 as io
import pandas as pd
import numpy as np

def test(path=None):
    '''
    Effectue un test pour deviner le fruit ou legume d'une photo.
    Affiche le nom du fruit ou legume dans la console.
    :param path: Chenin  de la photo a deviner.
    :return:
    '''

    if path is None:
        path = input('Veuillez saisir le chemin de la photo a tester.')

    list_target = ['apple',
                   'banana',
                   'beetroot',
                   'bell pepper',
                   'cabbage',
                   'capsicum',
                   'carrot',
                   'cauliflower',
                   'chilli pepper',
                   'corn',
                   'cucumber',
                   'eggplant',
                   'garlic',
                   'ginger',
                   'grapes',
                   'jalepeno',
                   'kiwi',
                   'lemon',
                   'lettuce',
                   'mango',
                   'onion',
                   'orange',
                   'paprika',
                   'pear',
                   'peas',
                   'pineapple',
                   'pomegranate',
                   'potato',
                   'raddish',
                   'soy beans',
                   'spinach',
                   'sweetcorn',
                   'sweetpotato',
                   'tomato',
                   'turnip',
                   'watermelon']

    # Test si le fichier existe
    try:
        with open(path):
            pass
    except IOError:
        print(f'{path} est un fichier incorrecte.')
        return

    # Deplacement vers un fichier temporaire pour traiter l'image
    os.mkdir('a_supprimer')
    os.mkdir('a_supprimer/supprimer')
    shutil.copy(path, 'a_supprimer/supprimer')

    # Creation du nom de l'image
    path_dir, path_csv = create_fichier_csv_dataset('a_supprimer')

    path_dir_uni, path_csv_uni = create_uniformized_dataset(dataset_path=path_dir,
                               csv_path=path_csv,
                               traitement=['RGB'],
                               size=[(227, 227)],
                               r=1,
                               ratio_train=1,
                               header=0)

    # Suppression des dossiers temporaires
    shutil.rmtree(os.path.join(os.getcwd(), 'a_supprimer'))
    shutil.rmtree(path_dir)

    # Chargement du modele
    model = keras.models.load_model('./best_model_Alexnet2.h5')

    # Chargement de l'image normalise
    df = pd.read_csv(path_csv_uni)  # Lecture nom des images: Name, label: ClassId

    list_image = df['Name'].to_list()
    image = np.asarray([io.imread(os.path.join(path_dir_uni, list_image[0]))])

    shutil.rmtree(path_dir_uni)

    y = model.predict(image) # Realise la prediction

    # Transformation des probabilites en decision
    y = list(y[0])
    index = y.index(max(y)) # Indexe de la valeur max = index pour definir le nom du fruit

    print(f"{list_target[index]}, certitude : {max(y)} %") # Transforme le vecteur en nom de fruit

if __name__ == '__main__':
    test()


