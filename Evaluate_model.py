import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from function_dataset import get_data
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer

os.environ[
    "KMP_DUPLICATE_LIB_OK"] = "TRUE"  # pour resoudre ce message d'erreur OMP: Error #15: Initializing libiomp5, but found libiomp5md.dll already initialized.


def get_recall(df_prediction):
    '''
    Estimation du recall : Par label combien ont ete trouve / combien sont non trouve
    :param df_prediction:
    :return:
    '''

    # Nombre de success / Label
    label_success = df_prediction.groupby(
        df_prediction['label']).prediction_vrai.sum().to_list()  # repartition par label du nombre de succes

    # Nombre de prediction total / Label
    label_tot = df_prediction.groupby(df_prediction['label']).label.count().to_list()
    label_recall = []
    [label_recall.append(label_success[i] * 100 / label_tot[i]) for i in range(len(label_tot))]

    return label_recall



def get_precision(df_prediction):
    '''
    Estimation du recall : Par label combien ont ete trouve et combien sont non trouve
    :param df_prediction:
    :return:
    '''


    # Estimation par label du nombre de fail est de success
    prediction_success = df_prediction.groupby(
        df_prediction['prediction']).prediction_vrai.sum().to_list()  # repartition par label du nombre de succes

    # nbr d'observation / label - les observation bien trouve = fail par categorie
    prediction_sum = df_prediction.groupby(df_prediction['prediction']).label.count().to_list()
    precision = []
    [precision.append(prediction_success[i] * 100 / prediction_sum[i]) for i in range(len(prediction_sum))]

    return precision
def evaluate_model(path_model: str, path_dataset: str, path_csv: str):
    '''
    Test le modele et affiche differents graphiques de resultats.
    :param path_csv: chemin abs du csv en relation avec les images du dataset train afin de reecoder les donnees comme lors de l'entrainement
    :param path_model: Chemin du modele a utiliser
    :param path_dataset: Chemin du dataset a utiliser
    :return:
    '''


    path_dataset = os.path.join(path_dataset)
    path_model = os.path.join(path_model)
    path_csv = os.path.join(path_csv)

    _, x_test, _, y_test = get_data(dataset_dir=path_dataset)

    # Encodage et de-encodage des labels matrice/name_label
    df = pd.read_csv(path_csv)  # Lecture nom des images: Name, label: ClassId
    ord_enc = LabelBinarizer() # Creation de l'encodeur
    ord_enc = ord_enc.fit(df.ClassId.to_list())
    key_label_label = np.sort(ord_enc.classes_) # recupere le nom des label

    # Creation du dictionnaire matrice colonne / nom du label
    dict_label = {}
    for each_label in key_label_label:
        dict_label[each_label] = ord_enc.transform([each_label])

    my_model = keras.models.load_model(os.path.join(path_model))

    score = my_model.evaluate(x_test, y_test, verbose=1)
    print(score)

    y_predict = my_model.predict(x_test) # y_predict est une matrice colonne

    # Re encodage de y_predict et de y_test en label nom de fruits et legumes
    y_predict_label = []
    y_test_label = []
    match = [] # Verifie la correspondance entre y_predict_label et y_test_label

    # transforme les pourcentages de prediction par label en vecteur one hot
    for sample in y_predict:
        for i, each_label_predict in enumerate(sample):
            if each_label_predict == max(sample):
                sample[i] = 1
            else:
                sample[i] = 0

    df_prediction = pd.DataFrame(columns=['label', 'prediction', 'prediction_vrai'])
    for i in range(len(y_predict)):
        # Recuperation du nom du fruit a partir du vecteur colone
        [y_predict_label.append(clef) for (clef, valeur) in dict_label.items() if (valeur.ravel() == y_predict[i]).all()]
        [y_test_label.append(clef) for (clef, valeur) in dict_label.items() if (valeur.ravel() == y_test[i]).all()]
        match.append(y_test_label[-1] == y_predict_label[-1]) # les 2 dernieres valeurs ajoute sont elles identiques ?

        # Ajout des derniere valeurs  au dataframe
        df_prediction = pd.concat([df_prediction, pd.DataFrame([(y_predict_label[-1], y_test_label[-1], match[-1])], columns=['label', 'prediction', 'prediction_vrai'])])

    # Analyse de recall du modele: genere list : %/label
    label_recall = get_recall(df_prediction)
    label_precision = get_precision(df_prediction)



    # Affichage des gra

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16, 10))
    label = range(len(label_recall))
    ax1.bar(x=label, height=label_recall, color ='green', label='Recall')
    ax2.bar(x=label, height=label_precision, color ='blue', label='Precision')

    ax1.grid(axis='y')
    ax1.set_xlabel('Label')
    ax1.set_ylabel("%")
    ax1.set_title("Recall of the model")
    ax1.xaxis.set_ticks(range(36))
    ax1.xaxis.set_ticklabels(dict_label.keys(),  rotation=90)
    ax1.legend()

    ax2.grid(axis='y')
    ax2.set_xlabel('Label')
    ax2.set_ylabel("%")
    ax2.set_title("Precision of the model")
    ax2.xaxis.set_ticks(range(36))
    ax2.xaxis.set_ticklabels(dict_label.keys(),  rotation=90)
    ax2.legend()

    plt.subplots_adjust(hspace = 0.5)


    plt.show()

if __name__ == '__main__':
    evaluate_model(path_model='./best_model/best_model_save.h5',
                   path_dataset='./Dataset_full_227_227_RGB',
                   path_csv='./Dataset_full_227_227_RGB/train/Dataset_full_227_227_RGB_train_data.csv')