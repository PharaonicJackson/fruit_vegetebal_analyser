import os
import fonction_modele_keras as f_k
import fonctions_utiles
from fonctions_utiles import combinaison
from function_dataset import get_data
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd

os.environ[
    "KMP_DUPLICATE_LIB_OK"] = "TRUE"  # pour resoudre ce message d'erreur OMP: Error #15: Initializing libiomp5, but found libiomp5md.dll already initialized.


def check_argument_carry_on_learning(**kwargs):
    '''
    Verifie que les arguments passes a la fonction carry_on_several_learning sont correctes.
    Verifie que les arguments sont des listes et que tout les arguments necessaires sont inclus.
    Leve une erreur sinon
    :param kwargs list: ensemble des parametres de l'entrainement
    :return: None
    '''
    list_parameter = ['try_scale', 'try_epoch', 'try_batch', 'fct_model', 'try_dataset', 'try_optimizer', 'try_loss',
                      'list_metrics']

    for each_parameter in list_parameter:
        if each_parameter not in kwargs.keys():  # Verification que tout les parametres sont indiques
            raise KeyError(f"{each_parameter} doit etre indique.")

        if not type(kwargs[each_parameter]) == list:
            raise TypeError(f"{each_parameter} doit etre une liste")

        if len(kwargs[each_parameter]) <= 0 and each_parameter != 'list_metrics':
            raise ValueError(f"{each_parameter} doit comporter au moins une valeur.")


def split_label_title(**kwargs):
    '''
    Afin de realiser plusieurs entrainements qui entrainent potentiellement plusieurs versions du meme parametres
    Je separe chaque parametre dans 2 dictionnaires :
        - (dict) dict_title
        - (dict) dict_label

    dict_label inclus les listes de parametres qui existent en differentes versions.
    Un apprentissage unique sera realise pour chaqu'une des combinaisons de ces parametres.
    Ces parametres apparaitront dans les labels du graphique final

    dict_titre inclus les listes de parametres qui existent en une unique version.
    Ces parametres apparaitront dans le titre du graphique final

    Si dict_label est vide car uniquement des parametres uniques fct_model sera integre aux labels

    'list_metrics' est exclus de ce process

    :param kwargs:
    :return: tuple (dict_title, dict_label)
    '''

    list_parameter = ['try_scale', 'try_epoch', 'try_batch', 'fct_model', 'try_dataset', 'try_optimizer', 'try_loss',
                      'list_metrics']

    dict_title = dict()
    dict_label = dict()

    for i, each_parameter in enumerate(list_parameter):  # Parcours des parametres a remplir.
        if len(kwargs[each_parameter]) > 1:  # Verifie la longueur des listes de chaque parametre.
            dict_label[list_parameter[i]] = kwargs[each_parameter]  # le nom du parametre est une clef du dictionnaire
        else:
            dict_title[list_parameter[i]] = kwargs[each_parameter]

    if len(dict_label) == 0:  # Si le dictionnaire label est vide,
        dict_label['fct_model'] = kwargs['fct_model']  # je transfers le parametre 'fct_model' au dict_label
        dict_title.pop('fct_model')  # au dict_title

    if len(dict_title) == 0:  # Si le dictionnaire titre est vide,
        dict_title['fct_model'] = kwargs['fct_model']  # je transfers le parametre 'fct_model' au dict_label
        dict_label.pop('fct_model')  # au dict_title

    return dict_title, dict_label


def create_name_and_folder_multi_learning(dict_label: dict, **kwargs):
    '''
    A partir des argument genere pour chaque combinaison unique de parametre :

    -   Cree le dossier log pour suivre l'apprentissage (si il n'existe pas deja).
    -   Cree le fichier log  de suivi de l'apprentissage relatif a cette combinaison de parametres
    -   Le titre du label de la courbe d'apprentissage relative a cette combinaison de parametres.
    :param dict_label: Dictionnaire des labels pour cette combinaison de parametres
    :param args: ['try_scale', 'try_epoch', 'try_batch', 'fct_model', 'try_dataset', 'try_optimizer', 'try_loss',
                      'list_metrics']
    :return: tuple (chemin du fichier log, titre du label,chemin du call back pour tensor board)
    '''

    # Creation du folder log pour enregistrer l'evolution de l'apprentissage
    # Si il existe, je le supprime pour en creer un nouveau et avoir un dossier vide
    folder = [f for f in os.listdir(os.getcwd()) if
              not os.path.isfile(os.path.join(os.getcwd(), f))]  # Liste les repository
    if 'callback' in folder:
        pass
        # shutil.rmtree(os.path.join(os.getcwd(),'callback')) # supprime repertoire
    else:
        os.mkdir('callback')
        os.mkdir('./callback/log')
        os.mkdir('./callback/cross_val')

    # Creation du nom unique pour le label de la courbe
    titre_label = ''  # Initialisation du titre du label
    # Iteration sur les labels pour creer un nom unique de chaque entrainement
    for each_label in dict_label.keys():
        # Supprime / car try_dataset est un chemin
        if each_label == 'try_dataset':
            valeur_label = os.path.basename(kwargs[each_label])
        else:
            valeur_label = kwargs[each_label]
        titre_label = f"{titre_label} {each_label} : {valeur_label}  "  # Titre du graph pyplotcb_checkpoint_wei_

    # Creation des nom des chemins pour les differents callbacks
    path_fichier_callback_CSVLogger = fonctions_utiles.unique_path_time_filename(dir='./callback', prefix='cb_cvs_logger_',
                                                                            suffix='.h5')

    path_callback_tensorboard = fonctions_utiles.unique_path_time_filename(dir='./callback/log', prefix='cb_tensorboard_')

    return titre_label, path_fichier_callback_CSVLogger, path_callback_tensorboard


def multi_learning(**kwargs):
    '''
    Realise 1 apprentissages sur la base des arguments en input.
    Accepte plusieurs argument par parametre. Dans ce cas, est cree autant de liste d'arguments
    que de combinaisons possibles
    Le modele est sauvegarde.
    different callback utilises pour optimiser l'apprentissage
    CSVLogger : Pour enregistrer les evolution
    Checkpoint : Pour sauvegarder les meilleurs modeles
    EarlyStopping : Pour arreter l'entrainement si necessaire


    :param kwargs: ['try_scale', 'try_epoch', 'try_batch', 'fct_model', 'try_dataset', 'try_optimizer', 'try_loss',
    'list_metrics']
    'try_scale' : Liste de pourcentage du dataset pour l'apprentissage.
    'try_dataset' : Liste de chemin absolu du dataset.
    'try_batch' : Liste de batch.
    'fct_model' : Liste de fonction qui retourne un modele keras.
    'try_optimizer' : Liste de d'optimizer a utiliser.
    'try_loss' : Liste de fonction de perte.
    :return: None
    '''

    check_argument_carry_on_learning(**kwargs)  # Verification des arguments.check_argument_carry_on_one_learning

    dict_titre, dict_label = split_label_title(**kwargs)  # Separe les parametres dans 2 dictionnaires

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

    for each_comb in combinaison(**kwargs):  # Ne retourne que 1 combinaison.
        print(f'Nouvelle combinaison : {each_comb.items()}')

        titre_label, path_callback_CVS, path_callback_tensorboard = create_name_and_folder_multi_learning(
            dict_label=dict_label, **each_comb)

        # Chargement des donnees
        x_train, x_val, y_train, y_val = get_data(dataset_dir=each_comb['try_dataset'],
                                                  pourc_dataset=each_comb['try_scale'])

        print("Nombre de donnees d'entrainement/val", len(x_train), len(x_val))
        # function_dataset.display_image(x_train[0:20])

        #  Chargement arguments du modele
        if len(x_train[0].shape) == 3:  # Si image en couleur
            (lx, ly, lz) = x_train[0].shape  # Argument de la fonction qui renvoie un objet modele keras.
        if len(x_train[0].shape) == 2:  # Si image en N/B
            lx, ly = x_train[0].shape
            lz = 1

        _model = eval(each_comb['fct_model'] + "(lx, ly, lz)")  # Creation du modele
        _model.compile(optimizer=each_comb['try_optimizer'],
                       loss=each_comb['try_loss'],
                       metrics=each_comb['list_metrics'])

        # Creation du callback pour enregistrer les differents resultats.
        # Call back qui mesure val_accuracy et accuracy
        learning_data = keras.callbacks.CSVLogger(path_callback_CVS,
                                                  separator=',')

        # Entrainement du modele
        _model.fit(x_train, y_train,
                   batch_size=each_comb['try_batch'],
                   epochs=each_comb['try_epoch'],
                   callbacks=[learning_data],
                   validation_data=(x_val, y_val))

        _model.save('modele_' + titre_label)
        # Affichage de la courbe associee
        df = pd.read_csv(path_callback_CVS)
        y = list(df.val_accuracy)
        x = range(len(y))
        ax1.plot(x, y, label=titre_label)  # Ajout de la courbe

    # Affichage du graphique
    tunning_plot(ax1=ax1, dict_titre=dict_titre)  # Ajout des noms sur le graphique


def tunning_plot(ax1: plt.Axes, dict_titre: dict):
    '''
    Defini titre, titre des axes et legende de ax1 sur la base duction des titres dict_titre
    Affiche le graphique.
    :param ax1: graphique plt
    :param dict_titre: Dictionnaire des titres a ajouter au graphique
    :return: None
    '''

    ax1.set_title("Evolution de l'apprentissage")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation rate')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.grid(axis='y')
    titre_figure = "Annalyse Apprentissage :"
    for key, value in dict_titre.items():
        # list_metrics, try_scale, try_epoch ne sont pas inclus dans le titre du graph:
        if key != 'list_metrics' and key != 'try_scale' and key != 'try_epoch':
            titre_figure = titre_figure + f"{key}: {value}  "
    ax1.set_title(titre_figure)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    opti_adam_0_0001 = keras.optimizers.Adam(
        learning_rate=0.0001)

    parameter_entrainement = {
        'try_scale': [0.3],
        'try_epoch': [12],
        'try_batch': [16],
        'fct_model': ['f_k.create_model_AlexNet8'],
        'try_dataset': ['./dataset_full_227_227_RGB'],
        'try_optimizer': [opti_adam_0_0001],
        'try_loss': ['sparse_categorical_crossentropy'],
        'list_metrics': ['accuracy']}

    multi_learning(**parameter_entrainement)
