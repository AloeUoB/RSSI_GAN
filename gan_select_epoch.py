import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from play_space import feature

from GAN.con_gan_rssi.model_wgan_gp import Discriminator, Generator, initialize_weights
from general_utils import uniq_count
from GAN.gan_utils import load_GAN_model, generate_fake_data,\
    MinMaxScaler, renormalization, windowing, get_mean_stat,\
    plot_line_rssi_gan, MLPClassifier, map_A, map_C, map_D,\
    binary_convert, num_room_house, train, test
from GAN.gan_clf_test import gan_select_epoch



if __name__ == "__main__":
    data_directory = os.path.join('..', 'aloe', 'localisation', 'data', ''.format(os.path.sep))
    save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    house_name = 'A'
    reduce_ap = False
    if house_name == 'A':
        col_idx_use = [1, 2, 3, 4, 5, 6, 7, 8]
    else:
        col_idx_use = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    col_idx_use_label = col_idx_use[len(col_idx_use) - 1] + 1

    if reduce_ap:
        if house_name == 'C':
            col_idx_use = [1, 2, 4, 7, 9, 10]

    house_file = 'csv_house_' + house_name + '_fp.csv'
    ori_data = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=col_idx_use)
    label = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=[col_idx_use_label])

    house_file = 'csv_house_' + house_name + '_fl.csv'
    ori_data_fl = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=col_idx_use)
    label_fl = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=[col_idx_use_label])

    # data normalisation
    norm_data, min_val, max_val = MinMaxScaler(ori_data)
    norm_data_fl, min_val_fl, max_val_fl = MinMaxScaler(ori_data_fl)
    # get window data
    windowed_data, windowed_label = windowing(norm_data, label, seq_len=20, hop_size=10)
    windowed_data_fl, windowed_label_fl = windowing(norm_data_fl, label_fl, seq_len=20, hop_size=10, shuffle=False)
    windowed_label = windowed_label - 1
    windowed_label_fl = windowed_label_fl - 1

    NUM_CLASSES = num_room_house[house_name]
    APs = len(col_idx_use)
    total_number = 600

    all_acc = []
    all_f1 = []
    idx_epoch = []

    for num_epochs in range(20, 1000, 20):
        acc, f1 = gan_select_epoch(house_name, reduce_ap, NUM_CLASSES, APs, total_number, num_epochs, save_directory,
                     windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, device)
        print(
            f"Epoch [{num_epochs}/{1200}]\t "
            f"Accuracy: {acc}\t "
            f"F1: {f1}\t")

        all_acc.append(acc)
        all_f1.append(f1)
        idx_epoch.append(num_epochs)

    min_f1 = min(all_f1)
    min_index = all_f1.index(min_f1)
    min_epoch = idx_epoch[min_index]
    print('min epoch:', min_epoch, 'min F1:', min_f1)

    max_f1 = max(all_f1)
    max_index = all_f1.index(max_f1)
    max_epoch = idx_epoch[max_index]
    print('max epoch:', max_epoch, 'max F1:', max_f1)

