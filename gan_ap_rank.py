import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import csv
import torch

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

from GAN.con_gan_rssi.model_wgan_gp import Discriminator, Generator, initialize_weights
from general_utils import uniq_count
from GAN.gan_utils import load_GAN_model, generate_fake_data, get_fake_rssi,\
    MinMaxScaler, renormalization, windowing, get_mean_stat, get_std_stat, get_stats,\
    plot_line_rssi_gan, MLPClassifier, map_A, map_B, map_C, map_D, house_map,\
    binary_convert, num_room_house, train, test
from play_space import feature, augment
from gan_evaluation import get_col_use,multiclass, rf_multiclass
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_mean_house(house_name, num_epochs):
    reduce_ap = False
    col_idx_use, col_idx_use_label = get_col_use(house_name, reduce_ap)

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
    windowed_data_fl, windowed_label_fl = windowing(norm_data_fl, label_fl, seq_len=20, hop_size=10)

    NUM_CLASSES = num_room_house[house_name]
    APs = len(col_idx_use)

    keep_mean_real = get_mean_stat(np.transpose(windowed_data, (0, 2, 1)), windowed_label - 1, house_name,
                                   house_map[house_name], APs, NUM_CLASSES,
                                   transpose=False, save=False, plot_name=None)
    keep_std_real = get_std_stat(np.transpose(windowed_data, (0, 2, 1)), windowed_label-1, house_name, house_map[house_name], APs, NUM_CLASSES,
                  transpose=False, save=False, plot_name=None)

    WINDOW_SIZE = 20
    total_number = 1000
    low_bound = None
    GANmodel = "conGAN-CNN_house_" + house_name
    model_name = "ConGAN_wgp_rep_house_" + house_name + "_reduce_" + str(reduce_ap)
    fake_data, y_fake, gen = get_fake_rssi(windowed_label, num_epochs, total_number, low_bound, NUM_CLASSES, APs,
                                           save_directory,
                                           model_name, device)

    fake_data_np = fake_data.view(len(fake_data), APs, WINDOW_SIZE).detach().numpy()
    y_fake_np = y_fake.cpu().detach().numpy()
    keep_mean_fake = get_mean_stat(fake_data_np, y_fake_np, house_name, house_map[house_name],
                                   APs, NUM_CLASSES,
                                   transpose=False, save=False, plot_name=None)
    keep_std_fake = get_std_stat(fake_data_np, y_fake_np, house_name, house_map[house_name],
                              APs, NUM_CLASSES,
                              transpose=False, save=False, plot_name=None)
    return keep_mean_real, keep_mean_fake, keep_std_real, keep_std_fake, NUM_CLASSES, APs

def ranking_csv(keep_mean, house_name, house_map, APs, rooms, datatype="real"):
    keep_rank = (-keep_mean).argsort()
    map = house_map[house_name]
    ap_map_A = {0: 'f', 1: 'd', 2: 'c', 3: 'b', 4: 'a', 5: 'h', 6: 'e', 7: 'g'}
    ap_map_B = {0: 'g', 1: 'h', 2: 'd', 3: 'j', 4: 'i', 5: 'b', 6: 'c', 7: 'e', 8: 'k', 9: 'a', 10: 'f'}
    ap_map_C = {0: 'g', 1: 'd', 2: 'f', 3: 'k', 4: 'c', 5: 'b', 6: 'a', 7: 'j', 8: 'h', 9: 'i', 10: 'e'}
    ap_map_D = {0: 'g', 1: 'e', 2: 'j', 3: 'k', 4: 'b', 5: 'a', 6: 'i', 7: 'f', 8: 'd', 9: 'h', 10: 'c'}
    ap_map = {'A': ap_map_A, 'B': ap_map_B, 'C': ap_map_C, 'D': ap_map_D}
    ap_map = ap_map[house_name]

    keep_rank_room = []
    for i in range(len(keep_rank)):
        keep_rank_room.append(map[i + 1])
        for ap in range(APs):
            keep_rank_room.append(map[ap_map[keep_rank[i][ap]]])

    print("Done AP ranking in house :", house_name, datatype)
    keep_rank_room = np.asarray(keep_rank_room).reshape(rooms, APs+1)
    pd.DataFrame(np.transpose(keep_rank_room)).to_csv("GAN/ap_ranking/House_" + house_name + "_ranking_" + datatype + ".csv")

if __name__ == "__main__":
    data_directory = os.path.join('..', 'aloe', 'localisation', 'data', ''.format(os.path.sep))
    save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # house_name = 'A'
    # num_epochs = 620  # A 620, B 520, C 440(reduce 160), D 620

    # house_all = ['C']#['A','B','C','D']
    # num_epochs_all = [440]#[620, 520, 440, 620]
    house_all = ['A','B','C','D']
    num_epochs_all = [620, 520, 440, 620]

    for i in range(len(house_all)):
        house_name =  house_all[i]
        keep_mean_real, keep_mean_fake, keep_std_real, keep_std_fake, NUM_CLASSES, APs = load_mean_house(house_all[i], num_epochs_all[i])
        # ranking_csv(keep_mean_real, house_all[i], house_map, APs, NUM_CLASSES, datatype="real")
        # ranking_csv(keep_mean_fake, house_all[i], house_map, APs, NUM_CLASSES, datatype="fake")

        keep_rank_real = (-keep_mean_real).argsort()
        keep_rank_fake = (-keep_mean_fake).argsort()
        keep_rank_room = np.concatenate(([keep_rank_real[0]], [keep_rank_fake[0]]))
        for i in range(1, NUM_CLASSES):
            keep_rank_room = np.concatenate((keep_rank_room, [keep_rank_real[i]], [keep_rank_fake[i]]))

        ap_map = {'A': [1,1, 2,2, 3,3, 4,4],
                  'B': [1,1, 2,2, 3,3, 4,4, 5,5, 6,6, 7,7, 8,8, 9,9 ,10,10, 11,11],
                  'C': [1,1, 2,2, 3,3, 4,4, 5,5, 6,6, 7,7, 8,8, 9,9],
                  'D': [1,1, 2,2, 3,3, 4,4, 5,5, 6,6, 7,7, 8,8, 9,9 ,10,10]}

        df_rank = pd.DataFrame(np.transpose(keep_rank_room), columns=ap_map[house_name])
        df_rank = df_rank.rename(house_map[house_name], axis='columns')

        # get ranking correlation for each room
        keep_cor = []
        for n in range(NUM_CLASSES):
            cor = df_rank[house_map[house_name][n + 1]].corr(method='spearman')
            rank_sim = "{:.4f}".format(cor.iloc[0][1])
            keep_cor.append(rank_sim)
            keep_cor.append(rank_sim)
        df_cor = pd.DataFrame([keep_cor],columns=ap_map[house_name])
        df_cor = df_cor.rename(house_map[house_name], axis='columns')
        df_rank_cor = df_rank.append(df_cor)
        # save as csv file
        pd.DataFrame(df_rank_cor).to_csv(
            "GAN/ap_ranking/House_" + house_name + "_ranking_correlation.csv")