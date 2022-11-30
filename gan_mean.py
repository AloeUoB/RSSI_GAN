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
    binary_convert, num_room_house, train, test, load_house_data
from play_space import feature, augment
from gan_evaluation import get_col_use,multiclass, rf_multiclass
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances

def get_mean_df(keep_mean, NUM_CLASSES, house_name, house_map, data_type):
    house_aps_label = {'A': ['f', 'd', 'c', 'b', 'a', 'h', 'e', 'g'],
                       'B': ['g', 'h', 'd', 'j', 'i', 'b', 'c', 'e', 'k', 'a', 'f'],
                       'C': ['g', 'd', 'f', 'k', 'c', 'b', 'a', 'j', 'h', 'i', 'e'],
                       'D': ['g', 'e', 'j', 'k', 'b', 'a', 'i', 'f', 'd', 'h', 'c']}
    rssi_plot = pd.DataFrame((keep_mean), columns=house_aps_label[house_name])
    rssi_plot = rssi_plot.rename(house_map[house_name], axis='columns')

    for room in range(NUM_CLASSES):
        if room == 0:
            d = {'rssi': keep_mean[room], 'ap': rssi_plot.columns, 'room': house_map[house_name][room + 1]+'('+data_type+')'}
            df_all = pd.DataFrame(data=d)
        else:
            d = {'rssi': keep_mean[room], 'ap': rssi_plot.columns, 'room': house_map[house_name][room + 1]+'('+data_type+')'}
            df = pd.DataFrame(data=d)
            df_all = pd.concat([df_all, df])

    return df_all

def get_mean_df_con(keep_mean_real, keep_mean_fake, NUM_CLASSES, house_name, house_map):
    house_aps_label = {'A': ['f', 'd', 'c', 'b', 'a', 'h', 'e', 'g'],
                       'B': ['g', 'h', 'd', 'j', 'i', 'b', 'c', 'e', 'k', 'a', 'f'],
                       'C': ['g', 'd', 'f', 'k', 'c', 'b', 'a', 'j', 'h', 'i', 'e'],
                       'D': ['g', 'e', 'j', 'k', 'b', 'a', 'i', 'f', 'd', 'h', 'c']}
    rssi_plot_real = pd.DataFrame((keep_mean_real), columns=house_aps_label[house_name])
    rssi_plot_real = rssi_plot_real.rename(house_map[house_name], axis='columns')

    rssi_plot_fake = pd.DataFrame((keep_mean_fake), columns=house_aps_label[house_name])
    rssi_plot_fake = rssi_plot_fake.rename(house_map[house_name], axis='columns')

    for room in range(NUM_CLASSES):
        if room == 0:
            d = {'rssi': keep_mean_real[room], 'ap': rssi_plot_real.columns, 'room': house_map[house_name][room + 1]+'(real)', 'dataset': 'real'}
            df_all = pd.DataFrame(data=d)
            d = {'rssi': keep_mean_fake[room], 'ap': rssi_plot_fake.columns, 'room': house_map[house_name][room + 1]+'(fake)', 'dataset': 'fake'}
            df = pd.DataFrame(data=d)
            df_all = pd.concat([df_all, df])
        else:
            d = {'rssi': keep_mean_real[room], 'ap': rssi_plot_real.columns, 'room': house_map[house_name][room + 1]+'(real)', 'dataset': 'real'}
            df = pd.DataFrame(data=d)
            df_all = pd.concat([df_all, df])
            d = {'rssi': keep_mean_fake[room], 'ap': rssi_plot_fake.columns, 'room': house_map[house_name][room + 1]+'(fake)', 'dataset': 'fake'}
            df = pd.DataFrame(data=d)
            df_all = pd.concat([df_all, df])

    return df_all

def get_mean_df3(keep_mean_real, keep_mean_fake, keep_mean_smote, NUM_CLASSES, house_name, house_map):
    house_aps_label = {'A': ['f', 'd', 'c', 'b', 'a', 'h', 'e', 'g'],
                       'B': ['g', 'h', 'd', 'j', 'i', 'b', 'c', 'e', 'k', 'a', 'f'],
                       'C': ['g', 'd', 'f', 'k', 'c', 'b', 'a', 'j', 'h', 'i', 'e'],
                       'D': ['g', 'e', 'j', 'k', 'b', 'a', 'i', 'f', 'd', 'h', 'c']}
    rssi_plot_real = pd.DataFrame((keep_mean_real), columns=house_aps_label[house_name])
    rssi_plot_real = rssi_plot_real.rename(house_map[house_name], axis='columns')

    rssi_plot_fake = pd.DataFrame((keep_mean_fake), columns=house_aps_label[house_name])
    rssi_plot_fake = rssi_plot_fake.rename(house_map[house_name], axis='columns')

    rssi_plot_smote = pd.DataFrame((keep_mean_smote), columns=house_aps_label[house_name])
    rssi_plot_smote = rssi_plot_smote.rename(house_map[house_name], axis='columns')

    for room in range(NUM_CLASSES):
        if room == 0:
            d = {'rssi': keep_mean_real[room], 'ap': rssi_plot_real.columns, 'room': house_map[house_name][room + 1]+'(real)', 'dataset': 'real'}
            df_real = pd.DataFrame(data=d)
            d = {'rssi': keep_mean_fake[room], 'ap': rssi_plot_fake.columns, 'room': house_map[house_name][room + 1]+'(GAN)', 'dataset': 'GAN'}
            df_gan= pd.DataFrame(data=d)
            d = {'rssi': keep_mean_smote[room], 'ap': rssi_plot_smote.columns,
                 'room': house_map[house_name][room + 1] + '(SMOTE)', 'dataset': 'SMOTE'}
            df_smote = pd.DataFrame(data=d)
            df_all = pd.concat([df_real, df_gan, df_smote])
        else:
            d = {'rssi': keep_mean_real[room], 'ap': rssi_plot_real.columns, 'room': house_map[house_name][room + 1]+'(real)', 'dataset': 'real'}
            df = pd.DataFrame(data=d)
            df_all = pd.concat([df_all, df])
            d = {'rssi': keep_mean_fake[room], 'ap': rssi_plot_fake.columns, 'room': house_map[house_name][room + 1]+'(GAN)', 'dataset': 'GAN'}
            df = pd.DataFrame(data=d)
            df_all = pd.concat([df_all, df])
            d = {'rssi': keep_mean_smote[room], 'ap': rssi_plot_smote.columns,
                 'room': house_map[house_name][room + 1] + '(SMOTE)', 'dataset': 'SMOTE'}
            df = pd.DataFrame(data=d)
            df_all = pd.concat([df_all, df])

    return df_all

def load_mean_house(data_directory, house_name, num_epochs):
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

if __name__ == "__main__":
    data_directory = os.path.join('..', 'aloe', 'localisation', 'data', ''.format(os.path.sep))
    save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))
    save_dir = os.path.join('..', 'aloe', 'GAN', 'sim_plot', ''.format(os.path.sep))
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    house_all = ['A', 'B', 'C', 'D']
    num_epochs_all = [620, 520, 440, 620]
    mean_plot = True

    keep_sim_mean_house = []
    keep_sim_std_house = []
    for i in range(0,4):
        house_name = house_all[i]

        keep_mean_real, keep_mean_fake, keep_std_real, keep_std_fake, NUM_CLASSES, APs = \
            load_mean_house(data_directory, house_all[i], num_epochs_all[i])

        windowed_data, windowed_label, APs, NUM_CLASSES = \
            load_house_data(data_directory, house_name, datatype='fp', reduce_ap=False)
        windowed_data_fl, windowed_label_fl, APs, NUM_CLASSES = \
            load_house_data(data_directory, house_name, datatype='fl', reduce_ap=False)



        from imblearn.over_sampling import SMOTE
        from collections import Counter

        X = windowed_data.reshape(windowed_data.shape[0], windowed_data.shape[1] * windowed_data.shape[2])
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, windowed_label)
        X_res = X_res.reshape(X_res.shape[0], X_res.shape[1] // windowed_data.shape[2],
                              X_res.shape[1] // windowed_data.shape[1])

        keep_mean_smote = get_mean_stat(np.transpose(X_res, (0, 2, 1)), y_res - 1, house_name,
                                       house_map[house_name], APs, NUM_CLASSES,
                                       transpose=False, save=False, plot_name=None)
        keep_std_smote = get_std_stat(np.transpose(X_res, (0, 2, 1)), y_res - 1, house_name,
                                     house_map[house_name], APs, NUM_CLASSES,
                                     transpose=False, save=False, plot_name=None)

        if mean_plot:
            sns.set_theme(rc={'figure.figsize': (12,7)})
            # concatenated = get_mean_df_con(keep_mean_real, keep_mean_fake, NUM_CLASSES, house_name, house_map)
            concatenated = get_mean_df3(keep_mean_real, keep_mean_fake, keep_mean_smote, NUM_CLASSES, house_name, house_map)
            sns.scatterplot(data=concatenated, x="room", y="rssi", hue="ap", style="dataset")
            plt.title("House " + house_name + ": mean", fontsize=14)
            plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
            plt.xticks(rotation=90, fontsize=14)
            plt.yticks(rotation=90, fontsize=12)
            plt.tight_layout()
            plt.savefig(save_dir + 'mean_house'+ house_name + '.png')
            plt.show()
            plt.close()

            sns.set_theme(rc={'figure.figsize': (12, 7)})
            # concatenated = get_mean_df_con(keep_std_real, keep_std_fake, NUM_CLASSES, house_name, house_map)
            concatenated = get_mean_df3(keep_std_real, keep_std_fake, keep_std_smote, NUM_CLASSES, house_name,
                                        house_map)
            sns.scatterplot(data=concatenated, x="room", y="rssi", hue="ap", style="dataset")
            plt.title("House " + house_name + ": SD")
            plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
            plt.xticks(rotation=90, fontsize=14)
            plt.yticks(rotation=90, fontsize=12)
            plt.tight_layout()
            plt.savefig(save_dir + 'std_house' + house_name + '.png')
            plt.show()


        sim_mean = euclidean_distances(keep_mean_real, keep_mean_fake)
        sim_std = euclidean_distances(keep_std_real, keep_std_fake)

        keep_sim_mean = []
        for i in range(NUM_CLASSES):
            keep_sim_mean.append(sim_mean[i][i])
        keep_sim_mean_house.append(keep_sim_mean)

        keep_sim_std = []
        for i in range(NUM_CLASSES):
            keep_sim_std.append(sim_std[i][i])
        keep_sim_std_house.append(keep_sim_std)

        compare_noise = False
        if compare_noise:
            noise_data = torch.randn(100 * NUM_CLASSES, 11, 20).uniform_(0, 1).detach().numpy()
            noise_label = np.full(100, 0)
            for i in range(1, NUM_CLASSES):
                noise_label = np.concatenate((noise_label, np.full(100, i)), axis=0)

            keep_mean_noise = get_mean_stat(noise_data, noise_label, house_name,
                                           house_map[house_name], APs, NUM_CLASSES,
                                           transpose=False, save=False, plot_name=None)

            sim_mean_real_noise = euclidean_distances(keep_mean_real, keep_mean_noise)
            sim_mean_fake_noise = euclidean_distances(keep_mean_fake, keep_mean_noise)

    # df_mean_real = get_mean_df(keep_mean_real, NUM_CLASSES, house_name, house_map,'real')
    # df_mean_fake = get_mean_df(keep_mean_fake, NUM_CLASSES, house_name, house_map,'fake')
    # concatenated = pd.concat([df_mean_real.assign(dataset='real'), df_mean_fake.assign(dataset='fake')])

    # fig, ax = plt.subplots()
    # sns.set_theme(rc={'figure.figsize': (8, 5)})
    # sns.scatterplot(data=df_mean_fake, x="room", y="rssi", hue="ap")
    # plt.ylim(-0.1, 0.8)
    # # plt.xlabel("Room",fontsize=14)
    # # plt.xticks(fontsize=12)
    # # plt.ylabel("RSSI VALUE",fontsize=14)
    # # plt.yticks(fontsize=12)
    # ax.xaxis.set_tick_params(rotation=30, labelsize=10)
    # plt.title('Mean RSSI House C')
    # plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    # plt.tight_layout()
    # plt.show()

    # p = sns.relplot(
    #     data=concatenated, x="room", y="rssi", hue="ap",
    #     col="dataset", kind="scatter")
    # p.fig.suptitle("House C means")

