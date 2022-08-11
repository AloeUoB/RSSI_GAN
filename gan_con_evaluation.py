import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

from GAN.con_gan_rssi.model_wgan_gp import Discriminator, Generator, initialize_weights
from general_utils import uniq_count
from GAN.gan_utils import load_GAN_model, generate_fake_data, get_fake_rssi,\
    MinMaxScaler, renormalization, windowing, get_mean_stat, get_stats,\
    plot_line_rssi_gan, MLPClassifier, map_A, map_B, map_C, map_D, house_map,\
    binary_convert, num_room_house, train, test
from play_space import feature
from gan_evaluation import get_col_use,multiclass, rf_multiclass


if __name__ == "__main__":
    data_directory = os.path.join('..', 'aloe', 'localisation', 'data', ''.format(os.path.sep))
    save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 160
    house_name = 'C'
    reduce_ap = True
    col_idx_use, col_idx_use_label = get_col_use (house_name, reduce_ap)

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
    WINDOW_SIZE = 20
    total_number = 1000
    GANmodel = "conGAN-CNN_house_" + house_name
    model_name = "ConGAN_wgp_rep_house_" + house_name + "_reduce_" + str(reduce_ap)
    fake_data, y_fake, gen = get_fake_rssi(windowed_label, num_epochs, total_number, NUM_CLASSES, APs, save_directory,
                                           model_name, device)
# extract feature FINGERPRINT DATA
    windowed_data_tp = np.transpose(windowed_data, (0, 2, 1))
    X_train_feature, y_train_feature = feature(windowed_data_tp, windowed_label, datatype='rssi')  # (n,Aps,window)
    if reduce_ap:
        torch.save((X_train_feature, y_train_feature),
                   data_directory + '/gan_data/house_' + house_name + '_train_10hop_reduce_'+str(reduce_ap)+'.pt')
        X_train_feature, y_train_feature = torch.load(
            data_directory + '/gan_data/house_' + house_name + '_train_10hop_reduce_'+str(reduce_ap)+'.pt')
    else:
        torch.save((X_train_feature, y_train_feature), data_directory + '/gan_data/house_'+house_name+'_train_10hop.pt')
        X_train_feature, y_train_feature = torch.load(data_directory + '/gan_data/house_'+house_name+'_train_10hop.pt')
# extract feature FREE LIVING DATA
    windowed_data_fl_tp = np.transpose(windowed_data_fl, (0, 2, 1))
    X_test_feature, y_test_feature = feature(windowed_data_fl_tp, windowed_label_fl, datatype='rssi') # (n,Aps,window)
    if reduce_ap:
        torch.save((X_test_feature, y_test_feature),
                   data_directory + '/gan_data/house_' + house_name + '_test_10hop_reduce_'+str(reduce_ap)+'.pt')
        X_test_feature, y_test_feature = torch.load(
            data_directory + '/gan_data/house_' + house_name + '_test_10hop_reduce_'+str(reduce_ap)+'.pt')
    else:
        torch.save((X_test_feature, y_test_feature), data_directory + '/gan_data/house_'+house_name+'_test_10hop.pt')
        X_test_feature, y_test_feature = torch.load(data_directory + '/gan_data/house_'+house_name+'_test_10hop.pt')
# extract feature FAKE DATA
    fake_data = fake_data.view(len(fake_data), APs, WINDOW_SIZE)
    X_fake_feature, y_fake_feature = feature(fake_data.cpu().detach().numpy(), y_fake.cpu().detach().numpy(), datatype='rssi')
    torch.save((X_fake_feature, y_fake_feature),data_directory + '/gan_data/house_'+house_name+'_feature_fake_' + str(model_name) + str(num_epochs) + '.pt')
    X_fake_feature, y_fake_feature = torch.load(
        data_directory + '/gan_data/house_'+house_name+'_feature_fake_' + str(model_name) + str(num_epochs) + '.pt')

    windowed_label = windowed_label - 1
    windowed_label_fl = windowed_label_fl - 1
    y_train_feature = y_train_feature - 1
    y_test_feature = y_test_feature - 1

    multi = True
    if multi:
        # mlp with RSSI input
        multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                   APs, NUM_CLASSES, GANmodel, device, test_set='flive', exp=1, runs=10, show_epoch=False, flatten=True)

        multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                   APs, NUM_CLASSES, GANmodel, device, test_set='flive', exp=2, runs=10, show_epoch=False, flatten=True)

        # mlp with features input
        multiclass(X_train_feature, y_train_feature, X_test_feature, y_test_feature,
                   X_fake_feature, y_fake_feature, APs, NUM_CLASSES,
                   GANmodel, device, test_set='flive', exp=1, runs=10, show_epoch=False, feature=True, flatten=False)
        multiclass(X_train_feature, y_train_feature, X_test_feature, y_test_feature,
                   X_fake_feature, y_fake_feature, APs, NUM_CLASSES,
                   GANmodel, device, test_set='flive', exp=2, runs=10, show_epoch=False, feature=True, flatten=False)

    rf = True
    if rf:
        # RF with features input
        rf_multiclass(X_train_feature, y_train_feature, X_test_feature, y_test_feature,
                      X_fake_feature, y_fake_feature, GANmodel, test_set='flive', exp=1, runs=10, feature=True)
        rf_multiclass(X_train_feature, y_train_feature, X_test_feature, y_test_feature,
                      X_fake_feature, y_fake_feature, GANmodel, test_set='flive', exp=2, runs=10, feature=True)

        # RF with RSSI input
        fake_data = fake_data.view(len(fake_data), APs, WINDOW_SIZE)
        fake_data = fake_data.cpu().detach().numpy()
        y_fake = y_fake.cpu().detach().numpy()

        fake_data = fake_data.reshape((-1, APs*WINDOW_SIZE))
        windowed_data = windowed_data.reshape((-1, APs*WINDOW_SIZE))
        windowed_data_fl = windowed_data_fl.reshape((-1, APs*WINDOW_SIZE))

        rf_multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                      GANmodel, test_set='flive', exp=1, runs=10, feature=False)
        rf_multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                      GANmodel, test_set='flive', exp=2, runs=10, feature=False)

    # GANmodel = "conGAN-CNN_house_" + house_name
    # save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))
    # model_name = "ConGAN_wgp_rep_house_" + house_name + "_reduce_" + str(reduce_ap)
    # # generator parameter
    # Z_DIM = 11
    # CHANNELS_RSSI = 1
    # FEATURES_GEN = 32
    # NUM_CLASSES = num_room_house[house_name]
    # APs = len(col_idx_use)
    # WINDOW_SIZE = 20
    # GEN_EMBEDDING = 100
    # # load pretrained generator
    # gen = Generator(Z_DIM, CHANNELS_RSSI, FEATURES_GEN, NUM_CLASSES, APs, WINDOW_SIZE, GEN_EMBEDDING).to(device)
    # gen = load_GAN_model(gen, save_directory, model_name, num_epochs, device)
    # # generate fake data
    # total_number = 1000
    # unique_y, counts_y = np.unique(windowed_label, return_counts=True)
    # # get number of fake data
    # num_fake = []
    # for i in range(0, NUM_CLASSES):
    #     if total_number - counts_y[i] > 0:
    #         num_fake.append(total_number - counts_y[i])
    #     else:
    #         num_fake.append(0)
    # fake_data, y_fake = generate_fake_data(gen, Z_DIM, num_fake=num_fake, total_room=NUM_CLASSES, aps=APs,
    #                                        device=device)