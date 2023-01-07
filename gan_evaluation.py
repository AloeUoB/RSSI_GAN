import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

from GAN.con_gan_rssi.model_wgan_gp import Discriminator, Generator, initialize_weights
from general_utils import uniq_count
from GAN.gan_utils import load_GAN_model, generate_fake_data, get_fake_rssi,\
    MinMaxScaler, renormalization, windowing, get_mean_stat, get_stats,\
    plot_line_rssi_gan, MLPClassifier, map_A, map_B, map_C, map_D, house_map,\
    binary_convert, num_room_house, train, test, plot_line_rssi_all, get_col_use
from play_space import feature

def confuse_acc(X_test, y_test, model):
    output = model(X_test)
    pred = output.argmax(1)
    pred = pred.cpu().detach().numpy()
    true = y_test.cpu().detach().numpy()
    cf_mat = confusion_matrix(true, pred)
    class_acc = []
    for i in range(len(np.unique(true))):
        class_acc.append(cf_mat[i][i] / np.sum(cf_mat[i]) * 100)

    return cf_mat, class_acc

def plot_class_acc(class_acc, house_map, plot_title, save_dir,color):
    label_plot = np.arange(1, len(class_acc)+1).tolist()
    for i in range(len(label_plot)):
        label_plot[i] = house_map[label_plot[i]]
    plt.bar(label_plot, class_acc, color = color)
    plt.title("%s" % (plot_title), fontsize=14)
    plt.xticks(fontsize=14, rotation=90)
    plt.yticks(fontsize=14)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(save_dir + plot_title + '.png')
    plt.show()

def multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
               house_name, house_map, APs, NUM_CLASSES, GANmodel, device, f1_type ='weighted', test_set='flive', exp=1, runs=1, epochs=200, show_epoch=True,
               feature=False, flatten=True, confusion_met=True, rt=False):

    X1, X2, y1, y2 = train_test_split(windowed_data, windowed_label, test_size=0.3, shuffle=True,
                                      stratify=windowed_label, random_state=42)
    X1, y1 = windowed_data, windowed_label

    # TRAIN set
    if exp == 1:  # Experiment 1
        X_train = torch.from_numpy(X1)
        y_train = torch.from_numpy(y1)
    if exp == 2:  # Experiment 2
        if not feature:
            fake_data = fake_data.view(len(fake_data), APs, 20)
            fake_data = torch.transpose(fake_data, 1, 2)
            fake_data = fake_data.cpu().detach().numpy()
            y_fake = y_fake.cpu().detach().numpy()
        X_train = torch.cat((torch.from_numpy(X1).to(device), torch.from_numpy(fake_data).to(device)), dim=0)
        y_train = torch.cat((torch.from_numpy(y1).to(device), torch.from_numpy(y_fake).to(device)), dim=0)
    if exp == 3:
        X_train = torch.from_numpy(X1)
        y_train = torch.from_numpy(y1)

    # TEST set
    if test_set == 'fp':
        X_test = torch.from_numpy(X2)
        y_test = torch.from_numpy(y2)
    if test_set == 'flive':
        X_test = torch.from_numpy(windowed_data_fl)
        y_test = torch.from_numpy(windowed_label_fl)

    if flatten:
        X_train = torch.reshape(X_train, (-1, APs*20))
        X_test = torch.reshape(X_test, (-1, APs*20))

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train.float(), y_train.float())
    test_dataset = torch.utils.data.TensorDataset(X_test.float(), y_test.float())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=128,
                                               shuffle=True,
                                               drop_last=True, )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=128,
                                              shuffle=False,
                                              drop_last=True, )

    model_type = "mlp"
    print(test_set, 'experiment', exp, model_type, 'model', GANmodel, 'feature',feature)
    F1_all_1 = []
    F1_all_2 = []
    mcc_all = []
    acc_all_1 = []
    for i in range(runs):
        # if model_type == "log":
        #     model = LogisticRegression(APs*20, NUM_CLASSES).to(device)
        if model_type == "mlp":
            if feature:
                model = MLPClassifier(APs*7, NUM_CLASSES).to(device)
            else:
                model = MLPClassifier(APs*20, NUM_CLASSES).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()
        epochs = epochs
        loss_threshold = 0.001
        keep_loss = []
        keep_avg_loss = 0
        for epoch in range(epochs):
            loss_epoch, accuracy_epoch, f1_epoch_w, f1_epoch_m = train(train_loader, model, optimizer, criterion)
            keep_loss.append(loss_epoch)
            if epoch % 10 == 0 and epoch != 0:
                if show_epoch:
                    print(keep_avg_loss - (sum(keep_loss) / 10))
                if abs(keep_avg_loss - (sum(keep_loss) / 10)) < loss_threshold:
                    if show_epoch:
                        print(
                            f"Epoch [{epoch}/{epochs}]\t "
                            f"Loss: {loss_epoch}\t "
                            f"Accuracy: {accuracy_epoch}\t "
                            f"F1_weighted: {f1_epoch_w * 100}\t"
                            f"F1_macro: {f1_epoch_m * 100}\t")
                    break
                else:
                    keep_avg_loss = sum(keep_loss) / 10
                keep_loss = []

            if show_epoch:
                print(
                    f"Epoch [{epoch}/{epochs}]\t "
                    f"Loss: {loss_epoch / len(train_loader)}\t "
                    f"Accuracy: {accuracy_epoch}\t "
                    f"F1: {f1_epoch_w * 100}\t")

        loss, accuracy, f1_weight, f1_mac, mcc = test(test_loader, model, criterion)
        F1_all_1.append(f1_weight)
        F1_all_2.append(f1_mac)
        mcc_all.append(mcc*100)
        acc_all_1.append(accuracy)

    print('Average f1_weighted: ', sum(F1_all_1) / len(F1_all_1))
    print('Average f1_macro: ', sum(F1_all_2) / len(F1_all_2))
    print('Average mcc: ', sum(mcc_all) / len(mcc_all))
    print('Average accuracy: ', sum(acc_all_1) / len(acc_all_1))

    # pred, true = model_test(X_test.float(), y_test, model)
    if confusion_met:
        _output = model(X_test.float())
        pred = _output.argmax(1)
        acc = (pred == y_test).sum().item() / y_test.size(0)
        print('Accuracy:', acc * 100)
        pred = pred.cpu().detach().numpy()
        true = y_test.cpu().detach().numpy()
        f1 = f1_score(true, pred, average='weighted')
        print('F1 score:', f1 * 100)
        print(confusion_matrix(true, pred))

    cf_mat, class_acc = confuse_acc(X_test.float(), y_test, model)
    if exp == 1:
        plot_title = 'House_'+house_name+' control'+ GANmodel
        color = 'blue'
    elif exp == 2:
        plot_title = 'House_' + house_name + ' GAN'+ GANmodel
        color = 'green'
    elif exp == 3:
        plot_title = 'House_' + house_name + ' SMOTE'+ GANmodel
        color = 'orange'

    save_dir = os.path.join('..', 'aloe', 'GAN', 'cf_plot', ''.format(os.path.sep))
    plot_class_acc(class_acc, house_map, plot_title, save_dir, color)

    if rt:
        return sum(acc_all_1)/len(acc_all_1), sum(F1_all_1)/len(F1_all_1), sum(F1_all_2)/len(F1_all_2)

def rf_multiclass(X_train_feature, y_train_feature, X_test_feature, y_test_feature,
                  X_fake_feature, y_fake_feature, GANmodel, house_name, house_map, test_set='fp', exp=1, runs=10, feature=True):

    X1, X2, y1, y2 = train_test_split(X_train_feature, y_train_feature, test_size=0.3, shuffle=True,
                                      stratify=y_train_feature, random_state=42)
    X1, y1 = X_train_feature, y_train_feature

    if test_set == 'fp':
        X_train, X_test, y_train, y_test = X1, X2, y1, y2
    if test_set == 'flive':
        X_train, y_train = X1, y1
        X_test, y_test = X_test_feature, y_test_feature

    if exp == 2:
        X_train = np.concatenate((X_train, X_fake_feature), axis=0)
        y_train = np.concatenate((y_train, y_fake_feature), axis=0)

    print(test_set, 'experiment', exp,'model', GANmodel,'feature',feature)

    clf = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=False)
    param_grid = {
        'min_samples_leaf': [1, 3, 5, 10],
        'n_estimators': [30, 50, 100, 200, 300, 500, 1000]}
    clf_grid = GridSearchCV(clf, param_grid=param_grid, cv=cv, refit=True, n_jobs=-1)
    clf_grid.fit(X_train_feature, y_train_feature)
    print('Best parameters are: {}'.format(clf_grid.best_params_))

    f1_all_1 = []
    f1_all_2 = []
    acc_all =[]
    for i in range(runs):
        clf_tune = RandomForestClassifier(min_samples_leaf=clf_grid.best_params_['min_samples_leaf'],
                                          n_estimators=clf_grid.best_params_['n_estimators'], )
        clf_tune.fit(X_train, y_train)
        y_pred = clf_tune.predict(X_test)
        f1_weight = f1_score(y_test, y_pred, average='weighted') * 100
        f1_mac = f1_score(y_test, y_pred, average='macro') * 100
        acc = accuracy_score(y_test, y_pred)* 100
        f1_all_1.append(f1_weight)
        f1_all_2.append(f1_mac)
        acc_all.append(acc)


    print('Average f1_weighted:', np.mean(f1_all_1))
    print('Average f1_macro:', np.mean(f1_all_2))
    print('Average acc', np.mean(acc_all))
    cf_mat = confusion_matrix(y_test, y_pred)
    print(cf_mat)

    class_acc = []
    for i in range(len(np.unique(y_test))):
        class_acc.append(cf_mat[i][i] / np.sum(cf_mat[i]) * 100)

    if exp == 1:
        plot_title = 'RF_House_'+house_name+' control'
        color = 'blue'
    elif exp == 2:
        plot_title = 'RF_House_' + house_name + ' GAN'
        color = 'green'
    elif exp == 3:
        plot_title = 'RF_House_' + house_name + ' SMOTE'
        color = 'orange'

    save_dir = os.path.join('..', 'aloe', 'GAN', 'cf_plot', ''.format(os.path.sep))
    plot_class_acc(class_acc, house_map, plot_title, save_dir, color)

def rf_classification_f1(X_train, y_train, X_test, y_test, APs,
                   GANmodel, exp=1,flatten=True, feature=True, confusion_met=False):

    if flatten:
        X_train = X_train.reshape(-1, APs * 20)
        X_test = X_test.reshape(-1, APs * 20)

    # print('experiment', exp,'model', GANmodel,'feature',feature)
    unique_y, counts_y = np.unique(y_train, return_counts=True)
    if counts_y[0] == 1 :
        X_train = np.concatenate((X_train,X_train,X_train))
        y_train = np.concatenate((y_train,y_train,y_train))

    clf = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=False)
    param_grid = {
        'min_samples_leaf': [1, 3, 5, 10],
        'n_estimators': [30, 50, 100, 200, 300, 500, 1000]}
    clf_grid = GridSearchCV(clf, param_grid=param_grid, cv=cv, refit=True, n_jobs=-1)
    clf_grid.fit(X_train, y_train)
    # print('Best parameters are: {}'.format(clf_grid.best_params_))

    f1_all_1 = []
    f1_all_2 = []
    acc_all =[]

    clf_tune = RandomForestClassifier(min_samples_leaf=clf_grid.best_params_['min_samples_leaf'],
                                      n_estimators=clf_grid.best_params_['n_estimators'], )
    clf_tune.fit(X_train, y_train)
    y_pred = clf_tune.predict(X_test)
    f1_weight = f1_score(y_test, y_pred, average='weighted') * 100
    f1_mac = f1_score(y_test, y_pred, average='macro') * 100
    acc = accuracy_score(y_test, y_pred)* 100
    f1_all_1.append(f1_weight)
    f1_all_2.append(f1_mac)
    acc_all.append(acc)

    if confusion_met:
        cf_mat = confusion_matrix(y_test, y_pred)
        print(cf_mat)
        class_acc = []
        for i in range(len(np.unique(y_test))):
            class_acc.append(cf_mat[i][i] / np.sum(cf_mat[i]) * 100)
        print(class_acc)

    return sum(f1_all_2) / len(f1_all_2)


def classification_f1(X_train, y_train, X_test, y_test, APs, NUM_CLASSES, epochs, train_Bsize, device, GANmodel,
                   exp=1, flatten=True, show_epoch=True, confusion_met=True):
    if flatten:
        X_train = torch.reshape(X_train, (-1, APs * 20))
        X_test = torch.reshape(X_test, (-1, APs * 20))

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train.float(), y_train.long())
    test_dataset = torch.utils.data.TensorDataset(X_test.float(), y_test.long())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_Bsize,
                                               shuffle=True, drop_last=True,)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                              shuffle=False, drop_last=True,)
    model_type = "mlp"
    if show_epoch:
        print('experiment', exp, model_type, 'GAN-model', GANmodel)
    F1_all_1 = []
    F1_all_2 = []
    mcc_all = []
    acc_all_1 = []
    loss_all = []

    model = MLPClassifier(APs * 20, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    loss_threshold = 0.001
    keep_loss = []
    keep_avg_loss = 0
    for epoch in range(epochs):
        epoch +=1
        loss_epoch, accuracy_epoch, f1_epoch_w, f1_epoch_m = train(train_loader, model, optimizer, criterion)
        keep_loss.append(loss_epoch)
        if epoch % 10 == 0 and epoch != 0:
            if show_epoch:
                print(keep_avg_loss - (sum(keep_loss) / 10))
            if abs(keep_avg_loss - (sum(keep_loss)/10)) < loss_threshold:
                if show_epoch:
                    print(
                        f"Epoch [{epoch}/{epochs}]\t "
                        f"Loss: {loss_epoch}\t "
                        f"Accuracy: {accuracy_epoch}\t "
                        f"F1_weighted: {f1_epoch_w * 100}\t"
                        f"F1_macro: {f1_epoch_m * 100}\t")
                break
            else:
                keep_avg_loss = sum(keep_loss)/10
            keep_loss = []

        if show_epoch:
            print(
                f"Epoch [{epoch}/{epochs}]\t "
                f"tLoss: {loss_epoch}\t "
                f"Accuracy: {accuracy_epoch}\t "
                f"F1_weighted: {f1_epoch_w * 100}\t"
                f"F1_macro: {f1_epoch_m * 100}\t")
    if show_epoch:
        print('final epoch',epoch)
    loss, accuracy, f1_weight, f1_mac, mcc = test(test_loader, model, criterion)
    F1_all_1.append(f1_weight)
    F1_all_2.append(f1_mac)
    mcc_all.append(mcc*100)
    acc_all_1.append(accuracy)
    loss_all.append(loss)

    if show_epoch:
        print('Average f1_weighted: ', sum(F1_all_1) / len(F1_all_1))
        print('Average f1_macro: ', sum(F1_all_2) / len(F1_all_2))
        print('Average mcc: ', sum(mcc_all) / len(mcc_all))
        print('Average accuracy: ', sum(acc_all_1) / len(acc_all_1))

    if confusion_met:
        print('This confusion matrix scores')
        _output = model(X_test.float())
        pred = _output.argmax(1)
        acc = (pred == y_test).sum().item() / y_test.size(0)
        print('Accuracy:', acc * 100)
        pred = pred.cpu().detach().numpy()
        true = y_test.cpu().detach().numpy()
        f1_w = f1_score(true, pred, average='weighted')
        f1_m = f1_score(true, pred, average='macro')
        print('F1_weighted score:', f1_w * 100)
        print('F1_macro score:', f1_m * 100)
        print(confusion_matrix(true, pred))

    return sum(F1_all_2) / len(F1_all_2)

if __name__ == "__main__":
    data_directory = os.path.join('..', 'aloe', 'localisation', 'data', ''.format(os.path.sep))
    save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))
    # data_directory = os.path.join('..', 'SimCLR', 'localisation', 'data', ''.format(os.path.sep))
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 440
    house_name = 'C'
    reduce_ap = False
    col_idx_use, col_idx_use_label = get_col_use (house_name, reduce_ap)
# load fingerprint data
    house_file = 'csv_house_' + house_name + '_fp.csv'
    ori_data = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=col_idx_use)
    label = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=[col_idx_use_label])
# load free living data
    house_file = 'csv_house_' + house_name + '_fl.csv'
    ori_data_fl = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=col_idx_use)
    label_fl = np.loadtxt(data_directory + 'csv_house_C_fl.csv', delimiter=",", skiprows=1, usecols=[col_idx_use_label])
# data normalisation
    norm_data, min_val, max_val = MinMaxScaler(ori_data)
    norm_data_fl, min_val_fl, max_val_fl = MinMaxScaler(ori_data_fl)
# get window data
    windowed_data, windowed_label = windowing(norm_data, label, seq_len=20, hop_size=10)
    windowed_data_fl, windowed_label_fl = windowing(norm_data_fl, label_fl, seq_len=20, hop_size=10)
    check_data = False
    if check_data:
        plot_line_rssi_gan(windowed_data[0], windowed_label[0], transpose=False, house=house_name,
                           house_map=map_D,
                           full_scale=False, ymin=-0.1, ymax=1.1, save=False, model_name='house_D_real')
        get_stats(data_directory, house_file)

    plot_all= False
    if plot_all:
        for i in range(len(windowed_data)):
            plot_line_rssi_all(windowed_data[i], windowed_label[i], transpose=False, house=house_name, house_map=house_map[house_name],
                                   reduce=False, full_scale=False, ymin=-0.1, ymax=1.1, save=True, model_name='_idx'+str(i))

    gen_fake = True
    if gen_fake:
        # GENERATE FAKE DATA
        NUM_CLASSES = num_room_house[house_name]
        APs = len(col_idx_use)
        total_number = 600
        GANmodel = "conGAN-CNN_house_" + house_name
        model_name = "ConGAN_wgp_rep_house_" + house_name + "_reduce_" + str(reduce_ap)
        fake_data, y_fake, gen = get_fake_rssi(windowed_label, num_epochs, total_number, NUM_CLASSES, APs, save_directory, model_name, device)

    # visualisation
    visualisation = True
    if visualisation:
        for i in range(NUM_CLASSES):
            fake_data_room = fake_data[y_fake == i]
            for j in range(5):
                fake_plot = fake_data_room[j].view(APs, 20)
                plot_line_rssi_gan(fake_plot.cpu().detach().numpy(), i+1, transpose=True, house=house_name, house_map=house_map[house_name],
                                   reduce=reduce_ap, full_scale=False, ymin=-0.1, ymax=1.1, save=True, save_dir='GAN/result_visual/', model_name=model_name+'_e'+str(num_epochs)+'_', plot_idx=j)



    # windowed_data_tp = np.transpose(windowed_data, (0, 2, 1))
    # X_train_feature, y_train_feature = feature(windowed_data_tp, windowed_label, datatype='rssi') #(n,Aps,window)
    # # torch.save((X_train_feature, y_train_feature), data_directory + '/gan_data/house_C_living_train_10hop.pt')
    # X_train_feature, y_train_feature = torch.load(data_directory + '/gan_data/house_C_living_train_10hop.pt')
    # # windowed_data_fl_tp = np.transpose(windowed_data_fl, (0, 2, 1))
    # # X_test_feature, y_test_feature = feature(windowed_data_fl_tp, windowed_label_fl, datatype='rssi')
    # # torch.save((X_test_feature, y_test_feature), data_directory + '/gan_data/house_C_living_test_10hop.pt')
    # X_test_feature, y_test_feature = torch.load(data_directory + '/gan_data/house_C_living_test_10hop.pt')
    #
    # # fake_data_tp = np.transpose(fake_data, (0, 2, 1))
    # # fake_data = fake_data.view(len(fake_data), 11, 20)
    # # X_fake_feature, y_fake_feature = feature(fake_data.detach().numpy(), y_fake.detach().numpy(), datatype='rssi')
    # # torch.save((X_fake_feature, y_fake_feature),data_directory + '/gan_data/house_C_feature_fake_' + str(GANmodel) + str(200) + '.pt')
    # X_fake_feature, y_fake_feature = torch.load(
    #     data_directory + '/gan_data/house_C_feature_fake_' + str(GANmodel) + str(200) + '.pt')
    #
    # windowed_label = windowed_label - 1
    # windowed_label_fl = windowed_label_fl - 1
    # y_train_feature = y_train_feature - 1
    # y_test_feature = y_test_feature - 1

    multi = False
    if multi:
        # mlp with RSSI input
        multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                   GANmodel, device, test_set='flive', exp=1, runs=10, show_epoch=False, flatten=True)

        multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                   GANmodel, device, test_set='flive', exp=2, runs=10, show_epoch=False, flatten=True)


        # mlp with features input
        multiclass(X_train_feature, y_train_feature, X_test_feature, y_test_feature,
                   X_fake_feature, y_fake_feature,
                   GANmodel, device, test_set='flive', exp=1, runs=10, show_epoch=False, feature=True, flatten=False)
        multiclass(X_train_feature, y_train_feature, X_test_feature, y_test_feature,
                   X_fake_feature, y_fake_feature,
                   GANmodel, device, test_set='flive', exp=2, runs=10, show_epoch=False, feature=True, flatten=False)

    rf = False
    if rf:
        # RF with features input
        rf_multiclass(X_train_feature, y_train_feature, X_test_feature, y_test_feature,
                  X_fake_feature, y_fake_feature, test_set='flive', exp=1, runs=10,feature=True)
        rf_multiclass(X_train_feature, y_train_feature, X_test_feature, y_test_feature,
                      X_fake_feature, y_fake_feature, test_set='flive', exp=2, runs=10,feature=True)

        # RF with RSSI input
        fake_data = fake_data.view(len(fake_data), 11, 20)
        fake_data = fake_data.cpu().detach().numpy()
        y_fake = y_fake.cpu().detach().numpy()
        fake_data = fake_data.reshape((-1, 220))
        windowed_data =  windowed_data.reshape((-1, 220))
        windowed_data_fl = windowed_data_fl.reshape((-1, 220))

        rf_multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                      test_set='flive', exp=1, runs=10, feature=False)
        rf_multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                      test_set='flive', exp=2, runs=10, feature=False)




