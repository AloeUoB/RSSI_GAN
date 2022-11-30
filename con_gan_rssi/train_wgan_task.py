import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier

from utils import gradient_penalty
from model_wgan_gp import Discriminator, Generator, initialize_weights

from GAN.gan_utils import load_GAN_model, generate_fake_data,\
    MinMaxScaler, renormalization, windowing, get_mean_stat,\
    plot_line_rssi_gan, MLPClassifier,\
    binary_convert, map_A, map_C, map_D, train, test,\
    num_room_house, house_map, load_house_data

from play_space import feature
from general_utils import uniq_count
import wandb

def MinMaxScaler(data):
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val
    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)
    return norm_data, min_val, max_val

def numpy_to_2D(X):
    X = torch.from_numpy(X)
    return X.unsqueeze(1)

def select_data(X, y, potion, no_label):  # X2_train, y_train
    idx = y.argsort()
    ls = y[idx]
    end_lab_idx = np.where(ls[1:] != ls[:-1])
    unique_y, counts_y = np.unique(y, return_counts=True)
    choose_idx = np.array([], dtype=int)
    for i in range(0, no_label):
        if i == 0:
            idx_of_lab_i = idx[0:end_lab_idx[0][0]+1]
        elif i == (no_label-1):
            idx_of_lab_i = idx[end_lab_idx[0][i - 1] + 1:len(ls)]
        else:
            idx_of_lab_i = idx[end_lab_idx[0][i - 1] + 1:end_lab_idx[0][i]+1]

        choose_idx_lab_i = idx_of_lab_i[0:math.ceil(counts_y[i] * potion/ 100)]
        choose_idx = np.append(choose_idx, choose_idx_lab_i)

    # get data for one shot training
    y_shot = y[choose_idx]
    X_shot = X[choose_idx]
    # keep the rest except for validation
    mask = np.ones(len(X), bool)
    mask[choose_idx] = False
    y_rest = y[mask]
    X_rest = X[mask]

    return X_shot, y_shot, X_rest, y_rest, choose_idx

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_directory = os.path.join('..', 'aloe', 'localisation', 'data', ''.format(os.path.sep))
    save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))

    house_name = 'A'
    reduce_ap = False
    # load training set
    rssi_fp, y_fp, APs, NUM_CLASSES = \
        load_house_data(data_directory, house_name, datatype='fp', reduce_ap=False)
    y_fp = y_fp-1
    rssi_fp = np.transpose(rssi_fp, (0, 2, 1))
    X_train, y_train = rssi_fp, y_fp
    X_train2D = numpy_to_2D(X_train)

    # load validation set
    rssi_fl, y_fl, APs, NUM_CLASSES = \
        load_house_data(data_directory, house_name, datatype='fl', reduce_ap=False)
    y_fl = y_fl - 1
    # custom choose first 10% of each class
    X_val, y_val, X_valid, y_valid, choose_idx = select_data(rssi_fl, y_fl, 10, num_room_house[house_name])
    # _, X_val, _y, y_val = train_test_split(rssi_fl, y_fl, test_size=0.1, shuffle=True, stratify=y_fl, random_state=1)
    # X_val = np.transpose(X_val, (0, 2, 1))
    X_val_feature, y_val_feature = feature(X_val, y_val, datatype='rssi')
    X_val = torch.from_numpy(X_val)
    X_val_flat = torch.reshape(X_val, (-1, len(X_val[0])*len(X_val[0,0])))

    # Hyperparameters etc.
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 64
    WINDOW_SIZE = 20
    CHANNELS_RSSI = 1
    GEN_EMBEDDING = 100
    Z_DIM = 11
    FEATURES_CRITIC = 32
    FEATURES_GEN = 32
    CRITIC_ITERATIONS = 5
    LAMBDA_GP = 10

    dataset_train = torch.utils.data.TensorDataset(X_train2D.float(), torch.from_numpy(y_train).int())
    dataset_val = torch.utils.data.TensorDataset(X_val_flat.float().to(device), torch.from_numpy(y_val).int().to(device))

    loader_train = torch.utils.data.DataLoader(dataset_train,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         drop_last=True,)

    loader_val = torch.utils.data.DataLoader(dataset_val,
                                               batch_size=len(X_val_flat),
                                               shuffle=False,
                                               drop_last=True, )

    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])
    gen = Generator(Z_DIM, CHANNELS_RSSI, FEATURES_GEN, NUM_CLASSES, APs, WINDOW_SIZE, GEN_EMBEDDING).to(device)
    critic = Discriminator(CHANNELS_RSSI, FEATURES_CRITIC, NUM_CLASSES, APs, WINDOW_SIZE).to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    # initializate optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    # for tensorboard plotting
    fixed_noise = torch.randn(32, Z_DIM, APs, 20).uniform_(0, 1).to(device)
    fixed_noise_plot = fixed_noise[0][0]
    # plot_line_rssi_one(fixed_noise_plot .cpu().detach().numpy(), 1, label_map=label_map, ap_map=ap_map, ymin=-0.1, ymax=1.1)
    model_name = "ConGAN_wgp_task_house_" + house_name +"_reduce_" + str(reduce_ap)
    writer_loss = SummaryWriter(f"logs/" + model_name + "/loss/")

    gen.train()
    critic.train()

    NUM_EPOCHS = 1200
    # wandb.init(project=model_name, entity="rssi-augmentation")
    # wandb.config = {
    #     "learning_rate": LEARNING_RATE,
    #     "epochs": NUM_EPOCHS,
    #     "batch_size": BATCH_SIZE,
    #     "window_size": WINDOW_SIZE,
    #     "channels_RSSI": CHANNELS_RSSI,
    #     "gen_embed_size": GEN_EMBEDDING,
    #     "noise_dimention": Z_DIM,
    #     "feature_critic": FEATURES_CRITIC,
    #     "feature_gen": FEATURES_GEN,
    #     "critic_iterations": CRITIC_ITERATIONS,
    #     "Lambda_GP": LAMBDA_GP,
    #     "number_APs": APs,
    #     "number_rooms" : NUM_CLASSES,
    # }
    keep_performance = 0
    best_epoch = 0
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, labels) in enumerate(loader_train):
            real = real.to(device)
            cur_batch_size = real.shape[0]
            labels = labels.to(device)

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, APs, 20).uniform_(0, 1).to(device)
                fake = gen(noise, labels)
                critic_real = critic(real, labels).reshape(-1)
                critic_fake = critic(fake, labels).reshape(-1)
                gp = gradient_penalty(critic, labels, real, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = critic(fake, labels).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # wandb.log({"Loss/disc_train_epoch": loss_critic.item()})
            # wandb.log({"Loss/gen_train_epoch": loss_gen.item()})
        print(f"Epoch [{epoch}/{NUM_EPOCHS}], Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}")

        plot_line_rssi_gan(noise[0][0].view(APs, 20).cpu().detach().numpy(), labels[0].cpu().detach().numpy() + 1,
                           house=house_name, house_map=house_map[house_name], ymin=-0.1, ymax=1.1, save=False,
                           reduce=False, save_dir='GAN/train_visual/', model_name='noise_fake_', transpose=True)

        plot_line_rssi_gan(fake[0].view(APs, 20).cpu().detach().numpy(), labels[0].cpu().detach().numpy() + 1,
                           house=house_name, house_map=house_map[house_name], ymin=-0.1, ymax=1.1, save=False,
                           reduce=False, save_dir='GAN/train_visual/', model_name='noise_fake_', transpose=True)

        with torch.no_grad():
            x_fake = fake[0].view(APs, 20)
            x_real = real[0].view(APs, 20)
            cur_labels_numpy = labels.cpu().detach().numpy()

            # plot_line_rssi_gan(x_fake.cpu().detach().numpy(), cur_labels_numpy[0] + 1, house=house_name,
            #                    house_map=house_map[house_name], ymin=-0.1, ymax=1.1, save=True,reduce=reduce_ap,
            #                    save_dir='GAN/train_visual/'+model_name+'/', model_name=str(epoch)+'_fake_', transpose=True)
            # plot_line_rssi_gan(x_real.cpu().detach().numpy(), cur_labels_numpy[0] + 1, house=house_name,
            #                    house_map=house_map[house_name], ymin=-0.1, ymax=1.1, save=True,reduce=reduce_ap,
            #                    save_dir='GAN/train_visual/'+model_name+'/', model_name=str(epoch)+'_real_', transpose=True)

            num_fake = []
            # unique_y, counts_y = np.unique(windowed_label, return_counts=True)
            total_number = 1000
            for i in range(0, NUM_CLASSES):
                num_fake.append(total_number)
            fake_val, fake_y_val = generate_fake_data(gen, Z_DIM, num_fake=num_fake, total_room=NUM_CLASSES, aps=APs, device=device)

            plot_line_rssi_gan(fake_val[0].view(APs, 20).cpu().detach().numpy(), labels[0].cpu().detach().numpy() + 1,
                               house=house_name, house_map=house_map[house_name], ymin=-0.1, ymax=0.5, save=False,
                               reduce=False, save_dir='GAN/train_visual/', model_name='noise_fake_', transpose=True)

        fake_val = fake_val.view(len(fake_val), APs, WINDOW_SIZE)
        fake_val = fake_val.cpu().detach().numpy()
        fake_y_val = fake_y_val.cpu().detach().numpy()
        fake_val = torch.from_numpy(fake_val).to(device)
        fake_y_val = torch.from_numpy(fake_y_val).to(device)
        fake_val_flat = torch.reshape(fake_val, (-1, APs*WINDOW_SIZE))

        dataset_fake = torch.utils.data.TensorDataset(fake_val_flat.float(), fake_y_val)
        loader_fake = torch.utils.data.DataLoader(dataset_fake,
                                                 batch_size=128,
                                                 shuffle=True,
                                                 drop_last=True,)
        # MLP CLASSIFIER
        clf = MLPClassifier(APs*WINDOW_SIZE, NUM_CLASSES).to(device)
        optimizer = torch.optim.Adam(clf.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()
        clf_epochs = 200
        for clf_epoch in range(clf_epochs):
            loss_epoch, accuracy_epoch, f1_epoch = train(loader_fake, clf, optimizer, criterion)
            # if clf_epoch % 50 == 0:
            #     print(
            #         f"Clf_Epoch [{clf_epoch}/{clf_epochs}]\t "
            #         f"Clf_Loss: {loss_epoch / len( loader_val)}\t "
            #         f"Clf_Accuracy: {accuracy_epoch * 100 / len( loader_val)}\t "
            #         f"Clf_F1: {f1_epoch * 100}\t")
            # wandb.log({"validation_train/loss_val_train/e"+str(epoch): loss_epoch/len(loader_val)})
            # wandb.log({"validation_train/accuracy_val_train/e"+str(epoch): accuracy_epoch*100/len(loader_val)})
            # wandb.log({"validation_train/F1_val_train/e"+str(epoch): f1_epoch*100})

        loss, accuracy, f1_mlp = test(loader_val, clf, criterion)
        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}]\t "
            f"MLP_Loss: {loss}\t "
            f"MLP_Accuracy: {accuracy }\t "
            f"MLP_F1: {f1_mlp}\t")
        # wandb.log({"validation_test/loss_val_test": loss})
        # wandb.log({"validation_test/accuracy_val_test": accuracy})
        # wandb.log({"validation_test/F1_val_test": f1_mlp})

        # RF CLASSIFIER
        # fake_val_tp = np.transpose(fake_val, (0, 2, 1))
        fake_val_feature, fake_y_val_feature = feature(fake_val.cpu().detach().numpy(), fake_y_val.cpu().detach().numpy(), datatype='rssi')
        clf = RandomForestClassifier(random_state=42)
        cv = StratifiedKFold(n_splits=3, shuffle=False)
        param_grid = {
            'min_samples_leaf': [1, 3, 5, 10],
            'n_estimators': [30, 50, 100, 200, 300, 500]}
        clf_grid = GridSearchCV(clf, param_grid=param_grid, cv=cv, refit=True, n_jobs=-1)
        clf_grid.fit(fake_val_feature, fake_y_val_feature)
        # print('Best parameters are: {}'.format(clf_grid.best_params_))

        clf_tune = RandomForestClassifier(min_samples_leaf=clf_grid.best_params_['min_samples_leaf'],
                                          n_estimators=clf_grid.best_params_['n_estimators'], )
        clf_tune.fit(fake_val_feature, fake_y_val_feature)
        y_pred = clf_tune.predict(X_val_feature)
        f1_rf = f1_score(y_val_feature, y_pred, average='weighted') * 100
        acc_rf = accuracy_score(y_val_feature, y_pred) * 100
        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}]\t "
            f"RF_Accuracy: {acc_rf}\t "
            f"RF_F1: {f1_rf}\t")

        # wandb.log({"validation_RF/accuracy": acc_rf})
        # wandb.log({"validation_RF/F1": f1_rf})

        if f1_mlp > keep_performance or f1_rf > keep_performance:
            torch.save(gen.state_dict(), save_directory + model_name + '_epoch' + str(epoch) + '.pt')
            if f1_mlp > f1_rf:
                keep_performance = f1_mlp
            if f1_rf > f1_mlp:
                keep_performance = f1_rf
            best_epoch = epoch

        if epoch != 0 and epoch % 20 == 0:
            torch.save(gen.state_dict(), save_directory + model_name + '_epoch' + str(epoch) + '.pt')

    torch.save(gen.state_dict(), save_directory + model_name + '_epoch' + str(NUM_EPOCHS) + '.pt')


