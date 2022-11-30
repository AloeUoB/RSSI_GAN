import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from utils import gradient_penalty
from model_wgan_gp import Discriminator, Generator, initialize_weights

from GAN.gan_utils import load_GAN_model, generate_fake_data,\
    MinMaxScaler, renormalization, windowing, get_mean_stat,\
    plot_line_rssi_gan, MLPClassifier,plot_line_rssi_house,\
    binary_convert, map_A, map_C, map_D, train, train_val, test,\
    num_room_house, house_map, load_house_data, select_data_potion

from train_wgan_task import select_data
from general_utils import uniq_count
import wandb
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from GAN.gan_evaluation import multiclass, rf_multiclass
from tqdm import tqdm

import copy
from train_wgan_loss_class import classification_loss, prep_data

def create_house_label(X, label):
    y = np.full(len(X), label)
    return X, y

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load dataset
    data_directory = os.path.join('..', 'aloe', 'localisation', 'data', ''.format(os.path.sep))
    save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))
    # load GAN training data
    reduce_ap = False
    house_all = ['B', 'C', 'D']
    X_house_all = np.array([]).reshape(-1, 20, 11)
    y_house_all = np.array([])
    house_label = 1
    for house_name in (house_all) :
        rssi_fp, y_fp, APs, NUM_CLASSES = \
            load_house_data(data_directory, house_name, datatype='fp', reduce_ap=False)

        X_house, y_house = create_house_label(rssi_fp, house_label)
        X_house_all = np.concatenate((X_house_all, X_house), axis=0)
        y_house_all = np.concatenate((y_house_all, y_house), axis=0)
        house_label +=1

    # X_train, X_val, y_train, y_val = train_test_split(X_house_all, y_house_all, test_size=0.40, random_state=42)
    # X_train, y_train = prep_data(X_train, y_train)
    X_train, y_train = prep_data(X_house_all, y_house_all) # need (N,11,20) for GAN training
    # X_val, y_val = prep_data(X_val, y_val)
    X_train2D = X_train.unsqueeze(1)

    # epochs = 2000
    # cf_loss = classification_loss(X_train, y_train, X_val, y_val, APs, 3, epochs, device, False,
    #                exp=1, runs=1, flatten=True, show_epoch=False, confusion_met=True)

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

    NUM_CLASSES = len(np.unique(y_train))

    dataset_train = torch.utils.data.TensorDataset(X_train2D.float(), y_train)
    loader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               drop_last=True, )

    # initialize gen and disc
    gen = Generator(Z_DIM, CHANNELS_RSSI, FEATURES_GEN, NUM_CLASSES, APs, WINDOW_SIZE, GEN_EMBEDDING).to(device)
    critic = Discriminator(CHANNELS_RSSI, FEATURES_CRITIC, NUM_CLASSES, APs, WINDOW_SIZE).to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    # initializate optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    model_name = "ConGAN_wgp_all_house"

    gen.train()
    critic.train()

    NUM_EPOCHS = 1200
    wandb.init(project=model_name, entity="rssi-augmentation")
    wandb.config = {
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "window_size": WINDOW_SIZE,
        "channels_RSSI": CHANNELS_RSSI,
        "gen_embed_size": GEN_EMBEDDING,
        "noise_dimention": Z_DIM,
        "feature_critic": FEATURES_CRITIC,
        "feature_gen": FEATURES_GEN,
        "critic_iterations": CRITIC_ITERATIONS,
        "Lambda_GP": LAMBDA_GP,
        "number_APs": APs,
        "number_rooms" : NUM_CLASSES,
    }
    house_name_map = {0:'B', 1:'C', 2:'D'}

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

            wandb.log({"Loss/disc_train_epoch": loss_critic.item()})
            wandb.log({"Loss/gen_train_epoch": loss_gen.item()})

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] \
                      Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
        )

        # plot_line_rssi_house(noise[0][0].view(APs, 20).cpu().detach().numpy(), labels.cpu().detach().numpy()[0] + 1,
        #                    house=house_name_map[labels.cpu().detach().numpy()[0]], house_map=house_map[house_name], ymin=-0.1, ymax=1.1, save=False,
        #                    reduce=False, save_dir='GAN/train_visual/', model_name='noise_fake_', transpose=True)

        plot_line_rssi_house(fake[0].view(APs, 20).cpu().detach().numpy(), labels.cpu().detach().numpy()[0] + 1,
                           house=house_name_map[labels.cpu().detach().numpy()[0]], house_map=house_map[house_name], ymin=-0.1, ymax=1.1, save=True,
                           reduce=False, save_dir='GAN/train_visual/ConGAN_wgp_all_house_/', model_name='all_house_fake_', plot_idx=epoch, transpose=True)

        # with torch.no_grad():
        #     x_fake = fake[0].view(APs, 20)
        #     x_real = real[0].view(APs, 20)
        #     cur_labels_numpy = labels.cpu().detach().numpy()
            # print(cur_labels_numpy[0])

            # plot_line_rssi_gan(x_fake.cpu().detach().numpy(), cur_labels_numpy[0] + 1, house=house_name,
            #                    house_map=house_map[house_name], ymin=-0.1, ymax=1.1, save=False, reduce=reduce_ap,
            #                    save_dir='GAN/train_visual/' + model_name + '/', model_name=str(epoch) + '_fake_',
            #                    transpose=True)
            # plot_line_rssi_gan(x_real.cpu().detach().numpy(), cur_labels_numpy[0] + 1, house=house_name,
            #                    house_map=house_map[house_name], ymin=-0.1, ymax=1.1, save=False, reduce=reduce_ap,
            #                    save_dir='GAN/train_visual/' + model_name + '/', model_name=str(epoch) + '_real_',
            #                    transpose=True)

        if epoch != 0 and epoch % 20 == 0:
            torch.save(gen.state_dict(), save_directory + model_name + 'gen_epoch' + str(epoch) + '.pt')
            torch.save(critic.state_dict(), save_directory + model_name + 'critic_epoch' + str(epoch) + '.pt')

    torch.save(gen.state_dict(), save_directory + model_name + 'gen_epoch' + str(NUM_EPOCHS) + '.pt')
    torch.save(critic.state_dict(), save_directory + model_name + 'critic_epoch' + str(NUM_EPOCHS) + '.pt')
