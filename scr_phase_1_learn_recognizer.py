import argparse
import os
import cv2
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from plot_learning_anonymization import *
import keras.backend as K
from keras.layers.merge import _Merge
import datetime
from functools import partial
import tensorflow as tf
from sklearn.metrics import roc_auc_score, mean_squared_error, roc_curve, accuracy_score
import pickle
import itertools
import matplotlib.gridspec as gridspec
from keras.initializers import RandomNormal
import random
from models_phase_1 import *
import tensorflow as tf
import time
from PIL import Image
from keras.layers import add
from collections import Counter
from skimage import exposure

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ############################################################################################
# #    ARGUMENT PARSING
# ############################################################################################

ap = argparse.ArgumentParser()

ap.add_argument('-i', '--input_folder', required=True, help='Input folder')
ap.add_argument('-o', '--output_folder', required=True, help='Output folder')
ap.add_argument('-sb', '--classes_batch', type=int, default=3, help='Tot. subjects in batch')
ap.add_argument('-is', '--images_class_batch', type=int, default=12, help='Images per subject in batch')
ap.add_argument('-igo', '--interval_write_output', type=int, default=20,
                help='Number of epochs to write generator examples')
ap.add_argument('-lrd', '--learning_rate', type=float, default=5e-5, help='Learning rate discriminators')
ap.add_argument('-it', '--index_trait', type=int, default=0, help='Index of the trait to infer')

ap.add_argument('-w', '--image_width', type=int, default=256, help='Image width')
ap.add_argument('-he', '--image_height', type=int, default=256, help='Image height')
ap.add_argument('-e', '--epochs', type=int, default=16, help='Number epochs')

args = ap.parse_args()
args.batch_size = args.classes_batch * args.images_class_batch

############################################################################################

plt.ion()
if not os.path.isdir(args.output_folder):
    os.mkdir(args.output_folder)

date_time_folder = os.path.join(args.output_folder, 'Recognizer_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
#date_time_folder = os.path.join(args.output_folder, 'Recognizer_2020_05_21_08_10_31')

to_train = False
if not os.path.isdir(date_time_folder):
    to_train = True
    os.mkdir(date_time_folder)

    args_var = vars(args)
    file_out = open(os.path.join(date_time_folder, 'configs.txt'), "a+")
    file_out.write('%s\n' % ap.prog)
    for k in args_var.keys():
        file_out.write('%s: %s\n' % (k, args_var[k]))
    file_out.close()

# ######################################

print('Building data tables...')


def get_data_tables(input_folder):
    folders = os.listdir(input_folder)

    dt_table_list = []
    for folder in folders:
        files = os.listdir(os.path.join(args.input_folder, folder))
        for file in files:
            file_parts = file.split('_')
            day = file_parts[0][:10]
            session = file_parts[0]
            frame = int(file_parts[1])
            id = int(file_parts[2+args.index_trait])
            if id < 0:
                continue
            dt_table_list.append([os.path.join(args.input_folder, folder, file), day, session, frame, id])

    ids = list(set(p[4] for p in dt_table_list))

    id_keep = []
    all_ids = [el[4] for el in dt_table_list]
    for i, id in enumerate(ids):
        id_keep.append([id, all_ids.count(id)])

    id_keep = list(filter(lambda id: id[1] >= args.images_class_batch, id_keep))
    id_keep = [el[0] for el in id_keep]

    dt_table_list = list(filter(lambda el: el[4] in id_keep, dt_table_list))
    days = list(set(p[1] for p in dt_table_list))
    sessions = list(set(p[2] for p in dt_table_list))

    dt_table = []
    for i, elt in enumerate(dt_table_list):
        if elt[4] not in id_keep:
            continue
        dt_table.append([days.index(elt[1]), sessions.index(elt[2]), elt[3], elt[4]])
    dt_table = np.asarray(dt_table).astype(int)

    dt_paths = [p[0] for p in dt_table_list]

    ids_table = []
    for id in id_keep:
        idx = np.where(dt_table[:, 3] == id)
        ids_table.append([id, idx[0]])

    return dt_paths, dt_table, ids_table


def get_input_batch_gen(dt_paths, dt_table, id_table):
    paths_batch = []
    ids_batch = []
    idx_ids = random.sample(range(len(id_table)), args.classes_batch)  # choose idx subjects
    for idx_id in idx_ids:
        idx_imgs = random.sample(range(len(id_table[idx_id][1])), args.images_class_batch)  # choose imgs/subject
        for i in idx_imgs:
            paths_batch.append(dt_paths[id_table[idx_id][1][i]])
            ids_batch.append(id_table[idx_id][0])

    imgs = np.zeros((len(paths_batch), args.image_height, args.image_width, 4)).astype('float')

    for i, path_img in enumerate(paths_batch):
        train_img = cv2.imread(path_img)
        train_img = cv2.resize(train_img, (args.image_width, args.image_height))

        imgs[i, :, :, 0:3] = train_img
        imgs[i, :, :, 3] = np.random.randint(0, 255, (args.image_height, args.image_width))
    imgs = imgs / 255.
    return imgs, ids_batch

def standardize_images(imgs):
    means = np.mean(imgs, axis=(1, 2), keepdims=True)
    stds = np.std(imgs, axis=(1, 2), keepdims=True)
    stds = np.maximum(stds, 1.0/(args.image_width*args.image_height*3))
    return (imgs - means) / stds

def get_input_random_gen(tot, dt_paths):
    paths_batch = []
    idxs = random.sample(range(len(dt_paths)), tot)
    for idx in idxs:
        paths_batch.append(dt_paths[idx])

    imgs = np.zeros((len(paths_batch), args.image_height, args.image_width, 4)).astype('float')

    for i, path_img in enumerate(paths_batch):
        train_img = cv2.imread(path_img)
        train_img = cv2.resize(train_img, (args.image_width, args.image_height))

        imgs[i, :, :, 0:3] = train_img
        imgs[i, :, :, 3] = np.random.randint(0, 255, (args.image_height, args.image_width))
    imgs = imgs / 255.
    return imgs


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""

    def _merge_function(self, inputs):
        alpha = K.random_uniform((args.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def get_weights_merge():
    aux_0 = np.random.uniform(low=0, high=1, size=(args.batch_size, 1, 1, 1))
    aux_1 = np.random.uniform(low=0, high=1, size=(args.batch_size, 1, 1, 1))
    alpha_0 = np.minimum(aux_0, aux_1)
    alpha_1 = 1.0 - np.maximum(aux_0, aux_1)
    alpha_2 = np.maximum(aux_0, aux_1) - np.minimum(aux_0, aux_1)
    return alpha_0, alpha_1, alpha_2


def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(args.gradient_penalty - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def cosine_similarity(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def cos_sim_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def get_slice(elt):
    return K.slice(elt, [0, 0, 0, 0], [-1, -1, -1, 3])

def identity_function(elt):
    return elt

def get_random(elt):
    return K.random_normal((args.batch_size, 1))

def diff_distances(vests):
    x, y = vests
    return x - y

def get_slice_output_shape(shape):
    return (shape[0], shape[1], shape[2], shape[3]-1)

def get_output_shape(shape):
    return (shape[0], shape[1], shape[2], shape[3])


def get_fool_shape(shape):
    return (shape[0], 1)

def get_pairwise_batch(imgs, ids_imgs):

    combs_aux = list(itertools.product(list(range(0, args.batch_size)), list(
        range(0, args.batch_size))))  # all combinations of instances/batch
    combs = list(filter(lambda c: c[0] != c[1], combs_aux))  # remove pairs of same instance

    combs_ids = [[ids_imgs[c[0]], ids_imgs[c[1]]] for c in combs]

    labels_disc_rec = [0 if (combs_ids[i][0] != combs_ids[i][1]) else 1 for i in range(len(combs))]  # "I"/"G": -1/1

    idx_m1 = [i for i, val in enumerate(labels_disc_rec) if val == 0]
    idx_1 = [i for i, val in enumerate(labels_disc_rec) if val == 1]

    tot_m1 = random.uniform(0.4, 0.6)
    tot_1 = 1 - tot_m1
    tot_m1 = min(len(idx_m1), round(tot_m1 * args.batch_size))
    tot_1 = min(len(idx_1), round(tot_1 * args.batch_size))

    shuffle(idx_m1)
    shuffle(idx_1)
    idx_m1 = idx_m1[:tot_m1]
    idx_1 = idx_1[:tot_1]

    labels_disc_rec = np.asarray([labels_disc_rec[elt] for elt in idx_m1+idx_1])
    combs_1 = [combs[elt] for elt in idx_m1+idx_1]

    in_disc_rec_1 = np.asarray([imgs[combs_1[i][0], :, :, :] for i in range(len(combs_1))])
    in_disc_rec_2 = np.asarray([imgs[combs_1[i][1], :, :, :] for i in range(len(combs_1))])
    return in_disc_rec_1, in_disc_rec_2, labels_disc_rec


def test(tot_tests, DR, dt_paths, dt_table, id_table, verbose):

    it = 0
    gt_disc_rec = []
    preds_disc_rec = []
    while it < tot_tests:
        [imgs, ids_imgs] = get_input_batch_gen(dt_paths, dt_table, id_table)
        imgs_1, imgs_2, gt = get_pairwise_batch(imgs, ids_imgs)

        preds = DR.predict([standardize_images(imgs_1[:, :, :, :3]), standardize_images(imgs_2[:, :, :, :3])])

        if np.isnan(preds).sum() > 0:
            lixo = 1

        gt_disc_rec.extend(gt)
        preds_disc_rec.extend(preds)

        it = it + args.batch_size

    auc_disc_recognition = roc_auc_score(gt_disc_rec, [p for p in preds_disc_rec])
    fpr, tpr, thresholds = roc_curve(gt_disc_rec, [p for p in preds_disc_rec])

    accuracy_scores = []
    for thresh in thresholds:
            accuracy_scores.append(
                accuracy_score(gt_disc_rec, [1 if m > thresh else 0 for m in preds_disc_rec]))

    #print('A/1-----------')
    sc_1 = np.asarray(preds_disc_rec)[np.asarray(gt_disc_rec) == 1]
    sc_0 = np.asarray(preds_disc_rec)[np.asarray(gt_disc_rec) == 0]
    #print(sc_1)
    #print('O/0-----------')
    #print(sc_0)
    if verbose:
        print('DR: Mu 1: %f; Mu 0: %f' % (np.mean(sc_1), np.mean(sc_0)))
        print('DR: ACC Expected = %f' % max(accuracy_scores))
        print('DR: FPR Expected = %f' % fpr[accuracy_scores.index(max(accuracy_scores))])
        print('DR: TPR Expected = %f' % tpr[accuracy_scores.index(max(accuracy_scores))])
        best_threshold_recognition = thresholds[accuracy_scores.index(max(accuracy_scores))]
        print('Threshold')
        print(best_threshold_recognition)
        fig_1 = plt.figure(1)
        plt.clf()
        plt.grid(which='major', axis='both')
        plt.hist(sc_0, 20, alpha=0.5, color='r', label='I', weights=np.zeros_like(sc_0) + 1. / sc_0.size)
        plt.hist(sc_1, 20, alpha=0.5, color='g', label='G', weights=np.zeros_like(sc_1) + 1. / sc_1.size)
        plt.axis('equal')
        plt.axis([0, 1, 0, 1])
        fig_1.show()
        plt.pause(0.01)
        plt.waitforbuttonpress()
        plt.savefig(os.path.join(date_time_folder, 'Histogram.png'))

    return auc_disc_recognition, max(accuracy_scores)

def train(DR_model, DR, dt_paths, dt_table, id_table):
    # Adversarial ground truths
    genuines = np.ones((args.batch_size, 1))
    impostors = np.zeros((args.batch_size, 1))

    discriminator_R_losses = []
    expected_results = []
    dr_l = []
    plt.ion()
    for epoch in range(args.epochs):

        [imgs, ids_imgs] = get_input_batch_gen(dt_paths, dt_table, id_table)

        imgs_1, imgs_2, labels = get_pairwise_batch(imgs, ids_imgs)

        dr_loss = DR_model.train_on_batch([standardize_images(imgs_1[:, :, :, :3]), standardize_images(imgs_2[:, :, :, :3])], labels)

        dr_l.append([dr_loss])


        # print("%d [DA loss: %f] [DO loss: %f] [G loss: %f]" % (
        #    epoch, np.mean([d[0] for d in da_l_tmp]), np.mean([d[0] for d in do_l_tmp]), g_loss[0]))

        if epoch % args.interval_write_output == 0:
            expected = test(200, DR, dt_paths, dt_table, id_table, False)

            expected_results.append(expected)
            discriminator_R_losses.append(np.mean(dr_l, axis=0))
            dr_l = []

            saved_model_flag = save_best_model(epoch, expected_results, DR, dt_paths, dt_table, id_table)

            if not saved_model_flag:
                print('\r %d' % epoch, end='')

            ep = range(1, epoch // args.interval_write_output + 2)
            fig_1 = plt.figure(1)
            plt.clf()
            gs = gridspec.GridSpec(1, 2, figure=fig_1)

            ax = fig_1.add_subplot(gs[0, 0])
            ax.plot(ep, [x[0] for x in discriminator_R_losses])
            ax.set_facecolor((0.9, 0.7, 0.9))
            ax.grid(True)
            ax.title.set_text('DR')

            ax = fig_1.add_subplot(gs[0, 1])
            ax.plot(ep[-10:], [x[0] for x in discriminator_R_losses[-10:]])
            ax.set_facecolor((0.9, 0.7, 0.9))
            ax.grid(True)
            ax.title.set_text('Last')

            fig_1.show()
            plt.pause(0.01)

            plt.savefig(os.path.join(date_time_folder, 'Learning.png'))


def save_best_model(ep, results, disc_rec, dt_paths, dt_table, id_table):
    criterium = [r[0] for r in results]
    accs = [r[1] for r in results]
    if criterium.index(max(criterium)) != len(criterium) - 1:
        return 0
    print('[%d] Saving best model. Expected AUC = %.4f, ACC = %.4f' % (ep, criterium[-1], accs[-1]))
    disc_rec.save(os.path.join(date_time_folder, 'Discriminator_recognition.h5'))
    return 1

# #####################################################################################################################
# MAIN()
# #####################################################################################################################

data_paths, data_table, ids_table = get_data_tables(args.input_folder)

# #################################################################
# create Gen + Discs. graphs

discriminator_R = create_discriminator_R_cross_entropy((args.image_height, args.image_width, 3))

# #################################################################
# create computational graphs for discriminator_R

input_1 = Input(shape=(args.image_height, args.image_width, 3))
input_2 = Input(shape=(args.image_height, args.image_width, 3))

out_Rec = discriminator_R([input_1, input_2])


disc_Rec_model = Model(inputs=[input_1, input_2], outputs=out_Rec)
disc_Rec_model.compile(loss=['binary_crossentropy'],
                     optimizer=RMSprop(lr=args.learning_rate))

if to_train:
    train(disc_Rec_model, discriminator_R, data_paths, data_table, ids_table)

discriminator_R = load_model(os.path.join(date_time_folder, 'Discriminator_recognition.h5'), compile=False)

test(1000, discriminator_R, data_paths, data_table, ids_table, True)


print('Done...')

