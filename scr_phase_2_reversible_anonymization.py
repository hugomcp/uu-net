from __future__ import print_function
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
import numpy.matlib
from models_phase_2 import *
import tensorflow as tf
import time
from PIL import Image
from keras.layers import add

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ############################################################################################
# #    ARGUMENT PARSING
# ############################################################################################

ap = argparse.ArgumentParser()

ap.add_argument('-i', '--input_folder', required=True, help='Input folder')
ap.add_argument('-o', '--output_folder', required=True, help='Output folder')

ap.add_argument('-sb', '--subjects_batch', type=int, default=3, help='Tot. subjects in batch')
ap.add_argument('-is', '--images_subject_batch', type=int, default=12, help='Images per subject in batch')
ap.add_argument('-td', '--tot_discriminator', type=int, default=5, help='Tot. iterations discriminator per generator')

ap.add_argument('-wg', '--weight_generator', type=float, default=1.0, help='Weight Generator term')
ap.add_argument('-wdfo', '--weight_discriminator_faceness_O', type=float, default=1.0,
                help='Weight Discriminator Faceness (O) term')
ap.add_argument('-wdfa', '--weight_discriminator_faceness_A', type=float, default=1.0,
                help='Weight Discriminator Faceness (A) term')

ap.add_argument('-igo', '--interval_write_generator_output', type=int, default=20,
                help='Number of epochs to write generator examples')

ap.add_argument('-lrd', '--learning_rate_discriminator', type=float, default=5e-5, help='Learning rate discriminators')
ap.add_argument('-lrg', '--learning_rate_generator', type=float, default=5e-5, help='Learning rate generator')

ap.add_argument('-w', '--image_width', type=int, default=256, help='Image width')
ap.add_argument('-he', '--image_height', type=int, default=256, help='Image height')
ap.add_argument('-e', '--epochs', type=int, default=16, help='Number epochs')

ap.add_argument('-dri', '--discriminator_R_ID_path', required=True, help='Discriminator Recogn. path')
ap.add_argument('-drg', '--discriminator_R_gender_path', required=True, help='Discriminator Recogn. path')
ap.add_argument('-dre', '--discriminator_R_ethnicity_path', required=True, help='Discriminator Recogn. path')
ap.add_argument('-drhc', '--discriminator_R_haircolor_path', required=True, help='Discriminator Recogn. path')
ap.add_argument('-drhs', '--discriminator_R_hairstyle_path', required=True, help='Discriminator Recogn. path')
ap.add_argument('-drb', '--discriminator_R_beard_path', required=True, help='Discriminator Recogn. path')
ap.add_argument('-drm', '--discriminator_R_moustache_path', required=True, help='Discriminator Recogn. path')


ap.add_argument('-w1', '--weight_mse', type=float, default=100, help='Weight of the MSE term')
ap.add_argument('-w2', '--weight_adv', type=float, default=1, help='Weight of the ADV term')
ap.add_argument('-w3', '--weight_ano', type=float, default=1, help='Weight of the ANO term')
ap.add_argument('-w4', '--weight_con', type=float, default=1, help='Weight of the CON term')
ap.add_argument('-w5', '--weight_div', type=float, default=1, help='Weight of the DIV term')
ap.add_argument('-w6', '--weight_dis', type=float, default=1, help='Weight of the DIS term')

args = ap.parse_args()
args.batch_size = args.subjects_batch * args.images_subject_batch

############################################################################################

plt.ion()
if not os.path.isdir(args.output_folder):
    os.mkdir(args.output_folder)

date_time_folder = os.path.join(args.output_folder, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
#date_time_folder = os.path.join(args.output_folder, '2020_06_19_10_45_55')

to_train = False
if not os.path.isdir(date_time_folder):
    to_train = True
    os.mkdir(date_time_folder)
    os.mkdir(os.path.join(date_time_folder, 'test'))
    os.mkdir(os.path.join(date_time_folder, 'learning'))
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
            id = int(file_parts[2])
            if id < 0:
                continue
            dt_table_list.append([os.path.join(args.input_folder, folder, file), day, session, frame, id])

    ids_sessions = [list(x) for x in set(tuple([p[1], p[2], p[4]]) for p in dt_table_list)]

    ids_sessions_keep = []
    all_ids_sessions = [[el[1], el[2], el[4]] for el in dt_table_list]
    for i, id in enumerate(ids_sessions):
        ids_sessions_keep.append([id, all_ids_sessions.count(id)])

    ids_sessions_keep = list(filter(lambda id: id[1] >= args.images_subject_batch, ids_sessions_keep))
    ids_sessions_keep = [el[0] for el in ids_sessions_keep]

    dt_table_list = list(filter(lambda el: [el[1], el[2], el[4]] in ids_sessions_keep, dt_table_list))
    days = list(set(p[1] for p in dt_table_list))
    sessions = list(set(p[2] for p in dt_table_list))

    dt_table = []
    for i, elt in enumerate(dt_table_list):
        if [elt[1], elt[2], elt[4]] not in ids_sessions_keep:
            continue
        dt_table.append([days.index(elt[1]), sessions.index(elt[2]), elt[3], elt[4]])
    dt_table = np.asarray(dt_table).astype(int)

    ids = np.unique(dt_table[:, 3])

    dt_paths = [p[0] for p in dt_table_list]

    ids_sessions_table = []
    for id in ids:
        idx = np.where(dt_table[:, 3] == id)  # all lines of this ID

        sessions = np.unique(dt_table[idx[0], 0:2], axis=0)  # all sessions of this ID

        idx_all = []
        for s in range(sessions.shape[0]):
            idx = np.where(
                (dt_table[:, 3] == id) & (dt_table[:, 0] == sessions[s, 0]) & (dt_table[:, 1] == sessions[s, 1]))
            idx_all.append(list(idx[0]))

        ids_sessions_table.append([id, idx_all])

    final_ids_sessions_table = []
    for elt in ids_sessions_table:
        if len(elt[1]) >= 2:
            final_ids_sessions_table.append(elt)

    return dt_paths, final_ids_sessions_table  # dt_paths=[path1, ...]  #ids_table=[id [[idx_session_1], [idx_session_2], ...], ...]


def standardize_image(img):
    mean = np.mean(img, axis=(1, 2), keepdims=True)
    std = np.std(img, axis=(1, 2), keepdims=True)
    return (img - mean) / std


def get_input_batch_gen(dt_paths, id_table):
    paths_batch_1 = []
    paths_batch_2 = []
    ids_batch = []
    idx_ids = random.sample(range(len(id_table)), args.subjects_batch)  # choose idx subjects
    for idx_id in idx_ids:
        idx_sessions = random.sample(range(len(id_table[idx_id][1])), 2)

        idx_imgs_1 = random.sample(range(len(id_table[idx_id][1][idx_sessions[0]])),
                                   args.images_subject_batch)  # choose imgs_1/subject
        idx_imgs_2 = random.sample(range(len(id_table[idx_id][1][idx_sessions[1]])),
                                   args.images_subject_batch)  # choose imgs_2/subject
        for i in idx_imgs_1:
            paths_batch_1.append(dt_paths[id_table[idx_id][1][idx_sessions[0]][i]])
            ids_batch.append(id_table[idx_id][0])

        for i in idx_imgs_2:
            paths_batch_2.append(dt_paths[id_table[idx_id][1][idx_sessions[1]][i]])

    imgs_1 = np.zeros((len(paths_batch_1), args.image_height, args.image_width, 4)).astype('float')
    imgs_2 = np.zeros((len(paths_batch_2), args.image_height, args.image_width, 4)).astype('float')

    for i, path_img in enumerate(paths_batch_1):
        train_img = cv2.imread(path_img)
        train_img = cv2.resize(train_img, (args.image_width, args.image_height))

        imgs_1[i, :, :, 0:3] = train_img
        imgs_1[i, :, :, 3] = np.random.randint(0, 255, (args.image_height, args.image_width))
    imgs_1 = imgs_1 / 255.

    for i, path_img in enumerate(paths_batch_2):
        train_img = cv2.imread(path_img)
        train_img = cv2.resize(train_img, (args.image_width, args.image_height))

        imgs_2[i, :, :, 0:3] = train_img
        imgs_2[i, :, :, 3] = np.random.randint(0, 255, (args.image_height, args.image_width))
    imgs_2 = imgs_2 / 255.
    return imgs_1, imgs_2, ids_batch


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


def get_input_paths(dt_paths):
    imgs = np.zeros((len(dt_paths), args.image_height, args.image_width, 4)).astype('float')

    for i, path_img in enumerate(dt_paths):
        train_img = cv2.imread(path_img)
        train_img = cv2.resize(train_img, (args.image_width, args.image_height))

        imgs[i, :, :, 0:3] = train_img
        imgs[i, :, :, 3] = np.random.randint(0, 255, (args.image_height, args.image_width))
    imgs = imgs / 255.
    return imgs


def sample_images(epoch, gen, dt_paths):
    r, c = 4, 4

    in_imgs = get_input_random_gen(r * c, dt_paths)
    [A_imgs, O_imgs] = gen.predict(in_imgs)

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(
                np.concatenate((cv2.cvtColor(np.uint8(in_imgs[cnt, :, :, :3] * 255), cv2.COLOR_BGR2RGB),
                                cv2.cvtColor(np.uint8(A_imgs[cnt, :, :, :]*255), cv2.COLOR_BGR2RGB),
                                cv2.cvtColor(np.uint8(O_imgs[cnt, :, :, :]*255), cv2.COLOR_BGR2RGB)), axis=1))
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(os.path.join(date_time_folder, 'learning', '%06d.png' % epoch))
    plt.close()


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
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


def wasserstein_loss(y_true, y_pred):
    result = K.mean(y_true * y_pred)
    return result


def chi2_loss(y_true, y_pred):
    h_true = tf.histogram_fixed_width(y_true, value_range=(0., 1.), nbins=100)
    h_pred = tf.histogram_fixed_width(y_pred, value_range=(0., 1.), nbins=100)
    h_true = tf.cast(h_true, dtype=tf.dtypes.float32) / tf.reduce_sum(y_true)
    h_pred = tf.cast(h_pred, dtype=tf.dtypes.float32) / tf.reduce_sum(y_pred)
    return K.sqrt(K.mean(K.square(h_true - h_pred)))


def same_intensity_loss(y_true, y_pred):
    sum_1 = K.mean(y_true, axis=(1, 2, 3))
    sum_2 = K.mean(y_pred, axis=(1, 2, 3))
    diff = K.abs(sum_1 - sum_2)
    return K.mean(diff)


def cosine_similarity(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)


def cos_sim_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def get_slice(elt):
    return K.slice(elt, [0, 0, 0, 0], [-1, -1, -1, 3])


def roll(elt):
    return tf.roll(elt, shift=-round(args.images_subject_batch / 2), axis=0)
    # return K.slice(elt, [2, 0, 0, 0], [-1, -1, -1, -1])


def standardize_imgs(elt):
    return tf.image.per_image_standardization(elt)


def sum_imgs(elts):
    return tf.add_n([elts[0], elts[1], elts[2], elts[3], elts[4], elts[5], elts[6]])


def diff_distances(vests):
    x, y = vests
    return x - y


def get_slice_output_shape(shape):
    return shape[0], shape[1], shape[2], shape[3] - 1


def get_slice_intra_session_shape(shape):
    return shape[0] - 2, shape[1], shape[2], shape[3]


def get_output_shape(shape):
    return shape[0], shape[1], shape[2], shape[3]


def get_output_shapes(shapes):
    shape, _, _, _, _, _, _ = shapes
    return shape[0], shape[1], shape[2], shape[3]


def label_noise(lab):
    noise = np.random.normal(0.0, 0.05, lab.shape)
    return lab + noise


def input_noise(inp):
    noise = np.random.normal(0.0, 0.05, inp.shape)
    return inp + noise


def get_pairwise_batch(imgs, ids_imgs):
    combs_aux = list(itertools.product(list(range(0, args.batch_size)), list(
        range(0, args.batch_size))))  # all combinations of instances/batch
    combs = list(filter(lambda c: c[0] != c[1], combs_aux))  # remove pairs of same instance

    combs_ids = [[ids_imgs[c[0]], ids_imgs[c[1]]] for c in combs]

    labels_disc_rec = [-1 if (combs_ids[i][0] != combs_ids[i][1]) else 1 for i in range(len(combs))]  # "I"/"G": -1/1

    idx_m1 = [i for i, val in enumerate(labels_disc_rec) if val == -1]
    idx_1 = [i for i, val in enumerate(labels_disc_rec) if val == 1]

    tot_m1 = random.uniform(0.4, 0.6)
    tot_1 = 1 - tot_m1
    tot_m1 = min(len(idx_m1), round(tot_m1 * args.batch_size))
    tot_1 = min(len(idx_1), round(tot_1 * args.batch_size))

    shuffle(idx_m1)
    shuffle(idx_1)
    idx_m1 = idx_m1[:tot_m1]
    idx_1 = idx_1[:tot_1]

    labels_disc_rec = np.asarray([labels_disc_rec[elt] for elt in idx_m1 + idx_1])
    combs_1 = [combs[elt] for elt in idx_m1 + idx_1]

    in_disc_rec_1 = np.asarray([imgs[combs_1[i][0], :, :, :] for i in range(len(combs_1))])
    in_disc_rec_2 = np.asarray([imgs[combs_1[i][1], :, :, :] for i in range(len(combs_1))])
    return in_disc_rec_1, in_disc_rec_2, labels_disc_rec


def save_best_model(ep, results, disc_face, gen):
    criterium = [r[1] for r in results]  # minimum MSE of O image
    if criterium.index(min(criterium)) != len(criterium) - 1:
        return 0
    print('Saving best model...')
    # disc_face.save(os.path.join(date_time_folder, 'Discriminator_faceness.h5'))
    gen.save(os.path.join(date_time_folder, 'Generator.h5'))
    # gen.save_weights(os.path.join(date_time_folder, 'Generator.w'))
    return 1


def get_random_idx_IDs(id_table, tot_IDs, tot_sessions, tot_imgs):
    ids = []
    while len(ids) < tot_IDs:
        while True:
            id = random.randint(0, len(id_table) - 1)
            if id in ids or len(id_table[id][1]) < tot_sessions:
                continue
            break
        ids.append(id)
    return ids


def test(tot_IDs, tot_sessions, tot_imgs, G, dt_paths, id_table):
    idx_ids = get_random_idx_IDs(id_table, tot_IDs, tot_sessions, tot_imgs)

    for idx in idx_ids:

        idx_sessions = random.sample(range(len(id_table[idx][1])), tot_sessions)

        for s in idx_sessions:
            if len(id_table[idx][1][s]) < tot_imgs:
                continue
            idx_imgs = random.sample(range(len(id_table[idx][1][s])), tot_imgs)

            pos = [id_table[idx][1][s][i] for i in idx_imgs]
            paths = [dt_paths[p] for p in pos]

            in_imgs = get_input_paths(paths)

            [A_imgs, O_imgs] = G.predict(in_imgs)
            for j in range(tot_imgs):
                # cv2.imwrite(os.path.join(date_time_folder, 'test',
                #                          'x_%03d_%03d_%06d.png' % (id_table[idx][0], s, id_table[idx][1][s][idx_imgs[j]])), np.uint8(in_imgs[j, :, :, :3] * 255))
                # cv2.imwrite(os.path.join(date_time_folder, 'test',
                #                          'a_%03d_%03d_%06d.png' % (
                #                          id_table[idx][0], s, id_table[idx][1][s][idx_imgs[j]])),
                #             np.uint8(A_imgs[j, :, :, :3] * 255))
                #
                # cv2.imwrite(os.path.join(date_time_folder, 'test',
                #                          'r_%03d_%03d_%06d.png' % (
                #                              id_table[idx][0], s, id_table[idx][1][s][idx_imgs[j]])),
                #             np.uint8(O_imgs[j, :, :, :3] * 255))

                fig, axs = plt.subplots(1, 3)
                axs[0].imshow(cv2.cvtColor(np.uint8(in_imgs[j, :, :, :3] * 255), cv2.COLOR_BGR2RGB))
                axs[0].axis('off')
                axs[1].imshow(cv2.cvtColor(np.uint8(A_imgs[j, :, :, :3] * 255), cv2.COLOR_BGR2RGB))
                axs[1].axis('off')
                axs[2].imshow(cv2.cvtColor(np.uint8(O_imgs[j, :, :, :3] * 255), cv2.COLOR_BGR2RGB))
                axs[2].axis('off')
                fig.savefig(os.path.join(date_time_folder, 'test', '%03d_%03d_%06d_' % (
                    id_table[idx][0], s, id_table[idx][1][s][idx_imgs[j]]) + paths[j].split('/')[-1]))
                plt.close()


def train(G_model, DF_model, G, DF, dt_paths, id_table):
    # Adversarial ground truths
    valid = -np.ones((args.batch_size, 1))
    valid_patch = np.ones((args.batch_size, 8, 8, 1))

    fake = np.ones((args.batch_size, 1))
    fake_patch = np.zeros((args.batch_size, 8, 8, 1))

    impostors = fake
    genuines = valid
    dummy = np.zeros((args.batch_size, 1))

    intra_session_labels = np.concatenate((np.tile(np.concatenate(
        (-np.ones((round(args.images_subject_batch / 2), 1)), np.ones((round(args.images_subject_batch / 2), 1)))),
        (args.subjects_batch - 1, 1)), np.zeros((args.images_subject_batch, 1))), axis=0)

    generator_losses = []
    discriminator_Faceness_losses = []
    g_l = []
    df_l = []
    plt.ion()
    saved_model_epoch = -1
    for epoch in range(args.epochs):

        df_l_tmp = []
        for _ in range(args.tot_discriminator):
            [imgs_1, _, _] = get_input_batch_gen(dt_paths, id_table)
            valid_t = valid_patch  # label_noise(valid_patch)
            fake_t = fake_patch  # label_noise(fake_patch)

            # imgs = label_noise(imgs)

            df_loss = DF_model.train_on_batch(imgs_1, [valid_t, fake_t, fake_t, dummy, dummy])
            df_l_tmp.append(df_loss)

        [imgs_1, imgs_2, _] = get_input_batch_gen(dt_paths, id_table)

        # print('GENERATOR---------------------')
        g_loss = G_model.train_on_batch([imgs_1, imgs_2], [imgs_1[:, :, :, :3],
                                                           valid_patch, valid_patch,
                                                           impostors, genuines, impostors, intra_session_labels,
                                                           imgs_1[:, :, :, :3], imgs_1[:, :, :, :3]])
        df_l.extend(df_l_tmp)
        g_l.append(g_loss)

        print('\r Epoch %d' % epoch, end='')

        # print("%d [DA loss: %f] [DO loss: %f] [G loss: %f]" % (
        #    epoch, np.mean([d[0] for d in da_l_tmp]), np.mean([d[0] for d in do_l_tmp]), g_loss[0]))

        if epoch % args.interval_write_generator_output == 0:

            generator_losses.append(np.mean(g_l, axis=0))
            discriminator_Faceness_losses.append(np.mean(df_l, axis=0))
            g_l = []
            df_l = []
            sample_images(epoch, G, dt_paths)

            saved_model_flag = save_best_model(epoch, generator_losses, DF, G)
            if saved_model_flag:
                saved_model_epoch = epoch // args.interval_write_generator_output

            ep = range(1, epoch // args.interval_write_generator_output + 2)
            fig_1 = plt.figure(1)
            plt.clf()
            gs = gridspec.GridSpec(4, 8, figure=fig_1)
            ax = fig_1.add_subplot(gs[0, :2])
            ax.plot(ep, [x[0] for x in generator_losses])
            ax.plot(ep[saved_model_epoch], generator_losses[saved_model_epoch][0], 'or')
            ax.set_facecolor((0.8, 0.8, 0.8))
            ax.grid(True)
            ax.title.set_text('G')

            ax = fig_1.add_subplot(gs[0, 2])
            ax.plot(ep, [x[1] for x in generator_losses])
            ax.set_facecolor((0.8, 0.8, 0.8))
            ax.grid(True)
            ax.title.set_text('G(MSE)')

            ax = fig_1.add_subplot(gs[0, 3])
            ax.plot(ep[-10:], [x[1] for x in generator_losses[-10:]], color='green')
            ax.set_facecolor((0.8, 0.8, 0.8))
            ax.grid(True)

            ax = fig_1.add_subplot(gs[0, 4])
            ax.plot(ep, [x[2] for x in generator_losses])
            ax.set_facecolor((0.8, 0.8, 0.8))
            ax.grid(True)
            ax.title.set_text('G(DF_A)')

            ax = fig_1.add_subplot(gs[0, 5])
            ax.plot(ep[-10:], [x[2] for x in generator_losses[-10:]], color='green')
            ax.set_facecolor((0.8, 0.8, 0.8))
            ax.grid(True)

            ax = fig_1.add_subplot(gs[0, 6])
            ax.plot(ep, [x[3] for x in generator_losses])
            ax.set_facecolor((0.8, 0.8, 0.8))
            ax.grid(True)
            ax.title.set_text('G(DF_O)')

            ax = fig_1.add_subplot(gs[0, 7])
            ax.plot(ep[-10:], [x[3] for x in generator_losses[-10:]], color='green')
            ax.set_facecolor((0.8, 0.8, 0.8))
            ax.grid(True)

            ax = fig_1.add_subplot(gs[1, 0])
            ax.plot(ep, [x[4] for x in generator_losses])
            ax.set_facecolor((0.8, 0.8, 0.8))
            ax.grid(True)
            ax.title.set_text('G(DR_A)')

            ax = fig_1.add_subplot(gs[1, 1])
            ax.plot(ep[-10:], [x[4] for x in generator_losses[-10:]], color='green')
            ax.set_facecolor((0.8, 0.8, 0.8))
            ax.grid(True)

            ax = fig_1.add_subplot(gs[1, 2])
            ax.plot(ep, [x[5] for x in generator_losses])
            ax.set_facecolor((0.8, 0.8, 0.8))
            ax.grid(True)
            ax.title.set_text('G(DR_O)')

            ax = fig_1.add_subplot(gs[1, 3])
            ax.plot(ep[-10:], [x[5] for x in generator_losses[-10:]], color='green')
            ax.set_facecolor((0.8, 0.8, 0.8))
            ax.grid(True)

            ax = fig_1.add_subplot(gs[1, 4])
            ax.plot(ep, [x[6] for x in generator_losses])
            ax.set_facecolor((0.8, 0.8, 0.8))
            ax.grid(True)
            ax.title.set_text('G(DR_intra)')

            ax = fig_1.add_subplot(gs[1, 5])
            ax.plot(ep[-10:], [x[6] for x in generator_losses[-10:]], color='green')
            ax.set_facecolor((0.8, 0.8, 0.8))
            ax.grid(True)

            ax = fig_1.add_subplot(gs[1, 6])
            ax.plot(ep, [x[7] for x in generator_losses])
            ax.set_facecolor((0.8, 0.8, 0.8))
            ax.grid(True)
            ax.title.set_text('G(DR_inter)')

            ax = fig_1.add_subplot(gs[1, 7])
            ax.plot(ep[-10:], [x[7] for x in generator_losses[-10:]], color='green')
            ax.set_facecolor((0.8, 0.8, 0.8))
            ax.grid(True)

            ax = fig_1.add_subplot(gs[2, 0])
            ax.plot(ep, [x[8] for x in generator_losses])
            ax.set_facecolor((0.8, 0.8, 0.8))
            ax.grid(True)
            ax.title.set_text('G(Chi_A)')

            ax = fig_1.add_subplot(gs[2, 1])
            ax.plot(ep[-10:], [x[8] for x in generator_losses[-10:]], color='green')
            ax.set_facecolor((0.8, 0.8, 0.8))
            ax.grid(True)

            ax = fig_1.add_subplot(gs[2, 2])
            ax.plot(ep, [x[9] for x in generator_losses])
            ax.set_facecolor((0.8, 0.8, 0.8))
            ax.grid(True)
            ax.title.set_text('G(Chi_O)')

            ax = fig_1.add_subplot(gs[2, 3])
            ax.plot(ep[-10:], [x[9] for x in generator_losses[-10:]], color='green')
            ax.set_facecolor((0.8, 0.8, 0.8))
            ax.grid(True)

            ax = fig_1.add_subplot(gs[2, 4])
            ax.plot(ep, [x[0] for x in discriminator_Faceness_losses])
            ax.set_facecolor((0.9, 0.9, 0.7))
            ax.grid(True)
            ax.title.set_text('DA')

            ax = fig_1.add_subplot(gs[2, 5])
            ax.plot(ep[-10:], [x[0] for x in discriminator_Faceness_losses[-10:]], color='green')
            ax.set_facecolor((0.9, 0.9, 0.7))
            ax.grid(True)

            ax = fig_1.add_subplot(gs[2, 6])
            ax.plot(ep, [x[1] for x in discriminator_Faceness_losses])
            ax.set_facecolor((0.9, 0.9, 0.7))
            ax.grid(True)
            ax.title.set_text('Real')

            ax = fig_1.add_subplot(gs[2, 7])
            ax.plot(ep[-10:], [x[1] for x in discriminator_Faceness_losses[-10:]], color='green')
            ax.set_facecolor((0.9, 0.9, 0.7))
            ax.grid(True)

            ax = fig_1.add_subplot(gs[3, 0])
            ax.plot(ep, [x[2] for x in discriminator_Faceness_losses])
            ax.set_facecolor((0.9, 0.9, 0.7))
            ax.grid(True)
            ax.title.set_text('A')

            ax = fig_1.add_subplot(gs[3, 1])
            ax.plot(ep[-10:], [x[2] for x in discriminator_Faceness_losses[-10:]], color='green')
            ax.set_facecolor((0.9, 0.9, 0.7))
            ax.grid(True)

            ax = fig_1.add_subplot(gs[3, 2])
            ax.plot(ep, [x[3] for x in discriminator_Faceness_losses])
            ax.set_facecolor((0.9, 0.9, 0.7))
            ax.grid(True)
            ax.title.set_text('O')

            ax = fig_1.add_subplot(gs[3, 3])
            ax.plot(ep[-10:], [x[3] for x in discriminator_Faceness_losses[-10:]], color='green')
            ax.set_facecolor((0.9, 0.9, 0.7))
            ax.grid(True)

            ax = fig_1.add_subplot(gs[3, 4])
            ax.plot(ep, [x[4] for x in discriminator_Faceness_losses])
            ax.set_facecolor((0.9, 0.9, 0.7))
            ax.grid(True)
            ax.title.set_text('Grad')

            ax = fig_1.add_subplot(gs[3, 5])
            ax.plot(ep[-10:], [x[4] for x in discriminator_Faceness_losses[-10:]], color='green')
            ax.set_facecolor((0.9, 0.9, 0.7))
            ax.grid(True)

            ax = fig_1.add_subplot(gs[3, 6])
            ax.plot(ep, [x[5] for x in discriminator_Faceness_losses])
            ax.set_facecolor((0.9, 0.9, 0.7))
            ax.grid(True)
            ax.title.set_text('Grad')

            ax = fig_1.add_subplot(gs[3, 7])
            ax.plot(ep[-10:], [x[5] for x in discriminator_Faceness_losses[-10:]], color='green')
            ax.set_facecolor((0.9, 0.9, 0.7))
            ax.grid(True)

            fig_1.show()
            plt.pause(0.01)

            plt.savefig(os.path.join(date_time_folder, 'Learning.png'))


# #####################################################################################################################
# MAIN()
# #####################################################################################################################


data_paths, ids_table = get_data_tables(args.input_folder)

# #################################################################
# create Gen + Discs. graphs

discriminator_Faceness = create_discriminator_patch_GAN((args.image_height, args.image_width, 3))

discriminator_R_ID = load_model(os.path.join(args.discriminator_R_ID_path, 'Discriminator_recognition.h5'))
discriminator_R_ID.name = 'model_R_ID'

discriminator_R_gender = load_model(os.path.join(args.discriminator_R_gender_path, 'Discriminator_recognition.h5'))
discriminator_R_gender.name = 'model_R_gender'

discriminator_R_ethnicity = load_model(
    os.path.join(args.discriminator_R_ethnicity_path, 'Discriminator_recognition.h5'))
discriminator_R_ethnicity.name = 'model_R_ethnicity'

discriminator_R_haircolor = load_model(
    os.path.join(args.discriminator_R_haircolor_path, 'Discriminator_recognition.h5'))
discriminator_R_haircolor.name = 'model_R_haircolor'

discriminator_R_hairstyle = load_model(
    os.path.join(args.discriminator_R_hairstyle_path, 'Discriminator_recognition.h5'))
discriminator_R_hairstyle.name = 'model_R_hairstyle'

discriminator_R_beard = load_model(os.path.join(args.discriminator_R_beard_path, 'Discriminator_recognition.h5'))
discriminator_R_beard.name = 'model_R_beard'

discriminator_R_moustache = load_model(
    os.path.join(args.discriminator_R_moustache_path, 'Discriminator_recognition.h5'))
discriminator_R_moustache.name = 'model_R_moustache'

generator = create_generator_unet((args.image_height, args.image_width, 4))

# #################################################################
# create computational graphs for discriminator_Faceness

generator.trainable = False

gen_input = Input(shape=(args.image_height, args.image_width, 4))
real_img = Lambda(get_slice, output_shape=get_slice_output_shape)(gen_input)

[img_A, img_O] = generator(gen_input)

scores_A = discriminator_Faceness(img_A)
scores_O = discriminator_Faceness(img_O)
scores_real = discriminator_Faceness(real_img)

interpolated_img_A = RandomWeightedAverage()([real_img, img_A])
validity_interpolated_A = discriminator_Faceness(interpolated_img_A)

interpolated_img_O = RandomWeightedAverage()([real_img, img_O])
validity_interpolated_O = discriminator_Faceness(interpolated_img_O)

partial_gp_loss_A = partial(gradient_penalty_loss, averaged_samples=interpolated_img_A)
partial_gp_loss_A.__name__ = 'gradient_penalty'

partial_gp_loss_O = partial(gradient_penalty_loss, averaged_samples=interpolated_img_O)
partial_gp_loss_O.__name__ = 'gradient_penalty'

disc_Faceness_model = Model(inputs=gen_input,
                            outputs=[scores_real, scores_A, scores_O, validity_interpolated_A, validity_interpolated_O])
# disc_Faceness_model.compile(loss=[wasserstein_loss, wasserstein_loss, wasserstein_loss, partial_gp_loss_A, partial_gp_loss_O],
#                      optimizer=RMSprop(lr=args.learning_rate_discriminator), loss_weights=[2, 1, 1, 10, 10])
disc_Faceness_model.compile(
    loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', partial_gp_loss_A, partial_gp_loss_O],
    optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[2, 1, 1, 10, 10])

# #################################################################
# create computational graph for Discriminator_R
discriminator_R_ID.trainable = False
discriminator_R_gender.trainable = False
discriminator_R_ethnicity.trainable = False
discriminator_R_haircolor.trainable = False
discriminator_R_hairstyle.trainable = False
discriminator_R_beard.trainable = False
discriminator_R_moustache.trainable = False

input_1 = Input(shape=(args.image_height, args.image_width, 3))
input_2 = Input(shape=(args.image_height, args.image_width, 3))

rec_ID = discriminator_R_ID([input_1, input_2])
rec_gender = discriminator_R_gender([input_1, input_2])
rec_ethnicity = discriminator_R_ethnicity([input_1, input_2])
rec_haircolor = discriminator_R_haircolor([input_1, input_2])
rec_hairstyle = discriminator_R_hairstyle([input_1, input_2])
rec_beard = discriminator_R_beard([input_1, input_2])
rec_moustache = discriminator_R_moustache([input_1, input_2])

scores = Lambda(sum_imgs)([rec_ID, rec_gender, rec_ethnicity, rec_haircolor,
                           rec_hairstyle, rec_beard, rec_moustache])

discriminator_R = Model(inputs=[input_1, input_2], outputs=[scores])

# #################################################################
# create computational graphs for generator

discriminator_Faceness.trainable = False
discriminator_R.trainable = False
generator.trainable = True

gen_input_1 = Input(shape=(args.image_height, args.image_width, 4))
gen_input_2 = Input(shape=(args.image_height, args.image_width, 4))
img_1 = Lambda(get_slice, output_shape=get_slice_output_shape)(gen_input_1)
img_2 = Lambda(get_slice, output_shape=get_slice_output_shape)(gen_input_2)

[img_A_1, img_O_1] = generator(gen_input_1)
[img_A_2, img_O_2] = generator(gen_input_2)

valid_A_1 = discriminator_Faceness(img_A_1)
valid_O_1 = discriminator_Faceness(img_O_1)

img_std_1 = Lambda(standardize_imgs, output_shape=get_output_shape)(img_1)
img_A_std_1 = Lambda(standardize_imgs, output_shape=get_output_shape)(img_A_1)
img_O_std_1 = Lambda(standardize_imgs, output_shape=get_output_shape)(img_O_1)
img_A_std_2 = Lambda(standardize_imgs, output_shape=get_output_shape)(img_A_2)

rec_A_1 = discriminator_R([img_std_1, img_A_std_1])
rec_O_1 = discriminator_R([img_std_1, img_O_std_1])

img_1_intra_session_1 = Lambda(roll)(img_A_std_1)
img_1_intra_session_2 = img_A_std_2

rec_A_inter_session = discriminator_R([img_A_std_1, img_A_std_2])
rec_A_intra_session = discriminator_R([img_1_intra_session_1, img_1_intra_session_2])

generator_model = Model([gen_input_1, gen_input_2], [img_O_1, valid_A_1, valid_O_1,
                                                     rec_A_1, rec_O_1, rec_A_inter_session, rec_A_intra_session,
                                                     img_A_1, img_O_1])
generator_model.compile(loss=['mse', 'binary_crossentropy', 'binary_crossentropy',
                              wasserstein_loss, wasserstein_loss, wasserstein_loss, wasserstein_loss,
                              chi2_loss, chi2_loss],
                        loss_weights=[args.weight_mse, args.weight_adv, args.weight_adv, args.weight_ano, args.weight_ano, args.weight_con, args.weight_div, args.weight_dis, args.weight_dis],
                        optimizer=RMSprop(lr=args.learning_rate_generator))

if to_train:
    train(generator_model, disc_Faceness_model, generator, discriminator_Faceness, data_paths, ids_table)

generator = tf.keras.models.load_model(os.path.join(date_time_folder, 'Generator.h5'), compile=False)

test(tot_IDs=170, tot_sessions=2, tot_imgs=8, G=generator, dt_paths=data_paths, id_table=ids_table)

print('Done...')
