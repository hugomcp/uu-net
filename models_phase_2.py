from keras.optimizers import RMSprop, Adam
from keras.models import *
from keras.layers import *
from SpectralNormalizationKeras import *
from keras.constraints import Constraint
from keras.initializers import RandomNormal


def create_discriminator(img_shape):
    model = Sequential()

    model.add(Conv2D(16, kernel_size=3, strides=1, input_shape=img_shape, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=1, input_shape=img_shape, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(1))

    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)


def create_discriminator_R(img_shape):
    input1 = Input(shape=img_shape)
    input2 = Input(shape=img_shape)
    input12 = concatenate([input1, input2])

    conv11 = Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same")(input12)
    conv11_a = LeakyReLU()(conv11)
    drop11 = Dropout(0.25)(conv11_a)

    conv12 = Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same")(drop11)
    conv12_bn = BatchNormalization(momentum=0.8)(conv12)
    conv12_a = LeakyReLU()(conv12_bn)
    drop12 = Dropout(0.25)(conv12_a)

    conv13 = Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same")(drop12)
    conv13_bn = BatchNormalization(momentum=0.8)(conv13)
    conv13_a = LeakyReLU()(conv13_bn)
    drop13 = Dropout(0.25)(conv13_a)

    conv14 = Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same")(drop13)
    conv14_bn = BatchNormalization(momentum=0.8)(conv14)
    conv14_a = LeakyReLU()(conv14_bn)
    drop14 = Dropout(0.25)(conv14_a)

    flat1 = Flatten()(drop14)

    output = Dense(1, activation='linear', kernel_constraint=None)(flat1)
    model = Model(inputs=[input1, input2], outputs=output)

    return model


def create_discriminator_R_cross_entropy(img_shape):
    input1 = Input(shape=img_shape)
    input2 = Input(shape=img_shape)
    input12 = concatenate([input1, input2])

    conv11 = Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same")(input12)
    conv11_bn = BatchNormalization(momentum=0.8)(conv11)
    conv11_a = LeakyReLU()(conv11_bn)
    drop11 = Dropout(0.25)(conv11_a)

    conv12 = Conv2D(64, kernel_size=3, strides=1, input_shape=img_shape, padding="same")(drop11)
    conv12_bn = BatchNormalization(momentum=0.8)(conv12)
    conv12_a = LeakyReLU()(conv12_bn)
    drop12 = Dropout(0.25)(conv12_a)

    conv13 = Conv2D(128, kernel_size=3, strides=1, input_shape=img_shape, padding="same")(drop12)
    conv13_bn = BatchNormalization(momentum=0.8)(conv13)
    conv13_a = LeakyReLU()(conv13_bn)
    drop13 = Dropout(0.25)(conv13_a)

    conv14 = Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same")(drop13)
    conv14_bn = BatchNormalization(momentum=0.8)(conv14)
    conv14_a = LeakyReLU()(conv14_bn)
    drop14 = Dropout(0.25)(conv14_a)

    conv15 = Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same")(drop14)
    conv15_bn = BatchNormalization(momentum=0.8)(conv15)
    conv15_a = LeakyReLU()(conv15_bn)
    drop15 = Dropout(0.25)(conv15_a)

    conv16 = Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same")(drop15)
    conv16_bn = BatchNormalization(momentum=0.8)(conv16)
    conv16_a = LeakyReLU()(conv16_bn)
    drop16 = Dropout(0.25)(conv16_a)

    flat1 = Flatten()(drop16)

    dense1 = Dense(128, activation='relu', kernel_constraint=None)(flat1)

    output = Dense(1, activation='sigmoid', kernel_constraint=None)(dense1)
    model = Model(inputs=[input1, input2], outputs=output)

    return model


def create_generator(input_shape):
    inputs = Input(input_shape)
    conv1 = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(inputs)
    conv1_bn = BatchNormalization()(conv1)
    conv1_o = LeakyReLU(alpha=0.2)(conv1_bn)

    conv2 = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(conv1_o)
    conv2_bn = BatchNormalization()(conv2)
    conv2_o = LeakyReLU(alpha=0.2)(conv2_bn)

    conv3 = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(conv2_o)
    conv3_bn = BatchNormalization()(conv3)
    conv3_o = LeakyReLU(alpha=0.2)(conv3_bn)

    conv4 = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(conv3_o)
    conv4_bn = BatchNormalization()(conv4)
    conv4_o = LeakyReLU(alpha=0.2)(conv4_bn)

    conv5 = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(conv4_o)
    conv5_bn = BatchNormalization()(conv5)
    conv5_o = LeakyReLU(alpha=0.2)(conv5_bn)

    conv6 = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(conv5_o)
    conv6_bn = BatchNormalization()(conv6)
    conv6_o = LeakyReLU(alpha=0.2)(conv6_bn)

    conv7 = Conv2D(3, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv6_o)

    conv8 = Conv2DTranspose(64, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(conv7)
    conv8_bn = BatchNormalization()(conv8)
    conv8_o = LeakyReLU(alpha=0.2)(conv8_bn)

    conv9 = Conv2DTranspose(128, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(conv8_o)
    conv9_bn = BatchNormalization()(conv9)
    conv9_o = LeakyReLU(alpha=0.2)(conv9_bn)

    conv10 = Conv2DTranspose(256, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(conv9_o)
    conv10_bn = BatchNormalization()(conv10)
    conv10_o = LeakyReLU(alpha=0.2)(conv10_bn)

    conv11 = Conv2DTranspose(128, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(conv10_o)
    conv11_bn = BatchNormalization()(conv11)
    conv11_o = LeakyReLU(alpha=0.2)(conv11_bn)

    conv12 = Conv2DTranspose(64, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(conv11_o)
    conv12_bn = BatchNormalization()(conv12)
    conv12_o = LeakyReLU(alpha=0.2)(conv12_bn)

    conv13 = Conv2DTranspose(32, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(conv12_o)
    conv13_bn = BatchNormalization()(conv13)
    conv13_o = LeakyReLU(alpha=0.2)(conv13_bn)

    conv14 = Conv2DTranspose(3, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv13_o)

    model = Model(input=inputs, output=[conv7, conv14])
    model.summary()

    return model


def create_generator_2(input_shape):
    inputs = Input(input_shape)
    conv1 = Conv2D(64, (5, 5), strides=2, use_bias=False, padding='same', kernel_initializer='he_normal')(inputs)
    conv1_bn = BatchNormalization()(conv1)
    conv1_o = LeakyReLU(alpha=0.2)(conv1_bn)

    conv2 = Conv2D(128, (5, 5), strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')(conv1_o)
    conv2_bn = BatchNormalization()(conv2)
    conv2_o = LeakyReLU(alpha=0.2)(conv2_bn)

    conv3 = Conv2D(256, (5, 5), strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')(conv2_o)
    conv3_bn = BatchNormalization()(conv3)
    conv3_o = LeakyReLU(alpha=0.2)(conv3_bn)

    conv4 = Conv2DTranspose(128, (5, 5), strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')(
        conv3_o)
    conv4_bn = BatchNormalization()(conv4)
    conv4_o = LeakyReLU(alpha=0.2)(conv4_bn)

    conv5 = Conv2DTranspose(64, (5, 5), strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')(
        conv4_o)
    conv5_bn = BatchNormalization()(conv5)
    conv5_o = LeakyReLU(alpha=0.2)(conv5_bn)

    conv6 = Conv2DTranspose(32, (5, 5), strides=2, use_bias=False, padding='same', kernel_initializer='he_normal')(
        conv5_o)
    conv6_bn = BatchNormalization()(conv6)
    conv6_o = LeakyReLU(alpha=0.2)(conv6_bn)

    conv7 = Conv2D(3, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv6_o)

    conv8 = Conv2D(64, (5, 5), strides=2, use_bias=False, padding='same', kernel_initializer='he_normal')(conv7)
    conv8_bn = BatchNormalization()(conv8)
    conv8_o = LeakyReLU(alpha=0.2)(conv8_bn)

    conv9 = Conv2D(128, (5, 5), strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')(conv8_o)
    conv9_bn = BatchNormalization()(conv9)
    conv9_o = LeakyReLU(alpha=0.2)(conv9_bn)

    conv10 = Conv2D(256, (5, 5), strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')(conv9_o)
    conv10_bn = BatchNormalization()(conv10)
    conv10_o = LeakyReLU(alpha=0.2)(conv10_bn)

    conv11 = Conv2DTranspose(128, (5, 5), strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')(
        conv10_o)
    conv11_bn = BatchNormalization()(conv11)
    conv11_o = LeakyReLU(alpha=0.2)(conv11_bn)

    conv12 = Conv2DTranspose(64, (5, 5), strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')(
        conv11_o)
    conv12_bn = BatchNormalization()(conv12)
    conv12_o = LeakyReLU(alpha=0.2)(conv12_bn)

    conv13 = Conv2DTranspose(32, (5, 5), strides=2, use_bias=False, padding='same', kernel_initializer='he_normal')(
        conv12_o)
    conv13_bn = BatchNormalization()(conv13)
    conv13_o = LeakyReLU(alpha=0.2)(conv13_bn)

    conv14 = Conv2D(3, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv13_o)

    model = Model(input=inputs, output=[conv7, conv14])
    model.summary()


def create_generator_3(input_shape):
    k_s = 7
    inputs = Input(input_shape)
    conv1 = Conv2D(64, (k_s, k_s), strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')(inputs)
    conv1_bn = BatchNormalization()(conv1)
    conv1_o = LeakyReLU(alpha=0.2)(conv1_bn)

    conv2 = Conv2D(128, (k_s, k_s), strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')(conv1_o)
    conv2_bn = BatchNormalization()(conv2)
    conv2_o = LeakyReLU(alpha=0.2)(conv2_bn)

    conv3 = Conv2D(256, (k_s, k_s), strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')(conv2_o)
    conv3_bn = BatchNormalization()(conv3)
    conv3_o = LeakyReLU(alpha=0.2)(conv3_bn)

    conv4 = Conv2D(128, (k_s, k_s), strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')(
        conv3_o)
    conv4_bn = BatchNormalization()(conv4)
    conv4_o = LeakyReLU(alpha=0.2)(conv4_bn)

    conv5 = Conv2D(64, (k_s, k_s), strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')(
        conv4_o)
    conv5_bn = BatchNormalization()(conv5)
    conv5_o = LeakyReLU(alpha=0.2)(conv5_bn)

    conv6 = Conv2D(32, (k_s, k_s), strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')(
        conv5_o)
    conv6_bn = BatchNormalization()(conv6)
    conv6_o = LeakyReLU(alpha=0.2)(conv6_bn)

    conv7 = Conv2D(3, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv6_o)

    conv8 = Conv2D(64, (k_s, k_s), strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')(conv7)
    conv8_bn = BatchNormalization()(conv8)
    conv8_o = LeakyReLU(alpha=0.2)(conv8_bn)

    conv9 = Conv2D(128, (k_s, k_s), strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')(conv8_o)
    conv9_bn = BatchNormalization()(conv9)
    conv9_o = LeakyReLU(alpha=0.2)(conv9_bn)

    conv10 = Conv2D(256, (k_s, k_s), strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')(conv9_o)
    conv10_bn = BatchNormalization()(conv10)
    conv10_o = LeakyReLU(alpha=0.2)(conv10_bn)

    conv11 = Conv2D(128, (k_s, k_s), strides=1, use_bias=False, padding='same',
                    kernel_initializer='he_normal')(conv10_o)
    conv11_bn = BatchNormalization()(conv11)
    conv11_o = LeakyReLU(alpha=0.2)(conv11_bn)

    conv12 = Conv2D(64, (k_s, k_s), strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')(
        conv11_o)
    conv12_bn = BatchNormalization()(conv12)
    conv12_o = LeakyReLU(alpha=0.2)(conv12_bn)

    conv13 = Conv2D(32, (k_s, k_s), strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')(
        conv12_o)
    conv13_bn = BatchNormalization()(conv13)
    conv13_o = LeakyReLU(alpha=0.2)(conv13_bn)

    conv14 = Conv2D(3, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv13_o)

    model = Model(input=inputs, output=[conv7, conv14])
    print('Generator_3:')
    model.summary()

    return model


def define_encoder_block(layer_in, n_filters, batchnorm=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)
    return g


def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    g = BatchNormalization()(g, training=True)
    if dropout:
        g = Dropout(0.5)(g, training=True)
    g = Concatenate()([g, skip_in])
    g = Activation('relu')(g)
    return g


def create_generator_unet(image_shape):
    init = RandomNormal(stddev=0.02)

    in_image = Input(shape=image_shape)

    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e5)
    b = Activation('relu')(b)
    d1 = decoder_block(b, e5, 512)
    d4 = decoder_block(d1, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    out_A = Activation('sigmoid')(g)

    de1 = define_encoder_block(out_A, 64, batchnorm=False)
    de2 = define_encoder_block(de1, 128)
    de3 = define_encoder_block(de2, 256)
    de4 = define_encoder_block(de3, 512)
    de5 = define_encoder_block(de4, 512)
    # bottleneck, no batch norm and relu
    db = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(de5)
    db = Activation('relu')(db)
    dd1 = decoder_block(db, de5, 512)
    dd4 = decoder_block(dd1, de4, 512, dropout=False)
    dd5 = decoder_block(dd4, de3, 256, dropout=False)
    dd6 = decoder_block(dd5, de2, 128, dropout=False)
    dd7 = decoder_block(dd6, de1, 64, dropout=False)
    # output
    dg = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(dd7)
    out_O = Activation('sigmoid')(dg)

    model = Model(in_image, [out_A, out_O])
    print('U-NET Generator:')
    model.summary()
    return model


def create_discriminator_patch_GAN(image_shape):
    init = RandomNormal(stddev=0.02)
    in_src_image = Input(shape=image_shape)


    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_src_image)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # second last output layer
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # patch output
    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model(in_src_image, patch_out)
    print('Path GAN Discriminator:')
    model.summary()
    return model
