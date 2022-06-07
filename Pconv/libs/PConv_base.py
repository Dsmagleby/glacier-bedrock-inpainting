import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras.layers import Conv2D, InputSpec

from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, LeakyReLU, UpSampling2D, Concatenate, BatchNormalization, Input
from tensorflow.keras.layers import Concatenate, MaxPooling2D, Conv2DTranspose, Dropout


class PConv2D(Conv2D):
    def __init__(self, *args, last_layer=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_layer = last_layer
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

    def build(self, input_shape):
        """
        Adapted from original _Conv() layer of Keras.
        Parameters
            input_shape: list of dimensions for [img, mask].
        """
        assert isinstance(input_shape, list)
        assert self.data_format == 'channels_last', "data format should be `channels_last`"
        channel_axis = -1

        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')

        self.input_dim = input_shape[0][channel_axis]

        # Image kernel:
        kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        self.kernel  = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='img_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # Image bias:
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        # Calculate padding size to achieve zero-padding
        self.pconv_padding = (
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)),
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)),
        )

        # Window size - used for normalization
        self.window_size = self.kernel_size[0] * self.kernel_size[1]

        # Mask kernel:
        self.kernel_mask = K.ones(shape=self.kernel_size + (self.input_dim, self.filters))

        self.built = True

    def call(self, inputs):
        assert isinstance(inputs, list) and len(inputs) == 2
        #images, masks = inputs

        # Masked convolution:
        images = K.spatial_2d_padding(inputs[0], self.pconv_padding, self.data_format)
        masks = K.spatial_2d_padding(inputs[1], self.pconv_padding, self.data_format)

        # Apply convolutions to mask
        mask_output = K.conv2d(
            masks, self.kernel_mask,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Apply convolutions to image
        img_output = K.conv2d(
            (images*masks), self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Calculate the mask ratio on each pixel in the output mask
        mask_ratio = self.window_size / (mask_output + 1e-8)

        # Clip output to be between 0 and 1
        mask_output = K.clip(mask_output, 0, 1)

        # Remove ratio values where there are holes
        mask_ratio = mask_ratio * mask_output

        # Normalize iamge output
        img_output = img_output * mask_ratio

        # Apply bias only to the image (if chosen to do so)
        if self.use_bias:
            img_output = K.bias_add(
                img_output,
                self.bias,
                data_format=self.data_format)

        if self.last_layer:
            return img_output

        # Apply activations on the image
        if self.activation is not None:
            img_output = self.activation(img_output)

        return [img_output, mask_output]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        assert self.data_format == 'channels_last'
        space = input_shape[0][1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        new_shape = (input_shape[0][0],) + tuple(new_space) + (self.filters,)
        if self.last_layer:
            return new_shape
        return [new_shape, new_shape]


def encoder_block(input_img, input_mask, filters, kernel_size, drop=0, batch_norm=True, freeze_bn=False, activation=None, count=''):
    """
    Encoder block of layers.

    Parameters
        input_img: tensor, input image, output of an Input layer or a previous
            encoder block.
        input_mask: tensor, input binary mask, output of an Input layer or a
            previous encoder block.
        filters: integer, number of output channels.
        kernel_size: integer, width and height of the kernel.
        strides: integer, stride in both directions.
        batch_norm: boolean, whether to apply BatchNorm to the feature map
            (before activation if applied).
        freeze_bn: boolean, whether to freeze the BatchNorm (fine-tuning stage).
        activation: boolean, whether to apply a ReLU activation at the end.
        count: string, block count to append to the end of layers' names.

    """
    if count != '':
        count = '_' + count

    pconv, mask = PConv2D(filters,
                            kernel_size,
                            strides=2,
                            padding='valid',
                            use_bias=True,
                            kernel_initializer='he_uniform',
                            #kernel_regularizer='l2',
                            name='pconv2d_enc'+count
                           )([input_img, input_mask])

    if batch_norm:
        pconv = BatchNormalization(name='bn_enc'+count)(pconv, training=not freeze_bn)

    pconv = ReLU(name='relu'+count)(pconv)
    pconv = Dropout(drop)(pconv) if drop else pconv

    return pconv, mask


def decoder_block(prev_up_img, prev_up_mask, enc_img, enc_mask, filters, drop=0, last_layer=False, count=''):
    """
    Decoder block of layers.

    Parameters
        prev_up_img: previous image layer to up-sample.
        prev_up_mask: previous mask layer to up-sample.
        enc_img: image from encoder stage to concatenate with up-sampled image.
        enc_mask: mask from encoder stage to concatenate with up-sampled mask.
        filters: integer, number of output channels in the PConv2D layer.
        count: string, block count to append to the end of layers' names.
        last_layer: boolean, whether this is the last decoder block (no mask will
            be returned, no BatchNorm and no activation will be applied).
    """
    if count != '':
        count = '_' + count

    up_img  = UpSampling2D(size=2, name='img_upsamp_dec' + count)(prev_up_img)
    up_mask = UpSampling2D(size=2, name='mask_upsamp_dec' + count)(prev_up_mask)
    conc_img  = Concatenate(name='img_concat_dec' + count, axis=3)([up_img, enc_img])
    conc_mask = Concatenate(name='mask_concat_dec' + count, axis=3)([up_mask, enc_mask])

    if last_layer:
        return PConv2D(filters, 3, strides=1, padding='valid', use_bias=True, kernel_initializer='he_uniform', last_layer=last_layer, name='pconv2d_dec'+count)([conc_img, conc_mask])

    pconv, mask = PConv2D(filters, 3, strides=1, padding='valid', use_bias=True, 
                          kernel_initializer='he_uniform', #kernel_regularizer='l2', 
                          name='pconv2d_dec'+count)([conc_img, conc_mask])
                          
    pconv = BatchNormalization(name='bn_dec'+count)(pconv)
    pconv = LeakyReLU(alpha=0.2, name='leaky_dec'+count)(pconv)
    pconv = Dropout(drop)(pconv) if drop else pconv

    return pconv, mask
