from xmlrpc.client import Boolean
import tensorflow as tf
import sys
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Concatenate, Lambda


def vgg16_feature_model(flayers, weights='imagenet'):
    """
    Feature exctraction VGG16 model.

    # Arguments
        flayers: list of strings with names of layers to get the features for.
            The length of `flayers` should be > 1, otherwise the output shape
            is one axis less.
        weights: ether "imagenet" or path to the file with weights.
    # Returns
        features_model: keras.models.Model instance to extract the features.

    # Raises
        AssertionError: in case of `flayers` is not a list.
        AssertionError: in case of length of 'flayers' < 2.
    """

    assert isinstance(flayers,list), "First argument 'flayers' must be a list"
    assert len(flayers) > 1, "Length of 'flayers' must be > 1."

    base_model = VGG16(include_top=False, weights=weights)

    vgg16_outputs = [base_model.get_layer(flayers[i]).output for i in range(len(flayers))]

    features_model = Model(inputs=[base_model.input], outputs=vgg16_outputs, name='vgg16_features')
    features_model.trainable = False
    features_model.compile(loss='mse', optimizer='adam')

    return features_model

class IrregularLoss(tf.keras.losses.Loss):
    def __init__(self, mask, vgg_model, norm=False, apply_penalty=False, area_scale=False, name='IrregularLoss'):
        super().__init__(name=name)
        self.vgg_model = vgg_model
        self.norm = norm
        self.apply_penalty = apply_penalty
        self.area_scale = area_scale
        self.mask = Concatenate()([mask, mask, mask])

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.latitude = 20.644


    def loss_l1(self, y_true, y_pred):
        if K.ndim(y_true) == 4:
            # images and vgg features
            return K.mean(K.abs(y_pred - y_true), axis=[1,2,3])
        elif K.ndim(y_true) == 3:
            # gram matrices
            return K.mean(K.abs(y_pred - y_true), axis=[1,2])
        else:
            raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")


    def gram_matrix(self, x):
        assert K.ndim(x) == 4, 'Input tensor should be 4D (B, H, W, C).'
        assert K.image_data_format() == 'channels_last', "Use channels-last format."

        # Permute channels and get resulting shape
        x = K.permute_dimensions(x, (0, 3, 1, 2))

        shape = K.shape(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3]

        # Reshape x and do batch dot product
        features = K.reshape(x, K.stack([B, C, H*W]))

        gram = K.batch_dot(features, features, axes=2)

        # Normalize with channels, height and width
        gram /= K.cast(C * H * W, x.dtype)

        return gram


    def loss_per_pixel(self, y_true, y_pred, mask):
        assert K.ndim(y_true) == 4, 'Input tensor should be 4D (B, H, W, C).'
        return K.mean(K.abs(mask * (y_pred - y_true)), axis=[1,2,3])


    def loss_perc(self, vgg_out, vgg_gt, vgg_comp):
        l = 0.
        for o, g, c in zip(vgg_out, vgg_gt, vgg_comp):
            l += self.loss_l1(o, g) + self.loss_l1(c, g)
        return l


    def loss_style(self, vgg_out, vgg_gt, vgg_comp):
        l = 0.
        for o, g, c in zip(vgg_out, vgg_gt, vgg_comp):
            gram_gt = self.gram_matrix(g)
            l += self.loss_l1(self.gram_matrix(o), gram_gt) + self.loss_l1(self.gram_matrix(c), gram_gt)
        return l


    def loss_tv(self, y_comp, mask_inv):
        assert K.ndim(y_comp) == 4 and K.ndim(mask_inv) == 4, 'Input tensors should be 4D (B, H, W, C).'

        # Create dilated hole region using a 3x3 kernel of all 1s.
        kernel = K.ones(shape=(3, 3, mask_inv.shape[3], mask_inv.shape[3]))
        dilated_mask = K.conv2d(mask_inv, kernel, data_format='channels_last', padding='same')

        # Cast values to be [0., 1.], and compute dilated hole region of y_comp
        dilated_mask = K.cast(K.greater(dilated_mask, 0), 'float32')
        P = dilated_mask * y_comp

        # Calculate total variation loss
        a = self.loss_l1(P[:,1:,:,:], P[:,:-1,:,:])
        b = self.loss_l1(P[:,:,1:,:], P[:,:,:-1,:])
        return a+b


    def call(self, y_true, y_pred):
        y_true = Concatenate()([y_true, y_true, y_true])
        y_pred = Concatenate()([y_pred, y_pred, y_pred])

        if self.norm:
            y_true = Lambda(lambda x: (x-self.mean) / self.std)(y_true)
            y_pred = Lambda(lambda x: (x-self.mean) / self.std)(y_pred)

        mask_inv = 1 - self.mask                                # KerasTensor
        y_comp   = self.mask * y_true + mask_inv * y_pred       # KerasTensor
        vgg_out  = self.vgg_model(y_pred)                       # Tensor
        vgg_gt   = self.vgg_model(y_true)                       # Tensor
        vgg_comp = self.vgg_model(y_comp)                       # Tensor

        l_valid = self.loss_per_pixel(y_true, y_pred, self.mask)     # KerasTensor
        l_hole  = self.loss_per_pixel(y_true, y_pred, mask_inv)      # KerasTensor
        l_perc  = self.loss_perc(vgg_out, vgg_gt, vgg_comp)          # KerasTensor
        l_style = self.loss_style(vgg_out, vgg_gt, vgg_comp)         # KerasTensor
        l_tv    = self.loss_tv(y_comp, mask_inv)                     # KerasTensor

        if self.apply_penalty:
            y_diff = y_true * mask_inv - y_pred * mask_inv
            mask_total = tf.math.count_nonzero(tf.greater_equal(mask_inv, 1.))
            penalty = tf.cast(tf.constant(2.), tf.float64) - (tf.math.count_nonzero(tf.greater(y_diff, 0.)) / mask_total)
            penalty = tf.cast(penalty, tf.float32)
        else:
            penalty = 0.0
        
        if self.area_scale: # 20.644 is average length of latitude tiles
            area_sum = tf.reduce_sum(mask_inv)/3 * 20.644 * 30. # estimated area
            y_diff = y_true * mask_inv - y_pred * mask_inv
            pred_sum = tf.reduce_sum(y_diff)/3 * 20.644 * 30. # predicted volume
            area = 2.0 - tf.math.exp(-((tf.pow(pred_sum - tf.pow(area_sum, 3/2), 2)) /  tf.pow(area_sum, 4/2)) )
        else:
            area = 0.0

        # If both penalty functions are applied, half their relative strength
        if self.apply_penalty and self.area_scale:
            penalty = penalty * 0.5
            area    = area * 0.5
        
        if self.apply_penalty or self.area_scale:
            return (l_valid + 6.*l_hole + 0.05*l_perc + 120.*l_style + 0.1*l_tv) * (penalty + area)
        else:
            return l_valid + 6.*l_hole + 0.05*l_perc + 120.*l_style + 0.1*l_tv

class MaskedMSE(tf.keras.losses.Loss):
    def __init__(self, mask, alpha, beta, name='MaskedMSE'):
        super().__init__(name=name)
        self.mask  = mask
        self.alpha = alpha
        self.beta  = beta

    def call(self, y_true, y_pred):
        mask_inv = 1 - self.mask
        masked = tf.reduce_mean(tf.square(mask_inv * (y_pred - y_true)), axis=-1)
        mse    = tf.reduce_mean(tf.square((y_pred - y_true)), axis=-1)
        return masked*self.alpha + mse*self.beta


def SSIM(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
def PSNR(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, 1.0))
def PSNRLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.psnr(y_true, y_pred, 1.0))


def positive_ratio(mask, y_true, y_pred):
    mask_inv = 1 - mask
    y_diff = (y_true * mask_inv) - (y_pred * mask_inv)
    mask_total = tf.math.count_nonzero(tf.greater_equal(mask_inv, 1.))
    penalty = tf.math.count_nonzero(tf.greater(y_diff, 0.)) / mask_total
    return 1.0 - penalty


class PositiveRatio(tf.keras.losses.Loss):
    def __init__(self, mask, name='PositiveRatio'):
        super().__init__(name=name)
        self.mask  = mask

    def call(self, y_true, y_pred):
        mask_inv = 1 - self.mask
        y_diff = (y_true * mask_inv) - (y_pred * mask_inv)
        mask_total = tf.math.count_nonzero(tf.greater_equal(mask_inv, 1.))
        penalty = tf.math.count_nonzero(tf.greater(y_diff, 0.)) / mask_total
        return 1.0 - penalty


class MaskedRMSE(tf.keras.losses.Loss):
    def __init__(self, mask, name='MaskedRMSE'):
        super().__init__(name=name)
        self.mask  = mask

    def call(self, y_true, y_pred):
        mask_inv = 1 - self.mask
        masked = tf.sqrt(tf.reduce_mean(tf.square(mask_inv * (y_pred - y_true)), axis=-1))
        return masked