from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

from Pconv.libs.PConv_base import decoder_block, encoder_block
from Pconv.libs.loss_functions import IrregularLoss, SSIM, PSNR, vgg16_feature_model


def pconv_model(fine_tuning=False, 
                lr=0.0002, 
                predict_only=False, 
                image_size=(256, 256), 
                drop=0, 
                vgg16_weights='imagenet',
                vgg_norm = True,
                loss_penalty = False,
                loss_area = False):
    """Inpainting model."""

    img_input  = Input(shape=(image_size[0], image_size[1], 1), name='input_img')
    mask_input = Input(shape=(image_size[0], image_size[1], 1), name='input_mask')
    y_true     = Input(shape=(image_size[0], image_size[1], 1), name='y_true')
    
    # Encoder:
    # --------
    e_img_1, e_mask_1 = encoder_block(img_input, mask_input, 32, 7, drop=0, batch_norm=False, count='1')
    e_img_2, e_mask_2 = encoder_block(e_img_1, e_mask_1, 64, 5, drop=drop, freeze_bn=fine_tuning, count='2')
    e_img_3, e_mask_3 = encoder_block(e_img_2, e_mask_2, 128, 5, drop=drop, freeze_bn=fine_tuning, count='3')
    e_img_4, e_mask_4 = encoder_block(e_img_3, e_mask_3, 256, 3, drop=drop, freeze_bn=fine_tuning, count='4')
    e_img_5, e_mask_5 = encoder_block(e_img_4, e_mask_4, 256, 3, drop=drop, freeze_bn=fine_tuning, count='5')
    e_img_6, e_mask_6 = encoder_block(e_img_5, e_mask_5, 256, 3, drop=drop, freeze_bn=fine_tuning, count='6')
    e_img_7, e_mask_7 = encoder_block(e_img_6, e_mask_6, 256, 3, drop=drop, freeze_bn=fine_tuning, count='7')
    e_img_8, e_mask_8 = encoder_block(e_img_7, e_mask_7, 256, 3, drop=drop, freeze_bn=fine_tuning, count='8')

    # Decoder:
    # --------
    d_img_9, d_mask_9   = decoder_block(e_img_8, e_mask_8, e_img_7, e_mask_7, 256, drop=drop, count='9')
    d_img_10, d_mask_10 = decoder_block(d_img_9, d_mask_9, e_img_6, e_mask_6, 256, drop=drop, count='10')
    d_img_11, d_mask_11 = decoder_block(d_img_10, d_mask_10, e_img_5, e_mask_5, 256, drop=drop, count='11')
    d_img_12, d_mask_12 = decoder_block(d_img_11, d_mask_11, e_img_4, e_mask_4, 256, drop=drop, count='12')
    d_img_13, d_mask_13 = decoder_block(d_img_12, d_mask_12, e_img_3, e_mask_3, 128, drop=drop, count='13')
    d_img_14, d_mask_14 = decoder_block(d_img_13, d_mask_13, e_img_2, e_mask_2, 64, drop=drop, count='14')
    d_img_15, d_mask_15 = decoder_block(d_img_14, d_mask_14, e_img_1, e_mask_1, 32, drop=drop, count='15')
    d_img_16 = decoder_block(d_img_15, d_mask_15, img_input, mask_input, 1, last_layer=True, count='16')
    
    model = Model(inputs=[img_input, mask_input, y_true], outputs=d_img_16)

    # This will also freeze bn parameters `beta` and `gamma`: 
    if fine_tuning:
        for l in model.layers:
            if 'bn_enc' in l.name:
                l.trainable = False
    
    if predict_only:
        return model

    vgg_model = vgg16_feature_model(['block1_pool', 'block2_pool', 'block3_pool'], weights=vgg16_weights)
    model.add_loss(IrregularLoss(mask_input, vgg_model, vgg_norm, loss_penalty, loss_area)(y_true, d_img_16))
    model.compile(Adam(learning_rate=lr), metrics=[SSIM, PSNR])

    return model