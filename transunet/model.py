import transunet.encoder_layers as encoder_layers
import transunet.decoder_layers as decoder_layers
from transunet.resnet_v2 import  resnet_embeddings
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import transunet.utils as utils
import tensorflow as tf
import math

tfk = tf.keras
tfkl = tfk.layers
tfm = tf.math

MODELS_URL = 'https://storage.googleapis.com/vit_models/imagenet21k/'
        
def load_pretrained(model, fname='R50+ViT-B_16.npz'):
    """Load model weights for a known configuration."""
    origin = MODELS_URL + fname
    local_filepath = tf.keras.utils.get_file(fname, origin, cache_subdir="weights")
    utils.load_weights_numpy(model, local_filepath)
    
def TransUNet(image_size=224, 
                patch_size=16, 
                hybrid=True,
                grid=(14,14), 
                hidden_size=768,
                n_layers=12,
                n_heads=12,
                mlp_dim=3072,
                dropout=0.1,
                decoder_channels=[256,128,64,16],
                n_skip=3,
                num_classes=3,
                final_act='sigmoid',
                pretrain=True,
                freeze_enc_cnn=True,
                name='TransUNet'):
    # Tranformer Encoder
    assert image_size % patch_size == 0, "image_size must be a multiple of patch_size"
    x = tf.keras.layers.Input(shape=(image_size, image_size, 3))

    # Embedding
    if hybrid:
        grid_size = grid
        patch_size = image_size // 16 // grid_size[0]
        if patch_size == 0:
            patch_size = 1

        resnet50v2, features = resnet_embeddings(x, image_size=image_size, n_skip=n_skip, pretrain=pretrain)
        if freeze_enc_cnn:
            resnet50v2.trainable = False
        y = resnet50v2.get_layer("conv4_block6_preact_relu").output
        x = resnet50v2.input
    else:
        y = x
        features = None

    y = tfkl.Conv2D(
        filters=hidden_size,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="embedding",
        trainable=True
    )(y)
    y = tfkl.Reshape(
        (y.shape[1] * y.shape[2], hidden_size))(y)
    y = encoder_layers.AddPositionEmbs(
        name="Transformer/posembed_input", trainable=True)(y)

    y = tfkl.Dropout(0.1)(y)

    # Transformer/Encoder
    for n in range(n_layers):
        y, _ = encoder_layers.TransformerBlock(
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"Transformer/encoderblock_{n}",
            trainable=True
        )(y)
    y = tfkl.LayerNormalization(
        epsilon=1e-6, name="Transformer/encoder_norm"
    )(y)

    n_patch_sqrt = int(math.sqrt(y.shape[1]))

    y = tfkl.Reshape(
        target_shape=[n_patch_sqrt, n_patch_sqrt, hidden_size])(y)

    # Decoder CUP
    if len(decoder_channels):
        y = decoder_layers.DecoderCup(decoder_channels=decoder_channels, n_skip=n_skip)(y, features)

    # Segmentation Head
    y = decoder_layers.SegmentationHead(num_classes=num_classes, final_act=final_act)(y)

    # Build Model
    model =  tfk.models.Model(inputs=x, outputs=y, name=name)
    
    # Load Pretrain Weights
    if pretrain:
        load_pretrained(model)
        
    return model
