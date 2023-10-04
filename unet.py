from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    UpSampling2D,
    Concatenate,
    BatchNormalization,
    LayerNormalization,
    Activation,
)
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.python.keras import backend as K


norm_dir = {
    "IN": tfa.layers.InstanceNormalization,
    "FRN": tfa.layers.FilterResponseNormalization,
    "GN_2": tfa.layers.GroupNormalization,
    "GN_4": tfa.layers.GroupNormalization,
    "BN": BatchNormalization,
    "LN": LayerNormalization,
    "NN": 1,
}

activation_dir = {
    "relu": tf.keras.activations.relu,
    "leaky_relu": tf.keras.layers.LeakyReLU(),
    "selu": tf.keras.activations.selu,
}


def conv_layer(
    x, norm_name, filters, activation, scale_gamma, center_beta, name, norm_active=True
):
    """Method to build the conv-layer of an unet with the given settings.
    Parameter:
        x (array) : input for convolution
        norm_name (string) : name of normalization
        filters (int) : filter size
        activation (string) : name of activation function
        scale_gamma (bool) : definition if parameter is learned
        center_beta (bool) : definition if parameter is learned
        name (string) : name of layer
    Return:
        x (array) : output of convolution
    """
    x = Conv2D(filters, 3, padding="same", name=name)(x)
    if norm_active:
        if "GN" in norm_name:
            split_factor = int(norm_name.split("_")[1])
            x = norm_dir[norm_name](
                groups=int(filters / split_factor),
                scale=bool(scale_gamma),
                center=bool(center_beta),
            )(x)
        elif norm_name == "NN":
            x = x
        elif norm_name == "FRN":
            x = norm_dir[norm_name](name=name + f"_{norm_name}")(x)
        else:
            x = norm_dir[norm_name](scale=bool(scale_gamma), center=bool(center_beta))(
                x
            )

    x = Activation(activation, name=name + "_activation")(x)
    return x


def unet(
    norm_name,
    layers,
    n_classes,
    size_img,
    fs,
    activation,
    scale_arr,
    center_arr,
):
    """Method to build the unet with the given settings of parameter.
    Parameter:
        norm_name (array) : array with strings for layer-specific normalization
        layers (int) : layer depth of U-Net
        n_classes (int) : number of classes to segment
        size_img (tuple) : tuple of image size
        fs (int) : filter size
        activation (string) : strings of activation function
        scale_arr (array) : array with booleans of layer-specific learning of normalization parameter
        center_arr (array) : array with booleans of layer-specific learning of normalization parameter
    Returns:
        model (tf.model) : U-Net model
    """
    activation = activation_dir[activation]
    filters = tf.constant(fs)

    skip_con = []
    model_input = Input(size_img)
    x = model_input

    # Down-sampling
    for i in range(layers):
        x = conv_layer(
            x,
            norm_name[i],
            filters * 2**i,
            activation,
            scale_arr[i],
            center_arr[i],
            f"enc_layer{i}_conv1",
        )
        x = conv_layer(
            x,
            norm_name[i],
            filters * 2**i,
            activation,
            scale_arr[i],
            center_arr[i],
            f"enc_layer{i}_conv2",
        )

        # Saving last convolution for skip connection
        skip_con.append(x)
        x = MaxPool2D()(x)

    x = conv_layer(
        x,
        norm_name[4],
        filters * 2 ** (i + 1),
        activation,
        scale_arr[4],
        center_arr[4],
        "latent_conv",
    )

    # Up-sampling
    for i in range(layers):
        x = UpSampling2D()(x)
        x = Concatenate(name=f"skip_{i}")([x, skip_con.pop()])
        x = conv_layer(
            x,
            norm_name[layers + 1 + i],
            filters * 2 ** (layers - i - 1),
            activation,
            scale_arr[layers + 1 + i],
            center_arr[layers + 1 + i],
            f"dec_layer{layers-i}_conv1",
        )
        x = conv_layer(
            x,
            norm_name[layers + 1 + i],
            filters * 2 ** (layers - i - 1),
            activation,
            scale_arr[layers + 1 + i],
            center_arr[layers + 1 + i],
            f"dec_layer{layers-i}_conv2",
        )

    if n_classes > 1:
        output_layer1 = Conv2D(
            1, kernel_size=1, activation="sigmoid", padding="same", name="final1"
        )(x)

        output_layer2 = Conv2D(
            1, kernel_size=1, activation="sigmoid", padding="same", name="final2"
        )(x)
        return Model(model_input, (output_layer1, output_layer2))

    else:
        output_layer1 = Conv2D(
            n_classes,
            kernel_size=1,
            activation="sigmoid",
            padding="same",
            name="final1",
        )(x)
        return Model(model_input, output_layer1)
