import argparse
import pathlib
import json
import random
import time
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
import math
import data_generator
import metrics
import unet
from unet import norm_dir
import git


with open("constants.json", "r") as constants_file:
    CONSTANTS = json.load(constants_file)

with open("datasets.json", "r") as datasets_file:
    DATASETS = json.load(datasets_file)


class TimeHistory(keras.callbacks.Callback):
    """Saving time epoch wise."""

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def main(
    train_gen,
    val_gen,
    ts_gen,
    num_tr,
    num_val,
    num_ts,
    dataset,
    individuals,
    epochs,
    layer,
    bs,
    num_classes,
    norm_name,
    generation,
    fs,
    activation,
    scale_arr,
    center_arr,
):
    """
    Runs the U-Net for all individuals over the period of the given number of generations. It returns a dictionary with all the information to each individual of all generations.

            Parameters:
                    train_gen (tensor) : tensor with the train images and masks
                    val_gen (tensor) : tensor with the val images and masks
                    ts_gen (tensor) : tensor with the test images and masks
                    num_tr (int) : number of train samples
                    num_val (int) : number of validation samples
                    num_ts (int) : number of test samples
                    dataset (string) : name of the dataset
                    individuals (int) : number of individuals per generation
                    epochs (int) : number of epochs to run the training of the U-Net
                    layer (int) : number of layers of U-Net
                    bs (int) : batch size
                    num_classes (int) : number of segmentation classes
                    norm_name (array) : array of the Normalization layer for all individuals
                    generation (int) : number of generations to execute evolutionary algorithm
                    fs (array) : array of the filter sizes for all individuals
                    activation (array) : array of the activation functions for all individuals
                    scale_arr (array) : information array if normalization parameters are trained or not for all individuals
                    center_arr (array) : information array if normalization parameters are trained or not for all individuals

            Returns:
                    file_config (json) : File containing the information regarding the configuration of each individual
                    file_metrics (json) : File containing the information regarding the metrics of each individual
    """

    defined_metrics = [metrics.dice_coefficient]

    file_metrics = {}
    file_config = {}

    for i in range(individuals):
        K.clear_session()
        results_metrics = {}
        result_configs = {}
        history_time = TimeHistory()

        stop_criterion = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            verbose=1,
            patience=8,
            mode="min",
        )

        model_ckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{DIR_OUTPUT}/model_{generation}_{i}.h5",
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        )

        unet_model = unet.unet(
            norm_name[i],
            layer,
            num_classes,
            (224, 224, 1),
            fs[i],
            activation[i],
            scale_arr[i],
            center_arr[i],
        )

        unet_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=metrics.dice_coef_loss,
            metrics=defined_metrics,
            run_eagerly=True,
        )

        # Save checkpoint only in first and last generation
        if generation == 0 or generation == 19:
            history_model = unet_model.fit(
                train_gen,
                batch_size=tf.constant(bs),
                steps_per_epoch=num_tr // bs,
                validation_data=val_gen,
                validation_steps=num_val // bs,
                epochs=epochs,
                callbacks=[model_ckpt, history_time, stop_criterion],
            )
        else:
            history_model = unet_model.fit(
                train_gen,
                batch_size=tf.constant(bs),
                steps_per_epoch=num_tr // bs,
                validation_data=val_gen,
                validation_steps=num_val // bs,
                epochs=epochs,
                callbacks=[history_time, stop_criterion],
            )

        val_images, val_labels = tuple(zip(*val_gen))
        hd95_monai = []
        bbIoU_val = []
        for b in range(np.array(val_images).shape[0]):
            val_prediction = unet_model.predict(val_images[b])
            hd95_monai.append(
                metrics.hd_95_monai(
                    np.asarray(val_labels[b]), np.asarray(val_prediction)
                ).numpy()
            )
            bbIoU_val.append(
                metrics.bb_IoU(
                    np.asarray(val_labels[b]), np.asarray(val_prediction)
                ).numpy()
            )

        if math.isnan(np.mean(hd95_monai)):
            hd95_value = 1000
        else:
            hd95_value = np.mean(hd95_monai)
        hd95_metric = 1 / (hd95_value + tf.keras.backend.epsilon())

        if num_classes > 1:
            results_metrics["metrics"] = history_model.history
            result_configs["epoch_time"] = np.mean(history_time.times)

            ind_max_val = np.argmin(history_model.history["loss"])
            final_dice = (
                history_model.history["final1_dice_coefficient"][ind_max_val]
                + history_model.history["final2_dice_coefficient"][ind_max_val]
            ) / 2
            result_configs["metrics_tr_vec"] = [
                (final_dice),
                (history_model.history["loss"][ind_max_val]),
            ]

            ind_max_val = np.argmin(history_model.history["val_loss"])
            final_val_dice = (
                history_model.history["val_final1_dice_coefficient"][ind_max_val]
                + history_model.history["val_final2_dice_coefficient"][ind_max_val]
            ) / 2

            result_configs["metrics_val_vec"] = [
                (final_val_dice),
                (float(np.mean(bbIoU_val))),
                (hd95_value),
                (history_model.history["val_loss"][ind_max_val]),
            ]
            result_configs["F_val"] = (
                np.sum(
                    final_val_dice
                    + float(np.mean(bbIoU_val))
                    + float(np.where(hd95_metric > 1, 1, hd95_metric))
                )
                / 3
            )
        else:
            results_metrics["metrics"] = history_model.history
            result_configs["epoch_time"] = np.mean(history_time.times)

            ind_max_val = np.argmin(history_model.history["loss"])
            final_dice = history_model.history["dice_coefficient"][ind_max_val]
            result_configs["metrics_tr_vec"] = [
                (final_dice),
                (history_model.history["loss"][ind_max_val]),
            ]

            ind_max_val = np.argmin(history_model.history["val_loss"])
            final_val_dice = history_model.history["val_dice_coefficient"][ind_max_val]
            result_configs["metrics_val_vec"] = [
                (final_val_dice),
                (float(np.mean(bbIoU_val))),
                (hd95_value),
                (history_model.history["val_loss"][ind_max_val]),
            ]

            result_configs["F_val"] = (
                np.sum(
                    final_val_dice
                    + float(np.mean(bbIoU_val))
                    + float(np.where(hd95_metric > 1, 1, hd95_metric))
                )
                / 3
            )

        result_configs["activation"] = activation[i]
        result_configs["filter_size"] = int(fs[i])
        result_configs["scale"] = np.array(scale_arr[i]).tolist()
        result_configs["center"] = np.array(center_arr[i]).tolist()
        result_configs["norms"] = norm_name[i]
        file_metrics[f"model_{i}"] = results_metrics
        file_config[f"model_{i}"] = result_configs
    return file_config, file_metrics


def select(config_file, individuals, layer, generation):
    """
    Returns the arrays of the new individuals for the next generation.

            Parameters:
                    config_file (dict): json file containing the information of individuals of previous generations
                    individuals (int): number of individuals
                    layer (int): number of U-Net layer
                    generation (int): current generation

            Returns:
                    arrays : Arrays with the information of the new selected population
    """

    list_eval = []
    for key, items in config_file.items():
        list_eval.append(items["F_val"])

    ind_entries = len(list_eval)
    fittest = np.argsort(list_eval)[int(ind_entries / 2) : ind_entries]

    crossover_info = {}
    crossover = []
    mutation = []
    children = []
    children_scale = []
    children_center = []
    child_activation = []
    child_filter_size = []

    for i in range(individuals):
        crossover_info[f"gen_{generation}_individual_{i}"] = {}
        mutation = bool(random.choices([0, 1], weights=(90, 10))[0])
        np.random.shuffle(fittest)
        parents_0 = int(fittest[0])
        parents_1 = int(fittest[1])

        norm_options = 2 * layer + 1

        parents_idx = [(parents_0, parents_1)]
        crossover = int(np.random.randint(1, norm_options - 1, 1))

        children.append(
            list(config_file[f"model_{parents_0}"]["norms"][0:crossover])
            + list(config_file[f"model_{parents_1}"]["norms"][crossover:])
        )

        children_scale.append(
            list(config_file[f"model_{parents_0}"]["scale"][0:crossover])
            + list(config_file[f"model_{parents_1}"]["scale"][crossover:])
        )

        children_center.append(
            list(config_file[f"model_{parents_0}"]["center"][0:crossover])
            + list(config_file[f"model_{parents_1}"]["center"][crossover:])
        )

        child_activation.append(
            np.random.choice(
                [
                    config_file[f"model_{parents_0}"]["activation"],
                    config_file[f"model_{parents_1}"]["activation"],
                ]
            )
        )
        child_filter_size.append(
            np.random.choice(
                [
                    config_file[f"model_{parents_0}"]["filter_size"],
                    config_file[f"model_{parents_1}"]["filter_size"],
                ]
            )
        )

        crossover_info[f"gen_{generation}_individual_{i}"]["child"] = children[i]
        crossover_info[f"gen_{generation}_individual_{i}"]["crossover_pt"] = crossover
        crossover_info[f"gen_{generation}_individual_{i}"]["parents"] = parents_idx

        crossover_info[f"gen_{generation}_individual_{i}"]["scale"] = children_scale[i]
        crossover_info[f"gen_{generation}_individual_{i}"]["center"] = children_center[
            i
        ]
        crossover_info[f"gen_{generation}_individual_{i}"]["mutation"] = mutation

        if mutation:
            mut_idx = np.random.randint(norm_options)
            mut_gene = list(norm_dir.keys())[np.random.randint(len(norm_dir))]
            children[i][mut_idx] = mut_gene
            crossover_info[f"gen_{generation}_individual_{i}"]["mutation_idx"] = mut_idx
            crossover_info[f"gen_{generation}_individual_{i}"][
                "mutation_gene"
            ] = mut_gene

    return (
        children,
        children_scale,
        children_center,
        crossover_info,
        child_activation,
        child_filter_size,
    )


if __name__ == "__main__":
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source",
        help="The path of dataset dir.",
    )
    parser.add_argument("-o", "--output", help="The path for output.")
    parser.add_argument("-d", "--dataset", nargs="+")
    parser.add_argument("-g", "--generations", type=int, default=20)
    parser.add_argument("-i", "--individuals", type=int, default=20)
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-ng", "--number_gpus", type=int, default=1)
    args = parser.parse_args()

    DIR_SOURCE = pathlib.Path(args.source)
    DIR_OUTPUT = pathlib.Path(args.output)
    LIST_DS = args.dataset
    NUM_GENERATIONS = args.generations
    INDIVIDUALS = args.individuals
    EPOCHS = args.epochs
    GPUS = args.number_gpus

    BATCH_SIZE = 64
    LAYER = 4

    DATASET = LIST_DS[0]
    NUM_CLASSES = DATASETS[DATASET]["num_classes"]

    # Generating TRAIN, VAL, TEST dataset
    tr_ds, val_ds, ts_ds, num_tr, num_val, num_ts = data_generator.main(
        DATASET, DIR_SOURCE, BATCH_SIZE, EPOCHS, DIR_OUTPUT
    )

    file_config = {}
    file_metrics = {}
    cross_info = []

    for generation in range(NUM_GENERATIONS):
        file_config["gitsha"] = sha
        file_config[f"Generation_{generation}"] = {}
        act_func = np.random.choice(["relu", "leaky_relu", "selu"], size=INDIVIDUALS)
        filter_size = np.random.choice([4, 8, 16], size=INDIVIDUALS)
        if generation == 0:
            norm_arr = []
            scale_arr = []
            center_arr = []
            [
                norm_arr.append(
                    list(np.random.choice([*norm_dir.keys()], size=LAYER * 2 + 1))
                )
                for x in range(INDIVIDUALS)
            ]
            [
                scale_arr.append(list(np.random.choice([0, 1], size=2 * LAYER + 1)))
                for y in range(INDIVIDUALS)
            ]
            [
                center_arr.append(list(np.random.choice([0, 1], size=2 * LAYER + 1)))
                for z in range(INDIVIDUALS)
            ]

        else:
            norm_arr, scale_arr, center_arr, cross_info, act_func, filter_size = select(
                file_config[f"Generation_{generation-1}"]["models"],
                INDIVIDUALS,
                LAYER,
                generation,
            )
            file_config[f"Generation_{generation}"]["crossover"] = cross_info

        conf, met = main(
            tr_ds,
            val_ds,
            ts_ds,
            num_tr,
            num_val,
            num_ts,
            DATASET,
            INDIVIDUALS,
            EPOCHS,
            LAYER,
            BATCH_SIZE,
            NUM_CLASSES,
            norm_arr,
            generation,
            filter_size,
            act_func,
            scale_arr,
            center_arr,
        )

        file_config[f"Generation_{generation}"]["models"] = conf
        file_metrics[f"Generation_{generation}"] = met

        with open(f"{DIR_OUTPUT}/config_{DATASET}.json", "w", encoding="utf-8") as f:
            json.dump(file_config, f)

        with open(f"{DIR_OUTPUT}/metrics_{DATASET}.json", "w", encoding="utf-8") as f:
            json.dump(file_metrics, f)
