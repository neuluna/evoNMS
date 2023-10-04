import os
import numpy as np
import tensorflow as tf
import fnmatch
import json
import albumentations as A
from tqdm import trange
import cv2


with open("datasets.json", "r") as dataset_file:
    dict_ds = json.load(dataset_file)


def data_generator(dataset, dir_dataset, DIR_OUTPUT, test_scenario=False):
    """Method for loading the dataset, normalizing the images/masks to 0 to 1 and arrange split them into train, val and test sets.
    Parameter:
        dataset (string) : name of dataset
        dir_dataset (string) : path to the dataset
        DIR_OUTPUT (string) : path to verify the correct data load (abundant)
    Returns:
        tr_img, tr_mask, val_img, val_mask, ts_img, val_ind (array) : arrays of loaded, augmented and split images/masks

    """

    print("Starting data load!")
    dataset_dict = dataset

    if dataset == "bagls":
        test_scenario = True

    train_img = (
        dir_dataset / dataset / dict_ds[dataset_dict]["train_img"][0],
        dict_ds[dataset_dict]["train_img"][1],
    )
    train_mask = (
        dir_dataset / dataset / dict_ds[dataset_dict]["train_mask"][0],
        dict_ds[dataset_dict]["train_mask"][1],
    )
    train_sample = (
        dir_dataset / dataset / dict_ds[dataset_dict]["train_samples"][0],
        dict_ds[dataset_dict]["train_samples"][1],
    )

    if test_scenario:
        tr_samples = 25000
    else:
        tr_samples = len(fnmatch.filter(os.listdir(train_sample[0]), train_sample[1]))

    indices_samples = []
    img = []
    mask = []
    item = 0

    for i in trange(tr_samples):
        mask_file = cv2.imread(f"{train_mask[0]}/{item}{train_mask[1]}", 0)
        mask_file = cv2.resize(mask_file, dsize=(224, 224))
        if np.max(mask_file) != 0:
            # only normalizing to 0 .. num_classes
            if np.max(mask_file) > 10:
                mask_file = mask_file / 255.0
            else:
                mask_file = mask_file / 1.0
            img_file = cv2.imread(f"{train_img[0]}/{item}{train_img[1]}", 0)
            img_file = cv2.resize(img_file, dsize=(224, 224))
            img_file = img_file / 255.0
            img.append(img_file)
            mask.append(mask_file)
            indices_samples.append(item)
        item += 1

    img_arr = np.array([tf.expand_dims(i, -1) for i in img])
    mask_arr = np.array([tf.expand_dims(i, -1) for i in mask])

    # Setting size of images to compute new after removing non informational images
    tr_samples = len(indices_samples)
    # Shuffle images and masks in same order
    np.random.seed(42)
    indices = np.arange(tr_samples)
    rand = indices
    np.random.shuffle(rand)
    img_arr = img_arr[rand]
    mask_arr = mask_arr[rand]

    ts_img = []
    # Dataset split if not test set not explicitly given
    if dict_ds[dataset_dict]["test_img"] == []:
        tr_ind = indices[0 : int(0.7 * tr_samples)]
        val_ind = indices[int(0.7 * tr_samples) : int(0.8 * tr_samples)]
        ts_ind = indices[int(0.8 * tr_samples) :]

        tr_img, tr_mask = img_arr[tr_ind], mask_arr[tr_ind]
        val_img, val_mask = img_arr[val_ind], mask_arr[val_ind]
        ts_img, ts_mask = img_arr[ts_ind], mask_arr[ts_ind]
    else:
        tr_ind = indices[0 : int(0.9 * tr_samples)]
        val_ind = indices[int(0.9 * tr_samples) :]

        tr_img, tr_mask = img_arr[tr_ind], mask_arr[tr_ind]
        val_img, val_mask = img_arr[val_ind], mask_arr[val_ind]

        test_path = (
            dir_dataset / dataset / dict_ds[dataset_dict]["test_img"][0],
            dict_ds[dataset_dict]["test_img"][1],
        )

        img = []
        for ts in os.listdir(test_path[0]):
            ts_file = cv2.imread(f"{test_path[0]}/{ts}", 0)
            ts_file = cv2.resize(ts_file, dsize=(224, 224))
            img.append(np.array(ts_file) / 255.0)
        ts_img = np.array([tf.expand_dims(i, -1) for i in img])

    # Add augmentation to train dataset
    transform = A.Compose([A.HorizontalFlip(p=0.5), A.Rotate(limit=10)])
    x = []
    x1 = []
    for i in range(len(tr_img)):
        transformed_train = transform(image=tr_img[i], mask=tr_mask[i])
        x.append(np.array(transformed_train["image"]))
        x1.append(np.array(transformed_train["mask"]))
    tr_img = np.array(x)
    tr_mask = np.array(x1)

    # One hot encoding

    if 2 in np.unique(tr_mask.ravel()):
        tr_mask = (tr_mask == 1, tr_mask == 2)
        val_mask = (val_mask == 1, val_mask == 2)

    print(len(tr_img), len(tr_mask), len(val_img), len(val_mask), len(ts_img))
    print("Dataload ended")
    return tr_img, tr_mask, val_img, val_mask, ts_img, val_ind


def main(dataset, dir_dataset, batch_size, epochs, DIR_OUTPUT):
    """Method to rearrange arrays as tensors.
    Parameter:
        dataset (string) : name of dataset
        dir_dataset (string) : path to images/masks of dataset
        batch_size (int) : size of batch
        epochs (int) : number of epochs
        DIR_OUTPUT (string) : path to verify the correct data load (abundant)
    Return:
        tr_gen, val_gen, ts_gen (tf.tensor) : images and masks arranged as tensor
        len(tr_img), len(val_img), len(ts_img) (int) : number of samples
    """

    tr_img, tr_mask, val_img, val_mask, ts_img, val_idx = data_generator(
        dataset, dir_dataset, DIR_OUTPUT
    )

    tr_gen = (
        tf.data.Dataset.from_tensor_slices((tr_img, tr_mask))
        .batch(batch_size, drop_remainder=True)
        .repeat(epochs)
        .prefetch(20)
    )

    val_gen = tf.data.Dataset.from_tensor_slices((val_img, val_mask)).batch(
        batch_size, drop_remainder=True
    )

    ts_gen = tf.data.Dataset.from_tensor_slices(ts_img)

    return tr_gen, val_gen, ts_gen, len(tr_img), len(val_img), len(ts_img)
