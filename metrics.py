import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.experimental.numpy as tnp
import monai


def dice_coefficient(y_true, y_pred, preprocessing=False):
    """
    Dice coefficient.
        Parameter:
            y_true (array) : array of ground truth masks
            y_pred (array) : array of predicted masks

        Returns:
            dsc (float) : dice similarity coefficient
    """
    NUM_CLASSES = y_pred.shape[0]
    smooth = 1e-7
    y_true_f = K.flatten(K.cast(y_true, "float32"))
    y_pred_f = K.flatten(K.cast(y_pred, "float32"))

    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2.0 * intersect / (denom + smooth)))


def dice_coef_loss(y_true, y_pred):
    """
    Dice loss to be minimized during training.
        Parameter:
            y_true (array) : array of ground truth masks
            y_pred (array) : array of predicted masks

        Returns:
            dsc_loss (float) : 1 - dice similarity coefficient
    """
    return 1 - dice_coefficient(y_true, y_pred)


def get_flat(reference, test):
    """Align masks in one array and create binary representations.
    Parameter:
        reference (array) : array of ground truth masks
        test (array) : array of predicted masks

    Returns:
        gt_img_arr (float) : 1d array of aligned ground truth masks
        pred_img_arr (float) : 1d array of aligned predicted masks
    """
    NUM_CLASSES = test.shape[-1]
    if NUM_CLASSES > 1:
        gt_img_arr = np.vstack((reference[0, ...], reference[1, ...]))
        pred_img_arr = np.vstack((test[0, ...], test[1, ...]))
    else:
        gt_img_arr = reference
        pred_img_arr = test

    gt_img_arr = np.array(
        [tf.where(gt_img_arr[i] > 0.5, 1, 0) for i in range(gt_img_arr.shape[0])]
    )
    pred_img_arr = np.array(
        [tf.where(pred_img_arr[i] > 0.5, 1, 0) for i in range(pred_img_arr.shape[0])]
    )
    return gt_img_arr, pred_img_arr


def draw_bb(image):
    """Draws a minimal rectangle based on the limits of non-zero pixels.
    Parameter:
        image (array) : mask image to get bounding box from

    Returns:
        ref_image (array) : 2d array with bounding box
    """
    try:
        y_idx = tnp.nonzero(image)[0]
        x_idx = np.nonzero(image)[1]
        ref_img = np.zeros(image.shape)
        cv2.rectangle(
            ref_img,
            (np.min(x_idx), np.min(y_idx)),
            (np.max(x_idx), np.max(y_idx)),
            (1.0, 1.0, 1.0),
            -1,
        )
    except:
        ref_img = np.zeros(image.shape)
    return ref_img


def get_bb(flat_ref, flat_test):
    """Calls the draw_bb to generate the bounding boxes.
    Parameter:
        flat_ref (array) : arrays with ground truth masks
        flat_test (array) : arrays with predicted masks

    Returns:
        bboxes_ref (array) : array with bounding boxes of ground truth mask images
        bboxes_test (array) : array with bounding boxes of predicted mask images
    """
    bboxes_ref = np.array([draw_bb(flat_ref[i]) for i in range(flat_ref.shape[0])])
    bboxes_test = np.array([draw_bb(flat_test[i]) for i in range(flat_ref.shape[0])])
    return bboxes_ref / 1.0, bboxes_test / 1.0


def bb_IoU(y_true, y_pred):
    """Flattens the mask arrays, generates the bounding boxes and calculates the IoU score of the bounding boxes.
    Parameter:
        y_true (array) : array of ground truth masks
        y_pred (array) : array of predicted masks

    Returns:
        iou (float) : IoU score of bounding boxes
    """
    flat_ref, flat_test = get_flat(y_true, y_pred)
    bb_true, bb_pred = get_bb(flat_ref, flat_test)
    return IoU(bb_true, bb_pred, True)


def IoU(y_true, y_pred, preprocessing=False):
    """Calculates the IoU score.
    Parameter:
        y_true (array) : array of ground truth masks
        y_pred (array) : array of predicted masks

    Returns:
        iou (float) : intersection over union score
    """
    DC = dice_coefficient(y_true, y_pred, preprocessing)
    return DC / (2 - DC)


def hd_95_monai(y_true, y_pred):
    """Calculates the hd95 coefficient by using the monai library.
    Parameter:
        y_true (array) : array of ground truth masks
        y_pred (array) : array of predicted masks

    Returns:
        hd95 (float) : hausdorff distance with a percentile of 95
    """
    return monai.metrics.compute_hausdorff_distance(
        np.expand_dims(y_true, axis=(0, 1)),
        np.expand_dims(y_pred, axis=(0, 1)),
        percentile=95.0,
    )
