# Evolutionary Normalization Optimization Boosts Semantic Segmentation Network Performance

This repository contains the source code for the project to determine how layer-specific normalization methods can influence the segmentation result of an U-Net by using an evolutionary algorithm approach.

## Structure of the Repository

### data_generator.py

Loads the dataset and normalizes the images/masks to 0 to 1 if it is a binary task. If it is a multi-class segmentation, the masks contain the class labels from 0 to num_classes. Returns the dataset split into the tensors train, val and test for further processing.

### main.py

- `main()`: runs the U-Net and the evaluation of the different individuals of a generation over multiple generations and sorting the models for the first and the last generation
- `select()`: Runs the selection and breeding of the new population

### metrics.py

- `dice_coefficient()`: calculates the Dice Similarity Coefficient (DSC)
- `dice_coef_loss()`: calculates the loss based on the DSC
- `get_flat()`: flattens the predicted masks
- `draw_bb()`: draws a minimal rectangle based on the predictions
- `get_bb()`: calls the draw_bb() function
- `bb_IoU()`: calculates the IoU score of the predicted bounding boxes
- `IoU()`: calculates the Intersection over Union score
- `hd_95_monai()`: calculates the Hausdorff Distance 95

### unet.py

- `conv_layer()`: builds one layer of the U-Net with the given settings and can be variable including normalization, activation and filter size.
- `unet()`: builds the U-Net architecture by using 4 Layers for encoding and Decoding by using a normal up-sampling.

### Util Files

To install this project, follow these steps:

- `constants.json`: defines all genes which can be chosen to build one U-Net individual
- `datasets.json`: gives an overview of the datasets and its structure
- `env.yml`: yaml-file to create an environment to run the code

## License

This repository is licensed under the terms of the [MIT License](./LICENSE).

## Citation

Please site the usage of the software as follows:

    Neubig, L., Kist, A.M. (2023). Evolutionary Normalization Optimization Boosts Semantic Segmentation Network Performance. In: Greenspan, H., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2023. MICCAI 2023. Lecture Notes in Computer Science, vol 14223. Springer, Cham. <https://doi.org/10.1007/978-3-031-43901-8_67>
