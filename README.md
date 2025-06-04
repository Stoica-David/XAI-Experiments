# XAI-Experiments

This project is structured into two main experiments, "First Experiment" and "Second Experiment," each with its own set of scripts designed for specific tasks. This README provides a step-by-step guide on how to execute each experiment successfully.

## First Experiment

The "First Experiment" focuses on various computations related to feature attribution, segmentation, and evaluation. Follow the steps below in the specified order to run this experiment.

**Execution Order for First Experiment:**

1.  **Generate Segmentation Masks:**
    * Run the script: `compute_segmentation_mask.py`
    * *Purpose:* This script is crucial as it generates the necessary segmentation masks that will be used by subsequent scripts for feature attribution.

2.  **Create Feature Attribution Heatmaps:**
    * Execute the following scripts sequentially to generate heatmaps for different attribution methods:
        * `compute_feature_attribution_gradCAM.py` (assuming `gradCAM` is implied, as specific files like `compute_feature_attribution_max-pooling.py`, `compute_feature_attribution_PCA-fusion.py`, `compute_feature_attribution_simple-average.py`, `compute_feature_attribution_weighted-average.py` are present, please choose the relevant ones based on your methodology for GradCAM, GradCAM++, and ScoreCAM).
        * `compute_feature_attribution_gradCAM++.py` (assuming a script for GradCAM++ exists, similar to the above).
        * `compute_feature_attribution_scoreCAM.py` (assuming a script for ScoreCAM exists, similar to the above).
    * *Purpose:* These scripts generate visual representations (heatmaps) of feature importance based on different attribution techniques.

3.  **Compute IoU and Dice Scores:**
    * Run the script: `compute_IoU_Dice_scores.py`
    * *Purpose:* This script calculates Intersection over Union (IoU) and Dice coefficients, which are common metrics for evaluating the accuracy of segmentation and attribution.

4.  **Perform Insertion and Deletion Tests:**
    * Execute the following scripts:
        * `run_insertion_tests.py`
        * `run_deletion_tests.py`
    * *Purpose:* These tests assess the robustness and fidelity of the attribution methods by measuring how predictions change with insertions or deletions of important features.

5.  **Compute Average Insertion and Deletion Curves:**
    * Finally, run these scripts to summarize the test results:
        * `compute_average_insertion_curve.py`
        * `compute_average_deletion_curve.py`
    * *Purpose:* These scripts aggregate the results from the insertion and deletion tests to provide average performance curves.

## Second Experiment

The "Second Experiment" focuses on preprocessing data and subsequently running training processes.

**Execution Order for Second Experiment:**

1.  **Run Preprocessing Algorithms:**
    * Navigate into the `Preprocessing Scripts` directory.
    * Execute each of the following scripts in their listed numerical order:
        * `1) cropped_background.py`
        * `2) blurred_background.py`
        * `3) blacked-out_background.py`
        * `4) CAM_augmentation.py`
        * `5) CAM_deletion.py`
        * `6) gradCAM_blur.py`
        * `7) gradCAM++_blur.py`
        * `8) scoreCAM_blur.py`
    * *Purpose:* These scripts perform various data transformations and augmentations necessary for preparing the dataset for training.

2.  **Run Trainings:**
    * After all preprocessing steps are completed, run the main training script: `run_trainings.py`
    * *Purpose:* This script initiates the model training process using the preprocessed data.

## Requirements

Before running any scripts, ensure all necessary dependencies are installed.

* Install required packages by running:
    ```bash
    pip install -r requirements.txt
    ```

**Note:** Ensure you have the correct Python environment set up and all paths are correctly configured for the scripts to locate necessary data and save outputs.
