# Brain Tumor MRI Image Classification

Deep learning classifier that labels a brain MRI as Glioma, Meningioma, Pituitary tumor or No Tumor.
A custom CNN is compared with three pretrained models (MobileNetV2, ResNet50, EfficientNetB0), and the best one is served in a Streamlit app.

> Educational project. Not a medical device. Do not use for real diagnosis.

## Dataset
Labeled MRI Brain Tumor Dataset (Roboflow, CC BY 4.0): 2,443 images (1,695 train / 502 validation / 246 test), four classes.

## Method
1. Data checks: class balance, image sizes, duplicates, leakage.
2. Preprocessing: resize to 224x224, model-specific normalization.
3. Augmentation (training only): flips, rotation, zoom, shift, brightness.
4. Models: custom CNN and 3 transfer-learning models (head training, then fine-tuning).
5. Training: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, class weights.
6. Evaluation on the untouched test set: accuracy, precision, recall, F1, confusion matrix, Grad-CAM.
7. Best model chosen on validation macro-F1, not on test.

## Results
| Model | Test Acc | Precision | Recall | F1 |
|---|---|---|---|---|
| Custom CNN | | | | |
| MobileNetV2 | | | | |
| ResNet50 | | | | |
| EfficientNetB0 | | | | |

## Run the app
Needs the `models/` folder created by the training notebook.

## Limitations
Small test set (246 images); one public dataset; not validated for clinical use.
