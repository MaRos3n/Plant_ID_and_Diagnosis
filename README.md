# Plant_ID_and_Diagnosis
project 3

# Plant and Ailment Identification Project

## Project Overview

This project aims to develop a machine learning model capable of identifying different plant species and diagnosing various plant diseases. The project utilizes a dataset of plant images categorized by species and health status, with labels indicating the specific plant and its condition.

## Datasets

The project merges several datasets into a master dataset, containing images of plants with corresponding labels that describe both the plant species and its health condition. The datasets include:

- **Flower Dataset:**  
  - 102 Oxford Flowers  
  - Compiled at the University of Oxford  
  - 102 species of flowers found in the UK  
  - 6,553 images

- **Leaf Dataset:**  
  - Leaf images  
  - Compiled at Shri Mata Vaishno Devi University  
  - 11 plant species in healthy and diseased states  
  - 4,503 images

- **Corn Datasets:**  
  - Corn diseases  
  - Combination of two other datasets (“PlantVillage” and “PlantDoc”)  
  - 4 states of health  
  - 4,189 images

- **Major Crops Datasets:**  
  - Major Crop Diseases  
  - First published in the “Computers & Electrical Engineering” journal on ScienceDirect  
  - 14 plant species  
  - 18 different states of health  
  - 61,487 images

The `combined_labels.csv` file contains the filenames of the images and their corresponding labels, which indicate the plant species and its health status.


## Model Architecture

The project uses the **VGG16** pre-trained model for supervised learning. The VGG16 model is fine-tuned to classify the images into their respective plant species and health conditions.

### Supervised Learning

- **Base Model:** VGG16 (pre-trained on ImageNet)
- **Custom Layers:**
  - Flatten layer
  - Dense layer with 1024 units and ReLU activation
  - Output layer with softmax activation (based on the number of unique classes)

### Unsupervised Learning

After the model is trained, unsupervised learning techniques such as PCA and K-Means clustering are applied to the extracted features to further analyze the data.

## Albumentations

### Overview

Albumentations is a fast and flexible image augmentation library that significantly enhances the diversity of training data. In this project, Albumentations is used to apply various transformations to the images, making the model more robust and better at generalizing to unseen data.

### Key Augmentations Applied

- **RandomRotate90:** Randomly rotates images by 90 degrees.
- **Flip:** Randomly flips images horizontally or vertically.
- **Transpose:** Swaps the axes of the image, effectively rotating it by 90 degrees and flipping it.
- **Noise Additions:** Adds either Gaussian noise or ISO noise to simulate different lighting conditions or sensor noise.
- **Blurring:** Applies various blurring techniques, including motion blur, median blur, and simple blur, to simulate out-of-focus or motion-blurred images.
- **Geometric Distortions:** Includes shift, scale, and rotate transformations, as well as optical distortion and grid distortion, to simulate real-world image variations.
- **Color Adjustments:** Alters hue, saturation, brightness, and contrast to simulate different environmental conditions.

### Usage

In the training process, Albumentations is integrated into a custom data generator that applies these augmentations to the training images in real-time. This ensures that each epoch sees a slightly different version of the data, improving the model’s ability to generalize.

## Training

The dataset is split into training, validation, and test sets:

- **Training Set:** Used to train the model.
- **Validation Set:** Used to tune hyperparameters and avoid overfitting.
- **Test Set:** Used to evaluate the model's performance on unseen data.

The model is trained for 7 epochs, with the following configurations:

- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy

## Fine-Tuning

After initial training, the model is fine-tuned by unfreezing the top layers of the VGG16 model and retraining it with a lower learning rate. 

This improves the model's accuracy and generalization.

## Evaluation

The model is evaluated on the test set, providing metrics such as loss and accuracy. 

Misclassified images are identified and analyzed to improve the model further.

## Future Work

The following features are planned for future development:

- **Deployment as a Mobile App:** Allow users to take pictures of plants using their mobile phones, identify the plant species, and diagnose its health status.
- **Integration of Additional Datasets:** Continuously improve the model by integrating more datasets.
- **Further Fine-Tuning:** Explore other pre-trained models and fine-tuning techniques to enhance accuracy.

## How to Run the Project

1. **Merge Datasets:** Run the script to merge individual datasets into the `master_dataset`.
2. **Train the Model:** Use the provided script to train the VGG16 model on the merged dataset.
3. **Fine-Tune the Model:** Unfreeze the top layers and fine-tune the model.
4. **Evaluate the Model:** Run the evaluation script to assess the model's performance on the test set.
5. **Analyze Misclassifications:** Use the misclassification analysis script to identify and understand errors in the model's predictions.

## Dependencies

- TensorFlow  
- Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  
- Pandas  
- Albumentations

Citations: 

Image Datasets:

"testflower"; 102_OxfordFlowers = https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset/data?select=dataset

"leaf_images"; (healthy & diseased) = https://www.kaggle.com/datasets/meetnagadia/collection-of-different-category-of-leaf-images

"corn_images"; (healthy & diseased) = https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset

"major_crop_leaf_images"; (healthy & diseased) = https://www.kaggle.com/datasets/rajibdpi/plant-disease-dataset

"OLID_images" = https://www.kaggle.com/datasets/raiaone/olid-i

"roboflow_plant_id" = https://universe.roboflow.com/muhammadrazi99/plant-recognition-2/dataset/20

"banana_nutrient_deficientcies" = https://data.mendeley.com/datasets/7vpdrbdkd4/1

    p, sunitha (2022), “Images of Nutrient Deficient Banana Plant Leaves”, Mendeley Data, V1, doi: 10.17632/7vpdrbdkd4.1

"crop_pest_and_disease" = https://data.mendeley.com/datasets/bwh3zbpkpv/1

    Mensah Kwabena, Patrick; Akoto-Adjepong, Vivian; Adu, Kwabena; Abra Ayidzoe, Mighty; Asare Bediako, Elvis; Nyarko-Boateng, Owusu; Boateng, Samuel; Fobi Donkor, Esther; Umar Bawah, Faiza; Songose Awarayi, Nicodemus; Nimbe, Peter; Kofi Nti, Isaac; Abdulai, Muntala; Roger Adjei, Remember; Opoku, Michael (2023), “Dataset for Crop Pest and Disease Detection”, Mendeley Data, V1, doi: 10.17632/bwh3zbpkpv.1

"ML_model_unseen_images" :
   - "tomato_late_blight" = https://growingsmallfarms.ces.ncsu.edu/2015/07/tomato-late-blight-detected-in-western-north-carolina/

   - "corn_rust" = https://smallgrains.ces.ncsu.edu/smallgrains-disease-identification-management/

   - "gardenia_healthy" = https://plants.ces.ncsu.edu/plants/gardenia/

   - "squah_powdery_mildew" = https://halifax.ces.ncsu.edu/2023/04/powdery-mildew-2023/

   - "apple_scab" = https://apples.ces.ncsu.edu/2023/07/apple-disease-update-week-of-july-17-2023/

- Norman, Ryan. "UNC AI Bootcamp."
- OpenAI. ChatGPT-4. OpenAI, 2024.