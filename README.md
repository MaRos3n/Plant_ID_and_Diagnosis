# Plant_ID_and_Diagnosis
project 3

# Plant and Ailment Identification Project

## Project Overview

This project aims to develop a machine learning model capable of identifying different plant species and diagnosing various plant ailments, including diseases and nutrient deficiencies. The project utilizes a dataset of plant images categorized by species and health status, with labels indicating the specific plant and its condition.

## Datasets

The project merges several datasets into a master dataset, containing images of plants with corresponding labels that describe both the plant species and its health condition. The datasets include:

- **Flower Dataset:** Contains images of various flower species with corresponding health status.
- **Leaf Dataset:** Contains images of various leaf species with corresponding health status.
- **Additional Datasets:** Other plant and ailment datasets that have been cleaned and merged into the master dataset.


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

## Training

The dataset is split into training, validation, and test sets:

- **Training Set:** Used to train the model.
- **Validation Set:** Used to tune hyperparameters and avoid overfitting.
- **Test Set:** Used to evaluate the model's performance on unseen data.

The model is trained for 10 epochs, with the following configurations:

- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy

## Fine-Tuning

After initial training, the model is fine-tuned by unfreezing the top layers of the VGG16 model and retraining it with a lower learning rate. This improves the model's accuracy and generalization.

## Evaluation

The model is evaluated on the test set, providing metrics such as loss and accuracy. Misclassified images are identified and analyzed to improve the model further.

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

To install the required dependencies, run:

```bash
pip install tensorflow keras numpy matplotlib scikit-learn pandas


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

# License

This `README.md` provides an overview of the project, details about the dataset and model architecture, instructions on how to run the project, and future work plans. Feel free to adjust the details to match your project's specifics and goals.




# Plant and Ailment Identification Project

## Project Overview

This project aims to develop a machine learning model capable of identifying different plant species and diagnosing various plant ailments, including diseases and nutrient deficiencies. The project leverages advanced machine learning methodologies, including transformer models and natural language processing (NLP) techniques, along with a new technology not covered in class, to solve this problem. The ultimate goal is to create a reliable model that can be further developed into a mobile application for real-time plant identification and diagnosis.

## Project Specifications

### Problem Identification

The core problem addressed by this project is the accurate identification of plant species and the diagnosis of their health status based on visual data. This is crucial for timely and effective plant care, particularly in agriculture and horticulture, where early detection of diseases or deficiencies can prevent significant crop loss.

### Dataset

The project utilizes multiple datasets of plant images, each labeled with the plant species and its health condition. These datasets have been merged into a single master dataset containing thousands of images across various plant species and health statuses. The merged dataset ensures sufficient data for training and validating the machine learning model.

### Model Implementation

The model implementation process includes the following steps:

1. **Data Preprocessing:**
   - Extraction, cleaning, and transformation of image data.
   - Export of cleaned data as CSV files for machine learning.

2. **Model Selection:**
   - **Base Model:** VGG16 pre-trained on ImageNet.
   - **Technologies Used:** TensorFlow, Keras, scikit-learn.
   - **Additional Technology:** [Specify any new technology used, such as PyTorch, GPT-4V, etc.]

3. **Model Training and Fine-Tuning:**
   - The model is trained on the merged dataset, with iterative optimization steps to improve accuracy.
   - The training process is documented with changes tracked in a CSV or directly in the script.

4. **Evaluation:**
   - The model is evaluated using a test set, with metrics like accuracy and loss being recorded.

### Model Optimization

- **Optimization Process:** 
  - The model was iteratively fine-tuned by adjusting hyperparameters and unfreezing top layers of the pre-trained model.
  - Performance changes were documented, and the final model's performance was reported.

- **Final Model Performance:**
  - The final model achieved a validation accuracy of 73.6% and a test accuracy of 80.6%, showing good generalization to unseen data.

## GitHub Documentation

The GitHub repository for this project is organized with the following considerations:

- **Repository Structure:**
  - Unnecessary files and folders have been removed.
  - An appropriate `.gitignore` file is in use to keep the repository clean.

- **README Customization:**
  - This README file provides a polished presentation of the project, including all necessary information for understanding and replicating the project.

## Presentation Requirements

The final presentation will cover the following points:

1. **Executive Summary:**
   - Overview of the project goals, including the identification of plants and diagnosis of ailments.

2. **Data Collection and Cleanup:**
   - Description of how the datasets were collected, cleaned, and merged.
   - Exploration processes and any challenges faced during data preparation.

3. **Approach:**
   - A detailed explanation of the model selection, training, and optimization processes.
   - Rationale behind the choice of VGG16 and the additional technology used.

4. **Future Development:**
   - Plans for future work, including the development of a mobile application for real-time plant identification.
   - Potential research directions if more time or resources were available.

5. **Results and Conclusions:**
   - The final model's performance, key insights gained, and conclusions drawn from the analysis.

6. **Slides:**
   - The presentation slides will be visually clean, professional, and effectively demonstrate the project.

## How to Run the Project

1. **Merge Datasets:** 
   - Run the script to merge individual datasets into the `master_dataset`.

2. **Train the Model:** 
   - Use the provided script to train the VGG16 model on the merged dataset.

3. **Fine-Tune the Model:** 
   - Unfreeze the top layers and fine-tune the model for better accuracy.

4. **Evaluate the Model:** 
   - Run the evaluation script to assess the model's performance on the test set.

5. **Analyze Misclassifications:** 
   - Identify and analyze misclassified images to further improve the model.

## Dependencies

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pandas

To install the required dependencies, run:

```bash
pip install tensorflow keras numpy matplotlib scikit-learn pandas


Training Data Generator:
Data Augmentation: Applied only to the training set to help the model generalize better.
rotation_range: Randomly rotate images by up to 20 degrees.
zoom_range: Randomly zoom inside pictures.
width_shift_range & height_shift_range: Randomly translate images horizontally and vertically.
horizontal_flip: Randomly flip images horizontally.
fill_mode: How to fill new pixels when an image is rotated or shifted.
rescale=1./255: Normalize pixel values between 0 and 1.

Validation and Test Data Generators:
No Data Augmentation: We want to evaluate the model on unaltered data.
shuffle=False: Maintains the order, which is useful for accurate evaluation and predictions.
flow_from_dataframe Parameters:
dataframe: The respective DataFrame (train_df, valid_df, test_df).
directory: The directory where images are stored.
x_col & y_col: Column names in the DataFrame for filenames and labels.
target_size: Resizes all images to the specified dimensions.
batch_size: Number of samples per batch.
class_mode='categorical': Suitable for multi-class classification.
