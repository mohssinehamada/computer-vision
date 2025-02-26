🌸 Flower Classification using ResNet-50

📌 1. Introduction

This project aims to classify images of flowers into 102 different categories using deep learning. We leverage Transfer Learning with a pre-trained ResNet-50 model to efficiently train on the Oxford 102 Flowers Dataset.

Using PyTorch, we download and preprocess the dataset, modify ResNet-50 for multi-class classification, train the model with GPU acceleration, and evaluate its accuracy while visualizing results.

📂 2. Dataset Details

The Oxford 102 Flowers Dataset contains 8,189 images categorized into 102 flower types. Labels are stored in .mat format. The dataset presents challenges due to similarities between flower species and variations in image backgrounds.

🛠 3. Code Breakdown

🔽 Step 1: Download & Extract Dataset

The function download_file(url, filename) checks if a dataset file exists locally. If not, it downloads and saves it. The dataset includes 102flowers.tgz for images, imagelabels.mat for labels, and setid.mat for dataset splits.

📊 Step 2: Define Custom Dataset Class

The FlowersDataset class reads image paths and labels, loads images from .jpg files, and applies transformations like resizing and normalization. It retrieves image labels from imagelabels.mat and maps them accordingly.

🏗 Step 3: Define Image Preprocessing

Images are resized to 224x224 pixels to match ResNet-50 input requirements. Normalization is applied using ImageNet’s mean and standard deviation to standardize the data.

🔥 Step 4: Load Pre-trained ResNet-50 Model

The ResNet-50 model is loaded with pre-trained ImageNet weights. The final fully connected layer is replaced with a new layer configured for 102 classes, and the model is moved to the available GPU or CPU device.

🎯 Step 5: Define Loss & Optimizer

The Cross-Entropy loss function is used for multi-class classification, and the Adam optimizer is selected with a learning rate of 0.0001 to optimize model training.

🎓 Step 6: Train the Model

The training loop runs for a specified number of epochs. The model processes batches of images, computes loss, performs backpropagation, and updates weights using the optimizer. The training loss is printed after each epoch.

✅ Step 7: Evaluate Performance

The evaluate_model() function computes model accuracy by comparing predictions to ground truth labels. Predictions are converted into NumPy arrays for easy analysis.

🎨 Confusion Matrix Visualization

The confusion matrix is plotted using Seaborn and Matplotlib. The matrix helps analyze model performance by displaying correct and incorrect classifications.

🔧 4. How to Run the Code

📌 1. Install Dependencies

Required libraries include PyTorch, Torchvision, Matplotlib, SciPy, and NumPy. These can be installed using pip install torch torchvision matplotlib scipy numpy.

📌 2. Run in Google Colab

Mount Google Drive using drive.mount('/content/drive') to access dataset files efficiently.

📌 3. Execute Notebook Cells

Run all notebook cells sequentially to download and extract the dataset, preprocess data, load ResNet-50, train the model, and evaluate its performance.

📈 5. Future Improvements

Enhancements include implementing data augmentation techniques such as random crops and rotations, tuning hyperparameters like batch size and learning rate, and experimenting with architectures such as EfficientNet for better performance.

📚 6. References

Relevant references include the Oxford 102 Flowers Dataset, ResNet research papers, and PyTorch official documentation.

🚀 Final Thoughts

This documentation provides a complete walkthrough of how to implement flower classification using ResNet-50 and PyTorch. Following these steps ensures successful training and evaluation of the model. Further improvements can be achieved by fine-tuning layers, experimenting with different architectures, or refining data preprocessing techniques.

