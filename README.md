# Transfer Learning with VGG19 on Caltech-101 Dataset

## Project Overview
This project demonstrates the use of transfer learning by leveraging the **VGG19 architecture pretrained on ImageNet** for multi-class image classification. The classification layer of VGG19 has been modified to handle 5 output classes from the **Caltech-101** dataset. The project focuses on retraining only the classification layer and analyzing the performance on training, validation, and test data. Additionally, neuron activations are visualized for deeper understanding, and hyper-parameter tuning is performed to improve accuracy.

## Dataset
- **Dataset**: A subset of the **Caltech-101** dataset, pre-divided into training, validation, and test sets.
- **Image Sizes**: All images are resized to **224x224 pixels** to match the input size required for VGG19.

## Tasks

1. **Modify VGG19 Model**:
   - The final classification layer is modified to have 5 output nodes corresponding to the 5 selected classes from the Caltech-101 dataset.
   - Only the classification layer is retrained; the rest of the network remains frozen to leverage the pretrained weights.
   
2. **Accuracy and Confusion Matrix**:
   - Observe the classification accuracy for training, validation, and test sets.
   - Compare the performance with the results obtained from a previous task (Task 2).
   - Present confusion matrices to visualize class-wise performance.
   
3. **Neuron Activation Visualization**:
   - For one image from each class, pass it through the CNN and identify the neuron in the last convolutional layer that is maximally activated.
   - Trace back to the specific patch in the image that causes this neuron to fire, and visualize these patches.
   
4. **Hyper-parameter Tuning** (Bonus):
   - Experiment with different hyper-parameters (learning rates, optimizers, etc.) to improve the overall accuracy of the model.

## Tools & Technologies
- **PyTorch**: Deep Learning framework used to implement the VGG19 model, modify the layers, and retrain the classification head.
- **VGG19**: Pretrained on **ImageNet**, modified for 5-class classification.
- **Confusion Matrix**: For performance comparison across training, validation, and test sets.
- **Image Visualization**: For visualizing neuron activations and patches.

## Key Highlights
- Modified the VGG19 model's classification layer to output 5 classes and retrained only this layer.
- Achieved competitive accuracy on the Caltech-101 subset using transfer learning.
- Visualized image patches that maximally activate neurons in the last convolutional layer.
- Explored various hyper-parameter tuning techniques to optimize classification accuracy.
