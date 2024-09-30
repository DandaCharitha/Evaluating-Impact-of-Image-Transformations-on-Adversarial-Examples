# Evaluating Impact of Image Transformations on Adversarial Examples

# Introduction

Deep learning has significantly advanced image classification across various industries, from autonomous navigation systems to healthcare diagnostics. However, the integration of these technologies raises serious privacy concerns, particularly regarding the mishandling of image data. For instance, in self-driving cars, employing adversarial training with manipulated images can enhance the model's object detection capabilities in challenging real-world scenarios. Similarly, in cybersecurity, training models with adversarial examples can bolster their ability to identify novel threats not represented in traditional datasets.

The focus on developing defenses against adversarial attacks has grown in the context of deep learning image classification tasks. These attacks introduce subtle perturbations that compromise the integrity of images, hindering deep neural networks (DNNs) from accurately classifying content. In this study, we examine three types of adversarial attacks—Fast Gradient Sign Method (FGSM), Randomized FGSM (RFGSM), and Projected Gradient Descent (PGD)—which exploit the vulnerabilities of DNNs by applying minimal noise to input images.

Using pre-trained models like ResNet50 and DenseNet121, known for their depth and feature extraction capabilities, this research aims to enhance defenses against adversarial attacks. By utilizing these established architectures, we strive to improve the reliability of image classification systems against emerging threats.

This project highlights the significance of gradient-based adversarial attacks, emphasizing that even minor alterations in input data can severely threaten model integrity. To counter this, we explore a novel defense strategy involving four image transformation techniques: Affine, Gaussian, Median, and Bilateral blurring. The main goal is to assess the robustness of DNNs against adversarial attacks while minimizing the need for additional training.

The findings reveal that applying these transformations not only enhances model accuracy in the face of adversarial perturbations but also streamlines the development process by eliminating the need for retraining with adversarial examples. This innovative approach promises to enable faster deployment of robust deep learning models, ensuring more reliable and trustworthy image classification systems for real-world applications.

# DataSet

The dataset used in this project is the ImageNet validation dataset, a subset of the larger ImageNet 2012-2017 dataset, which is an essential benchmark for computer vision tasks. ImageNet is a large-scale hierarchical image database that provides over 14 million images, each labeled to one of 1,000 object categories. The validation subset consists of 50,000 images, equally distributed across these 1,000 categories (50 images per category). These categories include a wide range of objects such as animals, vehicles, and everyday items, ensuring diversity and richness for testing complex AI models.(https://image-net.org/challenges/LSVRC/2012/index.php)

This dataset is widely used for evaluating the accuracy and generalization capabilities of deep learning models in image classification tasks. For this project, we utilized the 50,000-image validation set to assess the performance of two state-of-the-art convolutional neural network models, ResNet50 and DenseNet121, by measuring their ability to accurately classify the images across the given categories.

# Models Used

This project leverages two popular pre-trained models from the Torchvision library: ResNet50 and DenseNet121.

## ResNet50
ResNet50 is part of the ResNet family and is designed to address the vanishing gradient problem in deep networks using residual connections. With 50 layers, it captures complex features efficiently, promoting smoother gradient flow. Pre-trained on ImageNet, ResNet50 achieves an accuracy of 76.1%, making it highly effective for image classification tasks.

## DenseNet121
DenseNet121 introduces dense connectivity, where each layer receives inputs from all preceding layers. This design encourages feature reuse and improves gradient flow, making the model both efficient and effective at learning visual patterns. Pre-trained on ImageNet, DenseNet121 achieves an accuracy of 74.43%, excelling in environments with limited resources.

# Adversarial Attack Methods
In this project, the adversarial attack techniques are used to test the robustness of image classification models. These methods generate perturbed images designed to mislead machine learning models.

## Fast Gradient Sign Method (FGSM)
FGSM creates adversarial examples by adding a small perturbation to the original image in the direction of the gradient of the loss function. This method is efficient, using a hyperparameter ϵ (epsilon) to control the strength of the perturbation. FGSM exploits the model’s gradient to craft small, human-imperceptible changes that cause the model to misclassify the image.

## Randomized Fast Gradient Sign Method (RFGSM)
RFGSM introduces randomness into the FGSM process. It adds a random perturbation vector before applying the gradient-based noise, improving robustness. The strength of both the initial randomness and the final perturbation are controlled by hyperparameters α (alpha) and ϵ (epsilon) respectively.

## Projected Gradient Descent (PGD)
PGD is an iterative attack method that refines the perturbations over multiple steps. It ensures the perturbed image stays within a specified neighborhood of the original image, controlled by ϵ (epsilon), while maximizing the model’s loss through small, repeated updates. This makes PGD highly effective at finding adversarial examples.

# Image Transformation Techniques
To enhance model robustness and reduce adversarial noise, the four image transformation techniques were applied : Gaussian Blur, Median Blur, Bilateral Blur, and Affine Transformation.

## Gaussian Blur
Gaussian Blur smooths an image by averaging pixel intensities using a Gaussian function, reducing noise while preserving edges. This technique is useful for removing irrelevant details, improving object detection and feature extraction. It applies a kernel of either 5x5 or 3x3 size, with minimal accuracy improvement when downsizing the kernel.

## Median Blur
Median Blur replaces each pixel’s intensity with the median value of neighboring pixels, effectively reducing noise while retaining important image details. This technique softens fine textures but preserves key features, making it useful for cleaning up images without losing critical information.

## Bilateral Blur
Bilateral Blur applies filtering based on both spatial proximity and pixel intensity, allowing it to reduce noise while preserving edges. This method is effective for enhancing image quality and ensuring that essential structures, such as object edges, remain intact despite blurring.

## Affine Transformation
Affine Transformation alters images geometrically through translation, rotation, scaling, and shearing, using a transformation matrix. It preserves the overall structure and shape of objects, making it suitable for tasks like object detection and image registration while allowing adjustments to image positioning and orientation.

# Project Results

## Model Performance:
ResNet50 and DenseNet121 were tested on 50,000 images from the ImageNet validation dataset. ResNet50 correctly predicted 38,065 images, while DenseNet121 predicted 37,217, with 34,768 images accurately predicted by both models.

## Adversarial Image Generation:
Adversarial images were generated using FGSM, RFGSM, and PGD attacks:

ResNet50: FGSM created 29,820 adversarial images, RFGSM created 24,784.
DenseNet121: FGSM generated 31,588, RFGSM created 19,840.
Common Images: FGSM generated 34,768 adversarial images, RFGSM produced 21,766.

## Transformation Techniques:
Four transformations (Gaussian Blur, Median Blur, Bilateral Blur, and Affine Blur) were applied to adversarial images for defense:

ResNet50: FGSM-perturbed images had the highest accuracy, with Gaussian Blur at 83.1% and Affine Blur at 85.2%. RFGSM and PGD resulted in lower accuracy.
DenseNet121: FGSM also led in accuracy (62.12% with Median Blur), while RFGSM and PGD showed lower performance across all transformations.

## Transferability:
When adversarial images from ResNet50 were tested on DenseNet121, Gaussian Blur (72.61%) and Median Blur (75.91%) proved effective for FGSM-affected images. Affine Blur (71.1%) performed better for RFGSM images.

## Parameter Tuning:
Adjustments in kernel sizes and parameters in Gaussian and Median Blur led to small accuracy improvements (2% in some cases), but further increases in kernel size or α/β parameters often reduced accuracy.

## Top-3 Accuracy:
Top-3 accuracy significantly improved recovery rates:

ResNet50 with FGSM: Affine Blur showed a 14% increase.
DenseNet121 with FGSM: Median Blur recovered 81.63%, with a 24.5% improvement in top-3 accuracy.

## PGD Attack Results:
Affine Blur was the most effective against PGD attacks:
ResNet50: 93.13%
DenseNet121: 81.10%
Combined Model: 87.63%

These results show that FGSM attacks were more resilient to transformations than RFGSM and PGD. Affine and Median Blur proved particularly effective, and model ensembles enhanced recovery rates against adversarial attacks.

# Key Findings

FGSM Attacks: Applying Affine Blur to adversarial images correctly classified by both models recovered 90.3% top-3 accuracy with DenseNet121, showcasing its resilience.
RFGSM Attacks: DenseNet121 achieved 87.94% top-3 accuracy using Affine Blur.
PGD Attacks: Using Affine Blur, ResNet50 recovered 78.97%, while transferring adversarial images between ResNet50 and DenseNet121 models resulted in a 69.53% and 72.44% recovery, respectively. For top-3 accuracy, DenseNet121 recovered 81.10% of PGD-perturbed images.








