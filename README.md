# Lab2-CNN-VIT

---

# Deep Learning Lab: Computer Vision with PyTorch

In this lab, I worked on understanding and implementing various neural network architectures using the PyTorch library for computer vision tasks. I started by setting up a Convolutional Neural Network (CNN) to classify the MNIST dataset. This involved defining layers such as convolutional, pooling, and fully connected layers, and tuning hyperparameters like kernel size, padding, stride, and optimizer choice.

Additionally, I explored a more advanced architecture, Faster R-CNN, and compared its performance with the CNN model. I also fine-tuned pre-trained models like VGG16 and AlexNet on the MNIST dataset and evaluated their performance against the CNN and Faster R-CNN models.

Moving on to Part 2 of the lab, I implemented a Vision Transformer (ViT) from scratch for MNIST classification following a provided tutorial. I analyzed the results obtained from the ViT model and compared them with the results from Part 1, specifically with the CNN and Faster R-CNN models.

Overall, this lab allowed me to gain practical experience in building, training, and evaluating neural network architectures for computer vision tasks. It helped me understand the strengths and weaknesses of different models and how they perform on real-world datasets like MNIST.

## Directory Structure

```
.
├── data/                                       # Folder for storing datasets
│   └── mnist/                                  # MNIST dataset
├── src/                                        # Source code for different models
│   ├── cnn_classifier.ipynb                    # Implementation of CNN classifier
│   └── vision_transformer-mnist.ipynb          # Utility functions
├── LICENSE                                     # License information
└── README.md                                   # Instructions and overview
```

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- Matplotlib

## Instructions



1. Install the dependencies:

```
pip install -r requirements.txt
```

2. Download the MNIST dataset and place it in the `data/mnist/` directory.

3. Run the desired model script:

```
python src/cnn_classifier.ipynb 
python src/vision_transformer-mnist.ipynb
```

4. Follow the prompts to train the model and evaluate its performance.


## Acknowledgements

- The MNIST dataset: [Link](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- PyTorch documentation: [Link](https://pytorch.org/docs/stable/index.html)

## Contact Information

For any questions or issues, please contact mohammedelbakkalielammari@gmail.

---
