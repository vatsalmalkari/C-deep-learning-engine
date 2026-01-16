C++ Deep Learning Engine

"What if we didn't use PyTorch or TensorFlow? 

This project is a fully functional Convolutional Neural Network (CNN) library written entirely in C++ without any machine learning frameworks.

It implements Tensors, Automatic Differentiation (Backpropagation), Convolutional Layers, and Optimizers from scratch to recognize human emotions in real-time.

How to start: Run the Webcam Demo

You don't need to compile any C++ to see the AI in action. I have included a pre-trained model under models/fer_model.bin that you can run immediately using a Python script.

# Prerequisites

You need Python and OpenCV installed:
On terminal:
```
pip install opencv-python numpy

2. Run the Emotion Detector
On terminal in this folder and run:
```
python examples/run_webcam.py

What happens? A window will open, detect your face, and the AI will predict if you are Happy, Sad, Angry, or Neutral. 
This can detect multiple people's faces and their expressions too!

# Training the AI in C++

If you want to see the engine actually learn from data, follow these steps to train your own model.

**Download Data**
Download the FER-2013 dataset (Face Expression Recognition) from Kaggle.

Make a data folder

Extract fer2013.csv and place it inside the data/ folder.

**Compile Engine**
We use g++ to compile the training script. This links all our custom modules (Tensors, Convolution, Loss functions) together.

In your terminal:
```
g++ examples/train.cpp src/*.cpp modules/*.cpp -Iinclude -o train_network -O3 -std=c++17
```

**Start Training**

```
./train_network
```

Output: The loss will decrease and accuracy increases as the model iterates through all images.

Result: After training, it saves a new fer_model.bin file, which captures what it learned.

So how does it work: 
I did not just import a library; I built the library

**Phase 1: The Building Blocks** 

Before recognizing faces - it needed to recognize numbers

Enter the Tensor
The Tensor (src/tensor.cpp): In C++, there is no easy way to handle 3D grids of numbers like for images. I built a Tensor class that wraps a 1D vector but behaves like a multi-dimensional matrix.

Autograd: To learn, a neural network needs to know "how much did I mess up?" I implemented Gradient Descent manually, calculating derivatives for every operation to update the weights.

**Phase 2: The Neural Network Layers**

I replicated the components of a biological brain:

Conv2D (The Eyes): This layer slides a small 3x3 filter over the image to detect edges, curves, and textures.

ReLU (The Activation): A function that decides if a feature is important enough to pass forward (activates the neuron).
It outputs zero for any negative input and the input value itself (unchanged) for any positive input. f(x) = max(0, x)

Pooling (The Focus): Shrinks the image to focus on the most important features and ignore background noise.

Linear/ (The Decision): Takes all the detected features and votes on which emotion it sees.

**Phase 3: The Architecture**
We combined these layers into a pipeline similar to the famous VGG-Net architecture: Input:  **Image -> Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> MaxPool -> Flatten -> Output**

**Phase 4: C++ to Python**
Training: I trained in C++ because it is incredibly fast and efficient for heavy math. Python is an interpreted language while c++ is compiled

Inference: I saved the learned weights the actual model itself to a binary file fer_model.bin.

Application: We use Python for the webcam interface because it's easier to handle video streams, but the logic inside is running the exact same math we defined in C++.

**Note: this is still slower than using tensorflow and pytorch to build a cnn in python as those libraries are also written is c++ but uses specialized GPUs and other hardware to compute large data quickly**