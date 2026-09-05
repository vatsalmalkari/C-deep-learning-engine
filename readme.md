C++ Deep Learning Engine

<img width="1917" height="1080" alt="Image" src="https://github.com/user-attachments/assets/e9dbfaab-2527-44e7-b809-f8e9239cff0b" />

"What if we didn't use PyTorch or TensorFlow?"
This project is a lightweight, fully functional Convolutional Neural Network (CNN) framework 
written entirely from  scratch in C++. It features a custom Automatic Differentiation (Autograd) graph engine, 
an optimized custom Memory Arena Allocator, specialized neural network layers, 
and a  cross-entropy mathematical loss layout—all implemented without standard machine learning libraries.

## How to Start: Run the Webcam Demo
You don't need to wait for a full training cycle to see the AI in action. I have included a pre-trained model under models/fer_model.bin that you can run immediately using Python for camera streaming.
## 1. Prerequisites
Install Python and OpenCV to handle camera feeds:

pip install opencv-python numpy

## 2. Run the Real-Time Emotion Detector
Execute the script from the root directory:

python examples/run_webcam.py

What happens? A camera window will open, detect facial boundaries across multiple people simultaneously, and execute your model's weights to classify expressions into emotions in real-time.
##  Training the AI in C++
Follow these steps to feed raw data into the custom engine and watch it learn from scratch.
## 1. Download Data

Download the FER-2013 dataset (Facial Expression Recognition) from Kaggle.
Create a data/ folder in the project root directory.
Extract and place fer2013.csv directly inside that data/ folder.
(For convenience I have already added it)

## 2. Compile the Engine
We compile using Apple Clang with aggressive -O3 vectorizations and the -ffast-math optimization flag to instruct your CPU hardware to maximize floating-point math efficiency.
Run from the root directory:

clang++ -std=c++17 -O3 -ffast-math -Iinclude examples/train.cpp src/*.cpp -o train_network

## 3. Start Training

./train_network

What happens? The system initializes a 512 MB Transient Memory Arena to lock activations. 
As the loops execute, the loss drops consistently and training/validation accuracy metrics climb, culminating in an exported fer_model.bin payload file.
------------------------------
## How it Works: Building the Blueprint

## Phase 1: The Core Math Foundations
Before recognizing faces, the engine had to handle multi-dimensional grids of numbers and structural tracking:

The Tensor (src/tensor.cpp): Standard C++ vectors are flat and sequential. 
I implemented a robust Tensor abstraction layer that wraps raw memory blocks but executes operations across 3D shapes [Channels, Height, Width].

The Memory Arena (src/arena_allocator.cpp): Deep loops that create millions of hidden forward-pass activations quickly fragment system memory via OS heap malloc/free calls. 

I built a custom Memory Arena that pre-allocates a contiguous block of RAM. Activation tensors borrow and recycle memory slots from this arena, dropping OS allocation overhead down to an incredible 0.1% of runtime.

Autograd Engine (src/tensor.cpp): Implemented backpropagation via a dynamically generated directed acyclic graph (DAG). 

Tensors maintain lambda-closure functions (_gradfn) and parent pointers. When backward() is triggered, the graph runs a reverse topological sort, passing accumulated gradients down to target parameters via compound addition (+=).

## Phase 2: The Neural Network Layers
I replicated the components of a functional vision pipeline:

Conv2D: Slides feature filters across images to capture edge patterns, facial geometry, and textures.

ReLU: An element-wise non-linear activation function (f(x) = max(0, x)) that determines neuron signaling. 

Refactored to operate directly over raw pointers to prevent vector duplication.
Pooling: Executes a local spatial Max-Pooling window (2×2) to downsample dimensions, filtering out baseline noise while concentrating translation-invariant signals.

Flatten: Converts 3D feature arrays into 1D matrices through a zero-allocation, metadata-only reshaping step.

Linear: A fully connected decision layer executing Y = W ⋅ X + B.

Fused Cross-Entropy Loss (src/loss.cpp): Rather than chaining separate layer objects, I fused Softmax and Negative Log-Likelihood together using the Log-Sum-Exp numerical trick. 

This bypasses divisions by small numbers, avoiding exploding/vanishing gradients and ensuring safe, stable backpropagation ($\frac{\partial L}{\partial z_i} = p_i - y_i$).

## Phase 3: Architecture & Execution Pipeline
The layers are configured into a modern deep learning sequential layout:

Input Image (48x48) 
   └── Conv2D ──> ReLU ──> MaxPool (12 Filters)
   └── Conv2D ──> ReLU ──> MaxPool (24 Filters)
   └── Flatten ──> Linear Layer ──> Fused Cross-Entropy (7 Emotion Classes)

To maximize single-core processing capabilities, the execution loop handles Simulated Mini-Batching (aggregating structural loss derivatives across 32 shuffled samples per step), which stabilizes the training trajectory and boosts convergence efficiency.
------------------------------
## Performance Reality Check

C++ vs. Python Frameworks: While this project compiles directly to machine code and utilizes optimized memory arenas, industry-standard frameworks like PyTorch or TensorFlow will execute large datasets faster.

The Secret: Modern Python frameworks are wrappers around low-level C++/CUDA backends that delegate dense matrix operations to parallel threads on massive Graphics Processing Units (GPUs) or dedicated tensor hardware acceleration blocks.

Project Milestone: This engine serves as a single-threaded performance proof-of-concept. It manages to complete 50 full epochs over thousands of records, reaching an impressive 44.2% Training Accuracy and 41.3% Validation Accuracy with absolutely zero memory leaks.

------------------------------
## Project Directory Structure & Component Mapping

├── data/                    # Raw input training resources (User-created)
├── models/                  # Stored pre-trained network payloads
├── include/                 # Header declarations (.h files)
├── src/                     # C++ Implementation source code (.cpp files)
└── examples/                # Application entry points and webcam streaming

## Root Directory

train_network: The final optimized binary executable generated after running the compiler script.
README.md: Project documentation, architectural blueprints, installation guides, and performance write-ups.

## include/ (Header Architecture)
Contains the class signatures, member variables, and function declarations that dictate how components interact.

tensor.h: Declares the core Tensor class, shape/stride metadata vectors, autograd tracking pointers, and arithmetic operator overloads.

conv2d.h, linear.h, relu.h, pooling.h, flatten.h, dropout.h: Header file parameters defining layer variables, weights, biases, and forward() / backward() function contracts.

loss.h: Interface templates for standard error functions and the fused CrossEntropyLoss module.
sgd.h: Structure declarations for the Stochastic Gradient Descent tracking optimizer.
allocator.h / arena_wrapper.h: Signatures for the lightweight, memory-efficient block allocator framework.
fer_loader.h: Header parameters outlining raw dataset parsing routines.

## src/ (The Core Engine Core Code)
Contains the underlying execution logic and mathematical implementations of your system.

tensor.cpp: Implements basic matrix manipulations and the topological autograd sorting loop. The recursive graph 
traversal tracking loops used to backpropagate gradients via add_to_grad().

arena_allocator.cpp: Manages the persistent memory chunks. Contains the alignment protection calculations (align_forward) that safely allocate contiguous arrays without fragmentation or resource leaks.

conv2d.cpp: Executes high-yield 2D sliding matrix convolutions. Includes optimized rows-and-bounds hoisting logic to minimize inner boundary looping math during backpropagation.

linear.cpp: Performs dense fully connected weight projections ($Y = W \cdot X + B$). Includes proper matrix gradient summation loops to prevent parameter overwriting.

loss.cpp: Houses the fused log-sum-exp numerical cross-entropy pipeline. It calculates loss metrics and isolates backpropagation to safe, branchless substractions ($p_i - y_i$).

relu.cpp: Implements element-wise linear modifications. Refactored to map arrays via raw data pointers directly to eliminate temporary vector copies.

pooling.cpp: Executes local dimension downsampling. Uses localized index caching to accurately pass maximum values across layers during backpropagation steps.

dropout.cpp: Leverages a fast inline Xorshift32 PRNG engine to drop activation matrices dynamically without causing lock contentions.

sgd.cpp: Implements learning updates. Iterates across accumulated gradients, subtracting learning rate vectors from model parameter nodes.

## examples/ & System Interfaces
Houses real-world implementations, training configurations, and deployment interfaces.

train.cpp: The primary C++ network training script. It handles dataset splitting, setups optimization layers, shuffles index pools, tracks learning bounds, and controls mini-batch gradient accumulation over 50 epochs.

run_webcam.py: A deployment wrapper written in Python. It accesses your camera stream, handles bounding face boxExtractions via OpenCV, and passes face pixel tensors straight to your saved model payload to decode expressions live.



**Note: this is still slower than using tensorflow and pytorch to build a cnn in python as those libraries are also written is c++ but uses specialized GPUs and other hardware to compute large data quickly**

