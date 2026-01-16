#include "../include/tensor.h"
#include "../include/conv2d.h"
#include "../include/relu.h"
#include "../include/pooling.h"
#include "../include/flatten.h"
#include "../include/linear.h"
#include "../include/loss.h"
#include "../include/sgd.h"
#include "../include/fer_loader.h"
#include "../include/dropout.h" 
#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>

void save_model(const std::string& filename, const std::vector<std::pair<std::string, std::shared_ptr<Tensor>>>& params) {
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file for saving" << std::endl;
        return;
    }

    size_t num_tensors = params.size();
    file.write((char*)&num_tensors, sizeof(size_t));

    for (const auto& pair : params) {
        const auto& tensor = pair.second; 
        const auto& data = tensor->data();
        
        size_t data_size = data.size();
        file.write((char*)&data_size, sizeof(size_t));

        file.write((char*)data.data(), data_size * sizeof(float));
    }

    std::cout << "Model saved to " << filename << " (" << num_tensors << " tensors)" << std::endl;
    file.close();
}

int main() {
    std::cout << "Loading Data" << std::endl;
    std::vector<std::shared_ptr<Tensor>> all_x;
    std::vector<int> all_y;
    
    int total_load = 10000; 
    try {
        FERLoader::load("data/fer2013.csv", all_x, all_y, total_load); 
    } catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        return 1;
    }

    size_t split_idx = (size_t)(total_load * 0.8);
    std::vector<std::shared_ptr<Tensor>> train_x(all_x.begin(), all_x.begin() + split_idx);
    std::vector<int> train_y(all_y.begin(), all_y.begin() + split_idx);
    std::vector<std::shared_ptr<Tensor>> val_x(all_x.begin() + split_idx, all_x.end());
    std::vector<int> val_y(all_y.begin() + split_idx, all_y.end());

    std::cout << "   Training on " << train_x.size() << " samples." << std::endl;
    std::cout << "   Validating on " << val_x.size() << " samples." << std::endl;

    std::cout << "Building Model" << std::endl;
    
    Conv2D conv1(1, 12, 3, 1, 1);
    Relu relu1;
    Pooling pool1(2, 2);
    Dropout drop1(0.25f);

    Conv2D conv2(12, 24, 3, 1, 1);
    Relu relu2;
    Pooling pool2(2, 2);
    Dropout drop2(0.25f); 

    Flatten flatten;
    Linear fc(24 * 12 * 12, 7); 

    auto p1 = conv1.parameters();
    auto p2 = conv2.parameters();
    auto p3 = fc.parameters();
    
    p1.insert(p1.end(), p2.begin(), p2.end());
    p1.insert(p1.end(), p3.begin(), p3.end());

    SGD optimizer(p1, 0.001f); 

    int epochs = 50; 
    std::cout << "3. Starting Training (" << epochs << " epochs)..." << std::endl;

    for (int epoch = 0; epoch < epochs; epoch++) {
      
        drop1.train(); 
        drop2.train();

        float total_loss = 0.0f;
        int train_correct = 0;

        for (size_t i = 0; i < train_x.size(); i++) {
            
            auto out = conv1.forward(train_x[i]);
            out = relu1.forward(out);
            out = pool1.forward(out);
            out = drop1.forward(out);

            out = conv2.forward(out);
            out = relu2.forward(out);
            out = pool2.forward(out);
            out = drop2.forward(out); 

            out = flatten.forward(out);
            out = fc.forward(out);

            CrossEntropyLoss criterion;
            auto loss = criterion(out, train_y[i]);
            total_loss += loss->item();

            optimizer.zero_grad();
            loss->backward(); 
            optimizer.step();

            const auto& logits = out->data();
            int pred = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
            if (pred == train_y[i]) train_correct++;
        }

        drop1.eval(); 
        drop2.eval();
        
        int val_correct = 0;
        for (size_t i = 0; i < val_x.size(); i++) {
            auto out = conv1.forward(val_x[i]);
            out = relu1.forward(out);
            out = pool1.forward(out);
            out = drop1.forward(out);

            out = conv2.forward(out);
            out = relu2.forward(out);
            out = pool2.forward(out);
            out = drop2.forward(out); 

            out = flatten.forward(out);
            out = fc.forward(out);

            const auto& logits = out->data();
            int pred = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
            if (pred == val_y[i]) val_correct++;
        }

        float train_acc = (float)train_correct / train_x.size() * 100.0f;
        float val_acc = (float)val_correct / val_x.size() * 100.0f;
        
        std::cout << "Epoch " << epoch 
                  << " Loss: " << (total_loss/train_x.size()) 
                  << " Train Acc: " << train_acc << "%" 
                  << " Val Acc: " << val_acc << "%" << std::endl;
    }
    std::cout << "Saving trained model" << std::endl;
    
    save_model("fer_model.bin", p1);
    return 0;
}