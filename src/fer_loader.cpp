#include "../include/fer_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

void FERLoader::load(const std::string& csv_path, std::vector<std::shared_ptr<Tensor>>& images, std::vector<int>& labels, int limit) 
{
    std::ifstream file(csv_path);

    std::string line;
    
    std::getline(file, line);

    std::cout << "Loading FER-2013 Data" << std::endl;
    int count = 0;

    while (std::getline(file, line)) {
        if (limit != -1 && count >= limit) break;

        std::stringstream ss(line);
        std::string val_str;

        std::getline(ss, val_str, ','); 
        int label = std::stoi(val_str);
        labels.push_back(label);

        std::getline(ss, val_str, ',');
        std::stringstream pixel_ss(val_str);
        std::string pixel_val;
        std::vector<float> img_data;
        img_data.reserve(48 * 48);

        while (std::getline(pixel_ss, pixel_val, ' ')) {
            if (!pixel_val.empty()) {
                img_data.push_back(std::stof(pixel_val) / 255.0f);
            }
        }

        if (img_data.size() != 48 * 48) {
            std::cerr << "Warning: wrong image found at line " << count + 2 << std::endl;
            labels.pop_back(); 
            continue;
        }

        images.push_back(std::make_shared<Tensor>(img_data, std::vector<std::size_t>{1, 48, 48}));
        
        count++;
        if (count % 1000 == 0) std::cout << "Loaded " << count << " images \r" << std::flush;
    }
    std::cout << "\nLoaded " << count << " images successfully." << std::endl;
}