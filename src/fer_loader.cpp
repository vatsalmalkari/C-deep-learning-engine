#include "../include/fer_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>

void FERLoader::load(const std::string& csv_path, std::vector<std::shared_ptr<Tensor>>& images, std::vector<int>& labels, int limit) 
{
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open FER-2013 CSV at path: " + csv_path);
    }

    std::string line;
    // Skip CSV header (emotion,pixels,Usage)
    if (!std::getline(file, line)) {
        return;
    }

    std::cout << "Loading FER-2013 Data..." << std::endl;
    int count = 0;

    while (std::getline(file, line)) {
        if (limit != -1 && count >= limit) break;
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string label_str;
        std::string pixel_str;

        // Column 1: Emotion label
        if (!std::getline(ss, label_str, ',')) continue;
        
        // Column 2: Pixel data
        if (!std::getline(ss, pixel_str, ',')) continue;

        int label = std::stoi(label_str);

        // Fast string parsing for space-separated pixels
        std::vector<float> img_data;
        img_data.reserve(48 * 48);

        const char* p = pixel_str.c_str();
        char* end = nullptr;

        while (*p) {
            // Skip leading spaces or stray quotation marks
            while (*p == ' ' || *p == '\"') ++p;
            if (!*p) break;

            float val = std::strtof(p, &end);
            if (p == end) break; // Conversion stopped
            
            img_data.push_back(val / 255.0f);
            p = end;
        }

        // Validate complete image
        if (img_data.size() != 48 * 48) {
            std::cerr << "\nWarning: corrupt image found at count " << count << " (got " << img_data.size() << " pixels)" << std::endl;
            continue;
        }

        // Push both synchronously so indices always match 1:1
        labels.push_back(label);
        images.push_back(std::make_shared<Tensor>(img_data, std::vector<std::size_t>{1, 48, 48}));
        
        count++;
        if (count % 1000 == 0) {
            std::cout << "Loaded " << count << " images \r" << std::flush;
        }
    }
    std::cout << "\nLoaded " << count << " images successfully." << std::endl;
}
