#ifndef FER_LOADER_H
#define FER_LOADER_H

#include "tensor.h"
#include <string>
#include <vector>
#include <memory>

class FERLoader {
public:
    
    static void load(const std::string& csv_path, 
                     std::vector<std::shared_ptr<Tensor>>& images, 
                     std::vector<int>& labels,
                     int limit = -1);
};

#endif