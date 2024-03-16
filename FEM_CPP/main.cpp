#include <iostream>
#include <algorithm> // 包含 max_element, min_element
#include <numeric>   // 包含 accumulate
using namespace std;

int main() {
    double a;
    int size = 5;  // 明确指定数组的大小
    double list[size];

    for(int i = 0; i < size; i++) {
        std::cin >> a;
        list[i] = a;
    }

    // 使用指针，因为原生数组不提供.begin()和.end()
    auto max_it = std::max_element(list, list + size);
    auto min_it = std::min_element(list, list + size);
    double sum = std::accumulate(list, list + size, 0.0);
    double average = sum / size;

    // 确保解引用迭代器之前，数组不是空的
    cout << "list: " << list << endl;
    if (size > 0) {
        std::cout << "Max: " << *max_it << " Min: " << *min_it << std::endl;
    } else {
        std::cout << "Array is empty" << std::endl;
    }
    
    std::cout << "Average: " << average << std::endl;
    return 0;
}


int find_element(vector<int> &a)
{
    
// write a function, input a vector and find the largest and smallest
// element
    
}
