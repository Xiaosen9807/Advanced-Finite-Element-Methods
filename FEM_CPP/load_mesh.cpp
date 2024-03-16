#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cctype> // 包含 std::isdigit
#include <nlohmann/json.hpp>
#include "element.h"

using namespace std;




pair<vector<element>, vector<Node>> nodes_elements() {
    ifstream file("nodes_data.json");
    if (!file.is_open()) {
        throw runtime_error("Unable to open file");
    }

    nlohmann::json json_data;
    file >> json_data;

    vector<Node> all_nodes;
    vector<int> elementNodes;
    for (const auto& item : json_data) {
        if (item.contains("element nodes")) {
            elementNodes = item["element nodes"].get<vector<int>>();
        } else {
            Node node;
            node.id = item["id"];
            node.x = item["x"];
            node.y = item["y"];
            node.type = item["type"];
            node.BC[0] = item["BC"][0];
            node.BC[1] = item["BC"][1];
            all_nodes.push_back(node);
        }
    }

    vector<element> all_elements;
    int split = 3;  // Assuming 3 nodes per element
    int elem_num = elementNodes.size() / split;
    for (int i = 0; i < elem_num; ++i) {
        vector<Node*> this_nodes_lst;
        for (int j = 0; j < split; ++j) {
            this_nodes_lst.push_back(&all_nodes[elementNodes[i * split + j] - 1]);
        }
        element elem(this_nodes_lst);
        elem.id = i;
        all_elements.push_back(elem);
    }

    file.close();
    return make_pair(all_elements, all_nodes);
}


void nodes_info(int trunc = -1) {
    auto [elements, nodes] = nodes_elements(); // 使用结构化绑定来同时解包 pair

    // 检查是否有节点被读取
    if (nodes.empty()) {
        cout << "No nodes were read from the file." << endl;
    }

    if (trunc < 0 || trunc >= nodes.size()) {
	trunc = nodes.empty() ? 0 : nodes.size() - 1;
    }

    printf("nodes number: %zu\n", nodes.size());
    printf("Elements number: %zu\n", elements.size());

    // 输出节点信息
    for (int i=0; i<=trunc; i++){
	const Node& node = nodes[i];
        cout << "ID: " << node.id << "\n";
        cout << "Coords: [" << node.x << ", " << node.y << "]\n";
        cout << "Type: " << node.type << "\n";
        cout << "BC[0]: " << node.BC[0] << "  " << "BC[1]: " << node.BC[1] << "\n\n";
    }
    for (int i=0; i<=trunc; i++){
	element elem = elements[i];
	cout << "elem_id: " << elem.id << "\n";
	for (const int node_id : elem.node_ids){
	    cout << "node_id: " << node_id << " Coords: [" << nodes[node_id].x << ", " << nodes[node_id].y << "]  " <<"BC[0]: " << nodes[node_id ].BC[0] << "  " << "BC[1]: " << nodes[node_id ].BC[1] << "" << " \n";
	}

	cout << "\n\n";
    }

}

int main()
{
    
    nodes_info();
    return 0;
}
