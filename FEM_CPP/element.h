#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <Eigen/Dense>
#include <typeinfo>
#include <functional>

using namespace std;

class shape_func
{
public:
    int p;
    
    // 使用成员初始化列表来初始化成员变量p
    shape_func(int p) : p(p) {
        // 由于构造函数体为空，可以省略pass
    }

    // 假设您想在基类中定义func为虚函数，以便在派生类中重写
    double operator()(double xi, double eta) const {
        return func(xi, eta);
    }

protected:
    function<double(double, double)> func;
};

class T3_phi : public shape_func
{
public: T3_phi(int p) : shape_func(p) {
    switch (p) {
	case 0: func = [](double xi, double eta) -> double {return xi;};
	    break;
	case 1: func = [](double xi, double eta) -> double {return eta;};
	    break;
	case 2: func = [](double xi, double eta) -> double {return 1-xi-eta;};
	    break;
	default: throw invalid_argument("p must be 0, 1, or 2 in T3_phi");
    }
    
}
};

class T3_phipx : public shape_func
{
public: T3_phipx(int p) : shape_func(p) {
    switch (p){
    case 0: func = [](double xi, double eta) -> double {return 1;};
	break;
    case 1: func = [](double xi, double eta) -> double {return 0;};
	break;
    case 2: func = [](double xi, double eta) -> double {return -1;};
	break;
    default: throw invalid_argument("p must be 0, 1, or 2 in T3_phipx");
    }
}
};

class T3_phipy : public shape_func
{
public: T3_phipy(int p) : shape_func(p) {
    switch (p){
    case 0: func = [](double xi, double eta) -> double {return 0;};
	break;
    case 1: func = [](double xi, double eta) -> double {return 1;};
	break;
    case 2: func = [](double xi, double eta) -> double {return -1;};
	break;
    default: throw invalid_argument("p must be 0, 1, or 2 in T3_phipy");
    }
}
};

class Q4_phi : public shape_func
{
public: Q4_phi(int p) : shape_func(p) {
    switch (p){
    case 0: func = [](double xi, double eta) -> double {return (xi-1)*(eta-1)/4;};
	break;
    case 1: func = [](double xi, double eta) -> double {return (1+xi)*(1-eta)/4;};
	break;
    case 2: func = [](double xi, double eta) -> double {return (1+xi)*(1+eta)/4;};
	break;
    case 3: func = [](double xi, double eta) -> double {return (1-xi)*(eta+1)/4;};
	break;
    default: throw invalid_argument("p must be 0, 1, 2, or 3 in Q4_phi");
    }
}
};

class Q4_phipx : public shape_func
{
public: Q4_phipx(int p) : shape_func(p) {
    switch (p){
    case 0: func = [](double xi, double eta) -> double {return (eta-1)/4;};
	break;
    case 1: func = [](double xi, double eta) -> double {return (1-eta)/4;};
	break;
    case 2: func = [](double xi, double eta) -> double {return (1+eta)/4;};
	break;
    case 3: func = [](double xi, double eta) -> double {return -(1+eta)/4;};
	break;
    default: throw invalid_argument("p must be 0, 1, 2, or 3 in Q4_phipx");
    }
}
};

class Q4_phipy : public shape_func
{
public: Q4_phipy(int p) : shape_func(p) {
    switch (p){
    case 0: func = [](double xi, double eta) -> double {return (xi-1)/4;};
	break;
    case 1: func = [](double xi, double eta) -> double {return -(1+xi)/4;};
	break;
    case 2: func = [](double xi, double eta) -> double {return (1+xi)/4;};
	break;
    case 3: func = [](double xi, double eta) -> double {return (1-xi)/4;};
	break;
    default: throw invalid_argument("p must be 0, 1, 2, or 3 in Q4_phipy");
    }
}
};

struct Node {
    int id;
    double x, y;
    std::string type;
    array<double, 2> val;
    array<int, 2> BC;// 0 = No BC, 1 = Dirichlet, -1 = Neumann
};


class element {
public:
    int id;
    std::vector<Node*> node_lst;
    vector<int> node_ids;
    vector<array<double*, 2>> node_coords;
    Eigen::MatrixXd K_loc;
    string type;
    int num_nodes;
    vector<array<double*, 2>> node_values; // Store arrays of doubles

    vector<shape_func*> shape_func_lst;

    element(std::vector<Node*> node_lst) {
        K_loc = Eigen::MatrixXd::Zero(node_lst.size(), node_lst.size());
        num_nodes = node_lst.size();

        for (Node* node : node_lst) {
	    double *val_x_ptr = &node->val[0];
	    double *val_y_ptr = &node->val[1];
            node_values.push_back({val_x_ptr, val_y_ptr}); // Push the std::array

	    double *coords_x_ptr = &node->x;
	    double *coords_y_ptr = &node->y;
            node_coords.push_back({coords_x_ptr, coords_y_ptr}); // Push the std::array

	    node_ids.push_back(node->id);

        }
        type = node_lst[0]->type;
    }
};


vector<pair<vector<double>, double>> Gauss_points(element& elem, int order = 2) {
    const double sqrt3 = sqrt(3.0);
    vector<pair<vector<double>, double>> gaussPoints;

    if (elem.type == "Q4") {
        gaussPoints = {
            {{-1.0 / sqrt3, -1.0 / sqrt3}, 1.0},
            {{1.0 / sqrt3, -1.0 / sqrt3}, 1.0},
            {{-1.0 / sqrt3, 1.0 / sqrt3}, 1.0},
            {{1.0 / sqrt3, 1.0 / sqrt3}, 1.0}
        };
    } else if (elem.type == "T3") {
        gaussPoints = {
            {{1.0 / 6.0, 1.0 / 6.0}, 1.0 / 6.0},
            {{2.0 / 3.0, 1.0 / 6.0}, 1.0 / 6.0},
            {{1.0 / 6.0, 2.0 / 3.0}, 1.0 / 6.0}
        };
    } else {
        throw invalid_argument("Unsupported element type");
    }

    return gaussPoints;
};
    
    


