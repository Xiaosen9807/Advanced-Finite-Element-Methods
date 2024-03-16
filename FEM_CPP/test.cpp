#include <iostream>

class Base {
public:
    Base() {
        std::cout << "Base default constructor called" << std::endl;
    }

    Base(int x) {
        std::cout << "Base parameterized constructor called with x = " << x << std::endl;
    }
};

class Derived : public Base {
public:
    // 完全修改了构造函数，引入了两个新的参数
    Derived() : Base()
    {
	std::cout << "Derived default constructor called" << std::endl;
    }
    Derived(int y, std::string z) : Base(y) {  // 仍然需要在这里调用基类的构造函数
        std::cout << "Derived constructor called with y = " << y << " and z = " << z << std::endl;
    }
};

int main() {
    Base obj1;
    std::cout << "----------------" << std::endl;
    Base obj2(1);
    std::cout << "----------------" << std::endl;
    Derived d1;
    std::cout << "----------------" << std::endl;
    Derived d2(1, "hello");
    return 0;
}
