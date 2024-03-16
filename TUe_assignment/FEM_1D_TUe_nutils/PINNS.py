import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
from consts import * 
import os
import json
from sklearn.preprocessing import MinMaxScaler
def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

def denormalize(value, min_value, max_value):
    return value * (max_value - min_value) + min_value

class MinMax():
    def __init__(self, scale):
        self.min = min(scale)
        self.max = max(scale)
    def transform(self, x):
        return x * (self.max - self.min) + self.min
    
    


class PINNs(nn.Module):
    def __init__(self):
        super(PINNs, self).__init__()
        neuron = 40
        self.net = nn.Sequential(
            nn.Linear(1, neuron),  
            nn.ReLU(),         
            nn.Linear(neuron, neuron),
            nn.Tanh(),         
            # nn.ReLU(),         
            nn.Linear(neuron, neuron),
            nn.Tanh(),
            # nn.ReLU(),         
            # nn.Linear(neuron, neuron),
            # # nn.Tanh(), 
            # nn.ReLU(),         
            nn.Linear(neuron, 1), 
        )
        
    def forward(self, r):
        return self.net(r)

lossfunc = nn.MSELoss(reduction='sum')
lossfunc = nn.MSELoss()

def compute_loss(model, r, domain=[0,1], muJz=20, A0_ori=72, A1_ori=67, ws = {"w_pde":1, "w_data":1, "w_bc":1, "w_D1":1, "w_D2":1, "w_N":1}):
    r.requires_grad = True  # 允许对 r 求导
    w_pde = ws["w_pde"]
    w_data = ws["w_data"]
    w_bc = ws["w_bc"]
    w_D1 = ws["w_D1"]
    w_D2 = ws["w_D2"]
    w_N = ws["w_N"]

    # A_min = min(A0_ori, A1_ori) 
    # A_max = max(A0_ori, A1_ori)
    # scale_left = MinMaxScaler(feature_range=(A_min.detach().numpy(), A_max.detach().numpy()))
    # # print(A_min, A_max)
    # # exit()
    # # print(A0_ori)
    # A0_ori = torch.tensor([[A0_ori]], dtype=torch.float32)
    # A1_ori = torch.tensor([[A1_ori]], dtype=torch.float32)
    # # A0_ori = np.array([[A0_ori]])
    # # A1_ori = np.array([[A1_ori]])
    # # print(A0_ori, A1_ori)

    # # 对 A0_ori 进行缩放
    # A0_scaled = torch.tensor(scale_left.fit_transform(A0_ori), dtype=torch.float32)
    # # 对 A1_ori 进行缩放
    # A1_scaled = torch.tensor(scale_left.fit_transform(A1_ori), dtype=torch.float32)
    
    A_z = model(r)
    # 对 A_z 关于 r 求一阶导数
    dA_z_dr = torch.autograd.grad(A_z, r, torch.ones_like(A_z), retain_graph=True, create_graph=True)[0]
    
    left_pde = torch.autograd.grad(r * dA_z_dr, r, torch.ones_like(dA_z_dr),retain_graph=True, create_graph=True)[0] 
    # 计算 MSE 损失
    pde_loss = torch.nn.functional.mse_loss(left_pde, -r*muJz)
 
    # 边界条件损失
    r_0 = torch.tensor([0], dtype=torch.float32, requires_grad=True)
    A_0_pred = model(r_0)
    A_1_pred = model(torch.tensor([1], dtype=torch.float32))
    # print("A_0_pred, A_1_pred:", A_0_pred, A_1_pred)


    # Now use these tensors in your mse_loss calculations
    Dirichlet_BC_1 = lossfunc(A_0_pred, A0)
    Dirichlet_BC_2 = lossfunc(A_1_pred, A1)


    # 计算诺伊曼边界条件的损失（这里假设是导数为 0）
    Neumann_BC_grad = torch.autograd.grad(outputs=A_0_pred, inputs=r_0, grad_outputs=torch.ones_like(A_0_pred),retain_graph=True, create_graph=True)[0]

    expected_grad = torch.zeros_like(Neumann_BC_grad)
    Neumann_BC = lossfunc(Neumann_BC_grad, expected_grad)

    boundary_loss = w_D1 * Dirichlet_BC_1 + w_D2* Dirichlet_BC_2 + w_N*Neumann_BC
    # print("Dirichlet_BC_1:", Dirichlet_BC_1, "Dirichlet_BC_2:", Dirichlet_BC_2, "Neumann_BC:", Neumann_BC)

    real_solution = exact_solution(r)
    # print("real_solution[0]:", real_solution[0])
    # real_solution_scaled = scale_left.fit_transform(real_solution.detach().numpy())
    # print("real_solution_scaled[0]:", real_solution_scaled[0])
    # real_solution_scaled = torch.tensor(real_solution_scaled, dtype=torch.float32, requires_grad=False)

    real_loss = lossfunc(A_z, real_solution)
    
    loss = w_pde * pde_loss + w_data * real_loss + w_bc*boundary_loss
    # print("pde_loss:", pde_loss, "real_loss:", real_loss, "boundary_loss:", boundary_loss)
    # exit()

    return loss

def train(model, domain=[0, 1], muJz = 20, A0 = 71, A1 = 66,ws={"w_pde":1, "w_data":1, "w_bc":1, "w_D1":1, "w_D2":1, "w_N":1}, epochs=5000, lr=0.01, verbose = False):
    min_loss = float('inf')
    # for name, param in model.named_parameters():
    #     print(name)
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.dtype}")


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        r_train = torch.rand((50, 1), dtype=torch.float32)

        # 生成 50 个介于 0 和 1 之间的随机数
        r_samples = torch.rand(50, 1)

        # 将随机数缩放到 domain 的范围
        r_train = r_samples
        # r_train = r_samples * (domain[1] - domain[0]) + domain[0]
        # print(r_train)
        # exit()
        # print("A0, A1", A0, A1)

        loss = compute_loss(model, r=r_train, domain=domain, muJz=muJz, A0_ori=A0, A1_ori=A1, ws = ws)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 and verbose:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

def cal_error(A_z_pred, A_z_exact):
    return np.mean(np.abs(np.array(A_z_pred) - np.array(A_z_exact))) * 100



def generate_data(num=100):
    data = np.load("/Users/xusenqin/Desktop/Advanced-Finite-Element-Methods/FEM_1D_TUe_nutils/datasets/Poission.npz")
    # print('data', data["r"][:10], data["solution"][:10])
    index = np.random.choice(data["r"].shape[0], num, replace=False)
    r_data = data["r"].reshape(-1)  # 保持为一维数组以简化索引操作
    solution = data["solution"].reshape(-1)  # 同样保持为一维数组

    # 将找到的解转换为 torch 张量并返回
    return torch.tensor(r_data, dtype=torch.float32), torch.tensor(solution, dtype=torch.float32)

def exact_solution(x_input, region=1):
    area = (0.0053, 0.0063)
    scaler = MinMax(area)
    x = scaler.transform(x_input)
    # print("x_input", x_input, "x", x)
    mu0 = 1.257e-6
    Params = consts["Params"][region]
    mu = Params["mu"]
    Jz = Params["Jz"]
    A0 = Params["A"]
    B0 = Params["B"]
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    A = A0 + B0*torch.log(x) - 1/4 * mu* mu0 * Jz *x**2
    
    return A * 1e6

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(9807)
    # 初始化模型并开始训练
    verbose = True
    muJz = 20
    epoch = 1000
    domain = (0, 1)
    A0 = exact_solution(domain[0])
    A1 = exact_solution(domain[1])
    # print("A0, A1", A0, A1)
    # exit()

    ws = {"w_pde": .1, "w_bc": 20, "w_data": 30, "w_D1": 100, "w_D2": 1000, "w_N": 1000}
    ws = {"w_pde": 0, "w_bc": 0, "w_data": 30, "w_D1": 100, "w_D2": 1000, "w_N": 1000}
    print(ws)
    model = PINNs()
    # print(model(torch.Tensor([[0.0057], [0.0062]])))
    # print(model(torch.Tensor([0.0062])))
    # exit()
    start_time = time.time()
    train(model, domain=domain, epochs=epoch,  muJz=muJz, A0=A0, A1=A1, ws=ws, verbose=verbose)
    duration = time.time() - start_time
    
    print("Duration: {:.2f}s".format(duration))

    

    # 生成测试点
    r_test = torch.linspace(domain[0], domain[-1], 100).view(-1, 1)
    # 计算 PINN 的解
    A_min = min(A0, A1)  # 获取 A0_ori 和 A1_ori 中的最小值
    A_max = max(A0, A1)  # 获取 A0_ori 和 A1_ori 中的最大值

    A_z_pinn_pred = model(r_test).detach()
    # print(A_z_pinn_pred)
    A_z_pinn = denormalize(A_z_pinn_pred, min_value=A_min, max_value=A_max)# denormlize the A_z_pinn_pred
    A_z_pinn = A_z_pinn_pred


    A_z_exact = exact_solution(r_test)
    error = cal_error(A_z_pinn, A_z_exact)
    print("Error: {:.2f}%".format(error))

    plt.figure(figsize=(10, 6))
    plt.plot(r_test.numpy(), A_z_exact, label='Exact Solution', linewidth=2)
    plt.plot(r_test.numpy(), A_z_pinn, '--', label='PINN Solution', linewidth=2)
    print(A_z_pinn[-1], A_z_pinn_pred[-1])
    plt.xlabel('r', fontsize=14)
    plt.ylabel('Az', fontsize=14)
    x_position = 0.8  # Start of the x-axis
    y_position = max(A_z_exact)  # Position text at the minimum of the exact solution for visibility

    # Add text with a black background
    plt.text(x_position, y_position-1, "Duration: {:.2f}s".format(duration), fontsize=14, color='white', bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.5'))
    plt.text(x_position, y_position-1.5, "Error: {:.2f}%".format(error), fontsize=14, color='white', bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.5'))


    plt.legend(fontsize=14)
    plt.title('Comparison between PINN Solution and Exact Solution', fontsize=16)
    plt.grid(True)
    # 获取当前脚本文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建完整的文件路径
    file_path = os.path.join(current_dir, "PINN_Solution_w_pde{}.pdf".format(ws["w_pde"]))

    # 保存文件
    plt.savefig(file_path)
    plt.show()


