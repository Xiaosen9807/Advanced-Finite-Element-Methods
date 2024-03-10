import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import os

def normalize(value, min_value, max_value):
    """将值归一化到 [0, 1] 范围内。"""
    return (value - min_value) / (max_value - min_value)

def denormalize(value, min_value, max_value):
    """将归一化的值反归一化到原始范围。"""
    return value * (max_value - min_value) + min_value

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.neuron = 100
        neuron  = self.neuron
        self.net = nn.Sequential(
            nn.Linear(1, neuron),  
            nn.Tanh(),         
            nn.Linear(neuron, neuron),
            nn.Tanh(),         
            nn.Linear(neuron, neuron),
            nn.Tanh(),         
            nn.Linear(neuron, neuron),
            nn.Tanh(),        
            nn.Linear(neuron, 1), 
        )
        
    def forward(self, r):
        return self.net(r)

class muJz_net(nn.Module):
    def __init__(self):
        super(muJz_net, self).__init__()
        self.neuron = 100
        neuron  = self.neuron
        # 定义muJz网络
        self.muJz_net = nn.Sequential(
            nn.Linear(neuron, neuron),  
            nn.ReLU(),             
            nn.Linear(neuron, neuron), 
            nn.ReLU(),             
            nn.Linear(neuron, 1),  
        )
        
    def forward(self, r):
        muJz_input = torch.ones(1, self.neuron)  
        muJz_output = self.muJz_net(muJz_input)
        # 返回主网络输出和muJz网络输出
        model.muJz = muJz_output
        return muJz_output

lossfunc = nn.MSELoss(reduction='sum')

def compute_loss(model, muJz_model, r, muJz=20, A0_ori=72, A1_ori=67, ws = {"w_pde":1, "w_data":1, "w_bc":1, "w_D1":1, "w_D2":1, "w_N":1}):
    r.requires_grad = True  # 允许对 r 求导
    w_pde = ws["w_pde"]
    w_data = ws["w_data"]
    w_bc = ws["w_bc"]
    w_D1 = ws["w_D1"]
    w_D2 = ws["w_D2"]
    w_N = ws["w_N"]

    
    # 归一化 A0 和 A1
    # 假设 A 的最小值和最大值分别为 A_min 和 A_max
    A_min = min(A0_ori, A1_ori)  # 获取 A0_ori 和 A1_ori 中的最小值
    A_max = max(A0_ori, A1_ori)  # 获取 A0_ori 和 A1_ori 中的最大值


    # 归一化 A0_ori 和 A1_ori
    A0 = normalize(A0_ori, A_min, A_max)
    A1 = normalize(A1_ori, A_min, A_max)

    # 网络预测 A_z
    A_z = model(r)
    # 对 A_z 关于 r 求一阶导数
    dA_z_dr = torch.autograd.grad(A_z, r, torch.ones_like(A_z), retain_graph=True, create_graph=True)[0]
    
    left_pde = torch.autograd.grad(r * dA_z_dr, r, torch.ones_like(dA_z_dr),retain_graph=True, create_graph=True)[0] 
    
    model_muJz = muJz_model(r)

    pde_loss = torch.nn.functional.mse_loss(left_pde, -r*model_muJz)
    # print(model.muJz.requires_grad)
    # print(model_muJz[0][0])
    # exit()
 # 
    # 边界条件损失
    r_0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
    A_0_pred = model(r_0)
    A_1_pred = model(torch.tensor([[1.0]], dtype=torch.float32))
    # Convert A0 and A1 to tensors if they are not already
    A0_tensor = torch.tensor([A0], dtype=torch.float32)  # Make sure A0 is a tensor
    A1_tensor = torch.tensor([A1], dtype=torch.float32)  # Make sure A1 is a tensor

    # Ensure A0_tensor and A1_tensor have the same shape as A_0_pred and A_1_pred, respectively
    # This might involve unsqueezing if A_0_pred and A_1_pred are more than 1D
    A0_tensor = A0_tensor.expand_as(A_0_pred)
    A1_tensor = A1_tensor.expand_as(A_1_pred)
    # print(A0_tensor, A1_tensor)
    # print(A_0_pred, A_1_pred)
    # exit()

    # Now use these tensors in your mse_loss calculations
    Dirichlet_BC_1 = lossfunc(A_0_pred, A0_tensor)
    Dirichlet_BC_2 = lossfunc(A_1_pred, A1_tensor)


    # 计算诺伊曼边界条件的损失（这里假设是导数为 0）
    Neumann_BC_grad = torch.autograd.grad(outputs=A_0_pred, inputs=r_0, grad_outputs=torch.ones_like(A_0_pred),retain_graph=True, create_graph=True)[0]
    # 假设期望的导数值为 0，创建一个与 Neumann_BC_grad 形状相同且所有值都为 0 的张量
    expected_grad = torch.zeros_like(Neumann_BC_grad)
    Neumann_BC = lossfunc(Neumann_BC_grad, expected_grad)

    # 注意：您可能需要调整损失项之间的权重
    boundary_loss = w_D1 * Dirichlet_BC_1 + w_D2* Dirichlet_BC_2 + w_N*Neumann_BC
    # print("Dirichlet_BC_1:", Dirichlet_BC_1, "Dirichlet_BC_2:", Dirichlet_BC_2, "Neumann_BC:", Neumann_BC)
    # exit()

    real_solution = exact_solution(r, A0_ori, muJz)
    real_solution = normalize(real_solution, min_value=A_min, max_value=A_max)
    # print(real_solution)
    # print("A_z", A_z)

    real_loss = lossfunc(A_z, real_solution)
    
    loss = w_pde * pde_loss + w_data * real_loss + w_bc*boundary_loss

    # print("pde_loss:", pde_loss, "real_loss:", real_loss, "boundary_loss:", boundary_loss)
    # exit()

    # loss = real_loss
    
    # print(loss)
    # exit()
    return loss

def train(model, muJz_model, muJz = 20, A0 = 71, A1 = 66,ws={"w_pde":1, "w_data":1, "w_bc":1, "w_D1":1, "w_D2":1, "w_N":1}, epochs=5000, lr=0.001, verbose = False):
    min_loss = float('inf')
    for name, param in model.named_parameters():
        print(name)

    optimizer_PINN = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_muJz = torch.optim.Adam(muJz_model.parameters(), lr=lr*10)
    muJz_lst = []
    for epoch in range(epochs):
        optimizer_PINN.zero_grad()
        optimizer_muJz.zero_grad()
        
        r_train = torch.rand((50, 1), dtype=torch.float32)

        loss = compute_loss(model, model_muJz, r_train, muJz=muJz, A0_ori=A0, A1_ori=A1, ws = ws)
        
        # 反向传播和优化
        loss.backward()
        optimizer_PINN.step()
        optimizer_muJz.step()
        muJz_lst.append(model.muJz)
        if epoch % 100 == 0 and verbose:
            print(f'Epoch {epoch}, Loss: {loss.item()}, muJz: {model.muJz[0]}')

def cal_error(A_z_pred, A_z_exact):
    return np.mean(np.abs(np.array(A_z_pred) - np.array(A_z_exact))) * 100
# 计算解析解
def exact_solution(r, A0=71, muJz=20):
    return A0 - 1/4 * muJz * r**2

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(9807)
    # 初始化模型并开始训练
    verbose = True
    muJz = 20
    epoch = 5000
    A0 = 71
    A1 = 66
    ws = {"w_pde": .1, "w_bc": 20, "w_data": 30, "w_D1": 100, "w_D2": 1000, "w_N": 1000}
    # ws = {"w_pde": 0, "w_bc": 0, "w_data": 30, "w_D1": 100, "w_D2": 1000, "w_N": 1000}
    ws = {"w_pde": .1, "w_bc": 2, "w_data": 30, "w_D1": 1000, "w_D2": 1000, "w_N": 1000}
    print(ws)
    model = PINN()
    model_muJz = muJz_net()
    start_time = time.time()
    train(model, model_muJz, epochs=epoch,  muJz=muJz, A0=A0, A1=A1, ws=ws, verbose=verbose)
    duration = time.time() - start_time
    
    print("Duration: {:.2f}s".format(duration))

    # 生成测试点
    r_test = torch.linspace(0, 1, 100).view(-1, 1)
    # 计算 PINN 的解
    A_min = min(A0, A1)  # 获取 A0_ori 和 A1_ori 中的最小值
    A_max = max(A0, A1)  # 获取 A0_ori 和 A1_ori 中的最大值

    A_z_pinn_pred, muJz_pinn_pred= model(r_test).detach().numpy(), model_muJz(r_test).detach().numpy()
    A_z_pinn = denormalize(A_z_pinn_pred, min_value=A_min, max_value=A_max)# denormlize the A_z_pinn_pred


    A_z_exact = exact_solution(r_test.numpy())
    muJz_error = cal_error(muJz_pinn_pred, 20)
    print("muJz:", muJz_pinn_pred)
    print("muJz Error: {:.2f}%".format(muJz_error))
    A_z_error = cal_error(A_z_exact=A_z_exact, A_z_pred=A_z_pinn)
    print("Az Error: {:.2f}%".format(A_z_error))
    plt.figure()
    plt.plot(r_test, A_z_pinn, label="PINN")
    plt.plot(r_test, A_z_exact, label="PINN")
    plt.grid(True)
    # 获取当前脚本文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建完整的文件路径
    file_path = os.path.join(current_dir, "inverse_Solution_w_pde{}.pdf".format(ws["w_pde"]))

    # 保存文件
    plt.savefig(file_path)
    plt.show()

