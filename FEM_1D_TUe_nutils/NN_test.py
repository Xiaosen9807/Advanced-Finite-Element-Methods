import torch
import torch.nn as nn   
import numpy as np
import matplotlib.pyplot as plt
from consts import *
from sklearn.preprocessing import MinMaxScaler
import os

mu0 = 1.257e-6

class MinMax:
   def __init__(self, area):
       self.min = min(area)
       self.max = max(area)

   def transform(self, x):
       return x * (self.max - self.min) + self.min
 
    
def exact_solution(x_input, region=2):
    # area = [53, 63]
    # scaler = MinMax((area))
    # x = scaler.transform(x_input) #/ 10000
    x = x_input.clone().detach()
    # print(x[0][0])
    # exit()
    # x = x_input
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


def generate_data(region=1, num_points=1000):
    if region == 1:
        start, end = 0.0053, 0.0063,
    elif region == 2:
        start, end = 0.0063, 0.0101

    x_train = torch.linspace(start, end, num_points).unsqueeze(-1)  
    y_train = exact_solution(x_train, region)
    return x_train, y_train


class PINNs(nn.Module):
    def __init__(self, neuron=40):
        super(PINNs, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, neuron),
            nn.Tanh(),
            nn.Linear(neuron, neuron),
            nn.Tanh(),
            nn.Linear(neuron, 1),
        )

    def forward(self, x):
        return self.net(x)

lossfunc = nn.MSELoss(reduction="sum")

def compute_loss(model, x,  y_true,region, ws={"w_pde":1, "w_data":1, "w_bc":1, "w_D1":1, "w_D2":1, "w_N":1}):
    w_pde = ws["w_pde"]
    w_data = ws["w_data"]
    w_bc = ws["w_bc"]
    w_D1 = ws["w_D1"]
    w_D2 = ws["w_D2"]
    w_N = ws["w_N"]
    Params = consts["Params"][region]
    mu = Params["mu"]
    Jz = Params["Jz"]
    
    x.requires_grad = True
    Az = model(x)
    dA_dx =  torch.autograd.grad(Az, x, torch.ones_like(Az), retain_graph=True, create_graph=True)[0]
    drdA = torch.autograd.grad(x*dA_dx, x, torch.ones_like(Az), retain_graph=True, create_graph=True)[0]
    pde_loss = lossfunc(drdA, -x*mu*Jz*mu0)
    
    r_0 = x[0].clone().detach().requires_grad_(True)
    r_1 = x[1].clone().detach().requires_grad_(True)
    A_0_pred = model(r_0)
    A_1_pred = model(r_1)
   

    # Now use these tensors in your mse_loss calculations
    Dirichlet_BC_1 = lossfunc(A_0_pred, y_true[0])
    # print(y_true[0])
    # exit(0)
    Dirichlet_BC_2 = lossfunc(A_1_pred, y_true[-1])


    # 计算诺伊曼边界条件的损失（这里假设是导数为 0）
    Neumann_BC_grad = torch.autograd.grad(outputs=A_0_pred, inputs=r_0, grad_outputs=torch.ones_like(A_0_pred),retain_graph=True, create_graph=True)[0]

    expected_grad = torch.zeros_like(Neumann_BC_grad)
    Neumann_BC = lossfunc(Neumann_BC_grad, expected_grad)

    boundary_loss = w_D1 * Dirichlet_BC_1 + w_D2* Dirichlet_BC_2 + w_N*Neumann_BC 
    
    real_loss = lossfunc(Az, y_true)

    loss = w_pde * pde_loss + w_data * real_loss + w_bc*boundary_loss
    # print("Dirichlet_BC_1:", Dirichlet_BC_1, "Dirichlet_BC_2:", Dirichlet_BC_2, "Neumann_BC:", Neumann_BC)
    # print("pde_loss:", pde_loss, "real_loss:", real_loss, "boundary_loss:", boundary_loss)
    # exit()

    return loss


def train(model, x_data, y_data, region=1, epochs=5000, lr=0.01,ws={"w_pde":1, "w_data":1, "w_bc":1, "w_D1":1, "w_D2":1, "w_N":1}, verbose = False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = compute_loss(model, x_data, y_data, region, ws)
        # y_pred = model(x_data)
        # loss = lossfunc(y_pred, y_data)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 and verbose:
            print("epoch:", epoch, "loss:", loss.item())


def RAE(y_pred, y_true):
    error =  torch.mean(torch.abs(y_pred - y_true)/torch.abs(y_true)) * 100
    return float(error)


if __name__=="__main__":
    torch.manual_seed(9807)
    # x_train, y_train = generate_data(exact_solution)
    # 生成数据
    region = 2
    # 训练模型
    ws = {"w_pde": .001, "w_bc": 0.1, "w_data": 1, "w_D1": 4000, "w_D2": 10, "w_N": 0}
    # ws = {"w_pde": 0, "w_bc": 0, "w_data": 1, "w_D1": 100, "w_D2": 1000, "w_N": 1000}
    print(ws)
    epochs = 2000

    x_train, y_train = generate_data(region=region,  num_points=1000)

    # 创建 MinMaxScaler 实例
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    # 归一化 x_train 和 y_train
    x_train_normalized = torch.tensor(x_scaler.fit_transform(x_train), dtype=torch.float32)
    y_train_normalized = torch.tensor(y_scaler.fit_transform(y_train), dtype=torch.float32)

    model = PINNs()

    train(model=model, x_data=x_train_normalized, y_data=y_train_normalized, region=region, epochs=epochs, lr=0.01,ws=ws, verbose = True)
    model.eval()
    with torch.no_grad():
        # 生成测试数据并进行归一化
        x_test_normalized = torch.tensor(x_scaler.transform(x_train), dtype=torch.float32)
        y_pred_normalized = model(x_test_normalized)

        # 将预测结果反归一化
        y_pred = torch.tensor(y_scaler.inverse_transform(y_pred_normalized), dtype=torch.float32)
    error = RAE(y_pred=y_pred, y_true=y_train)
    
    print("Relative Absolute Error: {:.2f}%".format(error))

    r0 = x_test_normalized[0].clone().detach().requires_grad_(True) 
    A0 = model(r0)
    r1 = x_test_normalized[-1].clone().detach().requires_grad_(True) 
    A1 = model(r1)
    
    
    scale_x = x_scaler.scale_
    scale_y = y_scaler.scale_
    range_x = x_scaler.data_range_
    range_y = y_scaler.data_range_
    print('range_x:', range_x, 'range_y:', range_y)
    print('scale_x:', scale_x, 'scale_y:', scale_y)

    Neumann_grad_0 = torch.autograd.grad(outputs=A0, inputs=r0, grad_outputs=torch.ones_like(A0),retain_graph=True, create_graph=True)[0]
    Neumann_grad_1 = torch.autograd.grad(outputs=A1, inputs=r1, grad_outputs=torch.ones_like(A1),retain_graph=True, create_graph=True)[0]
    print("Neumann_grad_0:", Neumann_grad_0.detach().numpy() * scale_x / scale_y)
    print("Neumann_grad_1:", Neumann_grad_1.detach().numpy() * scale_x / scale_y)
    x_position = max(x_train)  # Start of the x-axis
    y_position = max(y_pred)  # Position text at the minimum of the exact solution for visibility

    # Add text with a black background
    # plt.text(x_position, y_position-1, "Duration: {:.2f}s".format(duration), fontsize=14, color='white', bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.5'))
    plt.text(x_position-range_x/5, y_position-range_y/5, "Error: {:.2f}%".format(error), fontsize=10, color='white', bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.5'), )



    # 可视化
    plt.plot(x_train.numpy(), y_train.numpy(), label='Exact Solution', linestyle='--')
    plt.plot(x_train.numpy(), y_pred.numpy(), label='PINN Predictions', color='red' )
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('A')
    plt.title('Comparison between PINN Solution and Exact Solution')
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建完整的文件路径
    file_path = os.path.join(current_dir, "PINN_region{}.pdf".format(region))
    plt.savefig(file_path)
    plt.show()
