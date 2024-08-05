% Analytical solution of a iron cored toroid
% For the explanation of the setup, see the Power Point slides 22-29 from
% the FEM course - 5LWF0
% Author: Stefan Geelen
% First version: June 2024
% Last edited: August 2024

%% Start
clear;
close all;

%% Parameters
MMF_i   = 105;                      % Imposed MMF = N*i
l       = [11, 2, 15,  2]*1e-3;     % Region thickness in [m]: l(k) = rho_(k+1) - rho_k
mu_r    = [1, 1, 2e3, 1];           % Relative permeability of each domain 
MMF     = [0, MMF_i, 0, -MMF_i];    % MMF in each domain

dr = 10e-5;              % Resolution of r for plots in [m]

%% Boundary conditions
A_0free = 0;            % Dirichlet BC: Az(r=rho_4) = Afree

%% Calculation variables
mu_0 = pi*4e-7;         % Permeability of vacuum
mu   = mu_0.*mu_r;      % Magnetic permeability of each domain

rho0 = [0, cumsum(l)];  % Radii of boundaries including origin

Jz = MMF./(pi*(rho0(2:end).^2 - rho0(1:end-1).^2)); % Current density in each domain 

%% Analytical solution
% Function input: rho0, mu, Jz, dr, A_0free

rho = rho0(2:end);      % Radii of domain interfaces only
N_dom = numel(rho);     % Number of domains
nu = 1./mu;             % Inverse of permeability of each domain

%% Matrix construction
% System of equations in matrix notation: M*x=y
M = zeros(2*N_dom);     % 2*Ndom unknowns to be solved (c_k and A_(0,k)) 
y = zeros(2*N_dom, 1);  

M(1, 1) = 1;            % Finite requirement of the field at r=0

for k = 1:(N_dom - 1)
    
    % Continuity of circumferential H at r=rho (solve for c_k)
    M(k+1, k)   =  1;
    M(k+1, k+1) = -1;
    y(k+1) = (rho(k)^2)*(Jz(k) - Jz(k+1))/2;

    % Continuity of Az at r=rho (solve for A_{0,k})
    M(k+N_dom, k)         =  log(rho(k))/nu(k);
    M(k+N_dom, k+1)       = -log(rho(k))/nu(k+1);
    M(k+N_dom, k+N_dom)   =  1;
    M(k+N_dom, k+N_dom+1) = -1;
    y(k+N_dom) = (rho(k)^2)*(Jz(k)/nu(k) - Jz(k+1)/nu(k+1))/4;

end

% Free choice for Az: Az(rho(Ndom)) = Afree
M(2*N_dom, N_dom)   = log(rho(N_dom))/nu(N_dom);
M(2*N_dom, 2*N_dom) = 1;
y(2*N_dom) = A_0free + (Jz(N_dom)*rho(N_dom)^2)/(4*nu(N_dom));

%% Solving the system of equations
x = M\y;

c   = x(1:N_dom);
A_0 = x(N_dom+1:end);

%% Calculation of Az, B and H in the domain
r  = []; 
Az = []; 
H_theta  = []; 
B_theta  = []; 
indx_dom = [];

for k = 1:N_dom
    
    r_loc       = rho0(k):dr:rho0(k+1);
    Az_loc      = A_0(k) - (Jz(k)*(r_loc.^2)/4 - c(k)*log(r_loc))/nu(k);
    H_theta_loc = Jz(k)*r_loc/2 - c(k)./r_loc;

    % Special case to prevent NaN after division by 0
    isr0 = (r_loc==0);
    if any(isr0)
        Az_loc(isr0)      = A_0(k);
        H_theta_loc(isr0) = 0;
    end
    
    r       = [r, r_loc];
    Az      = [Az, Az_loc];
    H_theta = [H_theta, H_theta_loc];
    B_theta = [B_theta, H_theta_loc/nu(k)];
    
end

%% Plotting
figure;
sgtitle('Analytical solution')
% Magnetic vector potential
subplot(3, 1, 1);
plot(r*1e3, Az*1e3, 'LineWidth', 2);
grid on;
xlabel('$r$ [mm]', 'Interpreter', 'latex');
ylabel('$A_z$ [mWb/m]', 'Interpreter', 'latex');
title('Magnetic vector potential distribution');

% Magnetic field density B
subplot(3, 1, 2);
plot(r*1e3, B_theta, 'LineWidth', 2);
grid on;
xlabel('$r$ [mm]', 'Interpreter', 'latex');
ylabel('$B_{\theta}$ [T]', 'Interpreter', 'latex');
title('Magnetic field density distribution');

% Magnetic field strength H
subplot(3, 1, 3);
plot(r*1e3, H_theta/1e3, 'LineWidth', 2.5);
grid on;
xlabel('$r$ [mm]', 'Interpreter', 'latex');
ylabel('$H_{\theta}$ [kA/m]', 'Interpreter', 'latex');
title('Magnetic field strength distribution');
