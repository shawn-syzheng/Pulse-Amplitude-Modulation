function y = Reservoir_Computing(u_train, y_target, u_test, TrainLen)
    % -------------------- Set Reservoir Parameter -----------------------------
N_res = 100;         % Reservoir Nodes
rho = 0.9;           % Reservoir Spectral Radius （Recomnadation rho < 1）
input_scaling = 0.1; % Input weight scaling factor
leaking_rate = 0.3;  % The leaking rate, which controls the time constant for state updates

% Generate random input weight matrix (size: N_res x 1, because the input is 1D)
Win = input_scaling * (2*rand(N_res,1) - 1);

% Generate Reservoir random connection matrix W (size: N_res x N_res)
W = rand(N_res, N_res) - 0.5;
% Adjust W so that its spectral radius is rho
eigsW = max(abs(eig(W)));
W = (W / eigsW) * rho;

% -------------------- Reservoir Training Phase -------------------------------
% Initialize the Reservoir state (vector x is N_res x 1)
x = zeros(N_res, 1);
% Collect the Reservoir state at each moment (during training)
X_train = zeros(N_res, TrainLen);

for n = 1:TrainLen
    u = u_train(n);
    % Update the status using leaky integration:
    x = (1 - leaking_rate) * x + leaking_rate * tanh( Win*u + W*x );
    X_train(:, n) = x;
end

% To enhance the expressiveness of linear regression, the input and constant terms can be added together
Feature_train = [ones(1, TrainLen); u_train; X_train];  % 尺寸：(1+1+N_res) x TrainLen

% Training the Readout layer using Ridge Regression
lambda_RC = 1e-4;  % Regularization parameter
I_RC = eye(size(Feature_train,1));
% Solution： W_out = y_target * Feature_train' / (Feature_train * Feature_train' + lambda*I)
W_out = y_target * Feature_train' / (Feature_train * Feature_train' + lambda_RC*I_RC);

% -------------------- Reservoir Testing Phase -------------------------------
TestLen = length(u_test);
y = zeros(1, TestLen);  % Output after storage and equalization
% Reinitialize or use the last training state (reinitialize here)
x = zeros(N_res,1);
for n = 1:TestLen
    u = u_test(n);
    x = (1 - leaking_rate) * x + leaking_rate * tanh( Win*u + W*x );
    feature = [1; u; x];  % Composing the feature vector for testing
    y_hat = W_out * feature;
    y(n) = y_hat;
end
end
