function [y] = NG_RC(data, target, TrainLen, L)
% NG-Reservoir Computing
%% 2. Set NG-RC (memory + polynomial order)
% TrainLen = 6000;
% L = 3;
%% Split training data and testing data
u_train = data(1:TrainLen);    % Receiver Training Sequence (Trainging data set)
y_target = target(1:TrainLen);     % Transmitter Symbol (Target data set)
u_test  = [u_train(end-L+2 : end) data(TrainLen+1:end)];   % Tesing data set

%% 2. Set NG-RC (memory + polynomial order)

dim_lin = L;          % Linear term
dim_quad = L*(L+1)/2;          % Second order term
dim_feature = 1 + L + dim_quad;
num_train_valid = length(u_train) - (L-1);

% Seting training feature space, rows=dim_feature, columns=num_valid
Phi_train = zeros(dim_feature, num_train_valid);

% Establish Training Feature
for n = L:length(u_train)
        x_n1 = u_train(n : -1 : n-L+1)'; 
        x_n2_matrix = x_n1* x_n1';
        x_n2_val = triu(x_n2_matrix);
        x_n2 = nonzeros(x_n2_val);
        phi_train = [1 x_n1' x_n2']';
        Phi_train(:, n-L+1) = phi_train;
end

[dimFeature, numTrainValid] = size(Phi_train);
% fprintf('Feature dimension = %d, Train samples = %d\n',...
%          dimFeature, numTrainValid);

%% 3. Ridge Regression training
lambda = 1e-4;
Ireg = eye(dimFeature);
W_out = (y_target(L:end) * Phi_train') / (Phi_train*Phi_train' + lambda*Ireg);

%% 4. Testing Phase
num_test_valid = length(u_test) - (L-1);
Phi_test = zeros(dim_feature, num_test_valid);
for n = L:length(u_test)
        y_n1 = u_test(n : -1 : n-L+1)'; 
        y_n2_matrix = y_n1* y_n1';
        y_n2_val = triu(y_n2_matrix);
        y_n2 = nonzeros(y_n2_val);
        phi_test = [1 y_n1' y_n2']';
        Phi_test(:, n-L+1) = phi_test;
end

Y_hat = W_out * Phi_test;   % (1 × numTestValid)
y = Y_hat;
