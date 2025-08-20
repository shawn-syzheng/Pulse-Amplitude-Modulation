function y = lstm(trainseq, payload, desire, L ,alpha, beta1, beta2, eps)
% 4. LSTM
adam.m_w_l = 0;  adam.v_w_l = 0;
adam.m_b_l = 0;  adam.v_b_l = 0;
adam.m_w_i = 0;  adam.v_w_i = 0;
adam.m_b_i = 0;  adam.v_b_i = 0;
adam.m_w_f = 0;  adam.v_w_f = 0;
adam.m_b_f = 0;  adam.v_b_f = 0;
adam.m_w_o = 0;  adam.v_w_o = 0;
adam.m_b_o = 0;  adam.v_b_o = 0;

% Xavier Initialization

w_l = (rand(1, L + 1)*2 - 1).* sqrt(1 / (L + 1)); % Linear Gate Weight (1×L)
w_i = (rand(1, L + 1)*2 - 1).* sqrt(1 / (L + 1)); % Input Gate Weight (1×L)
w_f = (rand(1, L + 1)*2 - 1).* sqrt(1 / (L + 1)); % Forget Gate Weight (1×L)
w_o = (rand(1, L + 1)*2 - 1).* sqrt(1 / (L + 1)); % Output Gate Weight (1×L)
b_l = 1 / (L + 1); % Linear Gate Bias
b_i = 1 / (L + 1); % Input Gate Bias
b_f = 1 / (L + 1); % Forget Gate Bias
b_o = 1 / (L + 1); % Output Gate Bias

c = 0; % Memory
h_1 = 0; % Hidden Gate
iteration_count=0;
% x = trainseq(1:training_length);
% d = desire(1:training_length);
x = trainseq;
d = desire;
delay = round(L/2);

for n = L:length(x)
    iteration_count= iteration_count+1;
    x_n = x(n : -1 : n - L + 1); %(1×L)
    Input = [h_1 x_n]; 
    g = w_l* Input' + b_l; % Signal Input Vector

    z_i = w_i* Input' + b_i; % Input Vector
    f_i = sigmoid(z_i); % Input Gate
    z_f = w_f* Input' + b_f; % Forget Vector
    f_f = sigmoid(z_f); % Forget Gate
    c = g.* f_i + c.* f_f; % Memory
    z_o = w_o* Input' + b_o; % Output Vector
    f_o = sigmoid(z_o); % Output Gate
    a = c.* f_o;
    h_1 = a;
    % define loss, suppose MSE wrt d(n)
    e = (1 / 2)*(a - d(n - delay))^2;
    e_history(n) = e;

    %% --- Backward pass ---
    % d(e)/d(a) = (a - d(n))
    delta = (a - d(n));

    % a = c* f_o
    da_dc = f_o;  % partial a wrt c
    da_do = c;             % partial a wrt f_o

    % => partial e wrt c
    dC = delta* da_dc;
    % => partial e wrt f_o
    dFo= delta* da_do;  
    % => partial e wrt z_o => dFo * f_o*(1-f_o)
    dz_o= dFo * f_o*(1 - f_o);

    % c = g*f_i + c_old*f_f
    % partial c wrt g   => f_i
    % partial c wrt f_i => g
    % partial c wrt f_f => c_old
    dg   = dC * f_i;
    df_i = dC * g;
    df_f = dC * c;

    % g = w_l * Input' + b_l => partial g wrt z_g=1
    dz_g= dg;

    % f_i= sigmoid(z_i) => partial f_i wrt z_i => f_i*(1-f_i)
    dz_i= df_i* f_i*(1-f_i);

    % f_f= sigmoid(z_f) => partial f_f wrt z_f => f_f*(1-f_f)
    dz_f= df_f* f_f*(1-f_f);

    % ============ partial e wrt w_l, w_i, w_f, w_o =============
    % w_l: size [1 x in_dim]
    % => dW_l = dz_g * Input
    dw_l= dz_g * Input;   % scalar * [1 x 5] => [1 x 5], check dimension
    db_l= dz_g;

    dw_i= dz_i * Input;
    db_i= dz_i;

    dw_f= dz_f * Input;
    db_f= dz_f;

    dw_o= dz_o * Input;
    db_o= dz_o;

    %% --- Update ---
    w_l= w_l - alpha* dw_l;
    b_l= b_l - alpha* db_l;
    w_i= w_i - alpha* dw_i;
    b_i= b_i - alpha* db_i;
    w_f= w_f - alpha* dw_f;
    b_f= b_f - alpha* db_f;
    w_o= w_o - alpha* dw_o;
    b_o= b_o - alpha* db_o;
    [w_l, adam.m_w_l, adam.v_w_l] = AdamUpdate(w_l, dw_l, adam.m_w_l, adam.v_w_l, ...
                                                   alpha,beta1,beta2,eps, iteration_count);
    [b_l, adam.m_b_l, adam.v_b_l] = AdamUpdate(b_l, db_l, adam.m_b_l, adam.v_b_l, ...
                                                   alpha,beta1,beta2,eps, iteration_count);

    [w_i, adam.m_w_i, adam.v_w_i] = AdamUpdate(w_i, dw_i, adam.m_w_i, adam.v_w_i, ...
                                                   alpha,beta1,beta2,eps, iteration_count);
    [b_i, adam.m_b_i, adam.v_b_i] = AdamUpdate(b_i, db_i, adam.m_b_i, adam.v_b_i, ...
                                                   alpha,beta1,beta2,eps, iteration_count);

    [w_f, adam.m_w_f, adam.v_w_f] = AdamUpdate(w_f, dw_f, adam.m_w_f, adam.v_w_f, ...
                                                   alpha,beta1,beta2,eps, iteration_count);
    [b_f, adam.m_b_f, adam.v_b_f] = AdamUpdate(b_f, db_f, adam.m_b_f, adam.v_b_f, ...
                                                   alpha,beta1,beta2,eps, iteration_count);

    [w_o, adam.m_w_o, adam.v_w_o] = AdamUpdate(w_o, dw_o, adam.m_w_o, adam.v_w_o, ...
                                                   alpha,beta1,beta2,eps, iteration_count);
    [b_o, adam.m_b_o, adam.v_b_o] = AdamUpdate(b_o, db_o, adam.m_b_o, adam.v_b_o, ...
                                                   alpha,beta1,beta2,eps, iteration_count);
end

% figure; plot(abs(e_history).^2);
% xlabel('n'); ylabel('MSE'); title('Train Error');
% grid on;

y = payload(1 : end);
y_test = y;
% y_test = [zeros(1, L-1) y];

for n = L:length(y_test)
    y_n = y_test(n : -1 : n - L + 1); %(1×L)
    Input = [h_1 y_n]; 
    g = w_l* Input' + b_l; % Signal Input Vector
    z_i = w_i* Input' + b_i; % Input Vector
    f_i = sigmoid(z_i); % Input Gate
    z_f = w_f* Input' + b_f; % Forget Vector
    f_f = sigmoid(z_f); % Forget Gate
    c = g* f_i + c* f_f; % Memory
    z_o = w_o* Input' + b_o; % Output Vector
    f_o = sigmoid(z_o); % Output Gate
    a = c* f_o;
    h_1 = a;
    y(n) = a;
end
