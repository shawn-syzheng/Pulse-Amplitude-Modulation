function y = LC_LSTM(trainseq, payload, desire, training_length, L, ...
                     alpha, beta1, beta2, eps)
% ---------------------------------------------------------
% 2-Gate LSTM variant WITHOUT any tanh:
%   - Combine input + forget => update gate (u)
%   - Keep output gate => f_o
%   - candidate c_tilde = (w_c * input + b_c) [pure linear]
%   - final output a = c * f_o [pure linear, no scaling]
%
%   Single pass, single-dim c/h, minimal code
% ---------------------------------------------------------

% =============== ADAM states ===============
adam.m_w_c = 0;  adam.v_w_c = 0;
adam.m_b_c = 0;  adam.v_b_c = 0;

adam.m_w_u = 0;  adam.v_w_u = 0;
adam.m_b_u = 0;  adam.v_b_u = 0;

adam.m_w_o = 0;  adam.v_w_o = 0;
adam.m_b_o = 0;  adam.v_b_o = 0;

% =============== Initialize weights ===============
inDim = L + 1;  % input = [h(t-1), x_n(1..L)]
% candidate c
w_c = (rand(1, inDim)*2 - 1).* sqrt(1/inDim); 
b_c = 0;

% update gate
w_u = (rand(1, inDim)*2 - 1).* sqrt(1/inDim); 
b_u = 0;

% output gate
w_o = (rand(1, inDim)*2 - 1).* sqrt(1/inDim); 
b_o = 0;

% =============== Hidden states ===============
c = 0;    % cell state
h_1 = 0;  % hidden output

iteration_count = 0;

% =============== Training data ===============
x = trainseq(1:training_length);
d = desire(1:training_length);

% =============== Single pass training ===============
for n = L : length(x)
    iteration_count = iteration_count + 1;

    % prepare input
    x_n = x(n : -1 : n - L + 1);  % (1xL)
    Input = [h_1, x_n];           % (1 x (L+1))

    %% ------ Forward ------
    % candidate cell = linear
    z_c = w_c * Input' + b_c;   % scalar
    c_tilde = z_c;             % no tanh => pure linear

    % update gate
    z_u = w_u * Input' + b_u;
    u   = sigmoid(z_u);

    c_old = c;
    % c(t) = u*c_tilde + (1-u)*c_old
    c = u.* c_tilde + (1-u).* c_old;

    % output gate
    z_o = w_o * Input' + b_o;
    f_o = sigmoid(z_o);

    % final output: a = c * f_o
    a = c .* f_o;
    h_1 = a;

    % MSE wrt d(n)
    e = 0.5 * (a - d(n))^2;
    e_history(n) = e;

    %% ------ Backward (1-step) ------
    delta = (a - d(n));

    % a = c * f_o
    % partial a wrt c => f_o
    % partial a wrt f_o => c
    da_dc = f_o;
    da_do = c;

    dC = delta .* da_dc;   % partial e wrt c
    dFo= delta .* da_do;   % partial e wrt f_o

    % f_o= sigmoid(z_o)
    dz_o= dFo .* f_o.*(1 - f_o);

    % c= u*c_tilde + (1-u)*c_old
    % partial c wrt c_tilde => u
    % partial c wrt u => c_tilde - c_old
    d_cTilde= dC .* u;
    d_u = dC .* (c_tilde - c_old);

    % c_tilde= z_c (pure linear => partial c_tilde wrt z_c => 1)
    dz_c= d_cTilde;   % no tanh => derivative = 1

    % u= sigmoid(z_u)
    dz_u= d_u .* (u.*(1-u));

    % gradient wrt weights
    dw_c= dz_c * Input;
    db_c= dz_c;

    dw_u= dz_u * Input;
    db_u= dz_u;

    dw_o= dz_o * Input;
    db_o= dz_o;

    % --- update with Adam ---
    [w_c, adam.m_w_c, adam.v_w_c] = AdamUpdate(w_c, dw_c, ...
        adam.m_w_c, adam.v_w_c, alpha,beta1,beta2,eps, iteration_count);
    [b_c, adam.m_b_c, adam.v_b_c] = AdamUpdate(b_c, db_c, ...
        adam.m_b_c, adam.v_b_c, alpha,beta1,beta2,eps, iteration_count);

    [w_u, adam.m_w_u, adam.v_w_u] = AdamUpdate(w_u, dw_u, ...
        adam.m_w_u, adam.v_w_u, alpha,beta1,beta2,eps, iteration_count);
    [b_u, adam.m_b_u, adam.v_b_u] = AdamUpdate(b_u, db_u, ...
        adam.m_b_u, adam.v_b_u, alpha,beta1,beta2,eps, iteration_count);

    [w_o, adam.m_w_o, adam.v_w_o] = AdamUpdate(w_o, dw_o, ...
        adam.m_w_o, adam.v_w_o, alpha,beta1,beta2,eps, iteration_count);
    [b_o, adam.m_b_o, adam.v_b_o] = AdamUpdate(b_o, db_o, ...
        adam.m_b_o, adam.v_b_o, alpha,beta1,beta2,eps, iteration_count);
end

% figure; plot(abs(e_history).^2);
% xlabel('n'); ylabel('MSE'); title('Train Error (No Tanh LSTM)');
% grid on;

%% ---------- Inference on payload ----------
y = payload;
for n = L : length(y)
    x_n = y(n : -1 : n - L + 1);
    Input= [h_1, x_n];

    z_c= w_c*Input' + b_c;
    c_tilde= z_c;   % no tanh
    z_u= w_u*Input' + b_u;
    u= sigmoid(z_u);

    c= u.* c_tilde + (1-u).* c;

    z_o= w_o*Input' + b_o;
    f_o= sigmoid(z_o);

    a= c.* f_o;
    h_1= a;
    y(n)= a;
end

end
