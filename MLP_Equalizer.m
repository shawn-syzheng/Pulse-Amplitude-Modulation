function y = MLP_Equalizer(x, payload, d, L, a, training_length) % Artificial Neural Network Equalizer

W1 = (rand(1, L)*2 - 1) * sqrt(1/L);
W2 = (rand(1, L)*2 - 1) * sqrt(1/L);
b1 = (rand(1, L)*2 - 1) * sqrt(1/L);
b2 = sqrt(1/L);

for n = L:length(x)
    x_n = x(n : -1 : n - L + 1);
    a1 = W1.* x_n + b1;
    z1 = tanh(a1);
    a2 = W2* z1' + b2;
    e = (1/2)* (a2-d(n))^2;
    e_NN(n) = e;
    delta2 = (a2 - d(n));
    delta1 = ( W2.* delta2 ) .* (1 - z1.^2 );
    dW2 = delta2.* z1;
    db2 = delta2;
    dW1 = delta1.* x_n;
    db1 = delta1;
    W2 = W2 - a* dW2;
    W1 = W1 - a* dW1;
    b2 = b2 - a* db2;
    b1 = b1 - a* db1;
end

x_test = payload(training_length + 1 : end);
x_test = [zeros(1, L-1) x_test];
rx4_test = zeros(1, length(x_test));
for n = L:length(x_test)
    y_n = x_test(n : -1 : n - L + 1);
    a1 = W1.* y_n + b1;
    z1 = tanh(a1);
    a2 = W2* z1' + b2;
    y(n) = a2;
end

figure;
plot(e_NN);
end
