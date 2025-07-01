function y = AE_Equalizer(x, payload, d, L, a, training_length)

% Network structure
hidden1 = 5;
hidden2 = 2;
hidden3 = 5;

% Xavier-like initialization
W1 = (randn(hidden1, L)) * sqrt(1/L);
b1 = (randn(hidden1, 1)) * sqrt(1/L);

W2 = (randn(hidden2, hidden1)) * sqrt(1/hidden1);
b2 = (randn(hidden2, 1)) * sqrt(1/hidden1);

W3 = (randn(hidden3, hidden2)) * sqrt(1/hidden2);
b3 = (randn(hidden3, 1)) * sqrt(1/hidden2);

W4 = (randn(1, hidden3)) * sqrt(1/hidden3);
b4 = (randn(1)) * sqrt(1/hidden3);

% Training
for n = L:training_length
    x_n = x(n : -1 : n - L + 1).';  % Lx1 input vector

    % === Forward ===
    a1 = W1 * x_n + b1;
    z1 = tanh(a1);

    a2 = W2 * z1 + b2;
    z2 = tanh(a2);

    a3 = W3 * z2 + b3;
    z3 = tanh(a3);

    a4 = W4 * z3 + b4;
    y_hat = a4;

    % === Loss ===
    e = y_hat - d(n);
    e_NN(n) = 0.5 * abs(e)^2;

    % === Backpropagation ===
    delta4 = e;                                      % 1x1
    delta3 = (W4.' * delta4) .* (1 - z3.^2);         % 5x1
    delta2 = (W3.' * delta3) .* (1 - z2.^2);         % 2x1
    delta1 = (W2.' * delta2) .* (1 - z1.^2);         % 5x1

    % === Gradients ===
    dW4 = delta4 * z3.';
    db4 = delta4;

    dW3 = delta3 * z2.';
    db3 = delta3;

    dW2 = delta2 * z1.';
    db2 = delta2;

    dW1 = delta1 * x_n.';
    db1 = delta1;

    % === Update ===
    W4 = W4 - a * dW4;
    b4 = b4 - a * db4;

    W3 = W3 - a * dW3;
    b3 = b3 - a * db3;

    W2 = W2 - a * dW2;
    b2 = b2 - a * db2;

    W1 = W1 - a * dW1;
    b1 = b1 - a * db1;
end

% === Testing ===
x_test = payload(training_length + 1 : end);
x_test = [zeros(1, L-1), x_test];  % zero-padding
y = zeros(1, length(x_test));

for n = L:length(x_test)
    x_n = x_test(n : -1 : n - L + 1).';

    a1 = W1 * x_n + b1;
    z1 = tanh(a1);

    a2 = W2 * z1 + b2;
    z2 = tanh(a2);

    a3 = W3 * z2 + b3;
    z3 = tanh(a3);

    a4 = W4 * z3 + b4;
    y(n) = a4;
end

% === Plot loss ===
figure; plot(e_NN);
xlabel('Iteration'); ylabel('Loss');
title('Training Loss Curve');

end
