function [y] = Volterra_Equalizer(x, payload, d, L, a)
    %  x: Training data
    %  payload: Payload data
    %  d: Desired Data
    %  L: Filter Length
    %  a: Learning Rate

    f_1 = zeros(1, L);
    f_2 = zeros(1, L*(L+1) / 2);
    f = [f_1 f_2]';
    error = zeros(1, length(x));
    
    %Training Phase
    for n = L:length(x)
        x_n1 = x(n : -1 : n-L+1)'; 
        x_n2_matrix = x_n1* x_n1';
        x_n2_val = triu(x_n2_matrix);
        x_n2 = nonzeros(x_n2_val);
        x_n = [x_n1' x_n2']';
        y = f' * x_n;
        e = d(n) - y;
        error(n) = e;

        f = f + a* e* x_n;
    end
    y = filter(f, 1, payload);

    figure;
    plot(abs(error).^2)
    xlabel('n'); ylabel('MSE'); title('Train Error');
    grid on;
