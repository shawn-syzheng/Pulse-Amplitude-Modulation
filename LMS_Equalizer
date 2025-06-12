function y = LMS_Equalizer(x, payload, d, L, a)

    f = zeros(1, L)';

    for n = L:length(x)
        x_n = x(n : -1 : n - L + 1).';
        z = f'* x_n;
        e = d(n) - z;
        f = f + a * e * conj(x_n);
        e_LMS(n) = abs(e).^2;
    end

    y = filter(f, 1, payload);
    figure;
    plot(e_LMS);
    xlabel('n'); ylabel('MSE'); title('Train Error');
    grid on;
