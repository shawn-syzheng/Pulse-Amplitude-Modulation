function [theta, m, v] = AdamUpdate(theta, grad, m, v, alpha,beta1,beta2, eps, iter)
    % Adam
    m= beta1*m + (1-beta1)* grad;
    v= beta2*v + (1-beta2)*(grad.^2);

    % bias correction
    m_hat= m/(1-beta1^iter);
    v_hat= v/(1-beta2^iter);

    % update
    theta= theta - alpha*( m_hat./ (sqrt(v_hat)+eps) );
end
