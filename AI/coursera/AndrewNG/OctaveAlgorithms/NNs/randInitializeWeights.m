function W = randInitializeWeights(L_in, L_out)

W = zeros(L_out, 1 + L_in);

INIT_EPS = 0.001;
W = rand(L_out, 1 + L_in) * 2 * INIT_EPS - INIT_EPS;  % one way

end
