function p = predict(Theta1, Theta2, X)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

% --------- %

% layer 1: bias + input nodes
a1 = [ones(m,1) X];

% layer 2
z2 = a1 * Theta1';  % weights
a2 = sigmoid(z2);  % values
a2 = [ones(size(a2,1),1) a2]; % bias + values

% layer 3
z3 = a2 * Theta2'; 
a3 = sigmoid(z3);  % values

[prob, p] = max(a3,[],2); 

end
