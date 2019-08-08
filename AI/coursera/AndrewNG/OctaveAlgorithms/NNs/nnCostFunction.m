function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ------ %

% Part 1: FP
a1 = [ones(m, 1) X];

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1), 1) a2];  % add bias node

z3 = a2 * Theta2';
a3 = sigmoid(z3);

hThetaX = a3;

yVec = zeros(m,num_labels);
for i = 1:m
    yVec(i,y(i)) = 1;
end
J = 1/m * sum(sum(-1 * yVec .* log(hThetaX)-(1-yVec) .* log(1-hThetaX)));
regulator_term = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * (lambda/(2*m));
J = J + regulator_term;

% Part 2: BP
for t = 1:m
	a1 = [1; X(t,:)'];

	z2 = Theta1 * a1;
	a2 = [1; sigmoid(z2)];

	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	yy = ([1:num_labels]==y(t))';

	delta_3 = a3 - yy;
	delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z2)];
	delta_2 = delta_2(2:end);

	Theta1_grad = Theta1_grad + delta_2 * a1';
	Theta2_grad = Theta2_grad + delta_3 * a2';
end

% Part 3: regularize grads

Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

% ------ %

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
