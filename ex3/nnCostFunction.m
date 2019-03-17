function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Theta1 has shape (25, 401) or (n2, n1+1)
% Theta2 has shape (10, 26) or (n3, n2+1) where ni is the number of units in layer i
% X has shape (5000, 401) or (m, n1+1)
X = [ ones(m, 1) X ];
% y is a label vector of size (m, 1), each can be a number from 1 to K

% Layer 2 or hidden layer
z2 = X * Theta1';
a2 = sigmoid(z2); % shape (m, n2)
a2 = [ ones(m, 1) a2 ];    % (m, n2+1)
% Layer 3 or output layer
a3 = sigmoid(a2 * Theta2'); % a3 = h shape (m, n3) and n3 = K

%% compute unregularized cost function
for c = 1:num_labels,
    yc = (y == c);
    hc = a3(:, c);
    J = J + (-1/m) * (yc' * log(hc) + (1 - yc)' * log(1-hc));
end

%% add regularization terms
temp1 = Theta1;
temp1(:, 1) = 0; % get rid of the bias terms = first column
temp2 = Theta2;
temp2(:, 1) = 0; % get rid of the bias terms = first column
J = J + lambda/(2*m) * (sum(sum(temp1 .^ 2)) + sum(sum(temp2 .^ 2)));

%% Back propagation: compute gradients

% output layer or layer 3: delta_3 should have the same shape as a3 of (m, n3)
delta3 = zeros(m, num_labels);
for c = 1:num_labels,
    delta3(:, c) = a3(:, c) - (y == c);
% Theta2 has shape (10, 26) or (n3, n2+1)
% a2 has shape (m, n2+1)
% hidden layer or layer 2: delta_2 then has shape (m, n2) 
delta2 = (delta3 * Theta2(:, 2:end)) .* sigmoidGradient(z2);

%Theta2_grad has the same shape as Theta2 (n3, n2+1)
Theta2_grad = (1/m) * delta3' * a2;

%Theta1_grard has the same shape as Theta1(n2, n1+1)
Theta1_grad = (1/m) * delta2' * X;

%% add regularization for gradient
Theta1_grad = Theta1_grad + (lambda/m) * temp1;
Theta2_grad = Theta2_grad + (lambda/m) * temp2;







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
