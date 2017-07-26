function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C=1.000000;
sigma=0.100000;

% return the found C & sigma
return; 

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vals = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_vals = [0.01 0.03 0.1 0.3 1 3 10 30];

for i=1:length(C_vals)
	for j=1:length(sigma_vals)
		C = C_vals(i);
		sigma = sigma_vals(j);
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		predictions = svmPredict(model, Xval);
		err = mean(double(predictions ~= yval));
		prediction_erros(i, j) = err;
		fprintf('C=%f, sigma=%f, prediction_erro=%f', C, sigma, err);
	end
end

min_err = min(min(prediction_erros));
[i, j] = find(prediction_erros == min_err);
C = C_vals(i);
sigma = sigma_vals(j);
fprintf('Best C=%f, sigma=%f, prediction_erro=%f', C, sigma, min_err);





% =========================================================================

end
