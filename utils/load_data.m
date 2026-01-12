function [X, Y] = load_data(filename, nFeature)

data = readmatrix(filename);
X = data(:, 1:nFeature);
Y = data(:, nFeature+1);

valid = ~any(isnan(X),2) & ~isnan(Y) & Y > 0;
X = X(valid,:);
Y = Y(valid);

end
