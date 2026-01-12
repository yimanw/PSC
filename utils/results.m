function m = calc_metrics(y, yhat)

y = y(:); yhat = yhat(:);
R2   = 1 - sum((y-yhat).^2) / sum((y-mean(y)).^2);
MSE  = mean((y-yhat).^2);
RMSE = sqrt(MSE);
MAE  = mean(abs(y-yhat));
MBE  = mean(yhat - y);

m = [R2, MSE, RMSE, MAE, MBE];
end
