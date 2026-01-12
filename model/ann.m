clc; clear; close all;
addpath(genpath('../utils'));

[X, Y] = load_data('../data/nano.xlsx', 6);

hidden_list = {[5],[10],[10 5]};
seed_range = 1:300;

cv = cvpartition(length(Y),'KFold',5);
best_score = -inf;

for seed = seed_range
    set_random_seed(seed)

    for h = 1:numel(hidden_list)

        metrics_val = zeros(cv.NumTestSets,5);

        for k = 1:cv.NumTestSets
            tr = cv.training(k);
            te = cv.test(k);

            [Xtr, mu, sg] = zscore_norm(X(tr,:));
            Xte = apply_zscore(X(te,:), mu, sg);

            net = fitnet(hidden_list{h});
            net.trainFcn = 'trainscg';
            net.divideFcn = 'dividetrain';
            net.trainParam.showWindow = false;

            net = train(net,Xtr',Y(tr)');
            yhat = net(Xte')';

            metrics_val(k,:) = calc_metrics(Y(te), yhat);
        end

        score = mean(metrics_val(:,1));
        if score > best_score
            best_score = score;
            best_param = {hidden_list{h}, seed};
        end
    end
end

disp('Best ANN [hidden, seed]:');
disp(best_param);
