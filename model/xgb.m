clc; clear; close all;
addpath(genpath('../utils'));

[X, Y] = load_data('../data/micro.xlsx', 6);

eta_list   = [0.03 0.05];
depth_list = [3 4];
seed_range = 1:500;

cv = cvpartition(length(Y),'KFold',5);
best_score = -inf;

for seed = seed_range
    set_random_seed(seed)

    for eta = eta_list
        for d = depth_list

            metrics_val = zeros(cv.NumTestSets,5);

            for k = 1:cv.NumTestSets
                tr = cv.training(k);
                te = cv.test(k);

                [Xtr, mu, sg] = zscore_norm(X(tr,:));
                Xte = apply_zscore(X(te,:), mu, sg);

                [ytr_hat, yte_hat] = train_xgb_once( ...
                    Xtr, Y(tr), Xte, Y(te), eta, d, seed);

                metrics_val(k,:) = calc_metrics(Y(te), yte_hat);
            end

            score = mean(metrics_val(:,1));
            if score > best_score
                best_score = score;
                best_param = [eta, d, seed];
            end
        end
    end
end

disp('Best XGBoost [eta, depth, seed]:');
disp(best_param);
