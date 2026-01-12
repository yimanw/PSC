clc; clear; close all;
addpath(genpath('../utils'));

[X, Y] = load_data('../data/pico.xlsx', 6);

kernel_list = {'matern52','matern32'};
seed_range = 1:300;

cv = cvpartition(length(Y),'KFold',5);
best_score = -inf;

for seed = seed_range
    set_random_seed(seed)

    for kf = 1:numel(kernel_list)

        metrics_val = zeros(cv.NumTestSets,5);

        for k = 1:cv.NumTestSets
            tr = cv.training(k);
            te = cv.test(k);

            [Xtr, mu, sg] = zscore_norm(X(tr,:));
            Xte = apply_zscore(X(te,:), mu, sg);

            mdl = fitrgp(Xtr,Y(tr), ...
                'KernelFunction',kernel_list{kf}, ...
                'Standardize',false);

            yhat = predict(mdl,Xte);
            metrics_val(k,:) = calc_metrics(Y(te), yhat);
        end

        score = mean(metrics_val(:,1));
        if score > best_score
            best_score = score;
            best_param = {kernel_list{kf}, seed};
        end
    end
end

disp('Best GPR [kernel, seed]:');
disp(best_param);
