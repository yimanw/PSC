clc; clear; close all;
addpath(genpath('../utils'));

[X, Y] = load_data('../data/pico.xlsx', 6);

seed_range = 1:500;
num_leaves_list = [5 15 31];
lr_list = [0.01 0.05];

cv = cvpartition(length(Y),'KFold',5);
best_score = -inf;

for seed = seed_range
    set_random_seed(seed)

    for nl = num_leaves_list
        for lr = lr_list

            metrics_val = zeros(cv.NumTestSets,5);

            for k = 1:cv.NumTestSets
                tr = cv.training(k);
                te = cv.test(k);

                [Xtr, mu, sg] = zscore_norm(X(tr,:));
                Xte = apply_zscore(X(te,:), mu, sg);

                model = train_lightgbm(Xtr, Y(tr), nl, lr, seed);
                yhat = predict(model, Xte);

                metrics_val(k,:) = calc_metrics(Y(te), yhat);
            end

            score = mean(metrics_val(:,1)); % R²
            if score > best_score
                best_score = score;
                best_param = [nl, lr, seed];
            end
        end
    end
end

disp('Best LightGBM [num_leaves, lr, seed]:');
disp(best_param);
