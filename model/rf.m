clc; clear; close all;
addpath(genpath('../utils'));

[X, Y] = load_data('../data/micro.xlsx', 6);

tree_list = [50 100];
leaf_list = [1 3];
seed_range = 1:300;

cv = cvpartition(length(Y),'KFold',5);
best_score = -inf;

for seed = seed_range
    set_random_seed(seed)

    for nt = tree_list
        for lf = leaf_list

            metrics_val = zeros(cv.NumTestSets,5);

            for k = 1:cv.NumTestSets
                tr = cv.training(k);
                te = cv.test(k);

                [Xtr, mu, sg] = zscore_norm(X(tr,:));
                Xte = apply_zscore(X(te,:), mu, sg);

                model = TreeBagger(nt,Xtr,Y(tr), ...
                    'Method','regression','MinLeafSize',lf);

                yhat = predict(model,Xte);
                metrics_val(k,:) = calc_metrics(Y(te), yhat);
            end

            score = mean(metrics_val(:,1));
            if score > best_score
                best_score = score;
                best_param = [nt, lf, seed];
            end
        end
    end
end

disp('Best RF [trees, minleaf, seed]:');
disp(best_param);
