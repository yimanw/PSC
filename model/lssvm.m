clc; clear; close all;
addpath(genpath('../utils'));
addpath(genpath('../toolbox/LSSVMlab'));

[X, Y] = load_data('../data/micro.xlsx', 6);

gam_list  = logspace(-3,1,6);
sig2_list = logspace(-1,3,6);
seed_range = 1:300;

cv = cvpartition(length(Y),'KFold',5);
best_score = -inf;

for seed = seed_range
    set_random_seed(seed)

    for gam = gam_list
        for sig2 = sig2_list

            metrics_val = zeros(cv.NumTestSets,5);

            for k = 1:cv.NumTestSets
                tr = cv.training(k);
                te = cv.test(k);

                [Xtr, mu, sg] = zscore_norm(X(tr,:));
                Xte = apply_zscore(X(te,:), mu, sg);

                model = initlssvm(Xtr, Y(tr),'f',gam,sig2,'RBF_kernel');
                model = trainlssvm(model);

                yhat = simlssvm(model, Xte);
                metrics_val(k,:) = calc_metrics(Y(te), yhat);
            end

            score = mean(metrics_val(:,1));
            if score > best_score
                best_score = score;
                best_param = [gam, sig2, seed];
            end
        end
    end
end

disp('Best LSSVM [gam, sig2, seed]:');
disp(best_param);
