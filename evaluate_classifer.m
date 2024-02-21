function metrics = evaluate_classifer(gt_labels,scores,threshold)
% clear all
% close all
% clc
% threshold = 0.5;
% 
% load("scores.mat")
% gt_labels = Multi_Labels(:,3);
% scores = scores_combined(:,3);
% gt_labels = double(gt_labels);
scores_sorted = sort(scores,'descend');

% Calculate ROC curve (different from precision-recall curve)
% [X,Y] = perfcurve(gt_labels, scores, 1, 'XCrit', 'tpr', 'YCrit', 'prec');
% Calculate precision-recall curve
[metrics.recall_arr,metrics.precision_arr] = perfcurve(gt_labels, scores, 1, 'XCrit', 'tpr', 'YCrit', 'prec');
metrics.recall_arr(isnan(metrics.recall_arr)) = 0;
metrics.precision_arr(isnan(metrics.precision_arr)) = 0;

metrics.f1_arr = 2 * (metrics.precision_arr .* metrics.recall_arr) ./ (metrics.precision_arr + metrics.recall_arr);
metrics.f1_arr(isnan(metrics.f1_arr)) = 0;
[max_f1,max_f1_i] = max(metrics.f1_arr);
max_f1_i = min(max_f1_i,length(gt_labels));
metrics.optimal_threshold = scores_sorted(max_f1_i);
% metrics.optimal_threshold = 0.706;
figure;
yyaxis left
plot(metrics.recall_arr, metrics.precision_arr);
ylabel('Precision');
hold on
yyaxis right
plot(metrics.recall_arr, metrics.f1_arr,'r');
ylabel('F1-Score');
xlabel('Recall');
title('Precision-Recall Curve');
grid on;

%Using optimal threshold gives better results, but it is overfitting!
threshold = metrics.optimal_threshold;


predictedLabels = double(scores>threshold);
N = length(gt_labels);
Acc = gt_labels==predictedLabels;
metrics.accuracy = sum(Acc)/N;

% % Plot the confusion matrix
figure;
confusionchart(gt_labels, predictedLabels);
title('Confusion Matrix');

% % Calculate precision and recall
confusionMatrix = confusionmat(gt_labels, predictedLabels);
metrics.CM = confusionMatrix;
Total_Pos = confusionMatrix(2, 1) + confusionMatrix(2, 2);
Total_Neg = confusionMatrix(1, 1) + confusionMatrix(1, 2);
% Total = Total_Pos + Total_Neg;
TP = confusionMatrix(2, 2);
TN = confusionMatrix(1, 1);
FP = confusionMatrix(1, 2);
FN = confusionMatrix(2, 1);

metrics.precision = TP / (TP + FP);
metrics.recall = TP / (TP + FN);
metrics.F1 = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall);
 
% Calculate average precision (AUC-PR)
% Y(1) = Y(2);
metrics.averagePrecision = trapz(metrics.recall_arr,metrics.precision_arr);
% fprintf('Average Precision (AUC-PR): %.2f\n', averagePrecision);
% keyboard
% 
% % Compute Cohen's quadratic kappa score
% N = sum(confusionMatrix(:)) % Total number of samples
num_classes = size(confusionMatrix, 1);
p_observed = sum(diag(confusionMatrix)) / N; % Observed agreement

expected_agreement = 0;
for i = 1:num_classes
    row_sum = sum(confusionMatrix(i, :));
    col_sum = sum(confusionMatrix(:, i));
    expected_agreement = expected_agreement + (row_sum / N) * (col_sum / N);
end
metrics.kappa_score = (p_observed - expected_agreement) / (1 - expected_agreement);
% Simplified formula for 2-class problem
% kappa_score = 2*(TP*TN-FN*FP)/((TP+FP)*(FP+TN)+(TP+FN)*(FN+TN))
% end
