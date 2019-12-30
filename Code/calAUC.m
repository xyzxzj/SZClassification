function [AucTest] = calAUC(predictScore, testLabel)
%% calAUC.m: calucate the AUC value
% input:
% predictScore: Probability that each test sample belongs to a positive sample
% testLabel: True label for each test sample
% output:
% AucTest
    testNum = length(testLabel);
    %change threshold, compute TPR and FPR
    TP_roc = zeros(testNum,1);
    FP_roc = zeros(testNum,1);
    TN_roc = zeros(testNum,1);
    FN_roc = zeros(testNum,1);
    TPR_roc = zeros(testNum,1);
    FPR_roc = zeros(testNum,1);
    predictLabelRoc = zeros(testNum,1);

    for k = 1:testNum 
        for g = 1:testNum
            if predictScore(g) >= predictScore(k)   % exceed threshold, predict label
                predictLabelRoc(g,1) = 1;
            else
                predictLabelRoc(g,1) = -1;
            end
        end
        for h = 1:testNum 
            if predictLabelRoc(h,1) == 1 && testLabel(h,1) == 1;
                TP_roc(k,:) = TP_roc(k,:) + 1;
            end
            if predictLabelRoc(h,1) == 1 && testLabel(h,1) == -1;
                FP_roc(k,:) = FP_roc(k,:) + 1;
            end
            if predictLabelRoc(h,1) == -1 && testLabel(h,1) == -1;
                TN_roc(k,:) = TN_roc(k,:) + 1;
            end
            if predictLabelRoc(h,1) == -1 && testLabel(h,1) == 1;
                FN_roc(k,:) = FN_roc(k,:) + 1;
            end
        end
        TPR_roc(k,:) = TP_roc(k,:)/(TP_roc(k,:) + FN_roc(k,:));   % True positive rate
        FPR_roc(k,:) = FP_roc(k,:)/(FP_roc(k,:) + TN_roc(k,:));   % False positive rate
    end
    TPR_n = TPR_roc;
    FPR_n = FPR_roc;
    PR = cat(2, FPR_n, TPR_n);
    PRSort = sortrows(PR, [1, 2]);
    Auc = zeros(testNum-1, 1);
    for n = 1: testNum-1
        Auc(n) = 0.5*(PRSort(n+1,1)-PRSort(n,1))*(PRSort(n,2)+PRSort(n+1,2));
    end
    AucTest = sum(Auc);
end