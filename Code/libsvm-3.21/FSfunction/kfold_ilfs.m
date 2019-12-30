%% file IMRC.m
% Inter-Modality Relationship Constrained Multi-Task Feature Selection
%
%% OBJECTIVE
% argmin_W { sum_i^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2)
%            + rho1 * \|W\|_1 + rho_L2 * ||fi*betai-fj*betaj||^2/d(fi,fj)}
%
function [save_name] = kfold_fsvFS(feature_label,throld,repeats,parameters,KFold,save_folder)
% clear;
% clc;
% close;
tic;
addpath('./utils'); % load utilities
addpath('./data');
addpath(genpath('./libsvm-3.21'));
load(feature_label);

save_folder=strcat(save_folder,'_',num2str(repeats))
if ~exist(save_folder)
    mkdir(save_folder) % 若不存在，在当前目录中产生一个子目录‘Figure’
end
fidname=strcat('UM_lasso_bn_246_',char(throld),'_',num2str(repeats),'.txt');
fide_path=fullfile(save_folder,fidname);
fid = fopen(fide_path, 'w');

for i=1:length(features)
    labels{1, i} = label;
end
% feature_label;
taskNums = length(features);
feature_lenght=size(features{1},2);

lambdaInd = 1;
for par = parameters
		for repeat=1:repeats
    		featselectindex{lambdaInd,repeat}=zeros(size(features{1},2),1);
            all_decision_values{lambdaInd,repeat}=zeros(size(labels{1},1),1);
            
            all_pre_label{lambdaInd,repeat}=zeros(repeats+1,size(features{1},1));
            temp= all_pre_label{lambdaInd,repeat};
            temp(repeats+1,:)=label(:,1);
            all_pre_label{lambdaInd,repeat}=temp;
        end  
	lambdaInd = lambdaInd + 1;
end

lambdaInd = 1;
for par_num = parameters
		for repeat=1:repeats
			%% K-fold cross valindation
			index = crossvalind('KFold', labels{1}, KFold);
			for selectIndex=1:KFold
			   %% split training and testing dataset
				testIndex = (selectIndex==index);
				trainIndex = ~testIndex;
				trainData = features{1}(trainIndex, :);
				testData= features{1}(testIndex, :);
				trainLabel = labels{1}(trainIndex, :);
				testLabel = labels{1}(testIndex, :);

				[trainNum, numF] = size(trainData);  % dimensionality.
				[testNum, ~] = size(testData);
                %%feature select
				[ranking, weights, subset] = ILFS(trainData, trainLabel , 4, 0 );
				
				selectedTrainData =trainData(:,ranking(1:par_num));
				selectedTestData  = testData(:,ranking(1:par_num));
                
                feature_selcet_length(lambdaInd,(repeat-1)*KFold+selectIndex) =size(selectedTestData,2);
				overmeanindex=featselectindex{lambdaInd,repeat};
				overmeanindex(ranking(1:par_num))=overmeanindex(ranking(1:par_num))+1;
				featselectindex{lambdaInd,repeat}=overmeanindex;
				%% classification
                % 构建线性核矩阵
                ktrain = selectedTrainData * selectedTrainData';
                Ktrain = [(1:trainNum)', ktrain];
                ktest = selectedTestData * selectedTrainData';
                Ktest = [(1:testNum)', ktest];
                % SVM train and test
                SKmodel = svmtrain(trainLabel, Ktrain, '-t 4 -b 1'); %#ok<*SVMTRAIN>
                [pre, acc, dec] = svmpredict(testLabel, Ktest, SKmodel, '-b 1');

                pre_label=all_pre_label{lambdaInd,repeat};
                pre_label(repeat,testIndex)=pre;
                all_pre_label{lambdaInd,repeat}=pre_label;
                
                decision_values=all_decision_values{lambdaInd,repeat};
				decision_values(testIndex)=decision_values(testIndex)+dec(:,1);
				all_decision_values{lambdaInd,repeat}=decision_values;
                Kfold_Acc(lambdaInd,(repeat-1)*KFold+selectIndex) = acc(1);
            end   
	end
	lambdaInd = lambdaInd + 1;
end
maxAcc = 0;
best_par_num=0;
lambdaInd = 1;
for par_num = parameters
		for repeat=1:repeats
			predictScore=all_decision_values{lambdaInd,repeat};
			predictLabel=predictScore;
			predictLabel(predictScore>0.5)=1;
			predictLabel(predictScore<=0.5)=-1;
			
			Acc = (sum(predictLabel(labels{1}==1)==1)+sum(predictLabel(labels{1}==-1)==-1))/size(labels{1},1);
			Sen = mean(predictLabel(labels{1}==1)==1);
			Spe = mean(predictLabel(labels{1}==-1)==-1);
			Auc = calAUC(predictScore, labels{1});
			
			Acc_list(repeat)=Acc;
			Sen_list(repeat)=Sen;
			Spe_list(repeat)=Spe;
			Auc_list(repeat)=Auc;
		end
		Kfold_Acc_std=std(Kfold_Acc(lambdaInd,:));
        feature_length_mean=mean(feature_selcet_length(lambdaInd,:))
        fprintf(fid, '****************************************\n');
		fprintf(fid, '********** par_num = %d *********\n', par_num);
		fprintf(fid, '* feature_length =%f, after selcet Mean length=%f *\n', size(features{1},2),feature_length_mean)
		fprintf(fid, ' Mean classification accuracy std: %0.4f%% \n', Kfold_Acc_std);
		fprintf(fid, ' Mean classification accuracy: %0.2f%% \n', 100*mean(Acc_list));
		fprintf(fid, ' Mean classification sensitivity: %0.2f%% \n', 100*mean(Sen_list));
		fprintf(fid, ' Mean classification specificity: %0.2f%% \n', 100*mean(Spe_list));
		fprintf(fid, ' Mean classification auc: %0.4f \n', mean(Auc_list));
		fprintf(fid, '****************************************\n');
        
		if maxAcc < mean(Acc_list)
			maxAcc = mean(Acc_list);
			maxSen = mean(Sen_list);
			maxSpe = mean(Spe_list);
			maxAuc = mean(Auc_list);
			best_par_num=par_num;
		end
	lambdaInd = lambdaInd + 1;
end

fprintf(fid, '*****************Optimal result****************\n');
fprintf(fid, '********** best_par_num = %d *********\n', best_par_num);
fprintf(fid, ' Mean classification accuracy: %0.2f%% \n', 100*maxAcc);
fprintf(fid, ' Mean classification sensitivity: %0.2f%% \n', 100*maxSen);
fprintf(fid, ' Mean classification specificity: %0.2f%% \n', 100*maxSpe);
fprintf(fid, ' Mean classification auc: %0.4f \n', maxAuc);
fprintf(fid, '****************************************\n');



save_name=strcat('UM_Inter_bn_246_',char(throld),'_',num2str(repeats),'.mat')
save_path=fullfile(save_folder,save_name);
save(save_path, 'all_decision_values', '-mat' )

save_name=strcat('UM_Inter_bn_246_all_pre_label',char(throld),'_',num2str(repeats),'.mat')
save_path=fullfile(save_folder,save_name);
save(save_path, 'all_pre_label', '-mat' )

save_name=strcat('UM_Inter_bn_246_featselectindex',char(throld),'_',num2str(repeats),'.mat')
save_path=fullfile(save_folder,save_name);
save(save_path, 'featselectindex', '-mat' )

time = toc;
fprintf(fid, 'total running times = %0.2f\n', time);
fprintf('total running times = %0.2f\n', time);
fclose(fid);
fprintf('vote_IMRC End\n');

