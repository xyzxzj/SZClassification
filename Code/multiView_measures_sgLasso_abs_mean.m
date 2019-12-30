
function [save_name] = multiView_measures_sgLasso_abs_mean(feature_label,throld,repeats,KFold,flag,save_folder)

tic;
addpath(genpath('./libsvm-3.21'));
addpath('./SLEP_package_4.1');
load(feature_label);

save_folder=strcat(save_folder,'_',num2str(repeats))
if ~exist(save_folder)
    mkdir(save_folder) 
end
fidname=strcat('class_result_bn_246_',char(throld),'_',num2str(repeats),'.txt');
fide_path=fullfile(save_folder,fidname);
fid = fopen(fide_path, 'w');

for i=1:length(features)
    labels{1, i} = label;
end

taskNums = length(features);
feature_lenght=size(features{1},2);

opts=[];
% Starting point
opts.init=2;        % starting from a zero point

opts.tol=1e-5; 
opts.tFlag=1;          % run .maxIter iterations
opts.maxIter=1e4;      % maximum number of iterations

% Normalization
opts.nFlag=0;         % without normalization

if strcmp(flag,'group_bn_local')
	opts.ind=get_group_5_bn_local();
end

lambdas =  [1 2 3 4 5 6 7 8 9 10];
lambdas2 = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0];

lambdaInd = 1;
for lambda = lambdas
	lambda2Ind = 1;
	for lambda2 = lambdas2
		for repeat=1:repeats
    		featselectindex{lambdaInd,lambda2Ind,repeat}=zeros(size(features{1},2),1);
            all_decision_values{lambdaInd,lambda2Ind,repeat}=zeros(size(labels{1},1),1);
            
            all_pre_label{lambdaInd,lambda2Ind,repeat}=zeros(repeats+1,size(features{1},1));%11*145
            temp= all_pre_label{lambdaInd,lambda2Ind,repeat};
            temp(repeats+1,:)=label(:,1);
            all_pre_label{lambdaInd,lambda2Ind,repeat}=temp;
        end  
		lambda2Ind = lambda2Ind + 1;
	end
	lambdaInd = lambdaInd + 1;
end

maxAcc = 0;
feature_length_best=0;
lambdaInd = 1;
for lambda = lambdas
	lambda2Ind = 1;
	for lambda2 = lambdas2
		for repeat=1:repeats
			%% K-fold cross valindation
			index = crossvalind('KFold', labels{1}, KFold);
%             index=1:1:145;
			for selectIndex=1:KFold
			   %% split training and testing dataset
				testIndex = (selectIndex==index);
				trainIndex = ~testIndex;
				trainData = features{1}(trainIndex, :);
				testData= features{1}(testIndex, :);
				trainLabel = labels{1}(trainIndex, :);
				testLabel = labels{1}(testIndex, :);

				[trainNum, d] = size(trainData);  % dimensionality.
				[testNum, ~] = size(testData);
                
                %%feature select               
                [W, funVal1, ValueL]= sgLeastR(trainData, trainLabel, [lambda,lambda2], opts);         
				Weight = W';
                Weight_abs=abs(W');
				selectedTrainData_w= trainData;
				selectedTestData_w = testData;
				selectedTrainData =selectedTrainData_w(:,Weight_abs>(mean(Weight_abs)));
				selectedTestData  = selectedTestData_w(:,Weight_abs>(mean(Weight_abs)));
                
                feature_selcet_length(lambdaInd,lambda2Ind,(repeat-1)*KFold+selectIndex) =size(selectedTestData,2);%每一次实验选择的特征长度
				overmeanindex=featselectindex{lambdaInd,lambda2Ind,repeat};
				overmeanindex(Weight_abs(:)>(mean(Weight_abs(:))))=overmeanindex(Weight_abs(:)>(mean(Weight_abs(:))))+1;
				featselectindex{lambdaInd,lambda2Ind,repeat}=overmeanindex;
				%% classification
                ktrain = selectedTrainData * selectedTrainData';
                Ktrain = [(1:trainNum)', ktrain];
                ktest = selectedTestData * selectedTrainData';
                Ktest = [(1:testNum)', ktest];
                % SVM train and test
                SKmodel = svmtrain(trainLabel, Ktrain, '-t 4 -b 1'); %#ok<*SVMTRAIN>
                [pre, acc, dec] = svmpredict(testLabel, Ktest, SKmodel, '-b 1');
               
                pre_label=all_pre_label{lambdaInd,lambda2Ind,repeat};
                pre_label(repeat,testIndex)=pre;
                all_pre_label{lambdaInd,lambda2Ind,repeat}=pre_label;
                decision_values=all_decision_values{lambdaInd,lambda2Ind,repeat};
				decision_values(testIndex)=decision_values(testIndex)+dec(:,1);
				all_decision_values{lambdaInd,lambda2Ind,repeat}=decision_values;
                Kfold_Acc(lambdaInd,lambda2Ind,(repeat-1)*KFold+selectIndex) = acc(1);
            end   
         end
		lambda2Ind = lambda2Ind + 1;
	end
	lambdaInd = lambdaInd + 1;
end

lambdaInd = 1;
for lambda = lambdas
	lambda2Ind = 1;
	for lambda2 = lambdas2
		for repeat=1:repeats
			predictScore=all_decision_values{lambdaInd,lambda2Ind,repeat};
			predictLabel=predictScore;
			predictLabel(predictScore>0.5)=1;
			predictLabel(predictScore<=0.5)=-1;
			
			Acc = (sum(predictLabel(labels{1}==1)==1)+sum(predictLabel(labels{1}==-1)==-1))/size(labels{1},1);
			Sen = mean(predictLabel(labels{1}==1)==1);
			Spe = mean(predictLabel(labels{1}==-1)==-1);
			Auc = calAUC(predictScore, labels{1});
			
			Acc_list(repeat)=Acc;
			Sen_list(repeat)=Sen
			Spe_list(repeat)=Spe
			Auc_list(repeat)=Auc;
		end
		Kfold_Acc_std=std(Kfold_Acc(lambdaInd,lambda2Ind,:));
        feature_length_mean=mean(feature_selcet_length(lambdaInd,lambda2Ind,:))
        fprintf(fid, '****************************************\n');
		fprintf(fid, 'all******* lambda = %f, lambda2 = %f *********\n', lambda, lambda2);
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
			lambda_best=lambda;
			lambda2_best=lambda2;
            feature_length_best=feature_length_mean;
		end
		lambda2Ind = lambda2Ind + 1;
	end
	lambdaInd = lambdaInd + 1;
end

fprintf(fid, '*****************Optimal result****************\n');
fprintf(fid, 'all******* lambda = %f, lambda2 = %f *********\n', lambda_best, lambda2_best);
fprintf(fid, ' selcet feature Mean length: %0.2f \n', feature_length_best);
fprintf(fid, ' Mean classification accuracy: %0.2f%% \n', 100*maxAcc);
fprintf(fid, ' Mean classification sensitivity: %0.2f%% \n', 100*maxSen);
fprintf(fid, ' Mean classification specificity: %0.2f%% \n', 100*maxSpe);
fprintf(fid, ' Mean classification auc: %0.4f \n', maxAuc);
fprintf(fid, '****************************************\n');



save_name=strcat('pre_probability_bn_246_',char(throld),'_',num2str(repeats),'.mat')
save_path=fullfile(save_folder,save_name);
save(save_path, 'all_decision_values', '-mat' )

save_name=strcat('all_pre_label_bn_246_',char(throld),'_',num2str(repeats),'.mat')
save_path=fullfile(save_folder,save_name);
save(save_path, 'all_pre_label', '-mat' )

save_name=strcat('featureSelectIndex_bn_246_',char(throld),'_',num2str(repeats),'.mat')
save_path=fullfile(save_folder,save_name);
save(save_path, 'featselectindex', '-mat' )

time = toc;
fprintf(fid, 'total running times = %0.2f\n', time);
fprintf('total running times = %0.2f\n', time);
fclose(fid);
fprintf('sgLasso End\n');

