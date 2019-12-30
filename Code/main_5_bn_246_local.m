delete(gcp('nocreate'))
clear;
clc;
close;
rehash

tic;
addpath('../Dataset');
addpath(genpath('./libsvm-3.21'));
addpath(genpath('./SLEP_package_4.1'));
repeats=1
KFold=145 
root=strcat('../Dataset')
save_root=strcat('./result');

folder_list_d = dir(root);
folder_list=folder_list_d(end:-1:1)
len = size(folder_list,1);
fprintf('feature_label_num=%d\n',len)
for i=1:len
      feature_name=folder_list(i).name;
     
     if isempty(strfind(feature_name,'feature'));
        continue;
     end
     if isempty(strfind(feature_name,'features_label_30'));
        continue;
     end
	flag='group_bn_local';
    throld=char(feature_name(16:17))
	feature_label = fullfile(root,feature_name);
       
 	%%%%%%%%%%%%%%%%%%%%%%%%%%vote_sglasso_svm%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	save_folder=strcat(save_root);
	save_path=multiView_measures_sgLasso_abs_mean(feature_label,throld,repeats,KFold,flag,save_folder);


end;
return






