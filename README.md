# License
Copyright (C) 2019 yizhen xiang(xyzxzj@csu.edu.cn),Jin Liu(liujin06@csu.edu.cn)

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, see http://www.gnu.org/licenses/.

yizhen xiang(xyzxzj@csu.edu.cn),Jin Liu(liujin06@csu.edu.cn) School of Computer Science and Engineering, Central South University ChangSha 410083, CHINA. 

# Type: Package Title: Schizophrenia Identification Using Multi-View Graph Measures of Functional Brain Networks
Description: This package aims to achieve the automatically diagnose subjects with schizophrenia based on multi-view graph measures of functional brain networks derived from their Resting-state functional magnetic resonance imaging (Rs-fMRI) brain scans.

Files: 1.Dataset

         1. features_label_30.mat: store 5 local graph measures extracted at the threshold of 0.3 of all subjects and the corresponding labels;

2. Code

   1. libsvm-3.21(http://www.csie.ntu.edu.tw/~cjlin/libsvm/): This is an open source SVM package, the linear SVM is implemented using the libsvm-3.21 package.

   2. SLEP_package_4.1[1]: This is a Sparse Learning with Efficient Projections package,which implements various of sparse learning algorithms. The sparse group lasso(sgLasso) method is implemented using this package.

   3. calAUC.m: function calculating AUC values.

   4. main_5_bn_246_local.m: This is a main function to load data and call the function "vote_Graph_sgLasso_abs_mean";

   5. vote_Graph_sgLasso_abs_mean.m: function conducting schizophrenia classification using multi-view graph measures;

   6. get_group_5_bn_local.m : function constructing grouping structures of 5 local graph measures according to the corresponding brain regions;

# How to run this project

This project must run in **Matlab >=2016a**, The following steps should be taken to run this project:
1. **Before running the codes, please first run "mexC.m" of the SLEP_package_4.1 to mex the related C functions. please note that if an error occurs in mex, maybe you need to download and install the C compiler to solve, such as TDM-GCC.**
2. After run "mexC.m" successfully, please run main_5_bn_246_local.m directly.
3. After main_5_bn_246_local.m finishes running, a folder named "result" will be generated. there are four files in this folder.
   1. all_pre_label_bn_246_30_1.mat: store the predict labels for all samples.
   2. featureSelectIndex_bn_246_30_1.mat: store the index of selected features in each round of experiments.
   3. pre_probability_bn_246_30_1.mat: store the predict probabilities for all samples.
   4. class_result_bn_246_30_1.txt：record the experimental results of each experiment and the optimal results of all experiments.
   

[1]Liu J, Ji S, Ye J. SLEP: Sparse learning with efficient projections[J]. Arizona State University, 2009, 6(491): 7.
