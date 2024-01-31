%% Hamidreza Abooei Mehrizi - 402617509
clc;
clear; 
close all;


addpath('./FSLib_v7.0.1_2020_2/lib'); % dependencies
addpath('./FSLib_v7.0.1_2020_2/methods'); % FS methods
addpath(genpath('./FSLib_v7.0.1_2020_2/lib/drtoolbox'));

%% Read data
load('.\hw3_TA_data_pattern.mat');

%% Create matrix - gather all data into one single matrix

TH = 26; % number of thresholds

all_region_matrix_gp1 = [];
all_region_matrix_gp2 = [];
all_region_matrix_gp3 = [];


for region = 1:116
    region_matrix_gp1 = [];
    region_matrix_gp2 = [];
    region_matrix_gp3 = [];

    for q = 1:TH
        region_matrix_gp1 = [region_matrix_gp1; All_gp1_local_feature{1,region}{q}];
        region_matrix_gp2 = [region_matrix_gp2; All_gp2_local_feature{1,region}{q}];
        region_matrix_gp3 = [region_matrix_gp3; All_gp3_local_feature{1,region}{q}];
    end

    all_region_matrix_gp1 = [all_region_matrix_gp1 ;region_matrix_gp1];
    all_region_matrix_gp2 = [all_region_matrix_gp2 ;region_matrix_gp2];
    all_region_matrix_gp3 = [all_region_matrix_gp3 ;region_matrix_gp3];
    
end

All_features = [all_region_matrix_gp1, all_region_matrix_gp2, all_region_matrix_gp3]';

num_sub_gp1 = 10;
num_sub_gp2 = 10;
num_sub_gp3 = 11;
sb = num_sub_gp1 + num_sub_gp2 + num_sub_gp3;

label = [ones(1,num_sub_gp1),ones(1,num_sub_gp2)*2,ones(1,num_sub_gp3)*3];

rl = randperm(sb); % random label
data = All_features(rl,:);
label = label(rl);

%% cross-validation
numfold = sb;
indices = crossvalind('kfold',sb,numfold); %leave-one-out

feature_selection_range = 1:10;
correct_num = zeros(numfold,numel(feature_selection_range));

confusion_matrix= zeros(3,3,numel(feature_selection_range));
for i = 1:numfold
    % Seperate train and test 
    X_train = data(indices~=i,:);
    Y_train = label(indices~=i);
    X_test  = data(indices==i,:);
    Y_test  = label(indices==i);

    % Normalization
    train_mean = mean(X_train);
    train_std = std(X_train);
    X_train = (X_train - train_mean)./train_std;
    X_test =  (X_test - train_mean)./train_std;

    numF = size(X_train,2);

    % Feature Selection

    ranking = fscchi2(X_train,Y_train);
%     ranking = spider_wrapper(X_train,Y_train,numF,lower('fisher'));

    X_train = X_train(:,ranking);
    X_test = X_test(:,ranking);
    
    for j = feature_selection_range
        X_train_fs = X_train(:,1:j);
        X_test_fs = X_test(:,1:j);
        
        % One-vs-all classification
        SVMModels = cell(3,1);
        Scores = zeros(1,3);
        for k = 1:3
            Y_train_new = (Y_train==k);
            SVMModels{k} = fitcsvm(X_train_fs,Y_train_new);
            [a,score]=predict(SVMModels{k},X_test_fs);
            Scores(k) = score(:,2); % Second column contains positive-class scores
        end
        [~,Y_pred] = max(Scores,[],2);
        correct_num(i,j) = correct_num(i,j) + (Y_pred==Y_test);
        confusion_matrix(Y_test,Y_pred,j) = confusion_matrix(Y_test,Y_pred,j)+1;
    end

end
accs = mean(correct_num);

figure();
plot(feature_selection_range, accs)
xlabel("Number of Features");
ylabel("Accuracy");
title("Accuracy per number of features. ");

[M,i] = max(accs);

fprintf("Best accuracy confusion matrix:\n");

disp(confusion_matrix(:,:,i));









