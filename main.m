clc;
clear;
close all;
addpath(genpath('.\'));
% data_sets = ["satimage_multiclass","wine","DNA","COIL20"];
%data_sets = ["procancer","endocrinecancer","cnscancer"];
data_sets=["ovary_can","breast_can","globun_can","brain_can1","lung_can","pomeroy", "nakayam","sing_procancer"];
%data_sets=["alon","bur","chin","chowdary","brain_can","gravier","west","sun","ship","Leukemia"];

%data_sets = ["mirflickr"];
data_set = data_sets(5);
save_results = 0;
xlsx_write = 0;

range = ["b1:j14"];
for f = 1:length(data_set)
    
   % percent = [0.9 0.8 0.7 0.6 0.5 ];
%percent=[0 0.1,0.2,0.3,0.4,0.5];
  %percent = [0.9 0.7 0.5 ];   
percent = [0.5];
%    
    for ii = 1:length(percent)
        %% Load a multi-label dataset
        dataset = data_set(1);
        load(strcat(dataset,".mat"));
       
        %% set all 0 to -1 in target
        target(target(:,:)==0) = -1;
        %===========for multi-class data==================
        target = target';
        %=============================
%         rng(9,'Twister');
%         scurr = rng;
%        
        %% Randomly select part of data
        max_num = 7000;
        if size(data,1) > max_num
            nRows = size(data,1);
            nSample = max_num;
            rndIDX = randperm(nRows);
            index = rndIDX(1:nSample);
            data = data(index, :);
            target = target(:,index);
        end
                
        
        %% randomisation of the dataset ()
        nRows = size(data,1);
        nSample = nRows;
        rndIDX = randperm(nRows);
        index = rndIDX(1:nSample);
        data = data(index, :);
        target = target(:,index);
        %% normalise
        
        normalise = 1;
        if normalise==1
            data = svdatanorm(data,'ker');
            %data = normalize(data,'zscore');
        end
        
        
        
        %% Perform n-fold cross validation and obtain evaluation results
        rng(15,'Twister');
        scurr = rng;
                             
        num_fold = 5; num_metric = 5; num_method = 1;
        indices = crossvalind('Kfold',size(data,1),num_fold);
               
        Results = zeros(num_metric+1,num_fold,num_method);
       for i = 1:num_fold
            
            test = (indices == i);
            a = i+1;
            if a>num_fold
                a = 1;
            end
            vald = (indices == a );
            train = ~test & ~vald;
            
            
            %disp(['Fold ',num2str(i)]);
            fprintf('Fold %d  ',i);
            kernel='rbf';
            
            data_train = data(logical(train+vald),:);
            target_train = target(:,logical(train+vald))';
            data_test = data(logical(test),:);
            target_test = target(:,logical(test))';
                                   
            per = percent(ii);
                        
%% ======================Proposed Algo=============================================================
            
           par.alpha=100; par.beta=0.3;;
            [target_train_incomp] = mask_target_entries(target_train, per);
          [Pre_Labels_train,Pre_Labels_test,time,obj] = agmlgs_fun(data_train,target_train_incomp,data_test,target_test,par);
            
            
 %% =======================Result Evaluation=======================================================
                                   
           Results(1,i,1) = time;
%           [ExactM_train,HamS_train,MacroF1_train,MicroF1_train,AvePre_train] = Evaluation(Pre_Labels_train',target_train');
           [ExactM_test,HamS_test,MacroF1_test,MicroF1_test,AvePre_test] = Evaluation(Pre_Labels_test',target_test');
          % [ExactM,HamS,MacroF1,MicroF1,AvePre,Rankingloss,Accuracy] = Evaluate(Pre_Labels_test',target_test')
            Results(2:end,i,1) = [ExactM_test,HamS_test,MacroF1_test,MicroF1_test,AvePre_test];         
                       
        end
        ignore = [];  Results(:,:,ignore) = [];
        meanResults = squeeze(mean(Results,2));
        stdResults = squeeze(std(Results,0,2) / sqrt(size(Results,2)));
        
        %addpath("D:\research-track\table generators latex");
        meanResults = three_decimals(meanResults);
        stdResults = three_decimals(stdResults);
        
        combined = (strcat(arrayfun(@num2str,meanResults,'un',0),'±',arrayfun(@num2str,stdResults,'un',0)));
        
        %% Save the evaluation results
        if save_results == 1
            filename=strcat("SR_ML_time_Results_",dataset,'.mat');
            save(filename,'meanResults','stdResults','-mat');
        end
        %% Show the experimental results
        disp(dataset); disp(per);
        disp([meanResults]);
        disp([stdResults]);
        %         disp('=========================standard deviation==================================');
        %         disp(stdResults);
        %d = {char(dataset),' ',' ',' ',' ',' ',' ', ' '};%,' ',' ',' ',' '};
        if xlsx_write == 1
            n = {dataset,'percentage of',' missing labels is ',per,'','','','',''};
            headers = {per,'algo1','algo2','algo3','algo4','algo5','algo6','algo7','algo8'}; %comparing algo names
            %columns = {'Time';'Exact match Train';'Exact match Test';'Hamming Loss Train';'Hamming Loss Test';'Macro F1 Train';'Macro F1 Test';...
            %    'Micro F1 Train';'Micro F1 Test';'Avg Precision Train';'Avg Precision Test';};
            columns = {'Time';'Exact match';'Hamming Loss';'Macro F1';...
                'Micro F1';'Avg Precision';};
            %data_as_cell1 = num2cell([meanResults]);
            %data_as_cell2 = num2cell([stdResults]);
            data_as_cell = combined;
            %info_to_write = [headers; data_as_cell1;n;data_as_cell2];
            info_to_write = [data_as_cell;];
            info_to_write = [columns info_to_write];
            info_to_write = [n; info_to_write];
            % xlswrite(strcat(dataset,".xls"), info_to_write,range(f));
            write_data = cell2table(info_to_write,"VariableNames",[" ","algo1","algo2","algo3","algo4","algo5","algo6","algo7","algo8"]);
            filename=strcat(dataset,num2str(per),'.xlsx');
            writetable(write_data,filename);
        end
    end
end


