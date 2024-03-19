function [ExactM,HamS,MacroF1,MicroF1,AvePre,Rankingloss,Accuracy] = Evaluate(Pre_Labels,test_target)
%% EVALUATION 
    ExactM = Exact_match(Pre_Labels,test_target);
    HamS = Hamming_score(Pre_Labels,test_target);
    
%     [~,~,~,ExampleBasedFmeasure]...
%         = ExampleBasedMeasure(test_target,Pre_Labels);
    [~,~,~,MacroF1]...
        = LabelBasedMeasure(test_target,Pre_Labels);  
    MicroF1 = MicroFMeasure(test_target,Pre_Labels);
    
%     RankS = Ranking_score(Outputs,test_target);
     AvePre = Average_precision(Pre_Labels,test_target);
%      AvePre = Average_precision2(Pre_Labels,test_target);
      Rankingloss = Ranking_loss(Pre_Labels,test_target);
      Accuracy = MultiLabelAccuracyEvaluation(Pre_Labels',test_target');
end
