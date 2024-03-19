function [recall, precision, f1score, specificity]= confusion_matrix_eval(Pre_labels,target_labels)
% Pre_label predicted by the classifier
% target_labels ground truth labels

%calculate values from confusion matrix
 [TP,FP,TN,FN] = TpFp_TnFn(Pre_labels,target_labels);

%calculate Recall
recall = TP/(TP+FN);

%calculate Precision
precision = TP/(TP+FP);

%calculate F1score
f1score = 2*(recall * precision) / (recall + precision);

%calculate Specificity
specificity = TN/(TN+FP);

end