function [TP,FP,TN,FN] = TpFp_TnFn(prdtn,Y)
TP =  full(sum(  (Y == 1) & (prdtn == 1),1)  );
FP =  full(sum(  (Y ~= 1) & (prdtn == 1),1)  );
TN =  full(sum(  (Y ~= 1) & (prdtn ~= 1),1)  ) ;
FN =  full(sum(  (Y == 1) & (prdtn ~= 1),1)  );
end