% 仅仅限于continuous的扫描

TRsec = 2;
nSlices = 30;
TA = TRsec/nSlices; %assumes no temporal gap between volumes
bidsSliceTiming=[0:TA:TRsec-TA]; %ascending
% if false %descending
   bidsSliceTiming = flip(bidsSliceTiming);
% end
% if true %interleaved
    order = [1:2:nSlices 2:2:nSlices]
    bidsSliceTiming(order) = bidsSliceTiming;
% end
%report results
fprintf('	"SliceTiming": [\n');
for i = 1 : nSlices
    fprintf('		%g', bidsSliceTiming(i));
    if (i < nSlices)
        fprintf(',\n');
    else
        fprintf('	],\n');
    end  
end