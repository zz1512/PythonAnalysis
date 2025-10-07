function my_result = kernel_measure_from_F_to_S(ds,~)
   
    my_result = struct();
    trial = size(ds.samples,1);
    sample_qian = ds.samples(1:trial/2,:);
    sample_hou = ds.samples(trial/2+1:trial,:);
    
     rall = 0;
     for x = 1:trial/2
          r= corr(sample_qian(x,:)',sample_hou(x,:)');
          rall = rall+r;
     end
    
    my_result.samples = 2*rall/trial;
end

