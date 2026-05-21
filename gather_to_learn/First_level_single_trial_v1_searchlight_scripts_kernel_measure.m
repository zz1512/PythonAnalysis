function my_result = kernel_measure(ds,argus)
   
    my_result = struct();
    
    sample_qian = ds.samples(1:35,:);
    sample_hou = ds.samples(36:70,:);
    
     rall = 0;
     for x = 1:35
          r= corr(sample_qian(x,:)',sample_hou(x,:)');
          rall = rall+r;
     end
    
    my_result.samples = rall/35;
end

