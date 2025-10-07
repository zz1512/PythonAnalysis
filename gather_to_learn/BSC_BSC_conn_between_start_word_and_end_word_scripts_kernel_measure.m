function my_result = kernel_measure(ds,argus)
   
    my_result = struct();
    
    sample_qian = mean(ds.samples(1:35,:),2);
    sample_hou = mean(ds.samples(36:70,:),2);
    
    r = corr(sample_qian,sample_hou);
    
    my_result.samples = atanh(r);
end
