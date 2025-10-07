function r = new_measure_dsm_corr(my_ds,args)
    
    r = struct();
    run3 = my_ds.samples(1:35,:);
    run5 = my_ds.samples(36:70,:);
    run6 = my_ds.samples(71:105,:);
    
    run56 = (run5+run6)/2;
    
    allr = 0;
     for x = 1:35
          rr = corr(run3(x,:)',run56(x,:)');
          allr = allr+rr;
      end
    
    r.samples = allr/35; 
end



