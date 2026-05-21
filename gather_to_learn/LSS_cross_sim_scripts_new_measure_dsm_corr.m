function r = new_measure_dsm_corr(my_ds,args)
%% 
    r = struct();
    run3 = my_ds.samples(1:35,:);
    run4 = my_ds.samples(36:70,:);
    
    allr=0;
     for x = 1:35
          rr = corr(run3(x,:)',run4(x,:)');
          allr = allr+rr;
      end
    
    r.samples = allr/35; 
end



