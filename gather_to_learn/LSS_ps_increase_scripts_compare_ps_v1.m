function r = compare_ps_v1(my_ds,args)
    
    r = struct();
    run4 = my_ds.samples(1:35,:);
    run5 = my_ds.samples(36:70,:);
    run6 = my_ds.samples(71:105,:);
    run1 =  my_ds.samples(106:140,:);
    run2 = my_ds.samples(141:175,:);
    
    run56 = (run5+run6)/2;
    run12 = (run1+run2)/2;
    
    
    allr1 = 0;
     for x = 1:35
          rr = corr(run4(x,:)',run12(x,:)');
          allr1 = allr1+rr;
     end
    
      allr2 = 0;
     for x = 1:35
          rr = corr(run4(x,:)',run56(x,:)');
          allr2 = allr2+rr;
      end
     
    r.samples = (allr2-allr1)/35; 
end



