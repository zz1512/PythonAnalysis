function r = new_measure_dsm_corr(my_ds,args)
    
    r = struct();
    
    trial = size(my_ds.samples,1)/3;
    run3 = my_ds.samples(1:trial,:);
    word1 = my_ds.samples(trial+1:trial*2,:);
    word2 = my_ds.samples(trial*2+1:trial*3,:);
    
    word = (word1+word2)/2;
    
    allr = 0;
     for x = 1:trial
          rr = corr(run3(x,:)',word(x,:)');
          allr = allr + atanh(rr);
      end
    
    r.samples = allr/trial; 
end



