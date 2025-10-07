function r = new_measure_dsm_corr(my_ds,args)
%% note  no demean here
    r = struct();
    
    run3 = my_ds.samples(1:35,:);
    run5 = my_ds.samples(36:70,:);
    run6 = my_ds.samples(71:105,:);
    run56 = (run5+run6)/2;
    %%
    run3_rsm = cosmo_pdist(run3, 'correlation');
    run3_rsm = cosmo_squareform(run3_rsm);
    %%
    run3_rsm=1-run3_rsm; 
    run3_rsm=atanh(run3_rsm);
    run3_rsm(run3_rsm==inf)=0;
    run3_rsm(run3_rsm==-inf)=0;
    run3_rsm=cosmo_squareform(run3_rsm);
    %%
    run56_rsm = cosmo_pdist(run56, 'correlation');
    run56_rsm = cosmo_squareform(run56_rsm);
    
    run56_rsm=1-run56_rsm; 
    run56_rsm=atanh(run56_rsm);
    run56_rsm(run56_rsm==inf)=0;
    run56_rsm(run56_rsm==-inf)=0;
    run56_rsm=cosmo_squareform(run56_rsm);
    
    r.samples = cosmo_corr(run3_rsm', run56_rsm', 'Pearson');
end



