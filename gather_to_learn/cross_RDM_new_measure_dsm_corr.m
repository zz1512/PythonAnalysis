function r = new_measure_dsm_corr(my_ds,args)
%% note  no demean here
    r = struct();
    my_d = cosmo_pdist(my_ds.samples, 'correlation');
    rsm=cosmo_squareform(my_d);
    %%
    trial_number=35;
    run3_rsm=rsm(1:trial_number,1:trial_number);
    run4_rsm=rsm(trial_number+1:trial_number+trial_number,trial_number+1:trial_number+trial_number);
    %%
    run3_rsm=1-run3_rsm; 
    run3_rsm=atanh(run3_rsm);
    run3_rsm(run3_rsm==inf)=0;
    run3_rsm(run3_rsm==-inf)=0;
    %%
    run4_rsm=1-run4_rsm; 
    run4_rsm=atanh(run4_rsm);
    run4_rsm(run4_rsm==inf)=0;
    run4_rsm(run4_rsm==-inf)=0;
    %%
    run3_rsm=cosmo_squareform(run3_rsm);
    run4_rsm=cosmo_squareform(run4_rsm);
    r.samples = cosmo_corr(run3_rsm', run4_rsm', 'Pearson');
end



