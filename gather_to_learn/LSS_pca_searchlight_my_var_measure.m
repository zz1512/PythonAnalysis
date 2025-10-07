function EV = my_var_measure(my_ds,args)
    EV=struct();
    total_explained=0;
    %%
    p1 = my_ds.samples(1:35,:);
    p2 = my_ds.samples(36:70,:);
    data = (p1+p2)*0.5;
    data = data';
    [~,~,~,~,explained,~] = pca(data);
        
   for n = 1:3
    total_explained=total_explained+explained(n);
    end
    
    EV.samples=total_explained;
end
