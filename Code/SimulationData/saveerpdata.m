function saveerpdata(erp_params,noise_params,leadfield,epochs_noise,savefolder,channels,noise_locs)

for erp_param=erp_params

% Source location and orientation:
% https://www.jneurosci.org/content/jneuro/24/42/9353.full.pdf
% Creates the different sources (20 in total) with different 
    [erps,sourceIdxes] = loaderps('P300_generators.csv',leadfield,'amplitude_stddev',erp_param,...
        'latency_stddev',erp_param,'pulsewidth_stddev',erp_param,'orientation_stddev',erp_param);
    erp_data = generate_scalpdata(erps,leadfield,epochs_noise);
    for noise_type=noise_params
        noise_type
        noise_components = generatenoise(leadfield,noise_type(1),noise_locs,'amplitude',5000*noise_type(2));
        noise_data = generate_scalpdata(noise_components,leadfield,epochs_noise);

        data        = struct();
        data.data=erp_data;
        data.index = {'e',':'};
        data.amplitude = 1;
        data.amplitudeType= 'relative';

        data = utl_check_class(data,'type','data');
        plot_erps(data,channels,["Cz"],epochs_noise);
        data.data=noise_data;
        plot_erps(data,channels,["Cz"],epochs_noise);
        %writematrix(data.data,fullfile(savefolder,strcat("stddev",string(erp_param)),strcat(string(noise_type(1)),'.csv')));
    end

end