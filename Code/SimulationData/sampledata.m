
startup()
n_epochs = 100;
sample_rate = 100; % in Hz
length_epochs = 1310; % in ms
% each source will get the same type of activation: brown coloured noise
epochs     = struct('n', n_epochs, 'srate', 100, 'length', 1310);
epochs_noise = struct('n',n_epochs,'srate',100,'length',1310);

channels = {'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3',...
    'P4', 'O1', 'O2', 'F7','F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2',...
    'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz'};


if ~exist ('leadfield','var')
    leadfield   = lf_generate_fromfieldtrip('labels', channels);
end
%sourceIdx = lf_get_source_nearest(leadfield,[0,0,0]);


% Source location and orientation:
% https://www.jneurosci.org/content/jneuro/24/42/9353.full.pdf
[erps,sourceIdxes] = loaderps('P300_generators.csv',leadfield,'amplitude_stddev',0.5,...
    'latency_stddev',50,'pulsewidth_stddev',100,'orientation_stddev',0.3);

% Add noise in equally spaced setup
noise_components = generatenoise(leadfield,10,'amplitude',2000,...
     'amplitude_stddev',1000);

erp_data = generate_scalpdata(erps,leadfield,epochs_noise);
noise_data = generate_scalpdata(noise_components,leadfield,epochs_noise);

compl_data = erp_data+noise_data;

% simulating data
data        = struct();
data.data = loaddata('6',[length(channels) Inf])';
data.index = {'e',':'};
data.amplitude = 1;
data.amplitudeType= 'relative';

data = utl_check_class(data,'type','data');
tempdata=data.data;
siz=size(tempdata);
cols=siz(2);

for col=1:2
    data.data=tempdata(:,col)'+noise_data(col,:);
    plot_signal_fromclass(data,epochs);
end
