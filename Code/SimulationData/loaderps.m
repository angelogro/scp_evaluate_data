function [components,sourceIdx]=loaderps(filename,leadfield,...
                                        varargin)
    p=inputParser;
    addRequired(p,'filename',@isstr);
    addRequired(p,'leadfield');
    addParameter(p,'amplitude_stddev',0,@isnumeric);
    addParameter(p,'latency_stddev',0,@isnumeric);
    addParameter(p,'pulsewidth_stddev',0,@isnumeric);
    addParameter(p,'location_stddev',0,@isnumeric);
    addParameter(p,'orientation_stddev',0,@isnumeric);
    
    parse(p, filename,leadfield,varargin{:});
    amplitude_stddev=p.Results.amplitude_stddev;
    latency_stddev=p.Results.latency_stddev;
    pulsewidth_stddev=p.Results.pulsewidth_stddev;
    location_stddev=p.Results.location_stddev;
    orientation_stddev=p.Results.orientation_stddev;
                                    
    T=readtable(filename);
    [n_rows,n_columns] = size(T);

    for i = 1:n_rows
        erp = struct('type', 'erp', ...
        'peakLatency',T.Latency_mean_(i), ...
        'peakLatencyDv',latency_stddev,...
        'peakWidth', T.Pulsewidth(i),...
        'peakWidthDv',pulsewidth_stddev,...
        'peakAmplitude', 100*T.Amplitude_mean_(i),...
        'peakAmplitudeDv',100*amplitude_stddev);
        location = [T.x(i)*(1+normrnd(0,location_stddev)),...
            T.y(i)*(1+normrnd(0,location_stddev)),...
            T.z(i)*(1+normrnd(0,location_stddev))];
        orientation = [T.OrientationX(i),...
            T.OrientationY(i),...
            T.OrientationZ(i)];
        sourceIdx(i) = lf_get_source_nearest(leadfield,location);
        components(i)=utl_create_component(sourceIdx(i),erp,leadfield);
        components(i).orientation = orientation/norm(orientation);
        components(i).orientationDv = [orientation_stddev,orientation_stddev,orientation_stddev];
    end
    
end