function [x,y,winlen] = windowEEGcorrelations(idx_openclose , TimeSeries, varargin)
if nargin>2
      windowlength= varargin{1};
      steplength = varargin{2};
else
      steplength = 1;
      windowlength = 256;
end

[N,Nvars] = size(TimeSeries);

windowstartId = (1:steplength:N-windowlength);
% vectorised EEG correlation data for model training
x = NaN( numel(windowstartId) , ((Nvars^2)-Nvars)/2 );
% target variable is avg. # of "open eye" observations in each window
y = NaN( numel(windowstartId) , 1 );
% store # of timesteps in each window, for observation importance weighting
winlen = NaN( numel(windowstartId) , 1 );
for si=windowstartId
    obsind = find(windowstartId==si);
    % correlation between EEGs in the window leading up to the prediction period
    EEGcor = corr( TimeSeries(si: si+windowlength,:));
    x(obsind,:) = EEGcor(tril(ones(14),-1)==1);
    response_t = si+windowlength+[1:steplength];
    y(obsind,1) = sum( idx_openclose(response_t(response_t<=N)) ); % total timesteps with CLOSED eyes
    winlen(obsind,1) = sum((response_t<=N)); %number of timesteps in the window.
end

end