
function EEGCloseEyes_CompreModels(varargin)
%% Load Probit model
PRB = load('EEGloseEyes_Modelling.mat','mdl_probit');
PRB_2 = load('EEGloseEyes_Validation.mat','mdl_probit','bestObservationweighting');

%% Load Lasso model
LAS = load('EEGloseEyes_Modelling.mat','coefLASSO','B0','B','idxLambdaMinDeviance','fitinfo');
LAS_2 = load('EEGloseEyes_Validation.mat','bestIntercept');

%% load Partial Least Squares regression model
PLS = load('EEGloseEyes_Modelling.mat','betaPLS');
PLS_2 = load('EEGloseEyes_Validation.mat','weightedThreshold');

%% load RNN LSTM model
RNN = load("EEGworkspace_45Hus3downsampleV_5_8_1_2_11_13_seqIn40seqOut5_Wregular_gradclip_minus1to1_noblink_B.mat",'NNModels','epochs', 'vars','trainingSettings' , 'DataX_batched', 'DataY_batched','Yind_c');
% Weighted FP-FN loss LSTM model
RNN_2 = load("EEGworkspace_45Hus3downsampleV_5_8_1_2_11_13_seqIn40seqOut5_Wregular_gradclip_minus1to1_sinwLoss_noblink_c.mat",'NNModels','epochs', 'vars','trainingSettings' , 'DataX_batched', 'DataY_batched','Yind_c');

%% extract EEG variables and transform them resprectively
% Load Correlation Window data
load('EEGloseEyes_Modelling.mat','windowlength','TimeSeries_cln','TimeSeries_clnnorm','idx_blink','idx_openclose');

%% transform raw data for modelling
% to correlation windows of 256 steps (2 secodns)
[x_corr,y_corr,~] = windowEEGcorrelations(  idx_openclose, TimeSeries_clnnorm );
xLogNorm_corr = normalize( exp(x_corr) ,1);


downsampleratio =3; % downsampling of 3
[Ts_X,Ts_Y] = DownsampleDifferenceSmoothNormaliseEEG(TimeSeries_cln , idx_openclose, downsampleratio, RNN.vars);
 
% create minibatches of the sequences of variables.
seqlen_in = 40; % input sequence length
seqlen_out=5; % target sequence length
Nbatches = 36/downsampleratio; % number of batches in each training epoch
[DataX_batched, DataY_batched, ~, Yind_c, ~] = batchdatasets( Nbatches, Ts_X , Ts_Y , 'seq2batch' ,seqlen_in, seqlen_out);


%% predict RNN LSTM
% Run an inference only loop over the dataset with the trained model
[~, Prediction, ~, ~] = ...
      TrainSequnceNNetV3(RNN.NNModels, RNN.epochs, RNN.trainingSettings , DataX_batched, DataY_batched, [], [],true, false);
% Predict LSTM with cost tradeoff loss function
PrdVect = reshape(Prediction(:,:,2), [numel(Prediction(:,:,1)) , 1] );
yfitRNNLSTM_downsampled = accumarray( Yind_c(:) , PrdVect ,[],@mean , NaN);
[~,yfitRNNLSTM] = DownsampleDifferenceSmoothNormaliseEEG(yfitRNNLSTM_downsampled , [], downsampleratio, [], 'upsample' );

%% predict RNN LSTM 2
[Ts_X,Ts_Y] = DownsampleDifferenceSmoothNormaliseEEG(TimeSeries_cln , idx_openclose, downsampleratio, RNN_2.vars);
 
% create minibatches of the sequences of variables.
[DataX_batched, DataY_batched, ~, Yind_c, ~] = batchdatasets( Nbatches, Ts_X , Ts_Y , 'seq2batch' ,seqlen_in, seqlen_out);

% Run an inference only loop over the dataset with the trained model
[~, Prediction_2, ~, ~] = ...
      TrainSequnceNNetV3(RNN_2.NNModels, RNN_2.epochs, RNN_2.trainingSettings , DataX_batched, DataY_batched, [], [],true, false);

PrdVect = reshape(Prediction_2(:,:,2), [numel(Prediction_2(:,:,1)) , 1] );

yfit_weightedRNNLSTM_downsampled = accumarray( Yind_c(:) , PrdVect ,[],@mean , NaN);
[~,yfit_weightedRNNLSTM] = DownsampleDifferenceSmoothNormaliseEEG(yfit_weightedRNNLSTM_downsampled , [], downsampleratio, [], 'upsample' );


%% predict Lasso
coefLASSO = LAS.coefLASSO;
yfit_Lasso = glmval(coefLASSO , x_corr ,'probit');

weightedcoefLASSO = [LAS_2.bestIntercept ; coefLASSO(2:end)];
yfit_weightedLasso = glmval(weightedcoefLASSO , x_corr ,'probit');


%% predict Probit
yfit_Probit = predict(PRB.mdl_probit, x_corr);

yfit_weightedProbit = predict(PRB_2.mdl_probit, x_corr);

%% Predict Partial Least Squares regression

yfit_PLS = [ones(size(xLogNorm_corr,1),1)*PLS.betaPLS(1) + xLogNorm_corr*PLS.betaPLS(2:end)];

weightedThreshold = PLS_2.weightedThreshold;

%% Collect predictions 
modelnames={'Probit','LassoProbit','PLS','RNNLSTM'};


YFIT(windowlength+1:14980, [1 2 3]) = [yfit_Probit, yfit_Lasso, yfit_PLS];
YFIT(1:windowlength, [1 2 3]) = NaN;
YFIT( : , [4] ) = yfitRNNLSTM;

modelnames_2=strcat('Weighted',modelnames);

YFIT_2(windowlength+1:14980, [1 2 3]) = [yfit_weightedProbit, yfit_weightedLasso, yfit_PLS+(weightedThreshold-0.5)];
YFIT_2(1:windowlength, [1 2 3]) = NaN;
YFIT_2( : , [4] ) = yfit_weightedRNNLSTM;

stepsperobs = 14980/117;

%% plots

%rows = models
% columns = types

% load('EEGModelingResults.mat','idx_openclose_downsampled','y_weighted','yfit_Probit','yfit_weightedProbit','yfit_Lasso','yfit_weightedLasso','yfitPLS','weightedThreshold','yfitRNNLSTM','yfitRNNLSTM_2')

CatYtargCorr = categorical(y_corr>=0.5);

Ytargrnn = categorical(idx_openclose>=0.5);




mnames = [{'Probit'}, ...
{{'Probit';['Closed class weighting: ',num2str(round(PRB_2.bestObservationweighting,1))]}};...
{'Lasso'} ,...
{{'Lasso';['Intercept shifted by ',num2str(round(100*(LAS.B0-LAS_2.bestIntercept)/LAS.B0,2)),'%']}} ;...
{'Partial Least Squares Regression'} , ...
{{'Partial Least Squares';['Closed Classification threshold \geq ',num2str(round(100*(PLS_2.weightedThreshold),1)),'%']}};...
{'RNN LSTM'} ,...
{{'RNN LSTM';['weighted error gradient']} }];

'$P\left(\mathrm{closed}\ge 0\ldotp 607\right)$';
['$P\left(\mathrm{closed}\ge',num2str(round(100*(PLS_2.weightedThreshold),1)),'%\right)$'];

switch varargin{1}

case "CompareModels"
      figure('Position',[228,175,1027,701])
      subplot(7,1,1:2)
      
      plotConsecutiveError( YFIT>=0.5 , idx_openclose>=0.5 , mnames(1:4) , stepsperobs )
      
      subplot(7,1,3:7)
      plot( movmean(normalize(TimeSeries_cln,1,'range',[0 1]) + [0:0.5:6.5],stepsperobs/2,1 ) ,'b'); grid minor; ylim([0 7.5]); yticklabels('')
      yyaxis right; area(idx_openclose,'FaceAlpha',0.1,'FaceColor','g','EdgeColor','none'); yticklabels('')
      xlabel('time (128^t^h sec)'); title('EEG data')
      
%       egadjust =[0:0.5:6.5];
%       for eg=1:14
%       x = 1:size(TimeSeries_cln,1);
%       y = movmean(normalize(TimeSeries_cln(:,eg),1,'range',[0 1]) + egadjust(eg),stepsperobs/2,1 );
%       h = plot(x,y); % capture the line handle when you plot it
%       cd = colormap('parula'); % take your pick (doc colormap)
%       cd = interp1(linspace(min(y),max(y),length(cd)),cd,y); % map color to y values
%       cd = uint8(cd'*255); % need a 4xN uint8 array
%       cd(4,:) = 255; % last column is transparency
%       drawnow
%       set(h.Edge,'ColorBinding','interpolated','ColorData',cd)
%       end
%       
%       
%       plotconfusion( categorical(yfit_weightedLS>=0.5), categorical(y_weighted>=0.5) ,{'Least Squares (Objective function)'} ,...
%             categorical(yfit_weightedLasso>=0.5), categorical(y_weighted>=0.5) ,{'Lasso (Intercept shift)'} ,...
%             categorical(yfit_weightedProbit>=0.5), categorical(y_weighted>=0.5 ),{'Probit (Observation weights)'} ,...
%             categorical(yfit_PLS>=weightedThreshold), categorical(y_weighted>=0.5) ,{'Partial Least Squares (Classification threshold)'} )
%       

case "CompareTradeoffModels"
%%
figure('Position',[70.1429 47.8571 1596 900]);
% PROBIT - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

yp = categorical(yfit_Probit>=0.5);
yp2 = categorical(yfit_weightedProbit>=0.5);

ii=1;
subplot(4,4,1 )
Cm.prb = confusionmat( CatYtargCorr , yp );
plotConfMat( Cm.prb' , {'Open','Closed'});
title(mnames{ii,1})

subplot(4,4, 2)
Cm.prb2 = confusionmat( CatYtargCorr , yp2 );
plotConfMat( Cm.prb2' , {'Open','Closed'});
title(mnames{ii,2})

subplot(4,4,3:4)
plotConsecutiveError( double([yp, yp2])-1 , double(CatYtargCorr)-1 , {'regular',string(mnames{ii,2}{end})} , 14980/117 )
set(get(gca,'title'),'String',[modelnames{ii},': ',get(get(gca,'title'),'String')]); xlabel('time (128^t^h sec)');

% LASSO - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
yl = categorical(yfit_Lasso>=0.5);
yl2 = categorical(yfit_weightedLasso>=0.5);

ii=2;
subplot(4,4,5 )
Cm.las = confusionmat( CatYtargCorr , yl );
plotConfMat( Cm.las' , {'Open','Closed'});
title(mnames{ii,1})

subplot(4,4, 6)
Cm.las2 = confusionmat( CatYtargCorr , yl2 );
plotConfMat( Cm.las2' , {'Open','Closed'});
title(mnames{ii,2})

subplot(4,4,[7:8])
plotConsecutiveError( double([yl, yl2])-1 , double(CatYtargCorr)-1 , {'regular',string(mnames{ii,2}{end})} , 14980/117 )
set(get(gca,'title'),'String',[modelnames{ii},': ',get(get(gca,'title'),'String')]); xlabel('time (128^t^h sec)');

% PARTIAL LEAST SQUARES - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
ypl = categorical(yfit_PLS>=0.5);
ypl2 = categorical(yfit_PLS>=weightedThreshold);

ii=3;
subplot(4,4, 9)
Cm.pls = confusionmat( CatYtargCorr , ypl );
plotConfMat( Cm.pls' , {'Open','Closed'});
title(mnames{ii,1})

subplot(4,4, 10)
Cm.pls2 = confusionmat( CatYtargCorr , ypl2 );
plotConfMat( Cm.pls2' , {'Open','Closed'});
title(mnames{ii,2})


subplot(4,4,[11:12])
plotConsecutiveError( double([ypl, ypl2])-1 , double(CatYtargCorr)-1 , {'regular',string(mnames{ii,2}{end})} , 14980/117 )
set(get(gca,'title'),'String',[modelnames{ii},': ',get(get(gca,'title'),'String')]); xlabel('time (128^t^h sec)');

% RNN LSTM - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

yr = categorical(yfitRNNLSTM>=0.5);
yr2 = categorical(yfit_weightedRNNLSTM>=0.5);

ii=4;
subplot(4,4,13 )
Cm.rnn = confusionmat( Ytargrnn , yr );
plotConfMat( Cm.rnn' , {'Open','Closed'});
title(mnames{ii,1})

subplot(4,4, 14)
Cm.rnn2 = confusionmat( Ytargrnn , yr2 );
plotConfMat( Cm.rnn2' , {'Open','Closed'});
title(mnames{ii,2})

subplot(4,4,[15:16])
plotConsecutiveError( double([yr, yr2])-1 , double(Ytargrnn)-1 , {'regular',string(mnames{ii,2}{end})} , 14980/117 )
set(get(gca,'title'),'String',[modelnames{ii},': ',get(get(gca,'title'),'String')]); xlabel('time (128^t^h sec)');


fnm = fieldnames( Cm );
for ii = 1:numel(fnm)
      FP_FNratio(ii) = (Cm.(fnm{ii})(2,1)/Cm.(fnm{ii})(1,2));
      Acc(ii) = sum([Cm.(fnm{ii})(1,1) , Cm.(fnm{ii})(2,2)])/( sum(Cm.(fnm{ii})(:)) );
end
CompareTbl = table(  string(num2str(round(FP_FNratio,2)')) , round(100*Acc,1)' ,'VariableNames',{'FN_FPratio','Accuracy'},...
      'rownames',{'Probit','Probit_W','Lasso','Lasso_W','PLS','PLS_W','LSTM','LSTM_W'}');

save('NNetconfmatrices','Cm','YFIT','YFIT_2','CompareTbl')
disp(CompareTbl)

      case "ModelSummaryTable"
            Summary = array2table([mean((YFIT>=0.5)==idx_openclose );  max(movsum(((YFIT>=0.5)~=idx_openclose ),[128/2 , 128/2],1)/128)],'VariableNames',modelnames,'RowNames',{'Accuracy';'MaxConsecutiveError'});
            
            disp( Summary )
            
            
      case "TradeoffModelSummaryTable"
            Summary = array2table([mean((YFIT_2>=0.5)==idx_openclose );  max(movsum(((YFIT_2>=0.5)~=idx_openclose ),[128/2 , 128/2],1)/128)],'VariableNames',modelnames,'RowNames',{'Accuracy';'MaxConsecutiveError'});
            
            disp( Summary )
            
end



function [Ts_X,Ts_Y] = DownsampleDifferenceSmoothNormaliseEEG(TimeSeries , idx_openclose, downsampleratio, vars, varargin)

if nargin>4
      % 1 variable at a time
      downsampleratio = 3;
      windowwidth = downsampleratio-1;
      NewTs = NaN(14980 , 4992);
      for it = 1: size(TimeSeries)
            ts_i = (it-1)*downsampleratio + [downsampleratio-windowwidth:downsampleratio + windowwidth];
            NewTs(ts_i,it) = TimeSeries(it);
      end
      NewTs = nanmean(NewTs,2);
      
      Ts_Y = NewTs;
      Ts_X=[];
      return
      
end

[N,Nvars] = size(TimeSeries);
% vars=[ [5 , 8] , [1 2 11 13] ];%[13 8 11 1 2];%(1:14);%[13 8 11 1 2];%(1:14); %[13 8 11 1 2]; % interesting Variables % [1 2 13 14];
dsr = downsampleratio; % downsampling ratio
dwn_TS = zeros( floor(N/dsr)-1 , Nvars+1 );
for sbp=vars
      dwn_TS(:,sbp) = arrayfun(@(ii) mean([TimeSeries(ii-(dsr-1):ii+(dsr-1),sbp)]) ,[dsr:dsr:numel(TimeSeries(:,1))-(dsr-1)]' );% [dsr:dsr:numel(TimeSeries(:,1))]' );%[dsr:dsr:numel(TimeSeries(:,1))-(dsr-1)]' );
end
dwn_TS(:,15) = arrayfun(@(ii) mean([idx_openclose(ii-(dsr-1):ii+(dsr-1), 1 )])  , [dsr:dsr:numel(TimeSeries(:,1))-(dsr-1)]' );%[dsr:dsr:numel(TimeSeries(:,1))]' );%[dsr:dsr:numel(TimeSeries(:,1))-(dsr-1)]' );
dwn_TS(:,15) = round(dwn_TS(:,15));

% reshape the downsampled EEGs: [timesteps, (none) , variable]
Ts_X = permute( dwn_TS( : , vars) , [1 3 2] );
% for each variable, calculate the moving sum of differences, and then normalize into the range [0 , 1]
for it=1:size(Ts_X,3)
      Xdiff = [0; diff(Ts_X(:,:,it)) ];
      Ts_X(:,:,it) =  movsum( Xdiff,[3 0]);
      Ts_X(:,:,it) = normalize( Ts_X(:,:,it) ,1,'range',[-1 1]);
end

Ts_Y = permute( dwn_TS( : , end ) , [1 3 2]);
Ts_Y = permute(dummyvar(categorical(Ts_Y)) , [1 3 2]);

end


      
function plotConsecutiveError( yFitsBinom , yTargetBinom , FitNames , observationlength )
plot( movsum( yFitsBinom~=yTargetBinom ,[observationlength/2 , observationlength/2],1)/observationlength ,'LineWidth',1);
title('Cumulative error across a 1 second window')
ylabel('Misclassification (sec)'); xlabel(['time (',num2str(round(1/observationlength,0)),'^t^h sec)']);
yyaxis right; area( (yTargetBinom),'FaceAlpha',0.1,'FaceColor','g','EdgeColor','none'); yticklabels('')
legend( FitNames,'Location',   'north')
grid minor
end





end

% load('EECloseEyes_EDA.mat','TimeSeries_clnnorm','TimeSeries_cln')
% figure;
% stepsperobs = 14980/117;
% egadjust =[0:0.5:6.5];
% for eg=1:14
% hold on;
% x = 1:size(TimeSeries_cln,1);
% y = movmean(normalize(TimeSeries_cln(:,eg),1,'range',[0 1]) + egadjust(eg),stepsperobs/2,1 );
% h = plot(x,y,'LineWidth',2); % capture the line handle when you plot it
% cd = colormap('jet'); % take your pick (doc colormap)
% cd = interp1(linspace(min(y),max(y),length(cd)),cd,y); % map color to y values
% cd = uint8(cd'*255); % need a 4xN uint8 array
% cd(4,:) = 255; % last column is transparency
% drawnow
% set(h.Edge,'ColorBinding','interpolated','ColorData',cd)
% end
% 
% set(gca,'Color',[0.35 0.35 0.35] ) % set(gca,'Color',[0.15 0.15 0.15] )
% grid minor
% grid on
% set(gca,'XTick',[0:2500:15000])
% set(gca,'GridColor',[1 1 1] )
% set(gca,'GridAlpha',0.15 )
% set(gca,'MinorGridColor',[1 1 1] )
% set(gca,'MinorGridAlpha',0.15 )
% set(gca,'MinorGridLineStyle','--' )
% 
% set(gca,'MinorGridAlpha',0.3 )
% set(gca,'MinorGridLineStyle','-' )
% set(gca,'Color',[0.25 0.25 0.25] )
% 
% 
% 
% set(gca,'GridLineStyle','-.' )
% set(gca,'MinorGridLineStyle','-.' )
% 
% xlabel('')
% ylabel('')
% % set(gca,'XTick',[])
% % set(gca,'YTick',[])
% set(gca,'ylim',[0 7.5])

