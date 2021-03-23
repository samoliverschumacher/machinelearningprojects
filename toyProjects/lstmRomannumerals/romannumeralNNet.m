clearvars;
Nobservations = 999;
Rom2Num = true; % convert roman numerals into numbers

RNGseed = 400;
rng(RNGseed)

for n=1:min([Nobservations,999])
    s = num2roman(n);
    if Rom2Num==true
        Data{n,2} = string(num2str(n)')';
        Data{n,1} = s;        
    else % numbers to roman numerals
    Data{n,1} = string(num2str(n)')';
    Data{n,2} = s;
    end
end
% get list of all potential tokens in the datasets
if Rom2Num==true
    Xkey = ["I" "V" "X" "L" "C" "D" "M"];
    Ykey = string(num2str([0:9]'))';    
else
Ykey = ["I" "V" "X" "L" "C" "D" "M"];
Xkey = string(num2str([0:9]'))';
end
% shuffle the data
Data = Data(randperm(size(Data,1)),:);

%% Convert String sequences into OneHot Encoded Data

batchsize = 60;
trainratio = 0.5;

train_idx = 1:floor(size(Data,1).*trainratio );
test_idx = find(all([1:size(Data,1)]~=train_idx',1));

train_idx_batches = 1:ceil((numel(train_idx)/batchsize));
test_idx_batches = 1:floor((numel(train_idx)/batchsize));


[DataX_batched,DataY_batched,XMask_batched,YMask_batched,Key_Y,EncoderSeq,DecoderSeq] = PrepareStringSequences( Data , Xkey, Ykey, batchsize);

DataX_batched_train = DataX_batched(train_idx_batches,:,:); 
DataY_batched_train = DataY_batched(train_idx_batches,:,:);
XMask_batched_train = XMask_batched(train_idx_batches,:,:); 
YMask_batched_train = YMask_batched(train_idx_batches,:,:);

DataX_batched_test = DataX_batched(test_idx_batches,:,:); 
DataY_batched_test = DataY_batched(test_idx_batches,:,:);
XMask_batched_test = XMask_batched(test_idx_batches,:,:); 
YMask_batched_test = YMask_batched(test_idx_batches,:,:);

DataX_test = cat(1,DataX_batched_test{:,:,:}); 
DataY_test = cat(1,DataY_batched_test{:,:,:});
DataX_train = cat(1,DataX_batched_train{:,:,:}); 
DataY_train = cat(1,DataY_batched_train{:,:,:});

InputDataSize = size(DataX_batched{1});
OutputDataSize = size(DataY_batched{1});

%% Select LSTM type
Attention = false;
BiLSTM = false;

%% Generate Model
if Attention
    [trainingSettings, epochs, NNModels] = test_1(InputDataSize,OutputDataSize);
    if BiLSTM
        [trainingSettings, epochs, NNModels] = BiLSTMAttn(InputDataSize,OutputDataSize);
    end
else
    [trainingSettings, epochs, NNModels] = test_2(InputDataSize,OutputDataSize);
end

%% Train the Neural Net

disp(' Test X and Y Data;') , disp(DecoderSeq(1:5,:)) , disp(EncoderSeq(1:5,:))
    [NNModels, Prediction, ErrorMetric1, ErrorMetric2] = ...
        TrainSequnceNNetV3(NNModels, epochs, trainingSettings , DataX_batched_train, DataY_batched_train, XMask_batched_train, YMask_batched_train);
    figure; plot(NNModels{end}.LossHistory)

if isempty(test_idx)
    [~, Prediction, ~, ~ ,SS,AttnScores] = ...
        TrainSequnceNNetV3(NNModels, epochs, trainingSettings, DataX_batched_train, DataY_batched_train, XMask_batched_train, YMask_batched_train , true);
    
    DatY_train_dec = one_hot_convert(cat(1,DataY_batched_train{:,:,:}),'decode',size(Prediction,3), 1 ,size(Prediction,3));
    DatY_train_dec = [DatY_train_dec(:,2:end) , repmat( numel(Key_Y) , [size(DatY_train_dec,1) , 1] ) ];
    
    SuccessId_train = all(one_hot_convert(Prediction(:,:,:),'decode',size(Prediction,3),  1  ,size(Prediction,3)) == DatY_train_dec ...
        | DatY_train_dec==numel(Key_Y) ,2);
    Train_Accuracy = mean( SuccessId_train ,1);
    YStrTrain =token2string( Key_Y,cat(1,DataY_batched_train{:}) );
    YStrTrain=cat(1,YStrTrain{:});
    if Attention
          figure;    successObs = find(SuccessId_train);VizAttn(AttnScores, successObs([96:99]+4), DataX_train , DataY_train , Key_Y, Key_Y)
    end
else
    
    %% Predict on the Test set
    
    [~, Prediction_test, ErrorMetric1_test, ErrorMetric2_test, SavedStates_test, AttnScores_test] = ...
        TrainSequnceNNetV3(NNModels, epochs, trainingSettings , DataX_batched_test, DataY_batched_test, XMask_batched_test, YMask_batched_test, true );
    
    DatY_test_dec = one_hot_convert( cat(1,DataY_batched_test{:,:,:}),'decode',size(Prediction_test,3), 1 ,size(Prediction_test,3));
    DatY_test_dec = [DatY_test_dec(:,2:end) , repmat( numel(Key_Y) , [size(DatY_test_dec,1) , 1] ) ];
    
    SuccessId_test = all(one_hot_convert(Prediction_test(:,:,:),'decode',size(Prediction_test,3),  1  ,size(Prediction_test,3)) ...
        == DatY_test_dec ...
        | DatY_test_dec==numel(Key_Y) ,2);
    Test_Accuracy = mean( SuccessId_test,1);
    "successObs = find(SuccessId_test);VizAttn(AttnScores, successObs(end-25), DataX_test , DataY_test , Key_X, Key_Y)";
    "CheckSoftmaxScores( Prediction_test(successObs(end-25),:,:) , DataY_test(successObs(end-25),:,:) , Key_Y )";
    YStrTest =token2string( Key_Y,DataY_test );
    YStrTest = cat(1,YStrTest{:});
    if Attention
        figure;    successObs = find(SuccessId_test);VizAttn(AttnScores_test, successObs([96:99]+4), DataX_test , DataY_test , Key_Y, Key_Y)
    end
end


function [trainingSettings, epochs, NNModels] = test_1(InputDataSize,OutputDataSize)
epochs = 200;
trainingSettings = struct();
trainingSettings.GDOptimizer = 'Adam';
trainingSettings.learnrate = 0.05;
trainingSettings.LossType = "MultiClassCrossEntropy";
trainingSettings.gradclip = true;
%% Topology
% input embedding
[NNLayerEncEmb] = GenerateNNetLayer( 16 , InputDataSize(1) , InputDataSize(3) , "dense" , "tanh" );
% Encoder
[NNLayerEnc1] = GenerateNNetLayer( 16 , InputDataSize(1) , NNLayerEncEmb.Nunits , "LSTM" , "tanh" ...
    , InputDataSize(2) , struct('resetstate',true,'predictsequence',false, 'InputMask',struct('hasMask',true) ) );

% Decoder Embedding
[NNLayerDecEmb] = GenerateNNetLayer( NNLayerEncEmb.Nunits , OutputDataSize(1) , OutputDataSize(3) , "dense" , "tanh" );
% Decoder 
[NNLayerDec1] = GenerateNNetLayer( NNLayerEnc1.Nunits , InputDataSize(1) , NNLayerDecEmb.Nunits , "LSTM" , "tanh" ...
    , OutputDataSize(2) , struct('resetstate', true ,'predictsequence',true,'SelfReferencing',true,'TeacherForcing',true,'teacherforcingratio',0.05) );
% Decoder Attention Layer
[NNLayerDec1] = AddAttentionLayer( NNLayerDec1 , size(NNLayerEnc1.Activations.HiddenOut) , "general" );

% Projection Layer to Output tokens
[NNLayerFinal] = GenerateNNetLayer( OutputDataSize(3) , OutputDataSize(1) , NNLayerDec1.Nunits , "dense" , "softmax" );
NNLayerDec1.ProjectionLayer = NNLayerFinal;
NNLayerDec1.ProjectionLayerOutput = zeros( size( permute( NNLayerDec1.ProjectionLayer.Activations.HiddenOut , [1 3 2] ) ) + [0 NNLayerDec1.Nstates-1 0 ] );

NNLayerDec1.EmbeddingLayer = NNLayerDecEmb;
NNLayerDec1.EmbeddingLayerInput = zeros( size( permute( NNLayerDec1.EmbeddingLayer.XInput , [1 3 2] ) ) + [0 NNLayerDec1.Nstates-1 0 ] );

NNModels = [{NNLayerEncEmb},{NNLayerEnc1},{NNLayerDec1}];
end

function [trainingSettings, epochs, NNModels] = test_2(InputDataSize,OutputDataSize)
epochs = 200;
trainingSettings = struct();
trainingSettings.GDOptimizer = 'Adam';
trainingSettings.learnrate = 0.05;
trainingSettings.LossType = "MultiClassCrossEntropy";
trainingSettings.gradclip = true;
%% Topology
% could start encoder with an embedding layer...
[NNLayerEncEmb] = GenerateNNetLayer( 16 , InputDataSize(1) , InputDataSize(3) , "dense" , "tanh" );
[NNLayerEnc1] = GenerateNNetLayer( 16 , InputDataSize(1) , NNLayerEncEmb.Nunits , "LSTM" , "tanh" ...
    , InputDataSize(2) , struct('resetstate',true,'predictsequence',false) );

% Decoder takes its own prior timesteps output as input in each step. when teacher forcing, it takes the target values 
% (like the encoder takes the source values)
% No teacher forcing

[NNLayerDecEmb] = GenerateNNetLayer( NNLayerEncEmb.Nunits , OutputDataSize(1) , OutputDataSize(3) , "dense" , "tanh" );

[NNLayerDec1] = GenerateNNetLayer( NNLayerEnc1.Nunits , InputDataSize(1) , NNLayerDecEmb.Nunits , "LSTM" , "tanh" ...
    , OutputDataSize(2) , struct('resetstate', true ,'predictsequence',true,'SelfReferencing',true,'TeacherForcing',true,'teacherforcingratio',0.05) );

% [NNLayerDec1] = AddAttentionLayer( NNLayerDec1 , size(NNLayerEnc1.Activations.HiddenOut) , "general" );


[NNLayerFinal] = GenerateNNetLayer( OutputDataSize(3) , OutputDataSize(1) , NNLayerDec1.Nunits , "dense" , "softmax" );
NNLayerDec1.ProjectionLayer = NNLayerFinal;
NNLayerDec1.ProjectionLayerOutput = zeros( size( permute( NNLayerDec1.ProjectionLayer.Activations.HiddenOut , [1 3 2] ) ) + [0 NNLayerDec1.Nstates-1 0 ] );
NNLayerDec1.EmbeddingLayer = NNLayerDecEmb;
% NNLayerDec1.EmbeddingLayerInput = zeros( size( permute( NNLayerDec1.EmbeddingLayer.XInput , [1 3 2] ) ) + [0 NNLayerDec1.Nstates-1 0 ] );

NNLayerDec1.EmbeddingLayerInput = zeros( size( permute( NNLayerDec1.EmbeddingLayer.XInput , [1 3 2] ) ) + [0 NNLayerDec1.Nstates-1 0 ] );

NNModels = [{NNLayerEncEmb},{NNLayerEnc1},{NNLayerDec1}];

end

function [trainingSettings, epochs, NNModels] = BiLSTMAttn(InputDataSize,OutputDataSize)
epochs = 200;

trainingSettings = struct();
trainingSettings.GDOptimizer = 'Adam';
trainingSettings.learnrate = 0.05;
trainingSettings.LossType = "MultiClassCrossEntropy";
trainingSettings.gradclip = true;
%% Topology
% could start encoder with an embedding layer...
[NNLayerEncEmb] = GenerateNNetLayer( 16 , InputDataSize(1) , InputDataSize(3) , "dense" , "tanh" );

[NNLayerEnc1] = GenerateNNetLayer( 16 , InputDataSize(1) , NNLayerEncEmb.Nunits , "LSTM" , "tanh" ...
    , InputDataSize(2) , struct('resetstate',true,'predictsequence',false,'BiLSTM',true,'MergeFcn', "concatenate" ,'InputMask',struct('hasMask',true) ) );

% Decoder takes its own prior timesteps output as input in each step. when teacher forcing, it takes the target values 
% (like the encoder takes the source values)

[NNLayerDecEmb] = GenerateNNetLayer( NNLayerEncEmb.Nunits , OutputDataSize(1) , OutputDataSize(3) , "dense" , "tanh" );

[NNLayerDec1] = GenerateNNetLayer( NNLayerEnc1.Nunits , InputDataSize(1) , NNLayerDecEmb.Nunits , "LSTM" , "tanh" ...
    , OutputDataSize(2) , struct('resetstate', true ,'predictsequence',true,'SelfReferencing',true,'TeacherForcing',true,'teacherforcingratio',0.05) );

[NNLayerDec1] = AddAttentionLayer( NNLayerDec1 , size(NNLayerEnc1.Activations.HiddenOut) , "general" );

[NNLayerFinal] = GenerateNNetLayer( OutputDataSize(3) , OutputDataSize(1) , NNLayerDec1.Nunits , "dense" , "softmax" );
NNLayerDec1.ProjectionLayer = NNLayerFinal;
NNLayerDec1.ProjectionLayerOutput = zeros( size( permute( NNLayerDec1.ProjectionLayer.Activations.HiddenOut , [1 3 2] ) ) + [0 NNLayerDec1.Nstates-1 0 ] );

NNLayerDec1.EmbeddingLayer = NNLayerDecEmb;
NNLayerDec1.EmbeddingLayerInput = zeros( size( permute( NNLayerDec1.EmbeddingLayer.XInput , [1 3 2] ) ) + [0 NNLayerDec1.Nstates-1 0 ] );

NNModels = [{NNLayerEncEmb},{NNLayerEnc1},{NNLayerDec1}];
end



