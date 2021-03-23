

% Req Dat
function [DataX_batched,DataY_batched,XMask_batched,YMask_batched,Key_Y,EncoderSeq,DecoderSeq] = PrepareStringSequences( Data , Xkey, Ykey, batchsize)
Nobservations = size(Data,1);

maxXlen = max(cellfun(@numel, Data(:,1)));
maxYlen = max(cellfun(@numel, Data(:,2)));


AllKey = unique([Ykey , Xkey]);
Ykey = AllKey ;
Xkey = AllKey ;



sz_key = numel(Xkey);

xOH = false( size(Data,1) , maxXlen , sz_key ); % examples , sequencelength , cardinality
yOH = false( size(Data,1) , maxYlen , sz_key ); % examples , sequencelength , cardinality
xOH_decode = NaN( size(Data,1) , maxXlen ); % examples , sequencelength , cardinality
yOH_decode = NaN( size(Data,1) , maxYlen ); % examples , sequencelength , cardinality

for obs = 1:size(Data,1)
    X = Data{obs,1};
    Y = Data{obs,2};
    Yi=[]; Xi=[];
    for seq = 1:size(Y,2)
        Yi(1,seq) = find(strcmp(Ykey, Y(1,seq) ));
    end
    for seq = 1:size(X,2)
        Xi(1,seq) = find(strcmp(Xkey, X(1,seq) ));
    end
        onehotX = one_hot_convert( Xi ,'encode', sz_key , 1 , sz_key );
        xOH(obs, 1:size(onehotX,2) , : ) = onehotX;
        xOH_decode(obs, 1:size(onehotX,2) ) = one_hot_convert( onehotX ,'decode', sz_key , 1 , sz_key );
        
        onehotY = one_hot_convert( Yi ,'encode', sz_key , 1 , sz_key );
        yOH(obs, 1:size(onehotY,2) , : ) = onehotY;
        yOH_decode(obs, 1:size(onehotY,2) ) = one_hot_convert( onehotY ,'decode', sz_key , 1 , sz_key );
end

% add Start of sentence and End of sentence tokens to Target & Input data.
XdatTokenised = zeros( size(xOH) + [0 1 2] ); % add 2 to max sequence length, add 2 to number of tokens (SOS & EOS)
XdatTokenised( : , 1:end-1 , 1:end-2 ) = xOH; % insert the tokens without EOS and SOS
% XdatTokenised( : , end , end-1 ) = true; % insert the EOS tokens (2nd from the end of the output Key Vector.

YdatTokenised = zeros( size(yOH) + [0 2 2] ); % add 2 to max sequence length, add 2 to number of tokens (SOS & EOS)
YdatTokenised( : , 2:end-1 , 1:end-2 ) = yOH; % insert the tokens without EOS and SOS
YdatTokenised( : , 1 , end-1 ) = true; % insert the SOS tokens (2nd from the end of the output Key Vector.
YMask = ones( size(YdatTokenised(:,:,1)) ); % mask timesteps after the sequence has finsihed
for obs=1:size(YdatTokenised,1)
    EOSid = find( [ ~any(permute( yOH( obs , : , :) , [ 3 2 1 ] ))  , 1 ] ,1,'first') + 1;
    YdatTokenised( obs , EOSid , end ) = 1;
    
    YMask( obs , 1:EOSid ) = 0;% unmask where there is tokens for the output sequence. 
    
    % Add Padding to Input Sequence
    EOSid = find( [ ~any(permute( xOH( obs , : , :) , [ 3 2 1 ] ))  , 1 ] ,1,'first');
    XdatTokenised( obs , EOSid , end ) = 1;
end

% Create batches

removeLastObs = rem( Nobservations ,batchsize);


% add masking to input data
XMask = ~any(XdatTokenised,3).*1;

% Batch data with uniform sequence lengths where posiible
for obs = 1:size(Data,1)
    [rw,cl]= find(~XMask(obs,:));
    Maxseqlen(obs,1) = max(cl);
end
[srtseqlen,ssI] = sort(Maxseqlen);

XMask = XMask(ssI(1:end-(removeLastObs)),:);
YMask = YMask(ssI(1:end-(removeLastObs)),:);



DataX = XdatTokenised(ssI(1:end-(removeLastObs)),:,:);
DataY = YdatTokenised(ssI(1:end-(removeLastObs)),:,:);

Key_X = [ Xkey , ["{sos}" "{eos}"] , "pad" ];
Key_Y = [ Ykey , ["{sos}" "{eos}"]  ,"pad" ];

[ Ystr ] = token2string( Key_Y , DataY );
[ Xstr ] = token2string( Key_X , DataX );

% Prepare Training Data

DataX_padded = cat(3 , DataX, XMask );
DataY_padded = cat(3 , DataY, YMask );

[DataX_batched, DataY_batched] = batchdatasets( batchsize , DataX_padded , DataY_padded );
[XMask_batched, YMask_batched] = batchdatasets( batchsize , XMask  , YMask );

% Create Padding variable
[ DecoderSeq ] = token2string( Key_Y , cat(1,DataY_batched{:}) ); DecoderSeq = cat( 1,DecoderSeq{:});
[ EncoderSeq ] = token2string( Key_X , cat(1,DataX_batched{:}) ); EncoderSeq = cat( 1,EncoderSeq{:});

end
