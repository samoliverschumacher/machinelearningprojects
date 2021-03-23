function [NNLayer] = AddAttentionLayer( DecoderNNlayer , EncoderSize , ScoringFcn )

    NNLayer = DecoderNNlayer;

    NNLayer.Attention = true;
    NNLayer.AttnInfo = struct('Enc_del_total',zeros( EncoderSize ),'EncoderInput',zeros( EncoderSize ) ...
        , 'Dec_del_total', zeros( size(DecoderNNlayer.Activations.HiddenOut) ) ...
        , 'AttentionScores' , zeros( size(NNLayer.XInput,1) , EncoderSize(2) , DecoderNNlayer.Nstates ) ...
        , 'ScoringFcn', ScoringFcn );%ScoringFcn = "dotproduct" or "general"
    
    if NNLayer.AttnInfo.ScoringFcn=="general"
        NNLayer.AttnInfo.Weights = IniWeights( "tanh" , [EncoderSize(3) , DecoderNNlayer.Nunits] , 'Glorot' );
        NNLayer.AttnInfo.dEdW = zeros( [ size( NNLayer.AttnInfo.Weights ) , EncoderSize(1) ] );
        NNLayer.AttnInfo.BP_pOut = zeros( [ size( NNLayer.AttnInfo.Weights ) , 2 ] );
        NNLayer.AttnInfo.Type = "general";
    end
    
    NNLayer.XInput = zeros( size(NNLayer.XInput) + [0 0 EncoderSize(3)] );
    
    szW = size( NNLayer.Weights.wIN.("forget") );
    szW(1) = szW(1) + EncoderSize(3);
    
    szdE = size(NNLayer.dEdW.wIN.("forget"));
    szdE(1) = szdE(1) + EncoderSize(3);
    
	szBP = size(NNLayer.BP_pOut.wIN.("forget"));
    szBP(1) = szBP(1) + EncoderSize(3);
        
    NNLayer.Connectivity = ones( szdE(1:2) );
    
    iniW_sigmoid = IniWeights( "sigmoid" , szW , 'Glorot' );
    iniW_tanh = IniWeights( "tanh" , szW , 'Glorot' );
    
for gate=["output","input","forget","activate"]
    
    NNLayer.dEdW.wIN.(gate) = zeros( szdE );
    NNLayer.BP_pOut.wIN.(gate) = zeros( szBP );
    if gate=="activate"
        NNLayer.Weights.wIN.(gate) = iniW_tanh;
    else
        NNLayer.Weights.wIN.(gate) = iniW_sigmoid;
    end
    
end