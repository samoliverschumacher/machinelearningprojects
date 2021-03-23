function VizAttn( At , obs , DataX , DataY , KeyX , KeyY)

% Variable Length Decoder Sequence

for o=obs(:)'%1:size(At,1)    
    [ EncoderSeq ] = token2string( KeyX , DataX(o,:,:) );
    [ DecoderSeq ] = token2string( KeyY , DataY(o,:,:) );
    endseq = max(find(EncoderSeq=="pad",1,'first')-1);
    if isempty(endseq) , endseq = numel(EncoderSeq); end
    AtSft=[];
    if numel(obs)>1
        plotcount = (numel(obs));
        rw = ceil(sqrt(plotcount));
        cl = ceil(plotcount./ceil(sqrt(plotcount)));
        subplot( rw , cl , find(obs==o) )
    end
    seqout = find(any(DataY(o,:,1:numel(KeyY)-1),3), 1, 'last' );
    for ii=1:seqout
        AtSft(o,1:endseq,ii) = Actvfcn(At(o,1:endseq,ii),false,3);
    end
    

    
    % EncLabels = strcat( cellstr(EncoderSeq) ,{'_t'}, cellstr(num2str([1:numel(EncoderSeq)]'))' );
    heatmap( squeeze(AtSft(o,1:endseq,:))' );
    set(gca,'XDisplayLabels',EncoderSeq(1:endseq)); xlabel('Input Sequence')
    set(gca,'YDisplayLabels',strcat(["..";DecoderSeq(1:seqout(end)-1)'],strcat( " => " ,DecoderSeq(1:seqout(end))'))); ylabel('Target Sequence')
end




