

function [ Ystr ] = token2string( Key , TokenSequence )
% Converts a tokenised observation back into the original form, as found in
% 'Key'
szF = size(TokenSequence,3);
if size(TokenSequence ,1)==1
    Ystr = Key( one_hot_convert( TokenSequence(1, any(TokenSequence(1,:,:),3) ,:) ,'decode', szF , 1 , szF ) );
else
    Ystr = cell(size(TokenSequence ,1),1);
    for obs = 1:size(TokenSequence ,1)
        Ystr{obs,:} = Key( one_hot_convert( TokenSequence(obs, any(TokenSequence(obs,:,:),3) ,:) ,'decode', szF , 1 , szF ) );
    end
end

end