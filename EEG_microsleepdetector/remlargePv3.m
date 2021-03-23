% Removes terms in Model, making sure to keep certain terms
% diagtbl has rsqd, max and mean pvals, formula, outlier leverage
function [Modelclean, Diag] = remlargePv3(Model, C2A_keeptrm, Tol)

try
    w = warning('query','last');
    id = w.identifier;
    warning('off',id);
    catch
end

nochanges = 0;
Modelclean = Model;
if ~isempty(Model.Coefficients.pValue)
    
    for ic = 1:length(Modelclean.Coefficients.pValue)
        if Modelclean.NumCoefficients <= 1
            if ic ==1
                nochanges = 1;
            end
            break
        else
            pvalsBefore = Modelclean.Coefficients.pValue;
            temp = logical(pvalsBefore>= Tol); %find high pvals
            keepid = ismember(Modelclean.CoefficientNames,C2A_keeptrm)';
            pvalid = temp;
            pvalid(keepid) = 0; %where a pval is high, but a 'keeper', set to 0
            if sum(pvalid) == 0
                if ic == 1
                    nochanges = 1;
                end
                break
            end
            [~,srtpvalI] = sort(pvalsBefore,'desc');
            laspvalloc = srtpvalI((pvalid(srtpvalI))); %laspvalloc=srtpvalI(laspvalloc);
            
            if strcmpi( Modelclean.CoefficientNames{laspvalloc(1)} , '(Intercept)')
                rmvpvalterm = '1';
            else
                rmvpvalterm = Modelclean.CoefficientNames{laspvalloc(1)};%Modelclean.PredictorNames{ arrayfun(@(K) any(contains( Modelclean.CoefficientNames{laspvalloc(1)} , Modelclean.PredictorNames(K) )) , [1:numel(Modelclean.PredictorNames)] ) };
            end
            
%             fic = Modelclean.VariableInfo.IsCategorical(Modelclean.Formula.Terms(laspvalloc(1) , :)==1);
%             
%             if any( Modelclean.VariableInfo.IsCategorical( cellfun(@(K) contains( strsplit([Modelclean.CoefficientNames{laspvalloc(1)}],{':','*'}) , K ) ,Modelclean.VariableNames ) ) )
%                 t1 = [strfind(rmvpvalterm,'_'):strfind(rmvpvalterm,':')-1];
%                 if isempty(t1)
%                     t2 = [strfind(rmvpvalterm,'_'):strfind(rmvpvalterm,'*')-1];
%                     rmvpvalterm(t2)=[];
%                     if isempty(t2)
%                         t3 = [strfind(rmvpvalterm,'_')];
%                         rmvpvalterm(t3:end) = [];
%                     end
%                 end
%             end
            Modelclean = removeTerms( Modelclean , Modelclean.Formula.Terms(laspvalloc(1) , :) );
            
            
%             Modelclean = removeTerms(Modelclean,rmvpvalterm); % refresh model less the large pval
        
            % storing Model Data
            [maxp, maxpid] = max(Modelclean.Coefficients.pValue);
            maxppred = cell2mat(Modelclean.CoefficientNames(maxpid));
            tempdiag(ic,:) = {Modelclean.Rsquared.Adjusted , maxp, maxppred, mean(Modelclean.Coefficients.pValue), Modelclean.Formula.LinearPredictor, max(Modelclean.Diagnostics.Leverage)};
        end
    end
    
    
    
    % disp('High pvals gone, aside from some key predictors')
    
    if nochanges == 0
        Diag = cell(size(tempdiag,1)+1,size(tempdiag,2)+2);
        Diag(2:end,2:4) = tempdiag(:,1:3); Diag(2:end,5:7) = tempdiag(:,4:6);%embedding diag data into Diag matrix
        Diag(2:end,1) = num2cell([1:size(tempdiag,1)]); %iteration counts
        
        commentA = cell(size(tempdiag,1),1);
        commentA(1:end) = {'.'};
        
        commentB = cell(size(tempdiag,1),1);
        commentB(1:end) = {'.'};
        
        [mp, mpid] = min(cell2mat(tempdiag(:,2)));
        commentA{mpid} = 'lowest max pval';
        
        [mrs, mrsid] = max(cell2mat(tempdiag(:,1)));
        commentB{mrsid} = 'Highest Rsqd';
        
        for ic = 1:length(commentA)
            Commentvect{ic} = strjoin([commentA(ic), commentB(ic)], ', ');
        end
        Commentvect = Commentvect';
        % commentvect = cellfun(@strjoin, [commentA, commentB])
        Diag(2:end,end) = Commentvect;
    else
        DiagHead = {'Iteration No.', 'rsqd', 'max pvals','max pval term', 'mean pvals', 'formula', 'max leverage outlier','Comment'};
        Diag(1,:) = DiagHead;
    end
else
    
    DiagHead = {'Iteration No.', 'rsqd', 'max pvals','max pval term', 'mean pvals', 'formula', 'max leverage outlier','Comment'};
    Diag(1,:) = DiagHead;
end

end





