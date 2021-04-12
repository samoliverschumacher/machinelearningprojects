%% EDA - Property prices

% import the data as a table
RawData = readtable('C:\Users\Sam Schumacher\Documents\Job Applications 2020\Applications\REA Group\Property pricing assignment\dataset.csv');

% preview the first 10 rows.
RawData(1:10,:)
%% 
% We've got a combination of strings, integers and floating point numbers.
% 
% Lets see the range of them, and their data types;
%%
summary(RawData)
%% 
% Loading the data into the workspace automatically decided what type of 
% varaible in each column of the CSV.
% 
% Variables like |*property_id |*has been classified the same way as |*slope 
% |*(as double), though they're a different type of variable: an ID will usually 
% be an integer, |*slope |*is definitely a floating point number.
% 
% by displaying the unique values returned when dividing all 52,248 rows 
% by 1, we can double check which of the variables are strictly integers;

% identify if 1st value in each column is numeric
numericIdx = arrayfun(@(X) isnumeric(RawData{1,X}) , 1:width(RawData) );

for ix = find(numericIdx)
      % unique list of values created by dividing all observations by 1
      remainderVals = unique( rem(RawData{:,ix} , 1) );
      % number of these values that are not NaNs (missing in the raw data)
      countRealRemainders = numel( remainderVals ) - sum(isnan(remainderVals));
      % display a count of unique decimals reamining after dividing each observation by 1
      fprintf('%s has %.0f unique remainders after division by zero \n', RawData.Properties.VariableNames{ix}, countRealRemainders )
end
%% 
% So, only |*slope|* and |*max_roof_height |*can be considered true continuous 
% variables. The rest have a remaineder of 0 when divided by 1, for all observations.

continuousIdx = ismember( RawData.Properties.VariableNames , {'slope','max_roof_height'});
% the other double type variables are discrete because they are all integers
discreteIdx = numericIdx & ~continuousIdx;
disp( RawData.Properties.VariableNames(discreteIdx))
%% 
% While these variables are all discrete, the magnitude of some of them 
% is still meaningful. i.e. |*year_built|*'s value could indicate temporal patterns 
% in the data, |*land, floorplate, bedrooms, bathrroms, garages|* will likely 
% have a impact on |*sale_price|*.
% 
% 
% 
% If |*property_id |*is an identifier there should be the same number of 
% unique identifiers as rows;

numel( unique(RawData.property_id))
%% 
% That's not the case. Is there only a small number of reccurences of any 
% one ID ?

t = tabulate(RawData.property_id);
figure;
histogram( t( t(:,2)>1 ,2) );
xlabel('number of times a property_id occurs','Interpreter','none')
ylabel('count of property ID''s')
title('Property IDs with multiple occurances in the dataset')
%% 
% This could indicate some proeprties were bought and sold multiple times 
% in the dataset. One |*property_id |*occcured 36 times.

% Show data with the property_id that occurs 36 times
RawData( RawData.property_id==t(t(:,2)==36,1) , : )
%% 
% This explains it; |*property_id|* |5543930| was a '|Strata|' |*property_type|*. 
% So, not all |*property_id'|*s define a unique property. This could give us insight 
% into the data;
% 
% *    Insight 1: *"Apartments with equivalent features (|*bathrooms|*, |*garages 
% |*etc.) sell for different prices with only |*sale_date |*changing. Once |*sale_date|* 
% is controlled for, the remaining variance could give a benchmark for the external 
% effects on |*sale_price |*that we do not have data for."
% 
% 
% 
% Only |*postcode |*& |*property_id|* will be converted from numeric to categoricals 
% for the purpose of exploring the data. Though, we must take note of this conversion 
% for |*postcode|*. Because this is a spatial category, it could later be converted 
% back to a numeric distance feature.

% identify cells of strings, property_id, and postcode as categorical variables
categoryIdx = ismember( RawData.Properties.VariableNames , {'property_id','postcode'}) ...
      | arrayfun(@(X) iscell(RawData{1,X}) , 1:width(RawData) );
disp( RawData.Properties.VariableNames(categoryIdx) )
%% 
% Now we can reconstruct the dataset with meaningful data types before exploring 
% the data.

Data = table();
% create categoricals
for ic = find(categoryIdx)
    Data.(RawData.Properties.VariableNames{ic}) = categorical(RawData{:,ic});
end
% create numeric variables
for inum = find(numericIdx & ~categoryIdx)
    Data.(RawData.Properties.VariableNames{inum}) = (RawData{:,inum});
end
Data.('sale_date') = RawData.sale_date;
disp(Data(1:5,:))
%% 
% First 5 columns are categoricals, last column is the matlab 'datetime' 
% class. Columns 6:14 are numeric integers and floating point numbers. |*sale_date 
% |*will be more meaningful as a continuous, numeric variable that starts at 0 
% for the earliest date:

Data.sale_date_numeric = datenum(Data.sale_date) - min(datenum(Data.sale_date));
% store column names as variable names
varNames = Data.Properties.VariableNames;
%% 
% Now is a good time to consider the the context & provenence of each of 
% the variables, before diving deeper into the EDA and missing the big picture.
% 
% 
% 
% *Listing common sense assumptions & relationships between variables:*
% 
% * |*property_id|*: multiple houses & multiple sale dates per ID
% * |*sale_price|*: have to asusme unadjusted for inflation
% * |*sale_date|*: should it be after |*year_built|*?

sum( year(Data.sale_date)<Data.year_built )
%% 
% 3 observations are not! 

disp( Data(year(Data.sale_date)<Data.year_built,:) )
%% 
% Same |*year_built|* but different locations and |*sale_dates|* is suspect. 
% With no way to verify if this is measurement error, and only small deviations 
% in time, all we can do is treat these observations with less weight, if modelling 
% technique allows.

lowDataQuality_id = year(Data.sale_date)<Data.year_built;
%% 
% * |*property_type|*: Should expect more strata with later year_built dates, 
% lower number of rooms/garages, lower prices but on larger land & floorplate. 
% Smaller floorplate to land ratio than Houses.
% * |*suburb, postcode, state|*: these are nested categories in that order. 
% * |*land, floorplate|*: larger land than floorplate

sum( (Data.land)<Data.floorplate )
%% 
% Wrong assumption! This finding is contradictory to the description of 
% the land and floorplate variables. Contacting the subject matter expert is required 
% here, as there are a considerable number of violations of this common sense 
% assumption. Until then, becasue these are all "Strata" |*property_types |*we'll 
% assume the meaning is; "_accumulated floorspace of all properties under that 
% same |*property_id|*, in square meters_", and treat these observations as no 
% different.
% 
% * |*bedrooms, bathrooms, garages|*: No specific assumptions to point out.
% * |*slope|*: expect normally distributed values: the land's gradient & orientation 
% of houses shouldn't have a bias.
% * |*max_roof_height|*: larger values for Strata (multi-level)
% * |*year_built|*: (as |*sale_date |*above)
%% Data Cleaning
% Data cleaning will be done before exploring the data relationships, to give 
% a clearer picture. This involves missing value handling, outlier treatments 
% and some variable transformations.
%% Missing Values
% 
% 
% Checking string categoricals for empty strings or missing observations;
%%
sum( Data{:,1:5}=="" | Data{:,1:5}==" " | ismissing(Data{:,1:5}) )
%% 
% No categoricals have empty category labels.
% 
% Inspecting only the numeric varialbes for missing values;

summary(Data(:,[6:15]))
%% 
% All but |*sale_price|* and |*sale_date|* have missing values. The most 
% missing values is in |*year_built|*, which has 1345 missing (2-3% of observations). 
% 
% 
%% Treatment for missing values
% 
% 
% One option for treating the missing variables when they are few is to remove 
% the row entirely. Though, if this method is used, we need to be sure there isn't 
% a pattern to where the missing variables occur. If the missing values occur 
% in a non-random way - removing them might be removing a hidden variable in itself. 
% 
% 
%% Checking for random occurance of missing values
% 

missingRow = any(ismissing(Data{:,7:14}),2);
sprintf( '%0.1f%% of rows have at least 1 missing variable', sum( missingRow ) ./ height(Data)*100 )
%% 
% Looking at correlations, and their associated pvalues between numeric 
% variables and incidence of rows where at least one variable has a missing value 
% will show if there is a significant underlying pattern to the missing values. 
% But first, a transform of |*sale_date |*to numeric from datetime class.

[rho,p] = corr([Data{:,[6:14,16]}, missingRow ],'rows','pairwise');
missingTbl = table( p(end,1:end-1)' , rho(end,1:end-1)' , 'VariableNames',{'pValue','correlation'} , ...
      'RowNames',varNames([6:14,16])');
disp( missingTbl )
%% 
% Sale price and year built have <0.05 pValues for their correlations to 
% missing rows. Though, |*sale_price |*and |*year_built |*could themselves be 
% correlated;

[rho,p] = corr(Data.sale_price, Data.year_built,'rows','pairwise')
%% 
% They are correlated. So missing values vary with sale price, but |*sale_price|* 
% goes down, when year goes up; they are colinear.
% 
% *    Insight #2: *"|*sale_price* |and |*year_built* |have a weak relationship 
% with occurance of missing values, but |*sale_price* |and |*year_built* |themselves 
% have a strong relationship."
% 
% 
% 
% Now checking whether there is a pattern in missing values that can be explained 
% by the categorical data by conducting an Analysis of Variance. Because we know 
% |*sale_price *|
% 
% Inspecting if variation in missing values is non random, by using an Analysis 
% of Variance (ANOVA) with the nested categories |*property_type, state, postcode 
% |*& |*suburb|*.
% 
% Because we've seen that |*sale_price |*and |*year_built |*are colinear, 
% and correlated with observations with missing rows, they will be added to the 
% ANOVA as continuos variables so that categories aren't hiding their effects.
% 
% 

% setting up the nested categories relationships matrix;
nestingM =  [0 0 0 0 0 0; ... % property
            0 0 0 1 0 0; ... % postcode -> is nested in state
            0 1 0 1 0 0; ... % suburb -> is nested in postcode, which is nested in state
            0 0 0 0 0 0;... % state
            0 0 0 0 0 0;... % sale_price -> a continuous variable
            0 0 0 0 0 0];... % year_built -> a continuous variable
% Anova with covariates sale price & year built. 
[p,tbl,stats,terms] = anovan( missingRow , {Data.property_type,Data.postcode,Data.suburb,Data.state,Data.sale_price,Data.year_built} ,...
      'varnames',varNames([2 4 3 5 6 14]),...
      'nested',nestingM ,'continuous',[5, 6]);
%% 
%     *Insight #3: *"Missing value rows occur very slightly more with properties 
% built more recently."
% 
% _(When controlling for |*state, postcode, suburb |*& |*sale_price |*- with 
% confidence level 3.3% (pValue))_
% 
% The conclusion here is that if we had to remove all rows that had _any_ 
% missing values, sales price prediction could be a biased model to the extent 
% that year_built is predictive of |*sales_price. |*But because |*year_built|* 
% is only slightly correlated with missing values, and only with 3.6% confidence 
% of the null hypotheses - removal of all rows with missing values might be too 
% heavy handed - especially when 7.6% of rows have a missing value/
% 
% 
%% Treating missing values without removal
% 
% 
% To keep observations with missing values, we've two choices: let the modelling 
% algorithm decide how to treat them, or impute the values with our own technique.
% 
% Treatment requires some common sense thinking about the variables;
% 
% * |*Land|:* An average could be taken for "|Houses" |/ "|Strata"| in the same 
% |*suburb |*or|* postcode|*. If too few observations exist in a nested group, 
% and we assume housing density changes smoothly with location, GPS data could 
% help to find the land-size averages of nearby suburbs.
% * |*floorplate|:* As above, we can take averages for in-suburb |*property_type|* 
% = "|Houses"|. For strata, it's a little more involved. floorplate will relate 
% to all observations under the same |*property_id|*. The number of observations 
% under the same |*property_id|* is an indication of either sequential buy-sell 
% events, and/or the number of apartments that "|Strata|" type |*property_id|* 
% holds. As we've seen in the '_common sense assumptions'_ above - floorplate 
% is of the entire building / |*property_id|* across all levels, we'd take the 
% average of |*property_id|* within |Strata.|
% * |*bedrooms*, *bathrooms*, *garages|:* Imputation needs to adhere to the 
% above also.
% * |*slope|:* estimated by the surrounding suburbs. Though if a suburb is less 
% than unanimously +ve or -ve in slope, estimation might be worse than leaving 
% the value as missing / removing from analysis. (House could be facing the otherway 
% on a uniformly sloped suburb)
% * |*max_roof_height|:* Could be averaged using |*property_type|*, # of apartments 
% in the |*property_id |*(method described for _floorplate_ ).
% * |*year_built|:* This could be done with suburb.
% 
% If the model built with missing value rows removed is not accurate enough, 
% these techniques above should be implemented.
% 
% 
%% Exploring the Shape of data
% 
% 
% Before exploring relationships between variables, we will view their shape 
% & make any normalising transforms. For the dependent variable, it will usually 
% give a better model, for the independent variables, understanding their shape 
% will be made easier. Some modelling techniques may benefit from this too.
% 
% *    Numeric Variables*
% 
% First, histograms to see bias, skew and extreme values of the numeric variables;
%%
numericVars = [6:14,16];
figure;
for vi = numericVars
      subplot(4,3,find(numericVars==vi) )
      histogram(Data{:,vi})
      title(varNames{vi},'Interpreter','none')
end
%% 
% *Dependent Variable*: *sale_price*
% 
% Positive skew*:* Could use a power law or exponential transform to make 
% marginal changes of equal size no matter what starting value.
% 
% *    Insight #4:_ *"_sale_price is positively skewed."
% 
% 
% 
% Outliers: some large postive outliers, may not need to remove after the 
% transform.
% 
% *Independent variables;*
% 
% * |*land, floorplate|*: both strongly positively skewwed. This is likely because 
% they are area measurements which scale with a 2nd power (Area = side*side). 
% Once a square root transform is done, analysis of extreme values will be more 
% meaningful.
% * |*bedrooms, bathrooms, garages|*: bedrooms & garages are negatively skewed, 
% though at low integer values, they will be left as raw before interpretting 
% variable relationships.
% * |*slope: |*normally distributed.
% * |*max_roof_height: |*This distribution has 2 modes; likely a function of 
% |*property_type|*, so adding interaction of these two variables could be helpful.
% * |*year_built:|* Negative skew, and some extreme outliers left of median. 
% A square or cube power transform could help normalise the data. Also, scalling 
% the data so it starts at 0 will be applied so cubd values don't become too extreme. 
% The higher frequency of values at the start of decades is suspicious - this 
% could be an estimation made in the absense of true information on |*year_built.*|
% * |*sale_date_numeric: |*This is nearly uniform, with a lack of data in more 
% recent years.
% 
% Now, making the mentioned transforms to help see data patterns;

Data.log_sale_price = log(Data.sale_price);
Data.sqrt_land = sqrt(Data.land);
Data.sqrt_floorplate = sqrt(Data.floorplate);
Data.cube_year_built = (Data.year_built - min(Data.year_built)).^3;
transformVarNames = {'log_sale_price','sqrt_land','sqrt_floorplate','bedrooms','bathrooms','garages','slope','max_roof_height','cube_year_built','sale_date_numeric'};
[is ,iL] = ismember(transformVarNames ,Data.Properties.VariableNames);
transformNumericVars = iL(is);
% replotting histograms of transformed variables
figure;
for vi = transformNumericVars
      subplot(4,3,find(transformNumericVars==vi) )
      histogram(Data{:,vi})
      title(transformVarNames{(transformNumericVars==vi)},'Interpreter','none')
end
%% 
% A few things have popped up;
% 
% * |*floorplate, land |*& |*max_roof_height|* has some clusters in very high 
% values that are now more obvious.
% * Taking the log of |*sale_price |* was a perfect transform to normality, 
% which will help with models assuming normal distribution of errors.
% 
% Now calculating the covariance matrix & associated pValues. Because there 
% are quite a few variables, we'll inspect visually those with strength in upper 
% 40% & pvalues smaller than 0.05
%%
[rho,p] = corr(Data{:,transformNumericVars},'rows','pairwise');
% select strong & significant relationships in the covariance matrix
strongCovID = (p<0.05) & abs(rho)>quantile(abs(rho(:)),0.6);
strongCovID(1) = true; % include dependent variable
figure;
% correlation plot showing red values for correlations with <0.05 pvalues
corrplot( Data(:,transformNumericVars(strongCovID(1,:))) ,'testR','on','varNames',varNames(numericVars(strongCovID(1,:))));
%% 
% The top row of plots is correlations with |*sale_price|*. The relationships 
% are;
% 
% * Higher |*bedroom, bathroom, garage |*values increase sale price, but |*bathroom|* 
% is strongest.
% * |*floorspace & max_roof_height |*decrease with sale price (likely explained 
% by |*property_type|*).
% * |*sale_date |*increases |*sale_price|* (_the transform done maxes the earliest 
% date = 0_).
% 
% *    Categorical Variables*
% 
% **
% 
% |A boxplot can show effect of grouping on the dependent variable. There 
% are so many categories of *suburb| |*and *postcode|, *so first |*state & property_type|* 
% will be viewed;
%%
figure;
boxplot(Data.sale_price,{Data.property_type,Data.state},"ColorGroup",Data.state,'Orientation','horizontal');
xlabel('sale_price','interpreter','none')
%% 
% Box plots do not overlap too much between |*property_type|* groupings 
% in each state. |*state|* grouping does separate values, though not as much.
% 
% Viewing suburb and poscode would be tricky because there are 326 suburbs. 
% But we ony need to view postcode, as subrub is nested in it.
%%
figure('Position',[740,119,583,685]);
boxplot(Data.sale_price, Data.postcode,'PlotStyle','compact','Orientation','horizontal','OutlierSize',1)
xlabel('sale_price','Interpreter','none')
ylabel('Poscodes')
set(gca,'YTickLabel',[])
title('Variation of sale_price with postcode','Interpreter','none')
%% 
% Using these categories as a predictor is trickier statistically that with 
% small number of categories, but can be done by assuming they are random variables, 
% Instead of fixed variables. A rationale for this could be;
% 
% "suburbs and postcodes divide the land arbitrarily. They are not an intrinsict 
% feature of the property"
% 
% So, we need to see that postcode median's are distributed normally. 

postcodeMedians = grpstats(Data(:,{'postcode','sale_price'}),'postcode','median');
figure; 
histogram(postcodeMedians.median_sale_price,20);
title('Distribution of suburbs median sale prices')
%% 
% Yep - that's normally distributed so we can model postcode and suburb 
% as random variables, unlike state and |*property_type|*.
%%

DataContrmvOutliers = Data{:,[6:14,16]}