function [varargout] = annotation_datadims( varargin )

annotation_inputs = varargin;
firstdimIdx = find(cellfun(@(x) isa(x,"double"), annotation_inputs),1);

if any( contains( annotation_inputs(firstdimIdx-1), {'doublearrow', 'arrow','textarrow','line'} ,'IgnoreCase', true ) )
      dimIndx = [firstdimIdx , firstdimIdx+1];
else        % textbox, rectangle, elipse
      dimIndx = [firstdimIdx];
end

position_as_data = [annotation_inputs{dimIndx}];
% [annotation_dims] = datapoints_to_dims( position_as_data );


annotation_inputs_new = annotation_inputs;
if numel(dimIndx)>1 % arrow / line
      [annotation_dims] = datapoints_to_dims( position_as_data );
      
      annotation_inputs_new{dimIndx(1)} = [annotation_dims(1:2)];
      annotation_inputs_new{dimIndx(2)} = [annotation_dims(3:4)];
else
      shapepos_XYWH =[ [position_as_data(1) , position_as_data(1)+position_as_data(3)] , ...
      [position_as_data(2) , position_as_data(2)+position_as_data(4)] ];

      [annotation_dims] = datapoints_to_dims( shapepos_XYWH );
      
      shapepos_XYWH_dims = [annotation_dims(1) , annotation_dims(3) , ...
            annotation_dims(2) - annotation_dims(1) , annotation_dims(4) - annotation_dims(3)];
      
      annotation_inputs_new{dimIndx(1)} = [shapepos_XYWH_dims(1:4)];
end

annotation( annotation_inputs_new{:} )


if isa(annotation_inputs{1},'matlab.ui.container.CanvasContainer')
      varargout{1} = annotation_inputs{1};
else
        varargout{1} = [];
end

      function [annotation_dims] = datapoints_to_dims( position_as_data )
            pos = get(gca, 'Position');
            xPositionStart = position_as_data(1);
            xPositionEnd= position_as_data(2);
            yPositionStart = position_as_data(2+1);
            yPositionEnd = position_as_data(2+2);
            
            annotation_dims = [(xPositionStart + abs(min(xlim)))/diff(xlim) * pos(3) + pos(1),...
                  (xPositionEnd + abs(min(xlim)))/diff(xlim) * pos(3) + pos(1) ,...
                  (yPositionStart - min(ylim))/diff(ylim) * pos(4) + pos(2),...
                  (yPositionEnd - min(ylim))/diff(ylim) * pos(4) + pos(2)];
      end

end