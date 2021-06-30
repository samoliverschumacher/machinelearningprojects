      function printNNetsettings( NetworkSettings )
	LogicalString = {'False','True'};
    disp('    Network Settings:')
    setting_names = fieldnames(NetworkSettings);
    numchar = max(cellfun(@numel ,setting_names )') - cellfun(@numel ,setting_names )' ;
    for cc=1:numel(setting_names), setting_names{cc,2} = [repmat(' ',[1,numchar(cc)]), setting_names{cc},' : ']; end
    
    for settingId = 1:numel(setting_names(:,1))
        switch class( NetworkSettings.(setting_names{settingId}) )
            case 'logical'
                fprintf([setting_names{settingId,2}, ' : %s \n'],LogicalString{NetworkSettings.(setting_names{settingId})+1})
            case 'function_handle'
                fprintf([setting_names{settingId,2}, ' : %s \n'],func2str(NetworkSettings.(setting_names{settingId})))
                
            case 'string'
                disp(setting_names{settingId,2}), disp(NetworkSettings.(setting_names{settingId}))
            otherwise
                disp(setting_names{settingId,2} ) , disp(NetworkSettings.(setting_names{settingId}) )
        end
    end
      end