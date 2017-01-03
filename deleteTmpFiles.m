function deleteTmpFiles()
    
    filename_list{1} = 'output/sigmaTDownSample.csv';
    filename_list{2} = 'output/reflectance.csv';
    filename_list{3} = 'output/reflectanceStderr.csv';
    filename_list{4} = 'output/densityMap.csv';
    
    for i = 1: length(filename_list)
        if exist(filename_list{i}, 'file')==2
            delete(filename_list{i});
        end
    end

end