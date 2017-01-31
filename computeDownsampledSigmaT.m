function output = computeDownsampledSigmaT(input, scale)

    % 4 to 1 to 4
%     output = imresize(input,1/scale,'box');
    
    % 2 to 1 to 2 in x direction 
%     output = imresize(imresize(input,[size(input,1) round(1/scale*size(input,2))],'box'),size(input),'box');
%     output = imresize(imresize(input,[size(input,1) round(1/scale*size(input,2))]),size(input),'box');
    
    % 4 to 1       
    output = imresize(input,[size(input,1) round(1/scale*size(input,2))],'box');
    
    disp([num2str(size(output,1)) ' x ' num2str(size(output,2))]);
    dlmwrite('output/sigmaTDownSample.csv', output, 'delimiter', ',', 'precision', 16);

end
