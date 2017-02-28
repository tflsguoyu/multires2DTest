function output = computeDownsampledSigmaT(input, scale, flag)

    % 4 to 1 to 4
%     output = imresize(input,1/scale,'box');
    
    % 2 to 1 to 2 in x direction 
%     output = imresize(imresize(input,[size(input,1) round(1/scale*size(input,2))],'box'),size(input),'box');
%     output = imresize(imresize(input,[size(input,1) round(1/scale*size(input,2))]),size(input),'box');
    
    % 4 to 1       
%     output = imresize(input,[size(input,1) round(1/scale*size(input,2))],'bilinear');
    output = downsample(input,scale,flag);
    
    disp([num2str(size(output,1)) ' x ' num2str(size(output,2))]);
    dlmwrite('output/sigmaTDownSample.csv', output, 'delimiter', ',', 'precision', 16);

end


function output = downsample(input,scale,flag)

    if scale > 1
        if strcmp(flag,'x_average')
            [r,c] = size(input);
            input_mean = mean(input,2);
            if mod(c,scale) ~= 0
                appended = repmat(input_mean, [1, scale-mod(c,scale)]);
                input = [input, appended];
            end
            c_new = size(input,2)/scale;
            output = zeros(r,c_new);
            for j = 1: c_new
                output(:,j) = mean(input(:,scale*(j-1)+1:scale*j),2);
            end

        elseif strcmp(flag,'x_sample')
            [~,c] = size(input);
            input_mean = mean(input,2);
            if mod(c,scale) ~= 0
                appended = repmat(input_mean, [1, scale-mod(c,scale)]);
                input = [input, appended];
            end
            output = input(:,1:scale:end);
        end

    else 
        output = input;
    end

end