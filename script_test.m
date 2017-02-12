function test
clc
scale = 3;
input = repmat([1:12], [3,1])
% output = imresize(input, [1 4], 'box')
output = downsample(input, scale, 'x_average')
mean(input(:))
mean(output(:))
figure;
plot(input,2*ones(1,length(input)),'b*')
hold on
plot(output, 2.5*ones(1,length(output)), 'r*')
plot(0,0)
plot(0,5)

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