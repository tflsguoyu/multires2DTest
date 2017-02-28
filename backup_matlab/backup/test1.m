function test
    clear
    close all
    
    maxDepth = 1000;
    x = zeros(maxDepth, 2);
    w = zeros(maxDepth, 2);

    x(1,:) = [0.5,0.5];
    w(1,:) = [cos(0.8),sin(0.8)];

    figure;
%     sigmaT_list = 3.1;
    sigmaT_list = 0.5:0.5:4.5;

    for index = 1:length(sigmaT_list)
        sigmaT = sigmaT_list(index)
        
        subplot(3,3,index)
        y = [];
        
        for samples = 1: 50000
            for dep = 1 : maxDepth - 1
                t = -log(rand)/sigmaT;
                x1 = x(dep, :) - t*w(dep, :);
                if x1(1) < 0.0 || x1(1) > 1.0 || x1(2) < 0.0 || x1(2) > 1.0
                    y = [y; x(3 : dep - 1, :)];
                    break;
                end
                theta = 2*pi*rand;
                w(dep + 1,:) = [cos(theta),sin(theta)];
                x(dep + 1,:) = x1;
            end
        end

        % density map
        densityMap = zeros(1001,1001);
        for i = 1: size(y,1)
            densityMap(round(y(i,2)*1000)+1, round(y(i,1)*1000)+1) = ...
                densityMap(round(y(i,2)*1000)+1, round(y(i,1)*1000)+1) + 1;
        end

        densityMap = conv2(densityMap, ones(20,20));
        densityMap = flip(densityMap);
        imagesc((densityMap))
        colorbar
        title(['sigmaT=' num2str(sigmaT) ' Samples=' num2str(samples)])
        
        % scattering points
%         hold on
%         scatter(y(:, 1), y(:, 2), 'b.')
%         
%         quiver(x(1,1), x(1,2), w(1,1), w(1,2), 'r', 'MaxHeadSize', 0.5);
%         axis equal
%         axis([-0.1 1.1 -0.1 1.1])
%         title(['sigmaT=' num2str(sigmaT) ' Samples=' num2str(samples)])
    end
end
