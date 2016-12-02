function test

    clear
    close all


    x(1,:) = [0.5,1];
    w(1,:) = [cos(1.6),sin(1.6)];


    figure;
    sigmaT_list = [0.1:0.5:4.1];
    % sigmaT_list = [1:0.5:5];

    for index = 1:length(sigmaT_list)

        sigmaT = sigmaT_list(index)
        subplot(3,3,index);  

        for samples = 1: 2000

            pathDepth = 1;
            
            while(1)
                t = -log(rand)/sigmaT;
                d = intersect(x(pathDepth,:), w(pathDepth,:));
                if d==0
                    break;
                elseif t>=d
                    theta = 2*pi*rand;
                    w(pathDepth+1,:) = [cos(theta),sin(theta)];
                    x(pathDepth+1,:) = x(pathDepth,:) - d * w(pathDepth,:);
                    break;
                else
                    theta = 2*pi*rand;
                    w(pathDepth+1,:) = [cos(theta),sin(theta)];
                    x(pathDepth+1,:) = x(pathDepth,:) - t * w(pathDepth,:);
                    pathDepth = pathDepth+1;
                end
            end

            plot(x(:,1), x(:,2),'b.');hold on;

%             y(samples,:) = x(end,:); % save final position of each path
            x(2:end,:) = [];
            w(2:end,:) = [];
        end

%         plot(y(:,1),y(:,2),'b.');hold on
        plot(x(1,1),x(1,2),'ro');
        quiver(x(1,1),x(1,2),w(1,1),w(1,2),'r','MaxHeadSize',0.5);
        axis equal
        axis([-0.1 1.1 -0.1 1.1])
        title(['sigmaT=' num2str(sigmaT) ' Samples=' num2str(samples)])


    end
    

%     % test intersection
%     [d,id] = intersect([0.5 0.5],[-1 -1])
    
    function [d,id] = intersect(x_12,w_12)
        
        k = inf(1,4);

        % line x = 0
        if w_12(1)~=0 
            k(1) = -x_12(1)/w_12(1);
        end
        
        % line x = 1
        if w_12(1)~=0 
            k(2) = (1-x_12(1))/w_12(1);
        end

        % line y = 0
        if w_12(2)~=0 
            k(3) = -x_12(2)/w_12(2);
        end

        % line y = 1
        if w_12(2)~=0 
            k(4) = (1-x_12(2))/w_12(2);
        end

        [k,I] = sort(k);
        d = -k(2);
        id = I(2);
        
    end


end