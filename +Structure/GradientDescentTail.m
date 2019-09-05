classdef GradientDescentTail
% GradientDescentTail Gradient descent algorithm becomes expensive if error
% calculation is done after every step of gradient/weight computation.
% Since the trend of algorithm is (generally) descending and fluctuation
% hence we take the minimum of last (TAIL_SIZE) iterations out of N iterations. 
    properties (GetAccess = public, SetAccess = private)
        Ebests=[]; %best tail errors
        %Ebests [curBestValue1, curBestValue2, ...]
        Wbests=cell(0); %best tail Weights
        Gbests=cell(0); %best tail gradients
    end
    
    properties (GetAccess = private, SetAccess = private)
        eSGDtail = [];
        wSGDtail = cell(0);
        gSGDtail = cell(0);
%         d = 0;
    end
    
    properties (Constant)
        TAIL_SIZE = 10;
    end
    
    methods
%         function e = getEin(idx,x,y,W)
%             e = 0;
%             for in = idx %take the full training sequence/data of this fold   
%                 e = e + log(1+exp(-y(in)*W'*x(:,in)));      
%             end                  
%             e = e/size(idx,2); %average error         
%         end        
        
        %private constructor
%         function obj = GradientDescentTail(varargin)
%             if(nargin == 1)
%                 obj.d = varargin{1};
%             else
%                 error('default constructor is not allowed\n');
%                 exit
%             end
%         end

        function obj = accumulateTail(obj,W,gr,Ein)
        % GradientDescentTail = ACCUMULATETAIL(GradientDescentTail,W(dx1),gr(dx1),Ein(1x1))
            obj.eSGDtail = [obj.eSGDtail Ein];
            obj.wSGDtail{end+1} =  W;
            obj.gSGDtail{end+1} = gr;
        
            if(size(obj.eSGDtail,2)<obj.TAIL_SIZE)
%                 obj.eSGDtail = [obj.eSGDtail Ein];
%                 obj.wSGDtail{end+1} =  W;
%                 obj.gSGDtail{end+1} = gr;
            else
                [tmpMin, tmpIdx] = min(obj.eSGDtail);
            
                obj.Ebests = [obj.Ebests tmpMin];
                obj.Wbests{end+1} = obj.wSGDtail{tmpIdx};
                obj.Gbests{end+1} = obj.gSGDtail{tmpIdx};
                
                obj.eSGDtail = [];
                obj.wSGDtail = cell(0);
                obj.gSGDtail = cell(0);
            end
        end
        
        function bestIdx = getBestIdx(obj)
            [~,bestIdx] = min(obj.Ebests);
        end
        
        function be=getBestE(obj)
            bestIdx = obj.getBestIdx(); %or getBestIdx(obj) - same things
            be = obj.Ebests(bestIdx);
        end
        function be=getBestW(obj)
            bestIdx = obj.getBestIdx(); 
            be = obj.Wbests{bestIdx};
        end
        function be=getBestG(obj)
            bestIdx = obj.getBestIdx(); 
            be = obj.Gbests{bestIdx};
        end
        
        function eb=get.Ebests(obj)
            eb = obj.Ebests;
        end
        function wb=get.Wbests(obj)
            wb = obj.Wbests;
        end
        function gb=get.Gbests(obj)
            gb = obj.Gbests;
        end
    end
end