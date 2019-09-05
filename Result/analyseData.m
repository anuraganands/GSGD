
%% How to Run:
% 1 - simply choose "TestResults" folder. 
function analyseData()
    clear all
    clc

    d = uigetdir(pwd, 'Select a folder');
    files = dir(fullfile(d, '*D.csv'));
    COL = 14;
    % 1 - RunNo,
    % 2 - error1, 
    % 3 - error2, 
    % 4 - error3, 
    % 5 - error4,	
    % 6 - error5,	
    % 7 - error6,	
    % 8 - error7,	
    % 9 - error8,	
    % 10 - error9,	
    % 11 - SR,
    % 12 - NFC,
    % results..
    % 12 - mean, median, best, worst, null, win%
    % 13 - NFCs
	% 14 - Ein {mean, median, best, worst} - new one
    
   
   
    total = size(files,1);
    Bests = cell(1);
    probNo = 0;
    totalRunsConducted = 30;
    fprintf(['file name,significance1,significance2,wins,Mean@NFCs,Median@NFCs,Best@NFCs,Worst@NFCs,' ...
        'EinMean@NFCs,EinMedian@NFCs,EinBest@NFCs,EinWorst@NFCs\n']);
    
    for(i = 1:total)
%         fprintf('proccessing file %s\n',files(i).name);
        [fid,msg] = fopen(files(i).name,'r');

        HL = 0;  %ignore header lines (first few lines)
        HC = 0;  %ignore columns (first few columns)
        result = textscan(fid, '', 'HeaderLines', HL, 'HeaderColumns', HC, 'Delimiter', ',');
        fclose(fid);
        
        EinNFCs =  result{end}(totalRunsConducted+1:totalRunsConducted+4)/1;
		Eins =  result{end-1}(totalRunsConducted+1:totalRunsConducted+4);
		NFCs = result{end-2}(totalRunsConducted+1:totalRunsConducted+4)/1;
		aggregates = result{end-3}(totalRunsConducted+1:totalRunsConducted+4);
        wins = result{end-3}(totalRunsConducted+6);
        sz = size(result,2);
        oneThird = floor(sz/3);
        twoThird = floor(sz*2/3);
        significance1 = result{oneThird}(totalRunsConducted+5); %1/3rd of iteration
        significance2 = result{twoThird}(totalRunsConducted+5); %2/3rd of iteration
        if(significance1<0.01)
            significance1 = 0.01;
        elseif (significance1<0.05)
            significance1 = 0.05;
        else
            significance1 = 0.0;
        end
        
        if(significance2<0.01)
            significance2 = 0.01;
        elseif (significance2<0.05)
            significance2 = 0.05;
        else
            significance2 = 0.0;
        end

        % %             '%0.1f@%0.1fk, %0.1f@%0.1fk, %0.1f@%0.1fk, %0.1f@%0.1fk,'...
% %             '%0.1f@%0.1fk, %0.1f@%0.1fk, %0.1f@%0.1fk, %0.1f@%0.1fk\n'],significance, wins, ...
        fprintf('%s,',files(i).name);
        fprintf(['%1.2f,%1.2f,%1.1f%%,' ...
            '%0.1f@%d,%0.1f@%d,%0.1f@%d,%0.1f@%d,'...
            '%0.3f@%d,%0.3f@%d,%0.3f@%d,%0.3f@%d\n'],significance1, significance2, wins, ...
            aggregates(1),NFCs(1), aggregates(2),NFCs(2), aggregates(3),NFCs(3), aggregates(4),NFCs(4),...
            Eins(1),EinNFCs(1), Eins(2),EinNFCs(2), Eins(3),EinNFCs(3), Eins(4),EinNFCs(4));
 
    end
  
    
end

% % There is an undocumented textscan() parameter 'headercolumns' to indicate the number of leading columns on the line to skip. Note for this purpose that "column" is determined using the same criteria used to determine "column" for the rest of textscan().
% % 
% % Possibly this HeaderColumns setting is only implemented if you use the undocumented format string '' (the empty string) which only works when all of the (unskipped) columns are numeric.
% % 
% % HL = 3;  %for example
% % HC = 10;  %in your case
% % result = textscan(fid, '', 'HeaderLines', HL, 'HeaderColumns', HC, 'Delimiter', ',');
% % If you want more control over your columns or do not like using undocumented parameters, then use an explicit format that throws away the unwanted data:
% % 
% % HL = 3;  %for example
% % HC = 10;  %in your case
% % NF = 1000; %1000 desired fields
% % lineformat = [repmat('%*s',1,HC) repmat('%f',1,NF)];
% % result = textscan(fid, lineformat, 'HeaderLines', HL, 'Delimiter', ',');

