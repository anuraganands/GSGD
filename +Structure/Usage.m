clear all
clc
import Structure.*;

sk = GradientDescentTail;
fprintf('Tail size: %d\n',sk.TAIL_SIZE); % set tailsize to 2
sk = sk.accumulateTail([5 2; 5 2; 5 2], [3 0; 3 0; 3 0], 9);
sk = sk.accumulateTail([6 3; 6 3; 6 3], [1 6; 1 6; 1 6], 7);
sk = sk.accumulateTail([4 1; 4 1; 4 1], [7 4; 7 4; 7 4], 2);
fprintf('(1)Best Ein (may need to reduce tail size otherwise will see []): %d\n',sk.Ebests(end));
sk = sk.accumulateTail([15 22; 15 22; 15 22], [13 8; 13 8; 13 8], 6);
sk = sk.accumulateTail([16 33; 15 10; 122 90], [11 7; 11 7; 11 7], 4);
sk = sk.accumulateTail([14 11; 14 11; 14 11], [77 4 ; 77 4; 77 4], 5);
fprintf('(2)Best Ein (may need to reduce tail size otherwise will see []): %d\n',sk.Ebests(end));
sk
bi = sk.getBestIdx;
% sk.Ebests(bi)
sk.getBestE
sk.getBestW
sk.getBestG
% [] + sk.getBestW
% % sk.Wbests(:,bi)
% % sk.Gbests(:,bi)


