% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.

function tld = tldTrainNN(pEx,nEx,tld)

nP = size(pEx,2); % get the number of positive example 
nN = size(nEx,2); % get the number of negative examples


%% ------------------ (BEGIN) --------------------
%% TODO: Update tld.pex and tld.nex

%% These are the positive and negative examples respectively retained
%% from all images seen so far.
%% given:
%% -----
%%  (pEx, nEx) -- Your are provided a set of positives and negatives from current image
%% respectively.
%%
%% to update:
%% ---------
%%  tld.pex, tld.nex -- Choose a good scheme to update the positives and negatives.
%%                      Naively extending tld.pex and tld.nex with all of pEx and nEx will
%%                      blow up computation. Choose wisely!

%% ------------------ (END) ----------------------
%positive examples 
if isempty(tld.pex)
    tld.pex = [tld.pex pEx];
else
    for i = 1:nP
        [conf, isin] = tldNN(pEx(:,i),tld);
        if conf <= tld.model.thr_nn && (isin(3) == 1 || isnan(isin(1)))%model originally labeled positive but NN disagrees
            tld.pex = [tld.pex pEx(:,i)]; 
        end
    end
end
if size(tld.pex,2)>400
   
   indexes = randperm(size(tld.pex,2)-200);
   tld.pex = [tld.pex(:,1:100) tld.pex(:,indexes(1:150)+50)];
end
%negative examples
if isempty(tld.nex)
    tld.nex = [tld.nex nEx];
else
    for i = 1:nN
        [conf, isin] = tldNN(nEx(:,i),tld);
        if (conf > 0.5 || isnan(conf)) && (isin(1) == 1 || isnan(isin(3))) %model originally labeled positive but NN disagrees
            tld.nex = [tld.nex nEx(:,i)];
        end
    end
end
if size(tld.nex,2)>400
   indexes = randperm(size(tld.nex,2)-50);
   tld.nex = [tld.nex(:,1:100) tld.nex(:,indexes(1:150)+50)];
end

