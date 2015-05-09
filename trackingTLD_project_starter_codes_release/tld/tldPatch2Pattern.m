% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.

function pattern = tldPatch2Pattern(patch,patchsize)

pattern = [];
%% ------------------ (BEGIN) -------------------
%% TODO: extract a feature from the patch after optionally resizing
%%       it to patchsize. Store this feature in pattern. At the simplest
%%       level, this could be a simple mean adjusted version
%%       of the resized patch itself.
%%       Some other features to try might be binary features such
%%       as LBP, BRISK, FREAK.

%% given:
%%-------
%% patch (M X N X 3) -- image patch
%% patchsize (2 X 2) -- pathchsize to resize the image patch to before
%%                      feature extraction
%% to update or compute:
%% --------------------
%% pattern (1 x tld.model.pattern_size) vector feature extracted from patch

%% -----------------  (END) ---------------------
    patch  = imresize(patch,patchsize); % 'bilinear' is faster
    %pattern = double(patch(:));
    %pattern = pattern - mean(pattern);
    pattern = efficientLBP(patch);
    pattern = reshape(pattern,[prod(size(pattern)),1]);
end

function [ LBP ] = naiveLBP(patch)
[m, n]=size(patch);
patch = padarray(patch,[1 1]);
size(patch)
for i=2:m
    for j=2:n
        J0=patch(i,j);
        I3(i-1,j-1)=patch(i-1,j-1)>J0;
        I3(i-1,j)=patch(i-1,j)>J0;
        I3(i-1,j+1)=patch(i-1,j+1)>J0; 
        I3(i,j+1)=patch(i,j+1)>J0;
        I3(i+1,j+1)=patch(i+1,j+1)>J0; 
        I3(i+1,j)=patch(i+1,j)>J0; 
        I3(i+1,j-1)=patch(i+1,j-1)>J0; 
        I3(i,j-1)=patch(i,j-1)>J0;
        %convert value to decimal
        LBP(i,j)=I3(i-1,j-1)*2^7+I3(i-1,j)*2^6+I3(i-1,j+1)*2^5+I3(i,j+1)*2^4+I3(i+1,j+1)*2^3+I3(i+1,j)*2^2+I3(i+1,j-1)*2^1+I3(i,j-1)*2^0;
    end
end
end

%implemented Nikolay S. 2014-01-09
function LBP= efficientLBP(inImg, varargin) 

    isRotInv=false;
    isChanWiseRot=false;
    filtR=generateRadialFilterLBP(8, 1);
    nClrChans=size(inImg, 3);

    inImgType=class(inImg);
    calcClass='single';

    isCalcClassInput=strcmpi(inImgType, calcClass);
    if ~isCalcClassInput
        inImg=cast(inImg, calcClass);
    end
    imgSize=size(inImg);

    nNeigh=size(filtR, 3);

    if nNeigh<=8
        outClass='uint8';
    elseif nNeigh>8 && nNeigh<=16
        outClass='uint16';
    elseif nNeigh>16 && nNeigh<=32
        outClass='uint32';
    elseif nNeigh>32 && nNeigh<=64
        outClass='uint64';
    else
        outClass=calcClass;
    end

    if isRotInv
        nRotLBP=nNeigh;
        nPixelsSingleChan=imgSize(1)*imgSize(2);
        iSingleChan=reshape( 1:nPixelsSingleChan, imgSize(1), imgSize(2) );
    else
        nRotLBP=1;
    end

    nEps=-3;
    weigthVec=reshape(2.^( (1:nNeigh) -1), 1, 1, nNeigh);
    weigthMat=repmat( weigthVec, imgSize([1, 2]) );
    binaryWord=zeros(imgSize(1), imgSize(2), nNeigh, calcClass);
    LBP=zeros(imgSize, outClass);
    possibleLBP=zeros(imgSize(1), imgSize(2), nRotLBP);
    for iChan=1:nClrChans  
        % Initiate neighbours relation filter and LBP's matrix
        for iFiltElem=1:nNeigh
            % Rotate filter- to compare center to next neigbour
            filtNeight=filtR(:, :, iFiltElem);

            % calculate relevant LBP elements via filtering
            binaryWord(:, :, iFiltElem)=cast( ...
                roundnS(filter2( filtNeight, inImg(:, :, iChan), 'same' ), nEps) >= 0,...
                calcClass );
            % Without rounding sometimes inaqulity happens in some pixels
            % compared to pixelwiseLBP
        end % for iFiltElem=1:nNeigh

        for iRot=1:nRotLBP
            % find all relevant LBP candidates
            possibleLBP(:, :, iRot)=sum(binaryWord.*weigthMat, 3);
            if iRot < nRotLBP
                binaryWord=circshift(binaryWord, [0, 0, 1]); % shift binaryWord elements
            end
        end

        if isRotInv
            if iChan==1 || isChanWiseRot
                % Find minimal LBP, and the rotation applied to first color channel
                [minColroInvLBP, iMin]=min(possibleLBP, [], 3);

                % calculte 3D matrix index
                iCircShiftMinLBP=iSingleChan+(iMin-1)*nPixelsSingleChan;
            else
                % the above rotation of the first channel, holds to rest of the channels
                minColroInvLBP=possibleLBP(iCircShiftMinLBP);
            end % if iChan==1 || isChanWiseRot
        else
            minColroInvLBP=possibleLBP;
        end % if isRotInv

        if strcmpi(outClass, calcClass)
            LBP(:, :, iChan)=minColroInvLBP;
        else
            LBP(:, :, iChan)=cast(minColroInvLBP, outClass);
        end
    end
end
function [radInterpFilt]=generateRadialFilterLBP(p, r)
%% Default params
if nargin<2
    r=1;
    if nargin<1
        p=8;
    end
end

%% verify params leget values
r=max(1, r);    % radius below 1 is illegal
p=round(p);     % non integer number of neighbours sound oucward
p=max(1, p);    % number of neighbours below 1 is illegal


%% find elements angles, aranged counter clocwise starting from "X axis"
% See http://www.ee.oulu.fi/mvg/files/pdf/pdf_6.pdf for illustration
theta=linspace(0, 2*pi, p+1)+pi/2;   
theta=theta(1:end-1);           % remove obsolite last element (0=2*pi)

%% Find relevant coordinates
[rowsFilt, colsFilt] = pol2cart(theta, repmat(r, size(theta) )); % convert to cartesian
nEps=-3;
rowsFilt=roundnS(rowsFilt, nEps);
colsFilt=roundnS(colsFilt, nEps);

% Matrix indexes should be integers
rowsFloor=floor(rowsFilt);
rowsCeil=ceil(rowsFilt);

colsFloor=floor(colsFilt);
colsCeil=ceil(colsFilt);

rowsDistFloor=1-abs( rowsFloor-rowsFilt );
rowsDistCeil=1-abs( rowsCeil-rowsFilt );
colsDistFloor=1-abs( colsFloor-colsFilt );
colsDistCeil=1-abs( colsCeil-colsFilt );

% Find minimal filter dimentions, based on indexes
filtDims=[ceil( max(rowsFilt) )-floor( min(rowsFilt) ),...
    ceil( max(colsFilt) )-floor( min(colsFilt) ) ];
filtDims=filtDims+mod(filtDims+1, 2); % verify filter dimentions are odd

filtCenter=(filtDims+1)/2;

%% Convert cotersian coordinates to matrix elements coordinates via simple shift
rowsFloor=rowsFloor+filtCenter(1);
rowsCeil=rowsCeil+filtCenter(1);
colsFloor=colsFloor+filtCenter(2);
colsCeil=colsCeil+filtCenter(2);

%% Generate the filter- each 2D slice for filter element  
radInterpFilt=zeros( [filtDims,  p], 'single'); % initate filter with zeros
for iP=1:p
    radInterpFilt( rowsFloor(iP), colsFloor(iP), iP )=...
        radInterpFilt( rowsFloor(iP), colsFloor(iP), iP )+rowsDistFloor(iP)+colsDistFloor(iP);
    
    radInterpFilt( rowsFloor(iP), colsCeil(iP), iP )=...
        radInterpFilt( rowsFloor(iP), colsCeil(iP), iP )+rowsDistFloor(iP)+colsDistCeil(iP);
    
    radInterpFilt( rowsCeil(iP), colsFloor(iP), iP )=...
        radInterpFilt( rowsCeil(iP), colsFloor(iP), iP )+rowsDistCeil(iP)+colsDistFloor(iP);
   
    radInterpFilt( rowsCeil(iP), colsCeil(iP), iP )=...
        radInterpFilt( rowsCeil(iP), colsCeil(iP), iP )+rowsDistCeil(iP)+colsDistCeil(iP);
    
    radInterpFilt( :, :, iP )=radInterpFilt( :, :, iP )/sum(sum(radInterpFilt( :, :, iP )));
end
% imshow(sum(radInterpFilt,3), []);

% Substract 1 at central element to get difference between central element and relevant
% neighbours: (5) T=p{s(g1-g0), s(g2-g0),...,s(gn-g0)}
radInterpFilt( filtCenter(1), filtCenter(2), : )=...
    radInterpFilt( filtCenter(1), filtCenter(2), : )-1; 
end

function outData=roundnS(inData, nEps)
quantVal=10^nEps;
outData=round(inData/quantVal)*quantVal;
end