function [h] = getkernel(y, g, lam, shift, win, tol, maxIter, mask, smin)
%% Infer the most likely discretized spike train underlying an AR(2) fluorescence trace
% Solves the sparse non-negative deconvolution problem
%  min 1/2|Ks-y|^2 + lam * |s|_1 subject to s_t = c_t-g c_{t-1} >= 0

%% inputs:
%   y:  T*1 vector, vth dimensional array containing the fluorescence intensities 
        %withone entry per time-bin.
%   g:  vector, shape (p,)
%       if p in (1,2): AR coefficients for AR(p) process 
%       else: kernel that models the fluorescence implulse response 
%   lam:  scalar, sparsity penalty parameter lambda. 
%   shift: integer scalar, number of frames by which to shift window from on run of
%       NNLS to the next, default-100
%   win: integer acalar, window size 
%   tol: scalar, tolerance parameters 
%   maxIter: scalar, maximum number of iterations before termination 
%   mask: T * 1 boolean vector, restrict potential spike times 
%   smin: scalar, minimum spike size 
%% outputs
%   c: T*1 vector, the inferred denoised fluorescence signal at each time-bin.
%   s: T*1 vector, discetized deconvolved neural activity (spikes) 

%% Authors: Pengcheng Zhou, Carnegie Mellon University, 2016
% ported from the Python implementation from Johannes Friedrich

%% References 
% Friedrich J et.al., NIPS 2016, Fast Active Set Method for Online Spike Inference from Calcium Imaging

%% input arguments  
T = length(y); 
y = reshape(y, [], 1); 

if ~exist('lam', 'var') || isempty(lam)
    lam = 0; 
end
if ~exist('shift', 'var') || isempty(shift)
    shift = 100; 
end
if ~exist('win', 'var') || isempty(win)
    win = 200; 
end

if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-9; 
end
if ~exist('maxIter', 'var') || isempty(maxIter)
    maxIter = []; 
end
if ~exist('mask', 'var') || isempty(mask)
    mask = true(T,1); 
end
if ~exist('smin', 'var') || isempty(smin)
    smin = 0; 
end
%% get the response kernel
w = win; 
K = zeros(w); 
[u, t] = meshgrid(1:w, 1:w);  
ind = 1+t-u;
if length(g)==1
    h = exp(log(g)*(0:(w-1)));
elseif length(g)==2
    temp = roots([1, -g(1), -g(2)]);
    d = max(temp);
    r = min(temp);
    h = (exp(log(d)*(1:w)) - exp(log(r)*(1:w))) / (d-r); % convolution kernel
else
    h = g;
end