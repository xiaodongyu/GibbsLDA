function [Phi, Theta, logPw_z]=learn_GibbsLDA(w, K, alpha, beta, max_iter, ...
    burn_in_iter, sampling_lag, plot_flag, plot_phi_sizes)
% function [Phi, Theta, z]=learn_GibbsLDA(w, K, alpha, beta)
% To do: perform Gibbs sampling for LDA inference
%
% Parameters:
% w: a M x Nd matrix that stores all the words in the M documents, one document each row
%      In this simple demo, we assume the number of words in each document
%      is the same, i.e., Nd
% K: number of topics
% alpha : hyperparameter of Dirichlet prior for Theta, represent the prior count of
%           topics in a document
% beta : hyperparameter of Dirichlet prior for Phi, represent the prior
%           count of words in a topic
%       I assume symmetric Dirichlet distribution, so both alpha and beta are scalars
% max_iter: maximum iterations
% burn_in_iter: burn-in iterations
% sampling_lag: sampling lag between two read-out or figure-showing
% plot_flag: flag to show figures or not; if show figures, they will also
%           be saved on local foloders
% plot_phi_sizes: the size to reshape the Phi in order to show it
%
% Output:
% Phi: a V x K matrix, each column is a topic-specified word distribution
% Theta: a K x M matrix, each column is a document-specified topic distribution
% 
% dimensionalities
% M : number of document
% K : number of topic
% V : size of vocabulary
% Nd : number of words in each document
% 
% example usage:
% [est_Phi, est_Theta, logPw_z]=learn_GibbsLDA(w, K, alpha, beta, 1000, 200, 100, 1, [4 4]);
% 
% Reference: 
% The equations of the conditional posterior is based on the technical note 
% by Yi Wang, "Gibbs Sampling and Latent Diriclet Allocation: The Gritty Details ", 
% available from http://dbgroup.cs.tsinghua.edu.cn/wangyi/lda/index.html
%
% Copyright (C) 2009 Xiaodong Yu, xdyu at umiacs umd edu;
% distributable under GPL, see README.txt


%% initialization
disp('initialization...');
[M,Nd] = size(w);
V = length(unique(w));
NWZ = zeros(V,K) + beta;
NZM = zeros(K,M) + alpha;
NZ = sum(NWZ);
z = zeros(M,Nd);
Phi = zeros(V,K);
Theta = zeros(K,M);

for m=1:M % for each document
    for n=1:Nd % for each word
        z(m,n) = find(mnrnd(1,ones(1,K)/K )==1); % draw topic for each word
        NZM(z(m,n),m) = NZM(z(m,n),m) + 1;
        NWZ(w(m,n),z(m,n)) = NWZ(w(m,n),z(m,n)) + 1;
        NZ(z(m,n)) = NZ(z(m,n)) + 1;
    end;
end;
% show the initial values of Phi
if plot_flag == 1
    figure(1); clf;
    colormap 'gray'
    for k=1:K
        subplot(1,K,k);
        Phi(:,k) = NWZ(:,k)/NZ(k);
        imagesc(reshape(Phi(:,k), plot_phi_sizes)); axis equal; axis tight;
        title(['\phi_' num2str(k)])
    end;
end;

%% sampling and burn-in 
disp('Gibbs sampling starts');

% read_out_Phi and read_out_Theta store the sum of read-out Phi and Theta
read_out_Phi = zeros(V,K);
read_out_Theta = zeros(K,M);
read_out_sampling_num = 0;
logPw_z = zeros(1,max_iter);
for iter = 1:max_iter
    fprintf('.'); % show progress
    for m=1:M % for each document
        for n=1:Nd % for each word
            % decrease three counts
            NZM(z(m,n),m) = NZM(z(m,n),m) - 1;
            NWZ(w(m,n),z(m,n)) = NWZ(w(m,n),z(m,n)) - 1;
            NZ(z(m,n)) = NZ(z(m,n)) -1;
            % update the posterior distribution of z, p(z_i)
            p  =zeros(1,K);
            for k=1:K
                p(k) = NWZ(w(m,n),k)/NZ(k) * NZM(k,m);
            end;
            p = p/sum(p);
            % draw topic for this word
            z(m,n) = find(mnrnd(1,p)==1); 
            % increase three counts
            NZM(z(m,n),m) = NZM(z(m,n),m) + 1;
            NWZ(w(m,n),z(m,n)) = NWZ(w(m,n),z(m,n)) + 1;
            NZ(z(m,n)) = NZ(z(m,n)) + 1;
        end;
    end;
    % keep the history of the log likelihood, p(w|z)
    for zz = 1:K
        logPw_z(iter) = logPw_z(iter) + log_multinomial_beta(NWZ(:,zz)) - log_multinomial_beta(ones(1,V)*beta);
    end;
    % check convergence and whether sampling_lag iterations since last read
    % out or figure showing
    if mod(iter,sampling_lag) == 0 || iter == 1
        if iter >= burn_in_iter % read out parameters after burn-in
            read_out_sampling_num = read_out_sampling_num + 1;
            for k=1:K
                read_out_Phi(:,k) = read_out_Phi(:,k) + NWZ(:,k)/NZ(k);
            end;
            for mm = 1:M
                read_out_Theta(:,mm) = read_out_Theta(:,mm) + NZM(:,mm)/sum(NZM(:,mm));
            end;
        end;
        % show the current Phi
        if plot_flag == 1
            figure(1); clf;
            colormap 'gray'
            for k=1:K
                subplot(1,K,k);
                Phi(:,k) = NWZ(:,k)/NZ(k);
                imagesc(reshape(Phi(:,k), plot_phi_sizes)); axis equal; axis tight; axis off;
                %title(['\phi_' num2str(k)])
            end;
            fig_fname = ['Phi_iter_' num2str(iter) '.jpg'];
            print('-djpeg', fig_fname);
        end;
        fprintf('\n'); % show progress
    end;
end

% finally, parameters are obtained by averaging the read-out values computed from
% the samples after the burn-in period
Phi = read_out_Phi/read_out_sampling_num;
Theta = read_out_Theta/read_out_sampling_num;

if plot_flag == 1
    % show the final results
    figure(1); clf;
    colormap 'gray'
    for k=1:K
        subplot(1,K,k);
        imagesc(reshape(Phi(:,k), plot_phi_sizes)); axis equal; axis tight; axis off;
        %title(['\phi_' num2str(k)])
    end;
    fig_fname = 'final_Phi_iter.jpg';
    print('-djpeg', fig_fname);

    % show the history of log likelihood, log(w|z), solid red dots
    % corresponding to the iterations when the figures
    figure(2); clf;
    plot(logPw_z,'b-');
    hold on;
    plot(sampling_lag:sampling_lag:max_iter, ...
        logPw_z(sampling_lag:sampling_lag:max_iter),'ro',...
        'MarkerSize', 10, 'MarkerFaceColor', 'r');
    xlabel('iterations'); ylabel('log(P(w|z))');
    print('-djpeg', 'logPw_z.jpg');

end;

function b = log_multinomial_beta(alpha)
% compute log of multinomial beta function, a multivariate extension of
% beta function
L = length(alpha);
b = 0;
for i=1:L
    b = b + (gammaln(alpha(i)));
end;
b = b-gammaln(sum(alpha));
