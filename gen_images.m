% scripts to generate the data of the graphical example as in the paper: 
% T. L. Griffiths and M. Steyvers. Finding scientific topics. Proceedings of
% the National Academy of Sciences of U.S., 2004.
% 
% Copyright (C) 2009 Xiaodong Yu, xdyu at umiacs umd edu;
% distributable under GPL, see README.txt

close all; clear;

% ground truth of topic-specified word distribution,  Phi
% the size of the vocabulary is 16 = 4x4, i.e., each position represent a word
Phi(:,1) = [1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0]/4;
Phi(:,2) = [0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0]/4;
Phi(:,3) = [0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0]/4;
Phi(:,4) = [0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1]/4;
Phi(:,5) = [1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0]/4;
Phi(:,6) = [0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0]/4;
Phi(:,7) = [0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0]/4;
Phi(:,8) = [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1]/4;

% show ground truth of Phi
figure; colormap 'gray'
for i=1:8
    subplot(1,8,i); imagesc(reshape(Phi(:,i), [4 4])); axis equal; axis tight; % axis off;
    %title(['\phi_' num2str(i)])
end;
print('-djpeg', 'Theta_ground_truth.jpg');

% hyperparameter: assume symmetric Dirichlet distribution
alpha = 5; 
beta = 1; 

M = 500; % number of document
K = 8; % number of topic
Nd = 100; % number of words in each document
V = 16; % size of the vocabulary

% ground truth of document-specified topic distribution,  Theta
z = zeros(M,Nd);
w = zeros(M,Nd);
for m=1:M
    theta = dirrnd(ones(1,K) * alpha); % draw topic mixture proportion
    for n=1:Nd 
        z(m,n) = find(mnrnd(1,theta)==1); % draw topic for each word
        w(m,n) = find(mnrnd(1,reshape(Phi(:,z(m,n)),[1 V]))==1); % draw word
    end;
    word_hist(:,:,m) = reshape(hist(w(m,:),16), [4 4]);
    %Docs_words = (m,:) = hist2elem(Docs(:,:,m));
    Theta(m,:) = theta;
end;

% show exmple documents
figure; colormap 'gray'
for i=1:25
    subplot(5,5,i); 
    imagesc(word_hist(:,:,i)); axis equal; axis tight; axis off;
end;
print('-djpeg', 'example_documents.jpg');

save('demo_data.mat','word_hist','w','z','Theta','Phi','K', 'alpha','beta');
