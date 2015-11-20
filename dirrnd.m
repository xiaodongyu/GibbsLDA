function theta = dirrnd(alpha)
% draws a sample from a dirichlet with the parameter vector alpha
theta = randg(alpha);
theta = theta/sum(theta);
