# Gaussian Mixture Model
何浩勋
16212799

## EM algorithm for the ML fitting of the parameter metric mixture model
The mixture model is expressed as,
$$ f(y_j;\Psi) = \sum\limits_{i=1}^{g} \pi_i f_i(y_j ;\theta_i) $$
where $\theta_i = (\mu_i,\sigma_i)$,

Let $\pi=(\pi_1,...,\pi_g)^T$, $\mu=(\mu_1,...,\mu_g)^T$, $\sigma=(\sigma_1,...\sigma_g)^T$, $\xi=(\mu^T,\sigma^T)^T$ , $\Psi = (\pi^T,\xi^T)^T$, $y=(y_1^T,...,y_n^T)$. 
The log likelihood for $\Psi$ is given by,
$$ \textrm{log}L(\Psi) = \sum\limits_{j=1}^{n} \textrm{log} \left[\sum\limits_{i=1}^{g} \pi_i f_i(y_j;\theta_i)\right] $$ Solving the likelihood equation,
$$ \partial \textrm{log} L(\Psi) / \partial \Psi = 0 $$
We have $\hat{\Psi}$, satisfies
$$
\hat{\pi}_i = \sum\limits_{j=1}^{n} \tau_i(y_j;\hat{\Psi}) / n \ \ \ \ (i=1,...,g)
$$ and
$$
\sum\limits_{i=1}^{g} \sum\limits_{j=1}^{n} \tau_i(y_j;\hat{\Psi}) \partial \textrm{log} f_i(y_j;\hat{\theta}_i) / \partial \xi
$$ where
$$
\tau_i(y_j;\hat{\Psi})=\pi_i f_i(y_j;\theta_i) / \sum\limits_{h=1}^{g} \pi_h f_h(y_j;\theta_h)
$$

If y is viewed as being incomplete, as the associated component vectors, $z=(z_1,...,z_n)$ are not available, $z_j$ is a g-dimensional vector with $z_{ij}=(z_j)_i = 1\ or\ 0$, according to whether $y_j$ is did or did not arise from the $i$th component of the mixture $(i=1,...,g\ ,j=1,...,n)$. The complete vector is therefore declared to be,
$$ y_c = (y^T,z^T) $$
The component-label vector $z_1,...,z_n$ are taken to be the realized values of the random vectors $Z_1,...Z_n$, where, for independent feature data, it is appropriate to assume that they are distributed unconditionally as,
$$ Z_1,...,Z_n \sim \textrm{Mult}_g(1,\pi) $$
This assumption means that the sitribution of the complete-data vector $Y_C$ implies the appropriate distribution of the incomplete-data vector Y. The complete-data log likelihood for $L_c(\Psi)$, is given by
$$ \textrm{log}L_c(\Psi) = \sum\limits_{i=1}^{g} \sum\limits_{j=1}^{n} z_{ij} \lbrace \textrm{log} \pi_i + \textrm{log} f_i(y_j;\theta_i) \rbrace $$

#### E-step
$$ w_{ij} = E_{\Psi^{(k)}}(Z_{ij} | y) = \pi_i^{(k)} f_i(y_j;\theta_i^{(k)}) / \sum\limits_{h=1}^{g} \pi_h f_h(y_j;\theta_h^{(k)}) $$

#### M-step
$$
\begin{array}{lllll}
\pi_i^{(k+1)} = \sum\limits_{j=1}^{n} w_{ij} / n &\ \ \ \ (i=1,...,g) \\ \\
\mu_i^{(k+1)} = \sum\limits_{j=1}^{n} w_{ij} y_j / \sum\limits_{j=1}^{n} w_{ij} &\ \ \ \ (i=1,...,g) \\ \\
\sigma_i^{(k+1)} = \sqrt{ \sum\limits_{j=1}^{n} w_{ij} (y_j - \mu_i^{(k+1)})^2 / \sum\limits_{j=1}^{n} w_{ij} } &\ \ \ \ (i=1,...,g) \\ \\
\end{array}
$$

#### Random starting values
Specigying a random start,
$$
\begin{array}{lllll}
\mu_1^{(0)},...,\mu_g^{(0)} \sim N(\bar{y},V) \\ \\
\Sigma_0^{(0)} = V \\ \\
\pi_1=...=\pi_g = 1/g
\end{array}
$$ where $V = \sum\limits_{j=1}^{n} (y_j-\bar{y})(y_j-\bar{y})^T/n$

#### Stopping criterion
$$
\begin{array}{lllll}
l^{(k)} = log L_c(\Psi^{(k)}) = \sum\limits_{i=1}^{g} \sum\limits_{j=1}^{n} z_{ij} \lbrace \textrm{log} \pi_i + \textrm{log} f_i(y_j;\theta_i) \rbrace \\ \\
a^{(k)} = (l^{(k+1)}-l^{(k)}) / (l^{(k)}-l^{(k-1)})\ \ \ \ , k>1\\ \\
l_A^{(k+1)} = l^{(k)} + \frac{1}{1-a^{(k)}} (l^{(k+1)}-l^{(k)})
\end{array}
$$ The EM algorithm can be stopped if
$$
|l_A^{(k+1)} - l_A^{(k)}| < tol
$$

#### Simulation

##### Example 1
I randomly generated $Y_1,Y_2,Y_3$, 
$$
\begin{array}{lll}
Y_1 \sim N(-2,1) \\
Y_2 \sim N(2,1) \\
Y_3 \sim N(6,1)
\end{array}
$$ $Y_1, Y_3$ each have 100 observations, $Y_2$ have 200 observations, that is, 400 observations total. $Y = (Y_1^T,Y_2^T,Y_3^T)$

The mixture model fits the real distribution well when $g \leq 4$, we can see that there are signs of overfit when $g>4$.

![](fig1.png)

##### Example 2
I randomly generated $Y_1,Y_2$, 
$$
\begin{array}{ll}
Y_1 \sim Beta(1,4) \\
Y_2 \sim Beta(4,1)
\end{array}
$$ $Y_1, Y_2$ each have 200 observations, that is, 400 observations total. $Y = (Y_1^T,Y_2^T)$

When $g>4$, I encountered a precision problem which causes 'Log(0) error', I had to stop EM algorithm and use the current result. However, the parameters seems converged.

![](fig2.png)