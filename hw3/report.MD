# Problem 8.16

$X$ is Bernoulli with $Pr(X=1)=1-Pr(X=0)=p$. $Y$ given $X=j$ is normal with mean $\mu_j$ and variance $\sigma^2$.

Consider now the monotone missing-data pattern with Y completely observed but n-r values of X missing and an ignorable mechanism.

Describe the E and M steps of EM algorithm for this problem.

### preparation
Let $\theta = (p,\mu_0,\mu_1,\sigma)$ , we have
$$
\large
\begin{array}{ccc}
f(y_i|x_i=j;\theta) = \frac{1}{\sqrt{2\pi}\sigma}
exp\left[-\frac{(y_i-\mu_j)^2}{2\sigma_2}\right]\\
\\
f(x_i=1|\theta)=1-f(x_i=0|\theta)=p
\end{array}
$$

### E-step
$$
\large
w_i=E[X_i|x,y,\theta^{(t)}] = \begin{cases}
x_i &, i\leq r\\
\frac{f(y_i|x_i=1;\theta^{(t)})f(x_i=1;\theta^{(t)})}{f(y_i|x_i=0;\theta^{(t)})f(x_i=0;\theta^{(t)})+f(y_i|x_i=1;\theta^{(t)})f(x_i=1;\theta^{(t)})} &, i>r
\end{cases}
$$

### M-step
$$
\large
\begin{array}{ccccccc}
p^{(t+1)}&=&\sum\limits_{i=1}^{n}{w_i} / n\\
\\
\mu_0^{(t+1)}&=&\sum\limits_{i=1}^{n}{(1-w_i)y_i} / {\sum\limits_{i=1}^{n}{(1-w_i)}}\\
\\
\mu_1^{(t+1)}&=&\sum\limits_{i=1}^{n}{w_iy_i} / {\sum\limits_{i=1}^{n}{w_i}}\\
\\
\sigma^{(t+1)}&=&\sqrt{\sum\limits_{i=0}^{n}[(1-w_i)(y_i-\mu_0^{(t+1)})^2 + w_i(y_i-\mu_1^{(t+1)})^2]/{n}}
\end{array}
$$