这个branch打算实验一个intuitive的采样方法：

因为：
x_t = sqrt(a_bar_t)x_0 + sqrt(1-a_bar_t)epsilon


x_{t-1} = sqrt(a_bar_{t-1})x_0 + sqrt(1-a_bar_{t-1})epsilon

那么

x_0 = (x_t - sqrt(1-a_bar_t)epsilon) / sqrt(a_bar_t)

x_{t-1} = (x_t - sqrt(1-a_bar_t)epsilon) / sqrt(a_bar_t) * sqrt(a_bar_{t-1}) + sqrt(1-a_bar_{t-1})epsilon 
= sqrt(a_bar_{t-1}) / sqrt(a_bar_t) * x_t + sqrt(1-a_bar_{t-1}) (sqrt(a_bar_{t-1}) / sqrt(a_bar_t) - 1)epsilon)

这个看上去特别像是DDIM，我需要花一些时间把这个搞清楚。

参考这个代码：https://github.com/Alokia/diffusion-DDIM-pytorch/blob/master/utils/engine.py#L161

终于成了。成功的关键是，t_prev这里当t-1<0的时候怎么处理。

当eta=0时是DDIM
当eta=1时是DDPM

DDIM paper说它比DDPM好。

