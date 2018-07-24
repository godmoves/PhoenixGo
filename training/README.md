## Test different ways to learn

- [ ] different weight of value part  
- [ ] cross entropy value part  
- [ ] ResNext as Policy-Value net  
- [ ] with/without ELF data
- [ ] imitate elf/phoenixgo
- [ ] global pooling

### Baseline

The last 10x128 network [39d46507](http://zero.sjeng.org/networks/39d465076ed1bdeaf4f85b35c2b569f604daa60076cbee9bbaab359f92a7c1c4.gz)
will be used as baseline to test the training results.

### Training data

Default training data set includes ELF data and Leela Zero data listed below.

#### Leela Zero data

[d351f06e](https://leela.online-go.com/training/train_d351f06e.zip)  
[050375ce](https://leela.online-go.com/training/train_050375ce.zip)  
[834f35fa](https://leela.online-go.com/training/train_834f35fa.zip)  
[7ff174e6](https://leela.online-go.com/training/train_7ff174e6.zip)  
[e1d466aa](https://leela.online-go.com/training/train_e1d466aa.zip)  
[12692a83](https://leela.online-go.com/training/train_12692a83.zip)  
[672342b5](https://leela.online-go.com/training/train_672342b5.zip)  
[2b80a9db](https://leela.online-go.com/training/train_2b80a9db.zip)  
[5839eb77](https://leela.online-go.com/training/train_5839eb77.zip)  
[d0187996](https://leela.online-go.com/training/train_d0187996.zip)  
[10bc1042](https://leela.online-go.com/training/train_10bc1042.zip)  

#### ELF data

[62b5417b](https://leela.online-go.com/training/train_62b5417b.zip)

### Learning rate schedule

When total loss drops less than 5% in 80000 steps, the learning rate will be
lowered (x0.1). If learning rate less than 1e-9, then the training process will
stop.

### Result

Tensorflow logs will be saved in `tf_training/leelalogs`, weight files and
Tensorflow checkpoint will be saved in `tf_training/weights` and training logs
are in `tf_training/traininglogs`. 

Check the real time learning process at [this site](http://101.231.109.4:6006/#scalars&run=test&_smoothingWeight=0&_ignoreYOutliers=false).
