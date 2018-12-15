## Different weights of the value part

According to the AlphaGo Zero paper, the total loss to optimize is:
```
total_loss = w1 * policy_loss + w2 * value_loss + w3 * reg_loss
```
And the default weights are:
```
w1 = 1.0
w2 = 1.0
w3 = 1e-4
```
But in the `method` part of the paper, they mention that they've used a different
setting for supervised learning due to overfitting:
```
w1 = 1.0
w2 = 0.01
w3 = 1e-4
```
So in this experiment we test the influences of different w2.

### Training setting

Steps per epoch: 800k

Value of w2: 0.01, 1, 10

Test setting: 1600 playouts, no ponder, Leela Zero master branch engine.

### Result

`w2 = 1`: baseline, we've trained this in `different_steps`.

`w2 = 0.01`: the final `policy_loss` is roughly the same as before, but
`value_loss` is way much larger, and it got **20%** winrate vs baseline.

`w2 = 10`: the NN can not converge, so I finally give this up.

### Summary

It seems that the training set is large enough (according to the AGZ paper), so
there is no need to use a smaller weight for the value part.
