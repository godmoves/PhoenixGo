## Test of init to loss vs lz mcts

### Weight

Old 10b resnet 1600k weight

### Game settings

5 seconds per move, noponder, threads 16, batch size 16. (about 22k-25k playouts)

### Result

```
lz_mcts v lz_init_loss (272 games)
board size: 19   komi: 7.5
               wins              black          white        avg cpu
lz_mcts         129 47.43%       58  42.65%     71  52.21%   1119.29
lz_init_loss    143 52.57%       65  47.79%     78  57.35%   1118.10
                                 123 45.22%     149 54.78%

player lz_init_loss: Leela Zero:0.16
player lz_mcts: Leela Zero:0.16
```

### Conclusion

The init to loss method seems promising because minigo gets a lot benefits form
that and the test of LZ also shows a roughly equal play strength under time parity,
even if all LZ games and networks are optimized for its original searching architecture.

### Archive

You can find all test games [here](https://github.com/godmoves/HappyGo/releases/tag/init_to_loss)
