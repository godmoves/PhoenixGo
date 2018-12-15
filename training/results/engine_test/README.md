## Performance test of different engines

The elf v0 weight is used as the baseline weight.  
Time settings: 5s per move, no ponder.

## Engine

- [Leela Zero](https://github.com/gcp/leela-zero/commit/488de437e9c669da9870aefa39749322661bb8bf)
- [ELF](https://github.com/pytorch/ELF/commit/751eb1cf85aac15e3c3c792a47afa1ac64ea810c)
- [HappyGo/Gigo](https://github.com/godmoves/HappyGo/commit/d5df02c8bc13295bcc94c89a05363eb3c6819c0a)

## Result

### ELF OpenGo vs Leela Zero

```
elf v leelaz (92 games)
board size: 19   komi: 7.5
         wins              black         white       avg cpu
elf        31 33.70%       9  19.57%     22 47.83%    601.82
leelaz     61 66.30%       24 52.17%     37 80.43%   1064.23
                           33 35.87%     59 64.13%

player elf: DF2:1.0
player leelaz: Leela Zero:0.15
```

### Gigo vs Leela Zero

```
gigo v leelaz (99 games)
board size: 19   komi: 7.5
         wins              black         white       avg cpu
gigo       30 30.30%       13 26.53%     17 34.00%    393.48
leelaz     69 69.70%       33 66.00%     36 73.47%    650.20
                           46 46.46%     53 53.54%

player gigo: gigo:1.15
player leelaz: Leela Zero:0.15
```

