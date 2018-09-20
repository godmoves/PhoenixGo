## Test PhoenixGo engine using its own weight

### Players

- phgo: PhoenixGo
- lzphgo: A modified Leela Zero engine for PhoenixGo weight

### TIme settings

5s per move, no ponder.

### Result

```
phgo v lzphgo (100 games)
board size: 19   komi: 7.5
         wins              black         white       avg cpu
phgo       63 63.00%       28 56.00%     35 70.00%    491.68
lzphgo     37 37.00%       15 30.00%     22 44.00%    602.20
                           43 43.00%     57 57.00%

player lzphgo: Leela Zero:0.15
player phgo: PhoenixGo:1.0
```
