# Logic Operation on 3 inputs (LOP3)
PTX Example:
```ptx
lop3.b32 d, a, b, c, immLut;
```

SASS Example (SM_86):
```sass
LOP3.LUT R3, R0, R1, R2, immLut, !PT;
```

For example, immLut = 0x80 is `and` operation:
- `d = a & b & c`

# Immediate Lookup Table (immLut)
The and operation's output is like this:
```
immLut 0x80 (10000000)
A B C | output
0 0 0 | 0
0 0 1 | 0
0 1 0 | 0
0 1 1 | 0
1 0 0 | 0
1 0 1 | 0
1 1 0 | 0
1 1 1 | 1
```

0x80 (0b10000000) is the binary representation of the immediate lookup table for the `and` operation.

0xc0(0b11000000) is the binary representation of the immediate lookup table for the `A & B` operation:
```
immLut 0xc0 (11000000)
  A B C | Output
  0 0 0 | 0
  0 0 1 | 0
  0 1 0 | 0
  0 1 1 | 0
  1 0 0 | 0
  1 0 1 | 0
  1 1 0 | 1
  1 1 1 | 1
```
