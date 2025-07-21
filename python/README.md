# Jul 20
## Q. How built-in str class stores its own data?
I understand that str class is immutable, and it must have raw data pointer to the contents and its length
as an internal data structure. My question is that when appending to str, is the memcpy of the original content and the appended content to the new memory location unavoidable?
```
s = ""
for i in range(1000):
    s += "a"
```
If `a = str(ptr=0x0000, length=5)` and `b = str(ptr=0x0005, length=6)`, then the new string c = a + b can be created as `c = str(ptr=0x0000, length=11)` without having to allocate new memory and memcpy. And if you want to have d = c + "!", then you can create d by *`(0x000b) = '!'; d = str(ptr=0x0000, length=12;`.

Is this intuition correct? Is this kind of optimization happening in the Python str class?
```
0x0000 "hello"
0x0005 " world"
```

As of Python 3.12.9, it turned out to be not correct. 
```
Python 3.12.9 (main, Feb  4 2025, 14:38:38) [Clang 19.1.7 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import sys
>>> sys.implementation
namespace(name='cpython', cache_tag='cpython-312', version=sys.version_info(major=3, minor=12, micro=9, releaselevel='final', serial=0), hexversion=51120624, _multiarch='darwin')
```

The script and the process I verified this:
```
$ git clone git@github.com:python/cpython.git --depth 1

# Looked up the tag 3.12.9 on github, and it turned out to be fdb81425a9ad683f8c24bf5cbedc9b96baf00cd2

$ git fetch origin fdb81425a9ad683f8c24bf5cbedc9b96baf00cd2 --depth 1
$ git checkout fdb81425a9ad683f8c24bf5cbedc9b96baf00cd2

# Asked claude to how to figure my questions out, and after few conversation strokes and failure, it finally generated the script like this:
$ python str-internal.py
```

In the end, currently only the optimizations for not allocating separate memory for the same string literals are applied, but not for the concatenation of two different strings.
```python
s1 = "hello"              # p1
s2 = "world"              # p2
s3 = s1 + s2              # p3
s4 = "hello" + "world"    # p4
s5 = "helloworld"         # p4 (same as s4)
s6 = "hello" "world"      # p4 (same as s4)
```
