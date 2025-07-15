# Preparation
In order to run the examples, I needed symbolic link to extra
```
$ ln -s ../extra ./examples/extra
```

## EfficientNet
Successfully run the example with EfficientNet model.
```
python examples/efficientnet.py ./test/models/efficientnet/Chicken.jpg
8 8.098451 hen
did inference in 1739.75 ms
```

## Stable Diffusion
Failed to run the example with Stable Diffusion model.
Resulted image file was a pure black image
```
...
loaded weights in 27427.50 ms, 4.26 GB loaded at 0.16 GB/s
got CLIP context (1, 77, 768)
got unconditional CLIP context (1, 77, 768)
running for [1, 167, 333, 499, 665, 831, 997] timesteps
decode (1, 512, 64, 64)
decode (1, 512, 128, 128)
decode (1, 512, 256, 256)
decode (1, 256, 512, 512)
(512, 512, 3)
saving /var/folders/fw/3lldg3j90z76c7wb_rrfsx4w0000gn/T/rendered.png
Exception ignored in: <function Tensor.__del__ at 0x1059e45e0>
Traceback (most recent call last):
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 4385, in _wrapper
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 177, in __del__
TypeError: 'NoneType' object is not callable
Exception ignored in: <function Tensor.__del__ at 0x1059e45e0>
Traceback (most recent call last):
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 4360, in _wrapper
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 177, in __del__
TypeError: 'NoneType' object is not callable
...
Exception ignored in: <function Tensor.__del__ at 0x1059e45e0>
Traceback (most recent call last):
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 4360, in _wrapper
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 177, in __del__
TypeError: 'NoneType' object is not callable
Exception ignored in: <function Tensor.__del__ at 0x1059e45e0>
Traceback (most recent call last):
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 4360, in _wrapper
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 177, in __del__
TypeError: 'NoneType' object is not callable
```

## LLaMA
Failed to run the example with LLaMA model.
Maybe I need to download the model weights first.
> You will need to download and put the weights into the weights/LLaMA directory, which may need to be created.

```
$ python3 examples/llama.py
using GPU backend
using LLaMA-7B model
Traceback (most recent call last):
  File "/Users/shunusami/workspace/tiny/tinygrad/examples/llama.py", line 447, in <module>
    llama = LLaMa.build(MODEL_PATH, TOKENIZER_PATH, model_gen=args.gen, model_size=args.size, quantize=args.quantize, device=device)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/examples/llama.py", line 196, in build
    tokenizer = MODEL_PARAMS[model_gen]['tokenizer'](model_file=str(tokenizer_path))
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/.devenv/state/venv/lib/python3.12/site-packages/sentencepiece/__init__.py", line 468, in Init
    self.Load(model_file=model_file, model_proto=model_proto)
  File "/Users/shunusami/workspace/tiny/.devenv/state/venv/lib/python3.12/site-packages/sentencepiece/__init__.py", line 961, in Load
    return self.LoadFromFile(model_file)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/.devenv/state/venv/lib/python3.12/site-packages/sentencepiece/__init__.py", line 316, in LoadFromFile
    return _sentencepiece.SentencePieceProcessor_LoadFromFile(self, arg)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
OSError: Not found: "/Users/shunusami/workspace/tiny/tinygrad/weights/LLaMA/tokenizer.model": No such file or directory Error #2
```

## YOLOv8
Just looked at it, but lots of TODOs, so maybe it's not runnable?
https://github.com/tinygrad/tinygrad/blob/master/examples/yolov8.py

Actually, it was pretty fast and easy to run the example with YOLOv8 model.
```
$ python3 examples/yolov8.py ./test/models/efficientnet/Chicken.jpg
No variant given, so choosing 'n' as the default. Yolov8 has different variants, you can choose from ['n', 's', 'm', 'l', 'x']
running inference for YOLO version n
ram used:  0.01 GB, head.cv2.2.2.bias                                 : 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 355/355 [00:01<00:00, 228.16it/s]
loaded weights in 1556.44 ms, 0.01 GB loaded at 0.01 GB/s
did inference in 4164ms
Objects detected:
- bird: 3
saved detections at outputs_yolov8/Chicken_output.jpg
```

## Whisper
I don't have pyaudio and torchaudio installed, so I couldn't run the example.

## Conversation
I don't have espeak installed, so I couldn't run the example.
> Make sure you have espeak installed and PHONEMIZER_ESPEAK_LIBRARY set.
