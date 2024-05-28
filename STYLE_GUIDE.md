#### Python
Generative Recommenders use [Yapf](https://github.com/google/yapf) to format our code.
The continuous integration check will fail if you do not use it.

Install them with:
```
pip install yapf
```

Be sure to run it before you push your commits, otherwise the CI will fail!

```
yapf --style=.style.yapf -ir ./**/*.py
```
