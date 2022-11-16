## Using Brunoflow
To use Brunoflow immediately, clone the repo and then run `pip install -e .` in a Python3.9+ environment.. Ideally, do this in a fresh conda environment! Since this is an editable install, you should be able to edit files directly and then those changes should automatically update when you use the library.

### Example code
```
import brunoflow as bf
x = bf.Node(val=10, name="x")
output = x ** 2
output.backprop()
print(x.grad) # expected: 20.0
print(x.compute_entropy().val) # expected: 0.0
```

## Modifying Brunoflow
### Pre-commit
Run `pre-commit install` to get nice pre-commit hooks for linting/formatting.

### Details on how Brunoflow works
See the README inside of the `brunoflow` directory.
