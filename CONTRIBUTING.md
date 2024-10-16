# Contributing

If you're interested in contributing to `palettecleanser`, please feel free to reach
out [to me](mailto:jiaming.justin.chen@gmail.com)! I'm more than willing and available to discuss any ideas and answer any questions you may have. There's no minimum level of contribution or skill required; we all start off somewhere.

### Style
`palettecleanser` utilizes [ruff](https://github.com/astral-sh/ruff) for general formatting and [pep8](https://www.python.org/dev/peps/pep-0008/) for overall style. Simply running the `Format document`, `Format imports` functions in [ruff](https://github.com/astral-sh/ruff) should work.

For any images saved in `palettecleanser`, please run the `compress_image_inplace` function prior to the pull request in order to minimize the size of the overall package. NOTE that this function will modify implace, so please make sure you have backups of your original if needed.

#### Full resolution
![full_res](images/pink_roses_full_res.jpg)
#### Compressed via. `Compress_image_inplace`
![compressed](images/pink_roses.jpg)


### Tests
Each new contribution should have corresponding test coverage. I'd like to retain 100% coverage on ```palettecleanser```. `pytest` is the current testing package.

### Commit messages
Please follow the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/#summary) standard for commit messages. Quoting from their website, the basic structure is:
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```
More details on the website. This is something I'm actively working towards as well but it should pay off in a cleaning and clearer commit tree.

### General steps - from [Dan Bader](https://github.com/dbader/readme-template)
  1. Fork ```palettecleanser``` (https://github.com/yourname/palettecleanser/fork)
  2. Create your feature branch (```git checkout -b feature/fooBar```)
  3. Commit your changes (```git commit -am 'Add some fooBar'```)
  4. Push to the branch (```git push origin feature/fooBar```)
  5. Create a new Pull Request
