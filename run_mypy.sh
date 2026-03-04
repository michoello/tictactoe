echo Running mypy
mypy . --explicit-package-bases --exclude 2015 --check-untyped-defs --disallow-untyped-defs --disallow-incomplete-defs
