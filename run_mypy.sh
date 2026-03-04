echo Running mypy
mypy . --exclude 2015 --check-untyped-defs --disallow-untyped-defs --disallow-incomplete-defs
