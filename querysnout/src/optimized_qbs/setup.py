from distutils.core import Extension, setup

module = Extension("cqbs", sources=["optimqbs/qbs.c", "optimqbs/cqbsmodule.c"])

setup(
    name="optimqbs",
    version="1.0",
    description="C implementation of QBSes",
    ext_modules=[module],
    packages=["optimqbs"],
)
