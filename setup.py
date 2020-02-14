"""
Setup file for package `twopoppy2`.
"""
import setuptools # noqa
from numpy.distutils.core import Extension
import pathlib
import warnings

PACKAGENAME = 'twopoppy2'

# the directory where this setup.py resides
HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":
    from numpy.distutils.core import setup

    extensions = [
        Extension(name='twopoppy2.fortran', sources=['twopoppy2/fortran.f90']),
        ]

    def setup_function(extensions):
        setup(
            name=PACKAGENAME,
            description='new implementation of two-population dust evolution model according to Birnstiel, Klahr, Ercolano, A&A (2012)',
            version='1.1.4',
            long_description=(HERE / "README.md").read_text(),
            long_description_content_type='text/markdown',
            url='https://github.com/birnstiel/twopoppy2',
            author='Til Birnstiel',
            author_email='til.birnstiel@lmu.de',
            license='GPLv3',
            packages=[PACKAGENAME],
            package_dir={PACKAGENAME: PACKAGENAME},
            package_data={PACKAGENAME: [
                'fortran.f90',
                ]},
            include_package_data=True,
            install_requires=['scipy', 'numpy', 'matplotlib', 'astropy'],
            zip_safe=False,
            ext_modules=extensions
            )
    try:
        setup_function(extensions)
    except BaseException:
        warnings.warn('could not compile the fortran routines! code will not work as no python solver implemented yet')
        setup_function([])
