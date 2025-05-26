# Installing Python files
python_sources = [
    '__init__.py',
    'constants.py',
    'main.py',
    'utils.py',
]
py.install_sources(python_sources, subdir: 'twopoppy2')

# Compiling Fortran module
source_files = ['fortran.f90']                                # Source files to be compiled
fortran_source = custom_target(
  '_fortranmodule.c',                                          # Target files to be created
  input : source_files,                                       # Source files
  output : ['_fortranmodule.c', '_fortran-f2pywrappers.f'],  # Output files
  command : [                                                 # Compile command to be executed
    py, '-m', 'numpy.f2py',                                   # Running f2py
    '-m', '_fortran',                                          # Module name
    '--lower',                                                # Convert Fortran to lower case
    '--build-dir', 'twopoppy2',                               # Build files in directory
    '@INPUT@'                                                 # Files to be compiled
  ]
)
# Building Python extension for Fortran module
py.extension_module(
  '_fortran',                                                  # Module name
  [source_files, fortran_source],                             # Source files
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,                                # Directories to include
  dependencies : [py_dep],                                    # Dependencies to be used
  install : true,                                             # Install extension
  subdir : 'twopoppy2',                     # Directory to install extension in
)