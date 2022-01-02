# -*- coding: utf-8 -*-

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy
import os

def find_in_path(name, path):
    """Find a file in a search path
    """
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """
    if "CUDAHOME" in os.environ:
        home = os.environ["CUDAHOME"]
        nvcc = os.path.join(home, "bin", "nvcc")
    else:
        nvcc = find_in_path("nvcc", os.environ["PATH"])
        if nvcc is None:
            print("The nvcc binary could not be "
                "located in your $PATH. Either add it to your path, "
                "or set $CUDAHOME")
            return {}
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {"home": home, "nvcc": nvcc,
                  "include": os.path.join(home, "include"),
                  "lib64": os.path.join(home, "lib64")}
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            print("The CUDA %s path could not be "
                                   "located in %s" % (k, v))
            return {}

    return cudaconfig

def user_input_cuda():
    """Get CUDA env from user input if not automatically found.
    """
    cudaconfig = {"home": "",
                  "nvcc": "",
                  "include": "",
                  "lib64": ""}
    print("CUDA not found in default paths.")
    print("Please input the following paths (leave them blank if you do not want to compile pycb's GPU code).")
    cudaconfig["nvcc"] = input("Insert nvcc path: ")
    cudaconfig["include"] = input("Insert path to CUDA includes: ")
    cudaconfig["lib64"] = input("Insert path to CUDA libraries: ")
    
    return cudaconfig

def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a weird functional
    subclassing going on.
    """
    # Tell the compiler it can processes .cu
    self.src_extensions.append(".cu")

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == ".cu":
            # use the cuda for .cu files
            self.set_executable("compiler_so", CUDA["nvcc"])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs["nvcc"]
        else:
            postargs = extra_postargs["gcc"]

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile
    
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

# get CUDA env
CUDA = locate_cuda()
if len(CUDA) == 0:
    CUDA = user_input_cuda()

# paths and libraries
cuda_src_dir = "cuda"
libraries_cuda = ["cuda", "cudart", "curand"]

# extensions
extensions = []
check_cuda_paths = [CUDA[k] == "" for k in CUDA.keys()]
if not all(check_cuda_paths):
    extensions.append(
        # cb_cuda
        Extension("cuda_fathon.cuda_fathon",
                  ["cython/cuda_fathon.pyx",
                   os.path.join(cuda_src_dir, "dfa_kernel.cu"),
                   os.path.join(cuda_src_dir, "mfdfa_kernel.cu"),
                   os.path.join(cuda_src_dir, "dcca_kernel.cu"),
                   os.path.join(cuda_src_dir, "ht_kernel.cu")],
                  library_dirs=[CUDA["lib64"]],
                  libraries=libraries_cuda,
                  language="c++",
                  runtime_library_dirs=[CUDA["lib64"]],
                  include_dirs=[numpy.get_include(), CUDA["include"]],
                  extra_compile_args={
                      "gcc": [],
                      "nvcc": ["-c", "--compiler-options", "-fPIC", "--compiler-options", "-Ofast"]
                      }
                  )
        )

for e in extensions:
    e.cython_directives = {"language_level": "3"}

setup(
  name="cufathon",
  version="0.1",
  author="Stefano Bianchi",
  author_email="stefanobianchi9@gmail.com",
  url="https://github.com/stfbnc/cufathon.git",
  ext_modules=extensions,
  cmdclass={"build_ext": custom_build_ext},
)
