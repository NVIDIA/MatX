from IPython.core.magic import register_cell_magic
import subprocess
import os
import threading

# Simple progress bar from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def run_command(cmd_list):
    process = subprocess.Popen(
        cmd_list,  # directly pass the command as a list
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
        universal_newlines=True
    )

    cur_line = 0

    def handle_stderr_line(line):
        nonlocal cur_line
        sp = line.split()
        if len(sp) == 0:
            return

        if sp[0] != '#$':
            print(line)
            return
        
        start_tokens = ('gcc', 'cudafe++', 'gcc', '"$CICC_PATH/cicc"', 'ptxas', 'fatbinary', 'rm', 'gcc', 'nvlink', 'fatbinary', 'rm', 'gcc', 'g++')
        if sp[1] != start_tokens[cur_line]:
            return

        printProgressBar(cur_line+1, len(start_tokens), prefix = 'Compiling...')
        cur_line += 1

    def read_stderr(pipe):
        for line in pipe:
            handle_stderr_line(line)


    def read_stdout(pipe):
        for line in pipe:
            print(line, end='')  # Print in real-time

    stderr_thread = threading.Thread(target=read_stderr, args=(process.stderr,))
    stdout_thread = threading.Thread(target=read_stdout, args=(process.stdout,))
    
    stderr_thread.start()
    stdout_thread.start()

    # Wait for process to complete
    process.wait()
    
    # Wait for threads to finish
    stderr_thread.join()
    stdout_thread.join()

    cur_line = 0  
    if process.returncode != 0:
        print("\nCompilation failed!")
        return False
 
    return True


def load_ipython_extension(ipython):
    # Register any magic commands or perform setup here
    from IPython.core.magic import register_cell_magic

    @register_cell_magic
    def run_matx(line, cell):
        output_code = f"""
        #include <matx.h>

        int main() {{
            {cell}
        }}
        """

        with open("/tmp/output.cu", "w") as f:
            f.write(output_code)
        with open("/tmp/output.cpp", "w") as f:
            f.write(output_code)    

        current_dir = os.getcwd()

        MATX_ROOT = '/MatX'

        gcc_cmd = f'g++ -std=c++17 -DMATX_DISABLE_CUB_CACHE -DMATX_ENABLE_FILEIO -DMATX_ENABLE_PYBIND11 -DMATX_EN_OMP -DMATX_EN_X86_FFTW -DMATX_NVTX_FLAGS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_DISABLE_ABI_NAMESPACE '\
                f'-I"{MATX_ROOT}/build/_deps/cccl-src/libcudacxx/lib/cmake/libcudacxx/../../../include" -I/usr/local/cuda/include -I{MATX_ROOT}/include -I{MATX_ROOT}/include/matx/kernels -I"{MATX_ROOT}/build/_deps/cccl-src/thrust/thrust/cmake/../.." -I"{MATX_ROOT}/build/_deps/cccl-src/cub/cub/cmake/../.." '\
                f'-isystem "{MATX_ROOT}/build/_deps/pybind11-src/include" -isystem /usr/include/python3.10 -isystem /usr/local/cuda/include'\
                f'-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_IGNORE_ABI_NAMESPACE_ERROR -fopenmp -DMATX_ROOT=\"\" -fvisibility=hidden -lcuda -lcufft -lcublas -lpybind11 -o /tmp/output /tmp/output.cpp'

        nvcc_cmd = f'nvcc -v -forward-unknown-to-host-compiler -Ofc -std=c++17 -DMATX_DISABLE_CUB_CACHE -DMATX_ENABLE_FILEIO -DMATX_ENABLE_PYBIND11 -DMATX_EN_OMP -DMATX_EN_X86_FFTW -DMATX_NVTX_FLAGS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_DISABLE_ABI_NAMESPACE --generate-code=arch=compute_80,code=[sm_80] '\
                    f'-I{MATX_ROOT}/include -I{MATX_ROOT}/include/matx/kernels -I"{MATX_ROOT}/build/_deps/cccl-src/lib/cmake/thrust/../../../thrust" -I"{MATX_ROOT}/build/_deps/cccl-src/lib/cmake/libcudacxx/../../../libcudacxx/include" -I"{MATX_ROOT}/build/_deps/cccl-src/lib/cmake/cub/../../../cub" -isystem "{MATX_ROOT}/build/_deps/pybind11-src/include" -isystem /usr/include/python3.10 -isystem "/usr/include/x86_64-linux-gnu/openblas64-openmp" -isystem "{MATX_ROOT}/build/_deps/cutensor-src/include" -isystem "{MATX_ROOT}/build/_deps/cutensornet-src/include" -isystem "{MATX_ROOT}/build/_deps/cudss-src/include" -isystem /usr/local/cuda/include'\
                    f'-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_IGNORE_ABI_NAMESPACE_ERROR -fopenmp -DMATX_ROOT=\"\" -fvisibility=hidden -lcuda -lcufft -lcublas -o /tmp/output /tmp/output.cu'

        if run_command(nvcc_cmd.split()):
            run_process = subprocess.run([f"/tmp/output"], capture_output=True, text=True)

            print(run_process.stdout)    

    ipython.register_magic_function(run_matx, 'cell')