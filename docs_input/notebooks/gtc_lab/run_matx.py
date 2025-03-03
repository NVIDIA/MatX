from IPython.core.magic import register_cell_magic
import subprocess
import os

def load_ipython_extension(ipython):
    # Register any magic commands or perform setup here
    from IPython.core.magic import register_cell_magic

    @register_cell_magic
    def run_matx(line, cell):
        filename = '/tmp/user_code.cu'
        with open(filename, 'w') as f:
            f.write(cell)

        exec(open('./exec_code.py').read())

    ipython.register_magic_function(run_matx, 'cell')