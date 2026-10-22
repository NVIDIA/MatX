.. _windowing:

Windowing
#########

Window functions are generators: they compute each sample when it is used, so applying a window to data fuses into the
same kernel. The ``window_functions`` example (``examples/window_functions.cu``) computes the frequency response of every
window below and reports its peak sidelobe level, 3 dB mainlobe width, and equivalent noise bandwidth. When MatX is built
with ``MATX_EN_VISUALIZATION=ON``, it also plots the responses.

Some windows are relatively expensive to compute per element, and a window generator repeats that computation for every
element it is applied to. If the computation is a bottleneck, materialize the window into a tensor by assigning the
generator to it, then apply the tensor instead, which replaces the computation with memory accesses. This is more likely
to help when the same window is applied many times, such as to every row of a batch or in repeated runs, because the
window is then computed only once. Whether it is faster depends on the window, data type, data size, and GPU, so
benchmark both forms.


.. toctree::
   :maxdepth: 1
   :glob:

   *