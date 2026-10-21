.. _windowing:

Windowing
#########

Window functions are generators: they compute each sample when it is used, so applying a window to data fuses into the
same kernel. The ``window_functions`` example (``examples/window_functions.cu``) computes the frequency response of every
window below and reports its peak sidelobe level, 3 dB mainlobe width, and equivalent noise bandwidth. When MatX is built
with ``MATX_EN_VISUALIZATION=ON``, it also plots the responses.

Because a window is recomputed in every kernel that uses it, an expensive window applied to a large tensor can be cheaper
to compute once into a tensor and then apply, e.g. ``(w = taylor<0>({N})).run(exec);`` followed by
``(out = in * w).run(exec);``.


.. toctree::
   :maxdepth: 1
   :glob:

   *