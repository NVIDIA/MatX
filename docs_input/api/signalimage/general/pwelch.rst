.. _pwelch_func:

pwelch
======

Estimate the power spectral density of a signal using Welch's method [1]_

.. doxygenfunction:: pwelch(const xType& x, const wType& w, index_t nperseg, index_t noverlap, index_t nfft, PwelchOutputScaleMode output_scale_mode, fsType fs)
.. doxygenfunction:: pwelch(const xType& x, index_t nperseg, index_t noverlap, index_t nfft, PwelchOutputScaleMode output_scale_mode, fsType fs)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/PWelch.cu
   :language: cpp
   :start-after: example-begin pwelch-test-1
   :end-before: example-end pwelch-test-1
   :dedent:

References
~~~~~~~~~~

  .. [1] \ P. Welch, "The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms," in IEEE Transactions on Audio and Electroacoustics, vol. 15, no. 2, pp. 70-73, June 1967, doi: 10.1109/TAU.1967.1161901.