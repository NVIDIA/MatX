FFT Convolution
###############

This example shows how to perform a convolution in the frequency domain using the convolution theorem:

.. math::
    h*x \leftrightarrow H \cdot X

The output of the FFT convolution is verified against MatX's built-in direct convolution to ensure the
results match.

The source code for this example can be found in examples/fft_conv.cu.

Background
----------
FFT convolution is generally preferred over direct convolution for sequences larger than a given size. This
size depends on the underlying hardware, but in general, a signal longer than a few thousand points will typically
be faster with an FFT convolution. However, FFT convolution requires setting up several intermediate buffers
that are not required for direct convolution. This is typically not a problem when the signal and filter size
are not changing since these can be allocated once on startup. 

Setup
-----
The first step for FFT convolution is allocating the intermediate buffers needed for each stage of the convolution. Assuming
the input signal and filter are in the time domain, the required buffers are:

1. Time domain input signal of length M (signalTimeData)
2. Time domain input filter of length L (filterTimeData)
3. Time domain output filtered signal of length M + L - 1 (timeOutData)
4. Frequency domain input signal of length M + L - 1 (signalFreqData)
5. Frequency domain input filter of length M + L - 1 (filterFreqData)

For this example the FFT size used is the same as the output signal length. Typically this may be rounded up to a 
power of two FFT for performance reasons, but this example is not attempting to optimize for performance::

    // Create time domain buffers
    tensor_t<complex, 1> sig_time({signal_size});
    tensor_t<complex, 1> filt_time({filter_size});
    tensor_t<complex, 1> time_out({filtered_size});

    // Frequency domain buffers
    tensor_t<complex, 1> sig_freq({filtered_size});
    tensor_t<complex, 1> filt_freq({filtered_size}); 

The next step initializes the data in both the signal and the filter to give reproducible results::

    // Fill the time domain signals with data
    for (index_t i = 0; i < signal_size; i++) {
        sig_time(i) = {-1*(2*(i%2)+1) * ((i%10) / 10.0f) + 0.1f, -1*((i%2) == 0) * ((i%10) / 5.0f) - 0.1f};
    }
    for (index_t i = 0; i < filter_size; i++) {
        filt_time(i) = {(float)i/filter_size, (float)-i/filter_size + 0.5f};
    }

Note that both ``sig_time`` and ``filt_time`` are views on the data objects using operator() to assign values on the host.

Once the data is initialized, we prefetch it onto the device to avoid the page fault when the kernel is launched::

    // Prefetch the data we just created
    sig_time.PrefetchDevice(0);
    filt_time.PrefetchDevice(0);

Since the expected output size of the full filtering operation is signal_len + filter_len - 1, both the filter and
signal time domain inputs are shorter than the output. This would normally require a separate stage of allocating buffers
of the appropriate size, zeroing them out, copying the time domain data to the buffers, and performing the FFT. However, MatX
has an API to do all of this automatically in the library using asynchronous allocations. This makes the call have a noticeable
performance hit on the first call, but subsequent calls will be close to the time without allocation.

To recognize that automatic padding is wanted, MatX uses the output tensor size compared to the input tensor size to determine
whether to pad the input with zeros. In this case the output signal (sig_time and filt_time) are shorter than the output tensors
(sig_freq and filt_freq), so it will automatically zero-pad the input.

.. _execution:

Execution
---------
With the setup complete, the actual convolution can be performed by an FFT, element-wise multiply, then IFFT::

    // Perform the FFT in-place on both signal and filter
    fft(sig_freq,  sig_freq);
    fft(filt_freq, filt_freq);

    // Perform the pointwise multiply. Overwrite signal buffer with result
    (sig_freq = sig_freq * filt_freq).run();

    // IFFT in-place
    ifft(sig_freq, sig_freq);

Note that for the multiply we're using the MatX multiply operator `*` that multiplies two tensors of equal size
and rank together, and stores the results in sig_freq. Because there is a 1:1 mapping of input and output elements, 
there are no race conditions when writing into the same buffer.

Verification
------------
The last step is verifying the results against a direct convolution. The results will likely not be identical when comparing
directly using floating point numbers because of the different operations and ordering involved with FFT versus direct
convolution. Instead, we compare the values with a small enough epsilon (0.001) for the magnitude of our output data::

    // Now the sig_freq view contains the full convolution result. Verify against a direct convolution
    matxDirectConv1D(time_out, sig_time, filt_time, matxConvCorrMode_t::MATX_C_MODE_FULL, 0);

    cudaStreamSynchronize(0);

    // Compare signals
    for (index_t i = 0; i < filtered_size; i++) {
        if (  fabs(time_out(i).real() - sig_freq(i).real()) > 0.001 || 
            fabs(time_out(i).imag() - sig_freq(i).imag()) > 0.001) {
            printf("Verification failed at item %lld. Direct=%f%+.2fj, FFT=%f%+.2fj\n", i,
            time_out(i).real(), time_out(i).imag(), sig_freq(i).real(), sig_freq(i).imag());
            return -1;
        }
    }

Conclusion
----------
While this example showed a single FFT convolution, filtering is commonly performed on a streaming set of data. By preparing
all of the buffers ahead of time, only the :ref:`execution` part of this example would need to be performed in the data path. This
allows for highly-optimized code using the simple syntax above.