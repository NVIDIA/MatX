.. _vector:

Vectorization and ILP
#####################

On both GPUs and CPUs vector instructions are important for achieving the highest performance possible. Vectorization allows
a single thread to perform multiple operations on multiple data points at once, allowing the hardware to achieve higher throughput.
This is especially true on newer generations of hardware where achieving peak memory bandwidth requires issuing vector instructions
that load or store multiple values at once.

ILP (Instruction Level Parallelism) is related to vectorization in that it allows a single thread to perform multiple operations, but
possibly in a serial manner. For example, a vectorized operation may load 16 values from memory into a single thread, and that thread
can launch 16 operations either in sequence or in parallel if the ISA allows it. This potentially allows for better hardware utilization
by hiding latency by keeping the execution pipelines busy.

Vectorization and ILP in MatX
-----------------------------

MatX performs vectorization and ILP for the user automatically when possible. This is done by analyzing the expression tree of the
user's code and looking for opportunities to vectorize and ILP at both runtime and compile time. As mentioned above, one candidate
for vectorization is loads and stores from memory. Modern NVIDIA GPUs can load 16 bytes in a single instruction, and some CPUs 
can load up to 64 bytes in a single instruction. MatX can analyze the tensors used in an expression for alignment and other properties
to determine if there are opportunities to issue vector loads and stores, depending on the executor. The loads and stores are not 
restricted to only the largest sizes, and may issue various width to meet the requirements of the expression.

Currently MatX couples vectorization with ILP. That is, ILP is dictated by the vectorization width. For example, if a vectorized
operation has a width of 4 floats (16 bytes), then the ILP will be 4. In the future this may change where a larger ILP may be used 
than the vectorization width if it is beneficial. This is common with block-based algorithms that reduce the need to synchronize 
across blocks by processing more data per block. In general, it's better to issue larger loads if possible, so as long as the ILP
is equal to or larger than the vectorization width, the load/store width will be the largest vectorizable width.

Limitations
-----------

Currently vectorization and ILP is not enabled on all operators. Specifically operators that rearrange the order of elements may 
have it disabled since it's not possible to use ILP with those. For example, an ``fftshift()`` operator rearranges the input 
elements, such that one thread needs values from a different thread and cannot be vectorized.

