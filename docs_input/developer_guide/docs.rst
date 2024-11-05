.. _devdocs:

Creating Documentation
######################

MatX documentation is based on the reStructuredText format, with Doxygen, Sphinx, and Breathe for parsing and display. This guide 
is for adding API functions to MatX and doesn't apply to other types of documentation.

Every API function should have its own .rst file placed in the appropriate path. Starting at the top level of docs_input/api, 
functions are categorized into more specific types inside of each directory. The current directory structure is:

.. code-block:: default
  
  ├── casting
  ├── creation
  │   ├── operators
  │   └── tensors
  ├── dft
  │   ├── dct
  │   ├── fft
  │   └── utility
  ├── io
  ├── linalg
  │   ├── decomp
  │   ├── eigenvalues
  │   ├── matvec
  │   └── other
  ├── logic
  │   ├── bitwise
  │   ├── comparison
  │   ├── logical
  │   └── truth
  ├── manipulation
  │   ├── basic
  │   ├── joinrepeat
  │   ├── rearranging
  │   └── selecting
  ├── math
  │   ├── arithmetic
  │   ├── complex
  │   ├── explog
  │   ├── extrema
  │   ├── misc
  │   ├── round
  │   ├── sumprod
  │   └── trig
  ├── polynomials
  ├── random
  ├── searchsort
  ├── signalimage
  │   ├── convolution
  │   ├── filtering
  │   ├── general
  │   └── radar
  ├── stats
  │   ├── avgvar
  │   ├── corr
  │   ├── hist
  │   └── misc
  ├── synchronization
  ├── viz
  └── window

Most new functions should fit into one of the categories above, if not, it can be discussed through a GitHub issues. MatX roughly follows the 
tree of NumPy, and using it as a starting point for categorizing functions can be helpful.

Once you find where your rst file will go, create the file with the same name as the function to add. Copying an existing rst file is a good 
starting point for editing.

Once inside the rst you will start by adding a label at the top for referencing, and the name of the function below with `=` as underlining. All
function names should be lower case. If not, the function should be changed.

.. code-block:: default

  .. _max_func:

  max
  ===

Below the header is where you describe the function and what it does. This is also where any caveats, limitations, etc, can be listed. Try to be 
as verbose as possible. More information is almost always better for libraries like this since this may be the only reference a user can find for 
this function.

Below the description is where the function prototypes are listed:

.. code-block:: default

  .. doxygenfunction:: max(const InType &in, const int (&dims)[D])
  .. doxygenfunction:: max(const InType &in)
  .. doxygenfunction:: max(Op t, Op t2)  

These use the Sphinx `doxygenfunction::` directive to reference the functions in the Doxygen output. Doxygen, Sphinx, and Breathe frequently have 
problems parsing complex C++ syntax. If it's not able to find the functions and they appear correct, try adding or removing qualifiers on the function 
names. If the parsing issues don't go away, the Sphinx C++ parser can be used as an alternative to Breathe:

.. code-block:: default

  .. cpp:function:: max(const InType &in, const int (&dims)[D])
  .. cpp:function:: max(const InType &in)
  .. cpp:function:: max(Op t, Op t2)

Make sure to list *all* function overloads here so the user can find all variants.

Testing-As-Documentation
########################

MatX uses a code/test-as-documentation methodology. All functions should have one or more unit tests, and their documentation should reference those 
tests. This ensures that the documentation is always working, and also allows the docs to change indirectly by modifying tests.

Inside the unit test find an example of the function being called, and wrap as much of the example as needed with syntax similar to the following:

.. code-block:: cpp

  // example-begin max-test-1
  auto t0 = make_tensor<TestType>({});
  auto t1o = make_tensor<TestType>({11});

  t1o.SetVals({(T)1, (T)3, (T)8, (T)2, (T)9, (T)10, (T)6, (T)7, (T)4, (T)5, (T)11});

  // Reduce all inputs in "t1o" into "t0" by the maximum of all elements
  (t0 = max(t1o)).run(exec);
  // example-end max-test-1

The comment *must* begin with `example-begin` and end with `example-end`. The name after that should follow the function name and the test number. Only 
include as much of the example as needed to be sufficient to understand how it works. Multiple examples/tests may be included listed sequentially.


Documentation is built/tested on every commit/CI run. If the documentation fails to build, CI will fail and the PR will not be accepted. Once the PR 
is merged, the documentation will automatically be pushed to the main documentation site.