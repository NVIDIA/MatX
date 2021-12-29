tensorShape_t
#################

The tensorShape_t is used to describe the rank and dimensions of data and view objects. It can be initialized with initializer-list syntax to improve clarity.
Note that ``tensorShape_t`` will be deprecated in a future release. Please use a container meeting the tensor descriptor interfaces, such as ``std::array``.

.. doxygenclass:: matx::tensorShape_t
    :members:
