Installation
============

The latest release of the package can be installed from github with:

.. code-block:: sh

    uv add "deepgenocompress @ git+https://github.com/ut-biomet/DeepCGP.git" --tag v0.1.0

or

.. code-block:: sh

    pip install "deepgenocompress @ git+https://github.com/ut-biomet/DeepCGP.git@v0.1.0"

To make sure you can actually use it this package, you need also need a
`keras backend <https://keras.io/getting_started/#configuring-your-backend>`_. This can be
installed separately or with as optional dependency:

.. code-block:: sh

    # use one of the below
    uv add "deepgenocompress[tensorflow] @ git+https://github.com/ut-biomet/DeepCGP.git" --tag v0.1.0
    uv add "deepgenocompress[torch] @ git+https://github.com/ut-biomet/DeepCGP.git" --tag v0.1.0
    uv add "deepgenocompress[jax] @ git+https://github.com/ut-biomet/DeepCGP.git" --tag v0.1.0

Similarly with ``pip``:

.. code-block:: sh

    pip install "deepgenocompress[tensorflow] @ git+https://github.com/ut-biomet/DeepCGP.git@v0.1.0"

.. note::

    If the backend is different than ``tensorflow`` you may need to explicitly specify it, cf.
    https://keras.io/getting_started/#configuring-your-backend
