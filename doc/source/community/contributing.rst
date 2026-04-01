Contributing
============

First of all, thank you for considering contributing to *TanaT*.
It is still an experimental toolkit, but it has received a warm welcome from various communities
interested in its functionalities.


Contributions are managed through GitHub Issues and Pull Requests.

We welcome contributions in the following forms:

- **Bug reports**: when filing an issue to report a bug, please use the search tool to ensure the bug hasn't been reported yet.
- **New feature suggestions**: if you think *TanaT* should include a new algorithm, please open an issue to ask for it (always check that the feature has not been asked for yet). Think about linking to a PDF version of the paper that first proposed the method when suggesting a new algorithm.
- **Bug fixes and new feature implementations**: if you feel you can fix a reported bug or implement a suggested feature yourself, do not hesitate to:

  1. fork the project;
  2. implement your bug fix;
  3. submit a pull request referencing the ID of the issue in which the bug was reported / the feature was suggested.

If you would like to contribute by implementing a new feature reported in the Issues, starting with
`Issues labelled "good first issue" <https://github.com/TanaT-Lab/TanaT/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`_
is a good idea.

When submitting code, please think about code quality and add proper docstrings with high code coverage.

More details on Pull requests
------------------------------

The preferred workflow for contributing to *TanaT* is to fork the
`main repository <https://github.com/TanaT-Lab/TanaT>`_ on GitHub, clone, and develop on a branch.

Steps:

1. Fork the `project repository <https://github.com/TanaT-Lab/TanaT>`_
   by clicking on the **Fork** button near the top right of the page. This creates
   a copy of the code under your GitHub user account. For more details on
   how to fork a repository see `this guide <https://help.github.com/articles/fork-a-repo/>`_.

2. Clone your fork of the *TanaT* repo to your local disk:

   .. code-block:: bash

      git clone git@github.com:YourLogin/TanaT.git
      cd TanaT

3. Create a ``my-feature`` branch to hold your development changes.
   Always use a feature branch; never work directly on ``main``:

   .. code-block:: bash

      git checkout -b my-feature

4. Develop the feature on your branch. Record your changes using ``git add`` and ``git commit``:

   .. code-block:: bash

      git add modified_files
      git commit

5. Push the changes:

   .. code-block:: bash

      git push -u origin my-feature

6. Follow `these instructions <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`_
   to create a pull request from your fork. This will notify the maintainers.

(If any of the above seems unfamiliar, please look up the
`Git documentation <https://git-scm.com/documentation>`_ on the web, or ask another contributor for help.)
