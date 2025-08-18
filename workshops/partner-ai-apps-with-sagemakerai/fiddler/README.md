# Fiddler <> SageMaker Demo 2024

For workshop admins, preconfigure a collection of Fiddler applications using the `AdminFiddlerSagemakerDemo.ipynb` notebook.

For workshop users, get your SageMaker App User Profile from your workshop administrator then follow along the `FiddlerSagemakerDemo.ipynb` notebook.

## Running the Notebook Locally

Run the following steps from a machine with access to a web browser.

1.  Clone this repository and `cd` into it.

    ```shell
    git clone git@github.com:fiddler-labs/fiddler-demo-dec-2024.git
    cd fiddler-demo-dec-2024
    ```

1.  Install the version of Python that will be used in the workshop.

    MacOS:

    ```shell
    brew install python@3.12
    ```

    Linux:

    ```shell
    sudo apt install python3.12
    ```

1.  Create a Python virtual environment and activate it.

    ```shell
    eval $(which python3.12) -m venv .venv
    source .venv/bin/activate
    ```

1.  Install JupyterLab and run it to open your browser to Jupyter Notebook.

    ```shell
    python -m pip install jupyterlab
    jupyter lab
    ```

1.  Select the `FiddlerSagemakerDemo.ipynb` Notebook from the sidebar and follow along!
