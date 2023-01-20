import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path
this_file_loc = Path(__file__).parent

def test_basic_example_runs():

    location_path = (this_file_loc.parent / 'examples').absolute()
    notebook_loc = location_path / 'basic_example.ipynb'

    with open(notebook_loc) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': location_path}})
    with open('executed_notebook.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def test_new_loader_runs():

    location_path = (this_file_loc.parent / 'examples').absolute()
    notebook_loc = location_path / 'new_data_loader.ipynb'

    with open(notebook_loc) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': location_path}})
    with open('executed_notebook.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def test_new_exporter_runs():

    location_path = (this_file_loc.parent / 'examples').absolute()
    notebook_loc = location_path / 'new_data_exporter.ipynb'

    with open(notebook_loc) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': location_path}})
    with open('executed_notebook.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)