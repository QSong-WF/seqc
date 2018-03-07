from typing import List
from jinja2 import Environment, FileSystemLoader
import os
import pandas as pd
import nbformat
import tempfile
from nbconvert.preprocessors import ExecutePreprocessor


class Notebook:

    def __init__(self, output_directory, *samples):
        self._output_directory: str = output_directory
        self._samples: List = samples
        self._this_dir = os.path.dirname(os.path.abspath(__file__))

    def merge_data(self, merged_sample_name='merged_counts.csv', remove_unmerged=False):
        """
        This function will merge any datasets provided as nested lists.
        Each top-level value is considered an input alias.
        Any second-level list is considered to be a group of files to be joined

        :param bool remove_unmerged: if True, this function will delete the unmerged files after
          completion
        :param str merged_sample_name: name of merged csv file
        :return None: The list of merged file names will replace the list passed to the class in
          self._datasets
        """
        # merge datasets
        dfs = [pd.read_csv(csv, index_col=0) for csv in self._samples]
        df = pd.concat(
            dfs,
            keys=list(range(len(self._samples))),
            names=['sample_number', 'cell_id']
        )

        # write merged datafiles using stem of the first file and underscore joined names
        merged_csv = '{dir}/{files}'.format(
            dir=self._output_directory,
            files=merged_sample_name,
        )

        df.to_csv(merged_csv)

        # delete original files, if requested
        if remove_unmerged:
            for csv in self._samples:
                os.remove(csv)

        # update file urns
        self._samples = merged_csv

    def write_template(self, notebook_filename: str):
        """write a filled ipython notebook to disk

        :param str notebook_filename:
        :return:
        """
        j2_env = Environment(loader=FileSystemLoader(self._this_dir), trim_blocks=True)
        rendered = j2_env.get_template('analysis_template.json').render(
            directory=self._output_directory,
            sample=self._samples,
        )
        with open(notebook_filename, 'w') as fdw:
            fdw.write(rendered)

    @classmethod
    def run_notebook(cls, notebook_filename):

        dir_ = tempfile.mkdtemp()
        with open(notebook_filename) as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': dir_}})

        with open(notebook_filename, 'wt') as f:
            nbformat.write(nb, f)
