Large-Scale Experiments
=======================

To assess the quality of an attack or Strong PUF design, it is often necessary to run several attacks for different
parameter settings and many instances. Often, attacks will run long or require much memory.

In pypuf, `studies` can be defined that run a predefined `experiment` for a parameter matrix. A CLI is provided that
can run single, subsets or all experiments.

An example study could look like this:

.. code-block:: python

    import sys
    from datetime import timedelta
    from typing import List

    from pypuf.batch import StudyBase


    class ExampleStudy(StudyBase):

        @staticmethod
        def parameter_matrix() -> List[dict]:
            N = {
                (64, 1): 4000,
                (64, 2): 8000,
                (64, 4): 30000,
                (128, 1): 10000,
                (128, 2): 20000,
                (128, 4): 60000,
            }

            return [
                {'n': n, 'k': k, 'N': N[n, k]}
                for n in [64, 128]
                for k in [1, 2, 4]
            ]

        def primary_results(self, results: dict) -> dict:
            return {
                'accuracy': results['accuracy'],
                'duration': results['duration'],
            }

        def run(self, n: int, k: int, N: int) -> dict:
            # Do some sort of experiment to determine so sort of result

            # If this runs long, maybe it's useful to store some data in self.log,
            # and store to disk with self.save_log()
            self.log = []
            self.log.append({
                'accuracy': 1.0  # just any information about the current state of affairs
            })  # can be any pickle-able data type!
            self._save_log()  # calls to save_log a throttled automatically, unless 'force' is used.

            # The result is represented by a dict
            return {
                'accuracy': 1.0,  # VERY GOOD experiment done here ...
                'duration': timedelta(seconds=3),  # .. and fast
                'additonal_info': 'foobar!',
                'is_example': True,
            }


    if __name__ == '__main__':
        ExampleStudy.cli(sys.argv)


Such a study can be run with ``python3 -m example_study debug 0 1``.
The parameters are defined as follows. In ``python3 -m <module> <results> <idx> <total>``,

#. ``<module>`` sets the name of the module to be run, i.e. the module where ``ExampleStudy.cli`` is called;
#. ``<results>`` sets the name of the result file in the current directory, in which all return values of the `run` method are stored,
#. ``<idx>`` is the zero-based index of the subset of parameters that shall be run,
#. ``<total>`` is the total number of subsets (blocks).

This interface nicely interact with SLURM, where a job file like this can be used to distribute pypuf studies
across a SLURM cluster:

.. code-block:: sh

    #!/bin/bash

    # 1. Adjust
    #    - email address;
    #    - job name;
    #    - time, memory, CPUs;
    #    - study length.
    # 2. Make sure to load the correct module
    # 3. Setup the virtual environment and cd and source correctly
    #
    # Then run with
    #  sbatch --array=0..39 this_file.sh  # adjust for study length
    #

    #SBATCH --mail-type=END
    #SBATCH --mail-user=<email>
    #SBATCH --job-name <jobname>
    #SBATCH --time=2-00:00:00
    #SBATCH --mem=8G
    #SBATCH --cpus-per-task=2
    #SBATCH --nodes=1

    # Usually High Performance Clusters will need to load Python before it is available
    module load python/3.7.6_tf_2.1.0

    # Navigate to your study file, if necessary
    cd ~/my_study/

    # Load your virtual environment where pypuf is installed
    source venv/bin/activate

    # Limit the number of CPUs used by numpy and tensorflow
    export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}
    export TF_NUM_INTEROP_THREADS=${SLURM_CPUS_PER_TASK}
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
    export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
    export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

    # run the study
    python3 -m example_study "${SLURM_JOB_NAME}" "${SLURM_ARRAY_TASK_ID}" 40
