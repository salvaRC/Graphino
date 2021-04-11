# Environment setup
- Git clone the repo 
- To get started with the code, please create a new conda environment as follows :
        
        conda env create -f env_gpu.yml 
        conda activate graphino  # graphino is the name of the environment    
    
- This will install all required dependencies, uses Python 3.7, and expects CUDA to be available. If no GPU is available, please 
    remove line 7 (*cudatoolkit*) from the [environment file](env_gpu.yml), and proceed as above.

- This should be enough for Pycharm, for command line stuff you may need to then also run ``python setup.py install``.

## Alternatives/If you encounter an error
- If you encounter, e.g., a cryptic ``InvalidArchiveError``, please try instead with 

    
    conda create -n graphino python=3.7
    conda activate graphino
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch  # ***
    pip install -r requirements.txt
    

*** look in  https://pytorch.org/ for the exact command (which depends on your CUDA version).

- If this does not work either, do not hesitate in contacting us (although it should work).
