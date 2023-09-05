
<div align="center">
<h1>
<b>
R-VOS Project 
</b>
</h1>
</div>

## Environment setup steps
1. Clone our repo. **https://github.com/Jack24658735/R-VOS**
2. Install conda env. with `conda env create --name *env_name* python=3.8`
3. Setup environment variables for CUDA_HOME (Change to your CUDA version)
    ```bash
    export CUDA_HOME=/usr/local/cuda-11.4
    ```
4. Install PyTorch and Torchvision with -f flag.
    ```bash
    pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
    ```
    * Note: On TWCC, we should install the newest torch by directly using `pip install torch && pip install torchvision` to avoid errors.s
5. Install packages needed for our project.
    ``` bash
    # Install packages from RVOS (Referformer, Onlinerefer)
    # Note: pillow version should be 8.4.0 to avoid error
    pip install -r requirements.txt
    pip install 'git+https://github.com/facebookresearch/fvcore' 
    pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    cd models/ops
    # remove build/ and dist/ if they exist
    rm -r build/ && rm -r dist && rm -r MultiScaleDeformableAttention.egg-info/
    python setup.py build install
    cd ../..

    # install SAM
    # Note: if you modify the SAM code, you should re-run this command.
    python -m pip install -e segment_anything
    
    # remove build/ and groundingdino.egg-info/ if they exist
    cd GroundingDINO/
    rm -r build/
    rm -r groundingdino.egg-info/

    # install GroundingDINO
    pip install -e .
    ```
## Run our code
* Prepare data & model weight (e.g., trained SAM checkpoint)
*  ```bash
    bash online_ytvos_train_sam_lora.sh ./outputs/
    ```
