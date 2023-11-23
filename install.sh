pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

python -m pip install -e segment_anything

pip install "mmcv==2.0.0" -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html

cd mmdetection
pip install -v -e .
cd ..

cd mmengine
pip install -v -e .
cd ..
