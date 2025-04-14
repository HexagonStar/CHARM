conda create -n charm 
conda activate charm

cd ../

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout d17fd7cd3b71c6a7bf7af34d8dc73135bb7ea8e9

# You may need to update the torch version based on your cuda version.
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .

pip install flash-attn==2.6.3

pip install accelerate==0.33.0 
pip install deepspeed==0.12.2
pip install transformers==4.43.4
pip install numpy==1.26.4 
pip install vllm

cd ../CHARM
