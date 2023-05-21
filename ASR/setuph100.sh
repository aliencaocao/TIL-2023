wget "https://developer.download.nvidia.com/compute/cudnn/secure/8.9.1/local_installers/11.8/cudnn-local-repo-ubuntu2004-8.9.1.23_1.0-1_amd64.deb?Bm31MNv21RSXjCZ4pDmDRdrNXUmSw9LIMxuDTF1GbEcBwTCJHMAjWcuX5KRUFTii0hGbqiP-pwqYw1d_o5XJHcVbJiyfsjkZ6d9Liity2Ra7s8WJeQmth7uSIaJzB9RB3f5v6D3SvrQe5kwdJFo6FgPRdenr32sZQq2DvRyfLfMNmaISFSqPnggrWSnUgpbWbD7fFClLRcDui_EtP8vuRgRjvM4=&t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczovL3d3dy5nb29nbGUuY29tLyJ9"
sudo dpkg -i 'cudnn-local-repo-ubuntu2004-8.9.1.23_1.0-1_amd64.deb?Bm31MNv21RSXjCZ4pDmDRdrNXUmSw9LIMxuDTF1GbEcBwTCJHMAjWcuX5KRUFTii0hGbqiP-pwqYw1d_o5XJHcVbJiyfsjkZ6d9Liity2Ra7s8WJeQmth7uSIaJzB9RB3f5v6D3SvrQe5kwdJFo6FgPRdenr32sZQq2DvRyfLfMNmaISFSqPngg'
sudo cp /var/cudnn-local-repo-ubuntu2004-8.9.1.23/cudnn-local-D953484A-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt-get install libcudnn8=8.9.1.23-1+cuda11.8
sudo apt-get install libcudnn8-dev=8.9.1.23-1+cuda11.8
rm -rf /usr/lib/python3/dist-packages/tensorflow-gpu
rm -rf /usr/lib/python3/dist-packages/tensorflow_gpu*
sudo cp /usr/bin/nvcc /usr/local/cuda/bin/nvcc
pip install transformers datasets evaluate hf_transfer kenlm@git+https://github.com/kpu/kenlm jiwer accelerate transformer_engine@git+https://github.com/NVIDIA/TransformerEngine.git@stable librosa soundfile safetensors tensorboard psutil opencv-python pydub