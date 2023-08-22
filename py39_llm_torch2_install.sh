# set up python libraries for LLM version

# To create requirements file from existing python env:
# pip freeze > py39_ratd_requirements.txt
# edit this file:
#	REMOVE the line attempting to install en_core_sci_sm
#	REMOVE the lines installing torch, torchvision and torchaudio

# Python env setup:
# cd full/path/to_base_dir_that_env_name_will_become_subdir_of  #ie /data/thar011/config
# python3.9 -m venv env_name 
# source env_name/bin/activate

pip3 install torch torchvision torchaudio

pip install -r requirements.txt


# only necessary if using scispacy: the RATD text_processing.py code DOESNT use it for anything...
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz

pip install accelerate
pip install bitsandbytes
pip install git+https://github.com/huggingface/transformers.git

pip install datasets


