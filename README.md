# QLora_PEFT
# QLora_PEFT



This was in the Mac:

   40  python -m venv qlora
   43  source qlora/bin/activate
   46  pip install bitsandbytes
   48  pip install torch
   50  pip install transformers
   52  pip install accelerate
   54  pip install scipy
   56  pip install peft
   58  pip install datasets
   63  pip install wandb



On the PC it seems that this is needed:


conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0  # for tensorflow version >2.5


