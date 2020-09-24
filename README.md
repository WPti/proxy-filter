What is Proxy-filter?
---------------

Proxy-filter is an easy-to-use tool for low-resource parallel corpus filtering tool. 
It is developed as a proxy learner on top of a transformer-based multilingual pre-trained language model (RoBERTa). 
Proxy-filter can easily score the noisy low-resource parallel corpus with high accuracy and low latency, covering 100 languages. 
Proxy-filter wins the runner-up position in WMT20 Parallel Corpus Filtering Task (https://www.statmt.org/wmt20/parallel-corpus-filtering.html).


Getting Started
---------------
- Proxy-filter has been tested with Python 3.6+ on Linux.
- Proxy-filter uses torch 1.4 under the hood to extract embeddings. You must install torch MANUALLY according to your CUDA version. Go check out https://pytorch.org/.
- For other requirements, you may install using pip: `pip install requirements.txt`
	
	
Method
---------------
Training the filtering model consists of two steps:
- generating a dataset for the proxy task [[1]](#1);
- finetuning pretrained xlm-roberta-large transformer [[2]](#2) with the huggingface interface [[3]](#3). 

`finetune.sh` will run these steps automatically.


Training a Filtering Model
--------------------------
After cloning the repo and installing the requirements, you can train a filtering model for a desired language pair (or domain).

Firstly you may edit finetune.sh and change the below data path to specify the directory of the training and validation data:

    --valid_src_data_dir  /VAILD_SRC_DATA_PATH/valid.ps  
    --valid_trg_data_dir  /VAILD_TRG_DATA_PATH/valid.en  
    --train_src_data_dirs /TRAIN_SRC_DATA_PATH/train.ps 
    --train_trg_data_dirs /TRAIN_TRG_DATA_PATH/train.en

Some insights:

     - It is usually enough to supply a ~5-10k parallel pair for training and a couple of hundreds for the validation;
     - Bigger datasets will increase the training time drastically and will not probably improve filtering performance;
     - A model can be biased towards training data sentence length in the filtering phase. So training data should not be dominated with long or short sentences;
     - You can feed multiple files to TRAINING data but not for the VALIDATION;
     - Training data can be subsampled by changing "training_examples_subsample" parameter (1.0 = 100%).


Running the finetune.sh will take 6-8 hours with 10k bitext on a single NVIDIA V100 gpu.
	

Checking the Training Logs for a Specific Model
-----------------------------------------------

If you want to check the details of a trained model you can check the training logs including datasets, parameters and performance by 
running tensorboard [[4]](#4) at the folder "../runs".

You can edit the prefix --output_dir parameter ../outputs/[name it to remember] if you plan to train multiple models.
	

Performance and Memory Adjustments
---------------------------------
Parameters:
- --max_seq_length
- --per_gpu_eval_batch_size
- --per_gpu_train_batch_size
    
The above parameters affect GPU memory usage and performance directly. Please adjust the batch size according to your GPU memory to prevent 
the "Out of Memory" error. 

Note that:

     - Lowering the max_seq_length more than 256 may cause accuracy drop;
     - Lowering the per_gpu_train_batch_size below 8 may cause performance issues; 
     - If you are limited on the GPU memory use gradient_accumulation_steps to keep gradient update batch size above 8. (gradient update batch size = per_gpu_train_batch_size*gradient_accumulation_steps )



Filtering
---------------

The filtering step will score each sentence pair that you provided to the model. A higher score means the model is more confident on that pair.
Proxy-filter can be simply run as: `bash filter.sh`. 

Note that you need to edit filter.sh or run filter.py with your own parameters:

     python ../code/filter.py 
     --max_seq_length 128 
     --batch_size 1024 
     --model_checkpoint_path ../outputs/[model-name-that-you-trained-before]
     --src_data /SRC_DATA_PATH/sents.ps  
     --trg_data /TRG_DATA_PATH/sents.en  
     --output_dir ../filter.output/  \

A "scores.txt" file will be produced at the "output_dir" location.

    
## References
<a id="1">[1]</a> 
Link to official submission will be here soon.

<a id="2">[2]</a> 
Model details, https://huggingface.co/transformers/model_doc/xlmroberta.html

<a id="3">[3]</a> 
Huggingface library, https://github.com/huggingface/transformers

<a id="4">[4]</a> 
Tensorboard, https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams