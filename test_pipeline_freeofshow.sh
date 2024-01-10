
OUTPUT_DIR='./results'
BOXNET_MODEL_PATH='./ckpt/boxnet.pt'
STABLE_MODEL_PATH='/data/zsz/ssh/models/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9/'
SEED=452800
MODE='gaussin_zero_one'   # 'gaussin_zero_one', 'zero_one' 
RATE=0.8

export options=" \
        --stable_model_path $STABLE_MODEL_PATH \
        --boxnet_model_path $BOXNET_MODEL_PATH \
        --output_dir $OUTPUT_DIR \
        --seed $SEED \
        --mask_mode $MODE \
        --max_guidance_rate $RATE
    "

python test_pipeline_freeofshow.py $options