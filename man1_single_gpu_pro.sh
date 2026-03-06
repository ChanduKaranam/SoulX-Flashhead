CUDA_VISIBLE_DEVICES=0

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python generate_video.py \
    --ckpt_dir models/SoulX-FlashHead-1_3B \
    --wav2vec_dir models/wav2vec2-base-960h \
    --model_type pro \
    --cond_image examples/man1.jpg \
    --audio_path examples/man.wav \
    --audio_encode_mode stream