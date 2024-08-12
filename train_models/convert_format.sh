cd src
python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /home/llama-3/llama3/Meta-Llama-3-8B \
    --model_size 8B \
    --output_dir /home/converted_models/llama-3-8b

# for llama1-2: transformers==4.29.2, 4.31.0 works
