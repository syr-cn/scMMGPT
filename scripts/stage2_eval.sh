#!/bin/bash
PROJECT_ROOT="path/to/root"

{
	name="stage2-eval";
	name="${name}-$(date +%b%d)";
	export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
	# export CUDA_VISIBLE_DEVICES="0,1,2,3"
	root=results/$name
	mkdir -p $root
	exec > $root/$name.out
	exec 2> $root/$name.err

	date +"%Y-%b-%d %H:%M:%S"
	BASEDIR=$(readlink -f "$0")
	echo "$BASEDIR"
	cat $BASEDIR
	echo -e "\n\n"

	python stage2.py \
		--filename $name \
		--devices "0,1,2,3,4,5,6,7" \
		--mode eval \
		--max_epochs 1 \
		--bert_name 'pubmedbert' \
		--precision "bf16" \
		--tune_gene_encoder \
		--tqdm_interval 100 \
		--num_query_token 32 \
		--img_first_flag True \
		--drop_ratio 0.02 \
		--init_lr 1e-5 \
		--warmup_steps 1000 \
		--cell_max_len 2048 \
		--batch_size 4 \
		--inference_batch_size 2 \
		--weight_decay 1e-3 \
		--accumulate_grad_batches 1 \
		--save_every_n_epochs 1 \
        --opt_model "path/to/opt_model" \
		--init_checkpoint "path/to/ckpt/ckpt.ckpt" \
		--root "$PROJECT_ROOT/data/cellxgene_hvalue/" \
		--llm_tune "lora" \
        --prompt "" \
    ;
}
