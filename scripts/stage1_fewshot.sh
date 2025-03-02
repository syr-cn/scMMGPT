#!/bin/bash
PROJECT_ROOT="path/to/root"

{
	name="stage1-fewshot-0.3";
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

	python stage1.py \
		--filename $name \
		--devices "0,1,2,3,4,5,6,7" \
		--mode 'continue train' \
		--gtm \
		--max_epochs 7 \
		--bert_name 'pubmedbert' \
		--precision "bf16" \
        --tune_gene_encoder \
		--num_query_token 32 \
		--drop_ratio 0.02 \
		--init_lr 1e-5 \
		--min_lr 1e-6 \
		--warmup_steps 1000 \
		--batch_size 12 \
		--batch_size_contrast 3 \
		--scheduler "linear_warmup_step_lr" \
		--weight_decay 1e-3 \
		--accumulate_grad_batches 1 \
		--qformer_contrast_func "contrast_batch" \
		--validation_every_step 100000 \
		--cell_max_len 2048 \
		--match_batch_size 8 \
		--tqdm_interval 100 \
		--save_every_n_epochs 5 \
		--retrieval_eval_epoch 1 \
		--retrieval_eval_step 10000 \
		--root "$PROJECT_ROOT/data/cellxgene_hvalue/" \
		--ckpt_path "path/to/ckpt/ckpt.ckpt" \
		--train_on_tabula_ratio 0.3 \
		--test_on_tabula \
        --tabula_train_indices_path  "$PROJECT_ROOT/data/tabula/train_indices.json" \
        --tabula_test_indices_path "$PROJECT_ROOT/data/tabula/test_indices.json" \
		--tabula_path "$PROJECT_ROOT/data/tabula" \
		--tabula_batchsize 2 \
	;
}