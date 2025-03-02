#!/bin/bash
PROJECT_ROOT="path/to/root"

{
	name="stage1-train";
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
		--mode train \
		--gtm \
		--max_epochs 1 \
		--bert_name 'pubmedbert' \
		--precision "bf16" \
		--tune_gene_encoder \
		--num_query_token 32 \
		--drop_ratio 0.02 \
		--init_lr 1e-5 \
		--warmup_steps 1000 \
		--batch_size 12 \
		--batch_size_contrast 3 \
		--weight_decay 1e-3 \
		--accumulate_grad_batches 1 \
		--qformer_contrast_func "contrast_batch" \
		--validation_every_step 120000 \
		--cell_max_len 2048 \
		--test_on_tabula \
		--tabula_path "$PROJECT_ROOT/data/tabula" \
		--tabula_batchsize 2 \
		--match_batch_size 8 \
		--tqdm_interval 10 \
		--save_every_n_epochs 1 \
		--retrieval_eval_epoch 1 \
		--retrieval_eval_step 10000 \
		--root "$PROJECT_ROOT/data/cellxgene_hvalue/" \
		--zeroshot_cls_datafiles "$PROJECT_ROOT/data/zeroshot_cls/pbmc_10k_new.h5ad" \
	;
}
