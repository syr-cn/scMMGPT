#!/bin/bash
PROJECT_ROOT="path/to/root"

{
	name="stage1-eval";
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
		--mode eval \
		--gtm \
		--lm \
		--max_epochs 1 \
		--precision "bf16" \
		--retrieval_eval_epoch 1 \
		--bert_name "pubmedbert" \
		--tune_gene_encoder \
		--num_query_token 8 \
		--rerank_cand_num 128 \
		--batch_size 12 \
		--test_on_tabula \
		--tabula_path "$PROJECT_ROOT/data/tabula/" \
		--tabula_batchsize 2 \
		--match_batch_size 64 \
		--tqdm_interval 10 \
		--save_every_n_epochs 1 \
		--init_checkpoint "path/to/ckpt/ckpt.ckpt" \
		--root "$PROJECT_ROOT/data/cellxgene_hvalue/"  \
	;
}
