from pathlib import Path
from datasets import Dataset, load_dataset
import os
import scanpy as sc
from scgpt import scbank
import gc
import shutil
import traceback
import warnings
import json
import random

class CellTextHelper:
    def __init__(self):
        celltext_file_obo = 'data/obo.json'
        celltext_file_wiki = 'data/cell_type_wiki.json'
        
        with open(celltext_file_obo, "r") as f:
            self.obo_map = json.load(f)
        self.obo_map = {v['name']:v for v in self.obo_map.values()}
        self.obo_map.update({k.lower():v for k, v in self.obo_map.items()})
        with open(celltext_file_wiki, "r") as f:
            self.wiki_map = json.load(f)
    
    def get_celltype_text_old(self, cell_type):
        cell_text = ''
        if cell_type in self.obo_map:
            if self.obo_map[cell_type]["def"]:
                cell_text += self.obo_map[cell_type]["def"]
        else:
            cell_type_lower = cell_type.lower()
            if cell_type_lower in self.obo_map:
                if self.obo_map[cell_type_lower]["def"]:
                    cell_text += self.obo_map[cell_type_lower]["def"]
        if cell_type in self.wiki_map and self.wiki_map[cell_type]:
            cell_text += '\n' + self.wiki_map[cell_type]
        return None
    
    def get_celltype_text(self, cell_type):
        cell_text = ''
        if cell_type in self.obo_map:
            if self.obo_map[cell_type]["def"]:
                cell_text += self.obo_map[cell_type]["def"]
                return cell_text
        else:
            cell_type_lower = cell_type.lower()
            if cell_type_lower in self.obo_map:
                if self.obo_map[cell_type_lower]["def"]:
                    cell_text += self.obo_map[cell_type_lower]["def"]
                    return cell_text
                
        return None

    def cellxgene_metadata_to_text(self, metadata):
        """
        Example metadata:
        - soma_joinid: 51152587
        - dataset_id: b0e547f0-462b-4f81-b31b-5b0a5d96f537
        - assay: 10x 5' v2
        - assay_ontology_term_id: EFO:0009900
        - cell_type: CD8-positive, alpha-beta cytotoxic T cell
        - cell_type_ontology_term_id: CL:0000794
        - development_stage: 27-year-old human stage
        - development_stage_ontology_term_id: HsapDv:0000121
        - disease: normal
        - disease_ontology_term_id: PATO:0000461
        - donor_id: KR_SGI_H006
        - is_primary_data: True
        - observation_joinid: bqkP6mQMA?
        - self_reported_ethnicity: Korean
        - self_reported_ethnicity_ontology_term_id: HANCESTRO:0022
        - sex: female
        - sex_ontology_term_id: PATO:0000383
        - suspension_type: cell
        - tissue: blood
        - tissue_ontology_term_id: UBERON:0000178
        - tissue_type: tissue
        - tissue_general: blood
        - tissue_general_ontology_term_id: UBERON:0000178
        - raw_sum: 6206.0
        - nnz: 2376
        - raw_mean_nnz: 2.611952861952862
        - raw_variance_nnz: 34.92556654261917
        - n_measured_vars: 33599
        """
        cell_text = ''
        # cell_text = random.choice([
        #     "This cell is identified as a {cell_type}, typically found in the {tissue}.",
        #     "The {cell_type}, a common cell type in the {tissue}, is what this sample is.",
        #     "Here we have a {cell_type} from the {tissue} region.",
        #     "A cell from the {tissue}, classified as {cell_type}, is observed.",
        #     "This is a {cell_type}, which is a cell type from the {tissue}.",
        #     "In the {tissue}, you will find cells like this {cell_type}.",
        #     "This sample is categorized under {cell_type} from the {tissue} tissue.",
        #     "Typical of the {tissue}, this {cell_type} is one of its unique cells.",
        #     "Observation shows a {cell_type}, a type of cell from the {tissue}.",
        #     "This {cell_type} is sourced from the {tissue} of a human.",
        #     "Characterized as a {cell_type}, this cell originates from the {tissue}.",
        #     "You are looking at a {cell_type}, one of the cell types from the {tissue}.",
        #     "This particular cell is a {cell_type} found in the {tissue}.",
        #     "Identified: {cell_type} from human {tissue}.",
        #     "The cell type here is {cell_type}, located in the {tissue}."
        # ]).format(cell_type=metadata["cell_type"], tissue=metadata["tissue"])
        cell_text = f"cell type: {metadata['cell_type']}."
        
        cell_type_text = self.get_celltype_text(metadata["cell_type"])
        if cell_type_text:
            # cell_text += f'\nInformation about the cell type {metadata["cell_type"]}: {cell_type_text}\n'  ## TODO: as cell_text
            cell_text += f' {cell_type_text}'

        # if 'sex' in metadata and 'development_stage' in metadata:
        #     cell_text += f' This cell is sampled from a {metadata["sex"]} human donor, and is in {metadata["development_stage"]} stage.'
        return cell_text

# Step 1: Preprocess the data

def preprocess(
    adata: sc.AnnData,
    main_table_key: str = "counts",
    include_obs = None,
    N=200000,
) -> sc.AnnData:
    """
    Preprocess the data for scBank. This function will modify the AnnData object in place.

    Args:
        adata: AnnData object to preprocess
        main_table_key: key in adata.layers to store the main table
        include_obs: dict of column names and values to include in the main table

    Returns:
        The preprocessed AnnData object
    """
    if include_obs is not None:
        # include only cells that have the specified values in the specified columns
        for col, values in include_obs.items():
            adata = adata[adata.obs[col].isin(values)]

    # filter genes
    sc.pp.filter_genes(adata, min_counts=(3 / 10000) * N)

    # TODO: add binning in sparse matrix and save in separate datatable
    # preprocessor = Preprocessor(
    #     use_key="X",  # the key in adata.layers to use as raw data
    #     filter_gene_by_counts=False,  # step 1
    #     filter_cell_by_counts=False,  # step 2
    #     normalize_total=False,  # 3. whether to normalize the raw data and to what sum
    #     log1p=False,  # 4. whether to log1p the normalized data
    #     binning=51,  # 6. whether to bin the raw data and to what number of bins
    #     result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    # )
    # preprocessor(adata)

    adata.layers[main_table_key] = adata.X.copy()  # preserve counts
    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)
    # adata.raw = adata  # freeze the state in `.raw`

    # apply a hard clip to the data for now
    print(
        f"original mean and max of counts: {adata.layers[main_table_key].mean():.2f}, "
        f"{adata.layers[main_table_key].max():.2f}"
    )
    # if isinstance(adata.layers[main_table_key], np.ndarray):
    #     adata.layers[main_table_key] = adata.layers[main_table_key].clip(0, 30)
    # else:  # assume it is a sparse matrix
    #     adata.layers[main_table_key].data = adata.layers[main_table_key].data.clip(0, 30)

    return adata

def process_adata_dir(input_dir, output_dir=None, N=200000, vocab=None, main_table_key="counts", token_col="gene_name", label_col=None):
    if output_dir is None:
        output_dir = Path(input_dir).parent / "databanks"
    # the input dir may looks like: ds_name/raw, which contains h5ad files
    if input_dir.endswith(".h5ad"):
        files = [Path(input_dir)]
    else:
        files = list(Path(input_dir).glob("*.h5ad"))
    print(f"Found {len(files)} files in {input_dir}")
    for f in files:
        try:
            adata = sc.read(f, cache=True)
            adata = preprocess(adata, main_table_key, N=N)
            print(f"read {adata.shape} valid data from {f.name}")
            print(f"{adata.obs_keys()=}")

            # TODO: CHECK AND EXPAND VOCABULARY IF NEEDED
            # NOTE: do not simply expand, need to check whether to use the same style of gene names

            # BUILD SCBANK DATA
            db = scbank.DataBank.from_anndata(
                adata,
                vocab=vocab,
                to=output_dir / f"{f.stem}.scb",
                main_table_key=main_table_key,
                token_col=token_col,
                immediate_save=False,
            )
            print(f"===")
            print(f"keys in db.data_tables: {db.data_tables.keys()}")
            print(f"===")
            if label_col:
                labels = list(adata.obs[label_col])
                db.data_tables['counts'].data = db.data_tables['counts'].data.add_column(label_col, labels)
            db.meta_info.on_disk_format = "parquet"
            # sync all to disk
            db.sync()
            # clean up
            del adata
            del db
            gc.collect()
        except Exception as e:
            traceback.print_exc()
            warnings.warn(f"failed to process {f.name}: {e}")
            shutil.rmtree(output_dir / f"{f.stem}.scb", ignore_errors=True)

    # or run scbank.DataBank.batch_from_anndata(files, to=args.output_dir)
    # test loading from disk
    # db = scbank.DataBank.from_path(args.output_dir)

    target_dir = output_dir / f"all_{main_table_key}"
    target_dir.mkdir(exist_ok=True)
    for f in files:
        output_parquet_dt = (
            output_dir / f"{f.stem}.scb" / f"{main_table_key}.datatable.parquet"
        )
        if output_parquet_dt.exists():
            os.symlink(output_parquet_dt, target_dir / f"{f.stem}.datatable.parquet")




"""
args.input_style = "binned"

if args.input_style == "binned":
    if args.input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif args.input_style == "log1p" or args.input_style == "normed_raw":
    if args.input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )

if args.input_emb_style == "category":
    args.mask_value = args.n_bins + 1
    args.pad_value = args.n_bins  # for padding gene expr values
    n_input_bins = args.n_bins + 2
else:
    args.mask_value = -1
    args.pad_value = -2
    n_input_bins = args.n_bins
"""

# Step 2: Prepare data with <cls> token appended at the beginning
def prepare_data_with_cls(data_source, vocab, pad_value=-2, num_proc=None):
    # data_source should be something ends with ds_name/databanks/all_counts
    parquet_files = [str(f) for f in Path(data_source).glob("*.parquet")]
    cache_dir = Path(data_source).parent / "cache"

    # load or make the dataset w/ <cls> appended at the beginning
    cls_prefix_datatable = Path(data_source) / "cls_prefix_data.parquet"
    if not cls_prefix_datatable.exists():
        print("preparing cls prefix dataset")
        raw_dataset = load_dataset(
            "parquet",
            data_files=parquet_files,
            split="train",
            cache_dir=str(cache_dir),
            num_proc=num_proc,
        )
        raw_dataset = _map_append_cls(raw_dataset, vocab, pad_value)
        raw_dataset.to_parquet(str(cls_prefix_datatable))
    raw_dataset = load_dataset(
        "parquet",
        data_files=str(cls_prefix_datatable),
        split="train",
        cache_dir=str(cache_dir),
            num_proc=num_proc,
    )
    return raw_dataset

def _map_append_cls(dataset: Dataset, vocab, pad_value) -> Dataset:
    dataset = dataset.map(
        lambda example: {
            "genes": [vocab["<cls>"]] + example["genes"],
            "expressions": [pad_value] + example["expressions"],
        },
        # batched=True,  # not using since then the map func needs to loop
        num_proc=len(os.sched_getaffinity(0)),
    )

    return dataset