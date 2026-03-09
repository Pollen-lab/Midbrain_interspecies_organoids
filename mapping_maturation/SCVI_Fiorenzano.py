#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
#from scar import setup_anndata
import warnings
import anndata
import numpy as np
#import bbknn
warnings.simplefilter("ignore")



# In[31]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import anndata
import scvi
import scanpy as sc

sc.set_figure_params(figsize=(4, 4))
scvi.settings.seed = 94705

import os

'''
library(Seurat)
library(SeuratDisk)
ad <- readRDS('/wynton/group/pollen/jding/Sara/human_harmony.THneurons.20241002.rds')
ad <- as.SingleCellExperiment(ad, assay = c("RNA"))
ad <- as.Seurat(ad)
SaveH5Seurat(ad, filename = "/wynton/group/pollen/jding/Sara/human_harmony.THneurons.20241002.h5Seurat")
Convert("/wynton/group/pollen/jding/Sara/human_harmony.THneurons.20241002.h5Seurat", dest = "h5ad")
'''

import scanpy as sc
adata_query = sc.read('/wynton/group/pollen/sara/20240313_GEX_D40-D100-D80.h5ad')
adata_query = adata_query[adata_query.obs['species'].isin(['chimp','human'])]
adata_query = adata_query[adata_query.obs['supervised_name'].isin(['DA neurons'])]


directory = '/wynton/group/pollen/jding/Sara/Fiorenzano'
adata_ref =  sc.read(os.path.join(directory,'human_harmony.THneurons.20241002.h5ad'))
adata_ref = adata_ref.raw.to_adata()
adata_ref.obsm['X_umap'] = adata_ref.obsm['X_UMAP']
adata_ref.obs['ident'] = adata_ref.obs['ident'] + 1
adata_ref.obs['ident'] = 'DA-' + adata_ref.obs['ident'].astype(str)
adata_ref.var_names = adata_ref.var['_index']

sc.pp.filter_cells(adata_ref, min_genes=100)
sc.pp.filter_genes(adata_ref, min_cells=3)

adata_ref.raw = adata_ref
adata_ref.layers['counts'] = adata_ref.X
adata_ref

sc.pp.normalize_total(adata_ref,exclude_highly_expressed=True)
sc.pp.log1p(adata_ref)
sc.pp.highly_variable_genes(adata_ref, n_top_genes=2500, batch_key="SeqRound", subset= True)



adata_query = adata_query.raw.to_adata()
adata_query.raw = adata_query
adata_query.layers['counts'] = adata_query.X
adata_query

adata_ref.var_names_make_unique()
adata_query.var_names_make_unique()

var = [x for x in adata_ref.var_names if x in adata_query.var_names]
adata_ref = adata_ref[:, var].copy()
adata_query = adata_query[:, var].copy()

print(adata_ref)

adata_query.obs['label'] = adata_query.obs['batch_name']
adata_ref.obs['label'] = adata_ref.obs['SeqRound']

scvi.model.SCVI.setup_anndata(adata_ref, batch_key="label", layer="counts")

arches_params = dict(
    use_layer_norm="both",
    use_batch_norm="none",
    encode_covariates=True,
    dropout_rate=0.2,
    n_layers=2,
)

vae_ref = scvi.model.SCVI(
    adata_ref,
    **arches_params
)
vae_ref.train()


adata_ref.obsm["X_scVI"] = vae_ref.get_latent_representation()
sc.pp.neighbors(adata_ref, use_rep="X_scVI")
sc.tl.leiden(adata_ref)
sc.tl.umap(adata_ref)

sc.pl.umap(
    adata_ref,
    color=[ "ident",'label'],
    frameon=False,
    save = 'adult-scvi'
)

# save the reference model
dir_path = "/wynton/home/pollenlab/jding/Sara/data/scanpy/scVI_adult_model/"
vae_ref.save(dir_path, overwrite=True)

# both are valid
scvi.model.SCVI.prepare_query_anndata(adata_query, dir_path)
#scvi.model.SCVI.prepare_query_anndata(adata_query, vae_ref)

# both are valid
vae_q = scvi.model.SCVI.load_query_data(
    adata_query,
    dir_path,
)
vae_q = scvi.model.SCVI.load_query_data(
    adata_query,
    vae_ref,
)

vae_q.train(max_epochs=200, plan_kwargs=dict(weight_decay=0.0))
adata_query.obsm["X_scVI"] = vae_q.get_latent_representation()

sc.pp.neighbors(adata_query, use_rep="X_scVI")
sc.tl.leiden(adata_query)
sc.tl.umap(adata_query)


adata_ref.obs["labels_scanvi"] = adata_ref.obs["ident"].values


dir_path = "/wynton/home/pollenlab/jding/Sara/data/scanpy/scVI_adult_model/"
vae_ref = scvi.model.SCVI.load(dir_path, adata_ref)

# unlabeled category does not exist in adata.obs[labels_key]
# so all cells are treated as labeled
vae_ref_scan = scvi.model.SCANVI.from_scvi_model(
    vae_ref,
    unlabeled_category="Unknown",
    labels_key="labels_scanvi",
)


vae_ref_scan.train(max_epochs=20, n_samples_per_label=100)

adata_ref.obsm["X_scANVI"] = vae_ref_scan.get_latent_representation()
sc.pp.neighbors(adata_ref, use_rep="X_scANVI")
sc.tl.leiden(adata_ref)
sc.tl.umap(adata_ref)

# save the reference model
dir_path_scan = "/wynton/home/pollenlab/jding/Sara/data/scanpy/scanvi_adult_model/"
vae_ref_scan.save(dir_path_scan, overwrite=True)

# again a no-op in this tutorial, but good practice to use
scvi.model.SCANVI.prepare_query_anndata(adata_query, dir_path_scan)


vae_q = scvi.model.SCANVI.load_query_data(adata_query,dir_path_scan,)


vae_q.train(
    max_epochs=100,
    plan_kwargs=dict(weight_decay=0.0),
    check_val_every_n_epoch=10,
)

adata_query.obsm["X_scANVI"] = vae_q.get_latent_representation()
adata_query.obs["predictions"] = vae_q.predict()

sc.pl.umap(adata_query,color=["predictions",'supervised_name'],frameon=False, legend_loc='on data',legend_fontsize ='xx-small',save= 'predictions')


import numpy as np
df = (
    adata_query.obs.groupby(['predictions',"species"])
    .size()
    .unstack(fill_value=0)
)


norm_df = df / df.sum(axis=0)

#norm_df = norm_df.loc[:,['RG/Astro','RG_DIV','IPC_EN','EN_ImN','EN', 'IPC_IN','IN_dLGE/CGE','OPC/Oligo','Technical']]
#norm_df = norm_df.loc[['Astrocyte-Fibrous','Astrocyte-Protoplasmic','RG-tRG','RG-oRG','RG-vRG','IPC-EN',
# 'EN-Newborn','EN-IT-Immature', 'EN-Non-IT-Immature','EN-L2_3-IT','EN-L4-IT','EN-L5-IT','EN-L5-ET','EN-L5_6-NP','EN-L6b','EN-L6-IT','EN-L6-CT','Cajal-Retzius cell',
# 'IPC-Glia','IN-dLGE-Immature','IN-CGE-Immature','IN-CGE-SNCG','IN-CGE-LAMP5','IN-CGE-VIP','IN-MGE-Immature','IN-MGE-PV','IN-MGE-SST',
#'OPC','Oligodendrocyte-Immature','Oligodendrocyte'],:]


# Optional: Visualize as a heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
ax = sns.heatmap(norm_df.T, cmap="gray_r",annot=False)
ax.set_yticks(np.arange(0.5, len(norm_df.T.index), 1)) 
ax.set_xticks(np.arange(0.5, len(norm_df.T.columns), 1)) 

# Drawing the frame 
for _, spine in ax.spines.items(): 
    spine.set_visible(True) 
    spine.set_linewidth(1) 

# Set x and y labels
plt.xlabel("In Vivo Atlas")
plt.ylabel("This Study")
plt.title("Fraction overlap of labels transferred cell types to the cell types defined in this study")
plt.tight_layout()
plt.savefig('./figures/SCVI.pdf')
plt.show()


adata_query.write(os.path.join(directory,'SCVI_query.h5ad'))


import numpy as np
df = (
    adata_query.obs.groupby(['predictions',"time_point"])
    .size()
    .unstack(fill_value=0)
)


norm_df = df / df.sum(axis=0)

#norm_df = norm_df.loc[:,['RG/Astro','RG_DIV','IPC_EN','EN_ImN','EN', 'IPC_IN','IN_dLGE/CGE','OPC/Oligo','Technical']]
#norm_df = norm_df.loc[['Astrocyte-Fibrous','Astrocyte-Protoplasmic','RG-tRG','RG-oRG','RG-vRG','IPC-EN',
# 'EN-Newborn','EN-IT-Immature', 'EN-Non-IT-Immature','EN-L2_3-IT','EN-L4-IT','EN-L5-IT','EN-L5-ET','EN-L5_6-NP','EN-L6b','EN-L6-IT','EN-L6-CT','Cajal-Retzius cell',
# 'IPC-Glia','IN-dLGE-Immature','IN-CGE-Immature','IN-CGE-SNCG','IN-CGE-LAMP5','IN-CGE-VIP','IN-MGE-Immature','IN-MGE-PV','IN-MGE-SST',
#'OPC','Oligodendrocyte-Immature','Oligodendrocyte'],:]


# Optional: Visualize as a heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
ax = sns.heatmap(norm_df.T, cmap="gray_r",annot=False)
ax.set_yticks(np.arange(0.5, len(norm_df.T.index), 1)) 
ax.set_xticks(np.arange(0.5, len(norm_df.T.columns), 1)) 

# Drawing the frame 
for _, spine in ax.spines.items(): 
    spine.set_visible(True) 
    spine.set_linewidth(1) 

# Set x and y labels
plt.xlabel("In Vivo Atlas")
plt.ylabel("This Study")
plt.title("Fraction overlap of labels transferred cell types to the cell types defined in this study")
plt.tight_layout()
plt.savefig('./figures/SCVI.timepoint.pdf')
plt.show()


