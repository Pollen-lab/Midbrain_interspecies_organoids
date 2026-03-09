#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# ### Reference mapping from D40Human with SCVI


# #### Linnarsson 1st trimester atlas as reference

# In[2]:


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



adata_ref = sc.read('/wynton/group/pollen/jding/Sara/linnarsson/MB.h5ad')
adata_ref.obs['CellClass'] = [x[2:-1] for x in adata_ref.obs['CellClass']]


sc.pl.umap(
    adata_ref,
    color=[ "CellClass",'Region','supervised_name'],
    frameon=False,
)


# ### Import query data
'''
# In[42]:
adata_query = sc.read('/wynton/group/pollen/sara/20240313_GEX_D40-D100-D80.h5ad')

adata_query.X = adata_query.raw.X
adata_query.raw = adata_query
adata_query.layers['counts'] = adata_query.X
#adata_query = adata_query[adata_query.obs['species'] == 'human']
print(adata_query.X)

# In[15]:
adata_ref.var_names_make_unique()
adata_query.var_names_make_unique()


# In[16]:
sc.pp.normalize_total(adata_query, target_sum=1e4,exclude_highly_expressed=True)
sc.pp.log1p(adata_query)
#sc.pp.scale(adata_query, max_value=10)
sc.pp.highly_variable_genes(adata_query, n_top_genes=2500, batch_key="batch_name", subset= True)
'''


adata_query = sc.read('/wynton/group/pollen/jding/Sara/linnarsson/SCVI_query.h5ad')
adata_ref.var_names_make_unique()
adata_query.var_names_make_unique()


# In[17]:


var = [x for x in adata_ref.var_names if x in adata_query.var_names]
adata_ref = adata_ref[:, var].copy()
adata_query = adata_query[:, var].copy()


# In[18]:


adata_ref


# In[19]:


adata_query.obs['sample_id'] = adata_query.obs['batch_name']

'''
# In[20]:


scvi.model.SCVI.setup_anndata(adata_ref, batch_key="sample_id", layer="counts")

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


# In[21]:
adata_ref.obsm["X_scVI"] = vae_ref.get_latent_representation()
sc.pp.neighbors(adata_ref, use_rep="X_scVI")
sc.tl.leiden(adata_ref)
sc.tl.umap(adata_ref)


# In[23]:


sc.pl.umap(
    adata_ref,
    color=[ "CellClass",'Region','supervised_name'],
    frameon=False,
)




# In[25]:
# save the reference model
dir_path = "/wynton/home/pollenlab/jding/Sara/data/scanpy/scVI_vivo_model/"
vae_ref.save(dir_path, overwrite=True)
'''

# load the reference model
directory='/wynton/home/pollenlab/jding/Sara/data/scanpy/'
dir_path = os.path.join(directory,'scVI_vivo_model/')
vae_ref = scvi.model.SCVI.load(dir_path, adata_ref)


# In[26]:


# both are valid
scvi.model.SCVI.prepare_query_anndata(adata_query, dir_path)
#scvi.model.SCVI.prepare_query_anndata(adata_query, vae_ref)


# In[27]:
adata_query.obs['sample_id'] = adata_query.obs['batch_name']

#add adata_ref for coembedding
adata_query = anndata.AnnData.concatenate(adata_query,adata_ref,join='outer',batch_categories=['This Study','In Vivo Atlas']) 


# In[28]:
# both are valid
vae_q = scvi.model.SCVI.load_query_data(
    adata_query,
    dir_path,
)
vae_q = scvi.model.SCVI.load_query_data(
    adata_query,
    vae_ref,
)


# In[29]:


vae_q.train(max_epochs=200, plan_kwargs=dict(weight_decay=0.0))
adata_query.obsm["X_scVI"] = vae_q.get_latent_representation()


# In[30]:


sc.pp.neighbors(adata_query, use_rep="X_scVI")
sc.tl.leiden(adata_query)
sc.tl.umap(adata_query)


# In[31]:
sc.pl.umap(adata_query, color=['supervised_name','sample_id','batch','CellClass','Region'],frameon=False,ncols=1,save='scvi-integration')


# ### Reference mapping with SCANVI

# In[32]:


adata_ref.obs["labels_scanvi"] = adata_ref.obs["supervised_name"].values


# In[33]:

directory='/wynton/home/pollenlab/jding/Sara/data/scanpy/'
dir_path = os.path.join(directory,'scVI_vivo_model/')
vae_ref = scvi.model.SCVI.load(dir_path, adata_ref)


'''
# In[34]:


# unlabeled category does not exist in adata.obs[labels_key]
# so all cells are treated as labeled
vae_ref_scan = scvi.model.SCANVI.from_scvi_model(
    vae_ref,
    unlabeled_category="Unknown",
    labels_key="labels_scanvi",
)


# In[ ]:


vae_ref_scan.train(max_epochs=20, n_samples_per_label=100)


# In[45]:


adata_ref.obsm["X_scANVI"] = vae_ref_scan.get_latent_representation()
sc.pp.neighbors(adata_ref, use_rep="X_scANVI")
sc.tl.leiden(adata_ref)
sc.tl.umap(adata_ref)


# In[46]:


sc.pl.umap(
    adata_ref,
    color=[ "CellClass",'Region','leiden'],
    frameon=False,
)


# In[ ]:


# save the reference model
dir_path_scan = "/wynton/home/pollenlab/jding/Sara/data/scanpy/scanvi_vivo_model/"
vae_ref_scan.save(dir_path_scan, overwrite=True)

'''
# In[47]:

# load the reference model
dir_path_scan = "/wynton/home/pollenlab/jding/Sara/data/scanpy/scanvi_vivo_model/"
vae_ref_scan = scvi.model.SCANVI.load(dir_path_scan,adata_ref)


# In[51]:


# again a no-op in this tutorial, but good practice to use
scvi.model.SCANVI.prepare_query_anndata(adata_query, dir_path_scan)


# In[53]:


vae_q = scvi.model.SCANVI.load_query_data(adata_query,vae_ref_scan)
#vae_q = scvi.model.SCANVI.load_query_data(adata_query,dir_path_scan,)


# In[49]:


vae_q.train(
    max_epochs=100,
    plan_kwargs=dict(weight_decay=0.0),
    check_val_every_n_epoch=10,
)


# In[ ]:


adata_query.obsm["X_scANVI"] = vae_q.get_latent_representation()
adata_query.obs["predictions"] = vae_q.predict()


# In[ ]:
sc.pl.umap(adata_query, color=['predictions','sample_id','batch','CellClass','Region'],frameon=False,ncols=1,save='scanvi-integration')


# In[83]:


adata_query.write('/wynton/group/pollen/jding/Sara/linnarsson/SCVI_integrated.h5ad')
