#Setup----
rm(list = ls()); gc()
folder = 'Midbrain/Figure5/V9/'; dir.create(folder)
object_folder = 'Midbrain/Object_creation/V5/'
peaks_folder = 'Midbrain/Consensus_peaks/D40_100/V8/'
dreamlet_folders <- list()
dreamlet_folders$DA_neurons <- '/media/jenelle/4TB_disk/Dropbox/Analysis/Dreamlet_R_4.4.0/Midbrain/ATAC/D40_D100_D80/V26/DA_neurons/'
dreamlet_folders$DA_STN_neurons_immature <- '/media/jenelle/4TB_disk/Dropbox/Analysis/Dreamlet_R_4.4.0/Midbrain/ATAC/D40_D100_D80/V26/DA_STN_neurons_immature/'
dreamlet_folders$Ventral_FB_MB_progenitors <- '/media/jenelle/4TB_disk/Dropbox/Analysis/Dreamlet_R_4.4.0/Midbrain/ATAC/D40_D100_D80/V26/Ventral_FB_MB_progenitors/'
sp_dreamlet_folders = c(human = '/media/jenelle/4TB_disk/Dropbox/Analysis/Dreamlet_R_4.4.0/Midbrain/ATAC/D40_D100_D80/V26/DA_neurons/human_only/',
                        chimp = '/media/jenelle/4TB_disk/Dropbox/Analysis/Dreamlet_R_4.4.0/Midbrain/ATAC/D40_D100_D80/V26/DA_neurons/chimp_only/')
rna_folder = '/media/jenelle/4TB_disk/Dropbox/Analysis/Dreamlet_R_4.4.0/Midbrain/Ancestral_genome/Figure4/Versions_main/V2/'
ucsc_tracks_folder = '/media/jenelle/4TB_disk/Dropbox/Analysis/Signac/UCSC_tracks/'
pollen_tracks_folder = '/media/jenelle/4TB_disk/Dropbox/Analysis/Signac/Pollen_tracks/'
annotation_folder = 'Genome_annotations/'
gwas_folder = 'GWAS_catalog/'
source("/media/jenelle/4TB_disk/Dropbox/R_analysis/MyFunctions.R")
source("/media/jenelle/4TB_disk/Dropbox/R_analysis/MyPlottingFunctions.R")
source("/media/jenelle/4TB_disk/Dropbox/R_analysis/MySeuratFunctions.R")
source("/media/jenelle/4TB_disk/Dropbox/R_analysis/MyDreamletFunctions.R")
source("/media/jenelle/4TB_disk/Dropbox/R_analysis/MyGenomicRangesFunctions.R")
source("/media/jenelle/4TB_disk/Dropbox/Analysis/Dreamlet_R_4.4.0/plotMyVolcano.R")
source("/media/jenelle/4TB_disk/Dropbox/Analysis/Dreamlet_R_4.4.0/plotStratify_mod.R")
library(Seurat)
source("/media/jenelle/4TB_disk/Dropbox/R_analysis/DimPlot_mod.R", echo=TRUE) #overwriting DimPlot to add stroke.size
library(Signac)
library(GenomicRanges)
library(SingleCellExperiment)
library(Matrix)
library(tidyverse)
library(stringr)
library(ensembldb)
library(txdbmaker)
library(patchwork)
library(qs)
library(SeuratWrappers)
library(cicero)
library(rtracklayer)
library(dreamlet)
library(extrafont)
library(rlang)
library(cowplot)
library(data.table)
library(ggrepel)
library(DescTools)
library(scattermore)
library(openxlsx)
library(dplyr)
loadfonts(device = "pdf")
info <- capture.output(sessionInfo()); writeLines(info, paste0(folder, "session_info.txt")) #save sessionInfo in case there are any errors due to versions of packages
coi = 'DA_neurons'
con = 'human_vs_chimp'
peak_width = 501
category_types = c("promoter", "exon", "intron", "intergenic")
p_thresh = 0.1
coaccess_filters = c(0,0.05,0.1,0.15,0.2,0.25,0.3)
cf_filter_to_plot = 0.15
species = c('human','chimp','rhesus')
species_names = c(human = "human",chimp = "chimp",rhesus = "macaque")
species_labels = c(human = "Human",chimp = "Chimp",rhesus = "Rhesus")
colors_species = c(human = '#F59121',chimp = '#3957A6',macaque = '#7E2859')
colors_polarize = c('human_specific' = '#F59121','chimp_specific' = '#3957A6',
                    'divergent' = '#079655', 'other' = 'black')
colors_species_ld <- list(
  human = c('#FFD7B5', '#B35900'), # Very light and very dark versions of #F59121
  chimp = c('#D1D7FF', '#1E3A73')  # Very light and very dark versions of #3957A6
)
tab20_colors <- c(
  "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", 
  "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", 
  "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", 
  "#17becf", "#9edae5"
)
celltype_order = c("DA_neurons","DA_STN_neurons_immature","FB_progenitors","Glial_progenitors_astrocytes","Hypothalamic_neurons",
                   "Lateral_MB_progenitors","Lateral_MB_progenitors_cycling","skip","MB_GABAergic_neurons","skip","MB_glutamatergic_neurons",
                   "MB_HB_FP_cells","MB_HB_glutamatergic_neurons","MB_HB_glutamatergic_neurons_immature","MB_HB_neurons_LHX1",
                   "Oculomotor_neurons","Progenitors_cycling", "STN_neurons","Ventral_FB_MB_progenitors","Ventral_FB_MB_progenitors_cycling")
color_mapping <- setNames(tab20_colors, celltype_order)
celltype_colors <- color_mapping[color_mapping != "skip"]

#Calculations (only run once)----
##Load for calculations
res_dl_da <- readRDS(paste0(dreamlet_folders$DA_neurons,'res_dl.rds'))
res_dl_immature <- readRDS(paste0(dreamlet_folders$DA_STN_neurons_immature,'res_dl.rds'))
res_dl_prog <- readRDS(paste0(dreamlet_folders$Ventral_FB_MB_progenitors,'res_dl.rds'))
res_dl <- res_dl_da
res_dl$DA_STN_neurons_immature <- res_dl_immature$DA_STN_neurons_immature
res_dl$Ventral_FB_MB_progenitors <- res_dl_prog$Ventral_FB_MB_progenitors
saveRDS(res_dl, paste0(folder,'res_dl.rds'))
species_allcon_peaks <- readRDS(paste0(peaks_folder,'species_allcon_peaks.rds'))
species_only_peaks <- readRDS(paste0(peaks_folder,'species_only_peaks.rds'))
de_genes_list <- readRDS(paste0(rna_folder,'de_genes.rds'))
all_genes_list <- readRDS(paste0(rna_folder,'all_genes.rds'))
multi_species <- qread(paste0(object_folder,'multi_species.rds'))

##Setup files
annotation_files <- c()
annotation_files[['human']] = paste0(annotation_folder,'Human_gencodev33.gtf')
annotation_files[['chimp']] = paste0(annotation_folder,'Chimp_gencodev33.gtf')
annotation_files[['rhesus']] = paste0(annotation_folder,'Rhesus_gencodev33.gtf')
refseq_files <- c()
refseq_files[['human']] = paste0(ucsc_tracks_folder,'hg38.ncbiRefSeq.gtf')
refseq_files[['chimp']] = paste0(ucsc_tracks_folder,'panTro6.ncbiRefSeq.gtf')
refseq_files[['rhesus']] = paste0(ucsc_tracks_folder,'rheMac10.ncbiRefSeq.gtf')
refseq_annotations <- vector(mode = "list", length = length(species)); names(refseq_annotations) = species
for (sp in species){
  refseq_annotations[[sp]] <- rtracklayer::import(refseq_files[[sp]])
  refseq_annotations[[sp]]$tx_id <- refseq_annotations[[sp]]$transcript_id
  refseq_annotations[[sp]]$gene_biotype <- refseq_annotations[[sp]]$type
}
saveRDS(refseq_annotations,paste0(folder,'refseq_annotations.rds'))

##Make master objects for da regions and all regions in DA neurons----
#Dreamlet results
celltypes = names(res_dl)
contrasts = coefNames(res_dl); base_coef = c('specieshuman','specieschimp','speciesmacaque','speciesorangutan','log_num_peaks')
contrasts <- contrasts[!sapply(contrasts, function(x) any(x == base_coef))]
da_peaks_list = list()
all_peaks_list = list()
for (celltype in celltypes) {
  celltype_list = list()
  celltype_list_all = list()
  for (con in contrasts) {
    df_con <- as.data.frame(topTable(res_dl[[celltype]], coef = con, number = Inf)) #using res_dl[[celltype]] here will adjust p values separately for each cell type instead of all together
    df_celltype_all <- df_con
    df_celltype_all$assay <- celltype; df_celltype_all$ID <- rownames(df_celltype_all); rownames(df_celltype_all) <- NULL
    df_celltype_all <- df_celltype_all %>% relocate(last_col(offset = 1), last_col())
    df_celltype <- df_celltype_all[df_celltype_all$adj.P.Val < p_thresh, ]
    df_celltype$gene_sign = paste0(sign(df_celltype$logFC),df_celltype$ID)
    celltype_list[[con]] = df_celltype
    celltype_list_all[[con]] = df_celltype_all
  }
  da_peaks_list[[celltype]] = celltype_list
  all_peaks_list[[celltype]] = celltype_list_all
}
saveRDS(da_peaks_list,paste0(folder,'da_peaks_list_',p_thresh,'.rds'))
saveRDS(all_peaks_list,paste0(folder,'all_peaks_list.rds'))

#Combine with peaks
con = 'human_vs_chimp'
celltypes = c('Ventral_FB_MB_progenitors','DA_STN_neurons_immature','DA_neurons')
dars_dreamlet <- list()
allrs_dreamlet <- list()
dars_gr= list()
dars_df=list()
allrs_gr=list()
allrs_df=list()
total_peaks_gr=list()
total_peaks_df=list()
spec_peaks_gr=c()
spec_peaks_df=c()
dars_and_spec_gr=c()
for (celltype in celltypes){
  #Dreamlet DARs in human
  dars_dreamlet[[celltype]] = da_peaks_list[[celltype]][[con]]
  dars_dreamlet[[celltype]]$name = dars_dreamlet[[celltype]]$ID
  allrs_dreamlet[[celltype]] = all_peaks_list[[celltype]][[con]]
  allrs_dreamlet[[celltype]]$name = allrs_dreamlet[[celltype]]$ID
  for (sp in species){
    #Dreamlet DARs
    result = combineDreamletAndGranges(species_allcon_peaks, sp, dars_dreamlet[[celltype]])
    dars_gr[[celltype]][[sp]] = result$dars_gr
    dars_df[[celltype]][[sp]] = result$dars_df
    #Dreamlet all peaks
    result = combineDreamletAndGranges(species_allcon_peaks, sp, allrs_dreamlet[[celltype]])
    allrs_gr[[celltype]][[sp]] = result$dars_gr
    allrs_df[[celltype]][[sp]] = result$dars_df
    }
}
#All CrossPeak peaks
for (sp in species){
  total_peaks_gr[[sp]] = species_allcon_peaks[[sp]]
  total_peaks_df[[sp]] <- as.data.frame(total_peaks_gr[[sp]]) %>%
    mutate(coords_name = paste(seqnames,start,end,sep = '-'))
}
#DARs and spec peaks passing expression threshold (only for DA neurons)
spec_res_proc = c()
species_spec = c('human','chimp')
for (sp in species_spec){
  spec_res_proc[[sp]] <- readRDS(paste0(sp_dreamlet_folders[[sp]],'res_proc.rds'))
}
for (sp in species_spec){
  data <- extractData(spec_res_proc[[sp]],coi)
  spec_peak_names = colnames(data)[str_starts(colnames(data),"Peak_")]
  spec_peaks_gr[[sp]] = species_only_peaks[[sp]][spec_peak_names]
  dars_and_spec_gr[[sp]] <- c(dars_gr[[coi]][[sp]],spec_peaks_gr[[sp]]) #Including species only only peaks
  peak_means <-as.data.frame(spec_res_proc[[sp]][[coi]]$voom.xy$x)
  peak_means$name = rownames(peak_means)
  peak_means <- peak_means %>%
    rename(AveExpr = "spec_res_proc[[sp]][[coi]]$voom.xy$x") %>%
    mutate(cell_type = coi)
  result = combineDreamletAndGranges(species_only_peaks, sp, peak_means)
  spec_peaks_df[[sp]] = result$dars_df
}
saveRDS(dars_dreamlet,paste0(folder,'dars_dreamlet.rds'))
saveRDS(dars_gr,paste0(folder,'dars_gr.rds'))
saveRDS(dars_df,paste0(folder,'dars_df.rds'))
saveRDS(allrs_gr,paste0(folder,'allrs_gr.rds'))
saveRDS(allrs_df,paste0(folder,'allrs_df.rds'))
saveRDS(total_peaks_gr,paste0(folder,'total_peaks_gr.rds'))
saveRDS(total_peaks_df,paste0(folder,'total_peaks_df.rds'))
saveRDS(spec_peaks_gr,paste0(folder,'spec_peaks_gr.rds'))
saveRDS(spec_peaks_df,paste0(folder,'spec_peaks_df.rds'))
saveRDS(dars_and_spec_gr,paste0(folder,'dars_and_spec_gr.rds'))
saveRDS(spec_res_proc,paste0(folder,'spec_res_proc.rds'))

##Load and format evolutionary signatures----
hs_variants <- readRDS(paste0(pollen_tracks_folder,'/human_specific_variants_V2.rds'))
hs_tracks_to_use = c('hCONDELs_indelsized','hIns','ZooHARs','HAQERs')
hs_tracks = hs_variants[hs_tracks_to_use]
n_deserts <- import.bed(paste0(pollen_tracks_folder,'/neanderthal_deserts/deserts_sorted_hg38.bed'))
hs_tracks$AADs = n_deserts
saveRDS(hs_tracks,paste0(folder,'hs_tracks.rds'))
cs_tracks_to_use = c('hDels')
cs_tracks = hs_variants[cs_tracks_to_use]
saveRDS(cs_tracks,paste0(folder,'cs_tracks.rds'))

##Annotation categories----
annotation_categories <- c()
sp_annotation <- c()
for (sp in species){
  sp_annotation[[sp]] <- readRDS(paste0(object_folder,sp,'_annotation.rds'))
  levels(sp_annotation[[sp]])[levels(sp_annotation[[sp]]) == "cds"] <- "CDS"
  # Define promoter regions
  tss <- GetTSSPositions(sp_annotation[[sp]])
  promoter_regions = expandRange(tss, upstream = 1000, downstream = 0)
  #Define exons, introns
  txdb <- makeTxDbFromGFF(annotation_files[[sp]], format="gtf")
  txbygene = transcriptsBy(txdb, by="gene")
  map <- relist(unlist(txbygene, use.names=FALSE)$tx_id, txbygene)
  exons_list = exonsBy(txdb, "tx", use.names=TRUE)
  exons = unlist(exons_list)
  introns_list = intronsByTranscript(txdb, use.names=TRUE)
  introns = unlist(introns_list)
  cds <- cdsBy(txdb, "tx")
  #Remove duplicate ranges as they would cause issues later
  promoter_regions = GenomicRanges::reduce(unique(promoter_regions), ignore.strand = TRUE); promoter_regions$category = 'promoter'
  exons = GenomicRanges::reduce(unique(exons), ignore.strand = TRUE);   exons$category = 'exon'
  introns = GenomicRanges::reduce(unique(introns), ignore.strand = TRUE); introns$category = 'intron'
  annotation_categories[[sp]] = c(promoter_regions,exons,introns)
}
saveRDS(annotation_categories,paste0(folder,'annotation_categories.rds'))
qsave(sp_annotation, paste0(folder,'sp_annotation.rds'))

##Export bed files for GREAT----
export.bed(dars_and_spec_gr$human,paste0(folder,'GREAT_dars_and_human_spec_test_',p_thresh,'.bed'))
allrs_and_human_spec_gr = c(spec_peaks_gr$human,allrs_gr[[coi]]$human)
export.bed(allrs_and_human_spec_gr,paste0(folder,'GREAT_allrs_and_human_spec_background.bed'))

##Integration for UMAP----
multi_species <- qread(paste0(object_folder,'multi_species.rds'))
num_con_peaks = length(species_allcon_peaks$human)
#Subset to consensus peaks
multi_species_con <-  list()
for (sp in species){
  multi_species_con[[sp]] = multi_species[[sp]]
  DefaultAssay(multi_species_con[[sp]]) <- 'peaks_consensus'
  multi_species_con[[sp]][["peaks_celltypes"]] <- NULL
  multi_species_con[[sp]] <- subset(multi_species_con[[sp]], features = rownames(multi_species_con[[sp]])[1:num_con_peaks])
}
rm(multi_species); gc()

#Rename peaks to human coords and process each species
species_umapplots = list()
for (sp in species){
  Annotation(multi_species_con[[sp]]) = Annotation(multi_species_con$human)
  #rename counts
  rownames(multi_species_con[[sp]]@assays$peaks_consensus@counts)[1:num_con_peaks] = rownames(multi_species_con$human@assays$peaks_consensus@counts)[1:num_con_peaks]
  #rename data
  rownames(multi_species_con[[sp]]@assays$peaks_consensus@data)[1:num_con_peaks] = rownames(multi_species_con$human@assays$peaks_consensus@data)[1:num_con_peaks]
  #rename meta.features
  rownames(multi_species_con[[sp]]@assays$peaks_consensus@meta.features)[1:num_con_peaks] = rownames(multi_species_con$human@assays$peaks_consensus@meta.features)[1:num_con_peaks]
  #process and UMAP
  multi_species_con[[sp]] <- RunTFIDF(multi_species_con[[sp]])
  multi_species_con[[sp]] <- FindTopFeatures(multi_species_con[[sp]], min.cutoff = 'q0') #need to use all features or else they won't be the same across species
  multi_species_con[[sp]] <- RunSVD(multi_species_con[[sp]])
  multi_species_con[[sp]] <- RunUMAP(object = multi_species_con[[sp]], reduction = 'lsi', dims = 2:30)
  multi_species_con[[sp]] <- FindNeighbors(object = multi_species_con[[sp]], reduction = 'lsi', dims = 2:30)
  Idents(multi_species_con[[sp]]) <- multi_species_con[[sp]]$cell_type
  species_umapplots[[sp]] <- DimPlot(object = multi_species_con[[sp]], label = FALSE, cols = tab20_colors, group.by = 'cell_type') +
    labs(x = 'UMAP1',y = 'UMAP2')+
    theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
          axis.text = element_blank(),
          axis.title = element_text(size = 6),
          axis.ticks = element_blank(),
          plot.title = element_blank()
    )
}
qs::qsave(multi_species_con, paste0(folder,'multi_species_con.rds'))

###Combine and process
multi_combined <- merge(multi_species_con$human, y = c(multi_species_con$chimp, multi_species_con$rhesus))
qs::qsave(multi_combined,paste0(folder,'multi_combined.rds'))
multi_combined <- FindTopFeatures(multi_combined, min.cutoff = 10)
multi_combined <- RunTFIDF(multi_combined)
multi_combined <- RunSVD(multi_combined)
multi_combined <- RunUMAP(multi_combined, reduction = "lsi", dims = 2:30)
qs::qsave(multi_combined,paste0(folder,'multi_combined.rds'))

###Integrate
# multi_combined <- qs::qread(paste0(folder, 'multi_combined.rds'))
# multi_species_con <- qs::qread(paste0(folder, 'multi_species_con.rds'))
# find integration anchors
integration.anchors <- FindIntegrationAnchors(
  object.list = list(multi_species_con[['human']], multi_species_con[['chimp']], multi_species_con[['rhesus']]),
  anchor.features = rownames(multi_combined),
  reduction = "rlsi",
  dims = 2:30,
  k.anchor = 20 #increased from default to integrate more strongly
)
rm(multi_species_con); gc()
qs::qsave(integration.anchors,paste0(folder,'integration_anchors.rds'))
# integrate LSI embeddings
lsi_reduction = multi_combined[["lsi"]]
rm(multi_combined); gc()
multi_integrated <- IntegrateEmbeddings(
  anchorset = integration.anchors,
  reductions = lsi_reduction,
  new.reduction.name = "integrated_lsi",
  dims.to.integrate = 1:30
)
# create a new UMAP using the integrated embeddings
multi_integrated <- RunUMAP(multi_integrated, reduction = "integrated_lsi", dims = 2:30)
qs::qsave(multi_integrated,paste0(folder,'multi_integrated.rds'))

##Enrichment of evolutionary signatures----
#Filter for only high confidence categories
categories = c('round1_indel_low', 'round2_indel_low')
dars_and_spec_gr$human_filter = filter_for_high_confidence(dars_and_spec_gr$human, categories, 2)
nodars_gr = setDiffForPeakNames(allrs_gr[[coi]]$human,dars_and_spec_gr$human)
nodars_gr_filter = filter_for_high_confidence(nodars_gr, categories, 2)
dars_and_spec_gr$chimp_filter = filter_for_high_confidence(dars_and_spec_gr$chimp, categories, 2)
nodars_gr_filter_chimp = species_allcon_peaks$chimp[nodars_gr_filter$name]
#Split any overlapping DA peaks
dars_and_spec_gr$human_reduce = split_overlapping_peaks(dars_and_spec_gr$human_filter) 
nodars_gr_reduce = split_overlapping_peaks(nodars_gr_filter)
dars_and_spec_gr$chimp_reduce = split_overlapping_peaks(dars_and_spec_gr$chimp_filter) 
nodars_gr_reduce_chimp = split_overlapping_peaks(nodars_gr_filter_chimp)
genome(dars_and_spec_gr$human) = 'hg38'
genome(dars_and_spec_gr$human_reduce) = 'hg38'
genome(nodars_gr_reduce) = 'hg38'
genome(dars_and_spec_gr$chimp) = 'panTro6'
genome(dars_and_spec_gr$chimp_reduce) = 'panTro6'
genome(nodars_gr_reduce_chimp) = 'panTro6'
fisher_test_result = c()
contingency_tables = c()
peaks_overlap_tracks_human = c()
tracks_overlap_peaks_human = c()
for (track in names(hs_tracks)){
  #DARs overlaps with evolutionary signature
  #Original peak set to find peaks that overlap (includes lower confidence categories for manual inspection)
  result = findOverlaps(dars_and_spec_gr$human,hs_tracks[[track]])
  peaks_overlap_tracks_human[[track]] = dars_and_spec_gr$human[queryHits(result)]
  tracks_overlap_peaks_human[[track]] = hs_tracks[[track]][subjectHits(result)]
  #Filtered and reduced peak set to get numbers for contingency table
  result = findOverlaps(dars_and_spec_gr$human_reduce,hs_tracks[[track]])
  A = length(result)
  B = length(dars_and_spec_gr$human_reduce) - A
  #Non-DARs overlap with evolutionary signature
  result = findOverlaps(nodars_gr_reduce,hs_tracks[[track]])
  C = length(result)
  D = length(nodars_gr_reduce) - C
  contingency_tables[[track]] <- matrix(c(A, B, C, D), nrow = 2, byrow = TRUE)
  result <- fisher.test(contingency_tables[[track]])
  fisher_test_result[[track]] = result$p.value
}
peaks_overlap_tracks_chimp = c()
tracks_overlap_peaks_chimp = c()
for (track in names(cs_tracks)){
  #DARs overlaps with evolutionary signature
  #Original peak set to find peaks that overlap
  result = findOverlaps(dars_and_spec_gr$chimp,cs_tracks[[track]])
  peaks_overlap_tracks_chimp[[track]] = dars_and_spec_gr$chimp[queryHits(result)]
  tracks_overlap_peaks_chimp[[track]] = cs_tracks[[track]][subjectHits(result)]
  #Reduced peak set to get numbers for contingency table
  result = findOverlaps(dars_and_spec_gr$chimp_reduce,cs_tracks[[track]])
  A = length(result)
  B = length(dars_and_spec_gr$chimp_reduce) - A
  #Non-DARs overlap with evolutionary signature
  result = findOverlaps(nodars_gr_reduce_chimp,cs_tracks[[track]])
  C = length(result)
  D = length(nodars_gr_filter_chimp) - C
  contingency_tables[[track]] <- matrix(c(A, B, C, D), nrow = 2, byrow = TRUE)
  result <- fisher.test(contingency_tables[[track]])
  fisher_test_result[[track]] = result$p.value
}
saveRDS(peaks_overlap_tracks_human,paste0(folder,'peaks_overlap_tracks_human.rds'))
saveRDS(peaks_overlap_tracks_human,paste0(folder,'peaks_overlap_tracks_chimp.rds'))
saveRDS(tracks_overlap_peaks_human,paste0(folder,'tracks_overlap_peaks_human.rds'))
saveRDS(tracks_overlap_peaks_chimp,paste0(folder,'tracks_overlap_peaks_chimp.rds'))
saveRDS(contingency_tables,paste0(folder,'contingency_tables.rds'))
saveRDS(fisher_test_result,paste0(folder,'fisher_test_result.rds'))

##Cicero----
multi_integrated <- qs::qread(paste0(folder,'multi_integrated.rds'))
cicero_species = c('human','chimp')
cicero_celltypes = c('Ventral_FB_MB_progenitors','DA_STN_neurons_immature','DA_neurons') #links in these cell types
# Filter to keep peaks expressed in >1% of DA neuron lineage celltypes in at least one species and all promoter peaks and all dreamlet peaks for selected cell types
sp_conns = c()
cicero_cds = c()
peaks_to_keep_names_con = c()
for (sp in cicero_species){
  #Get promoter peaks
  promoter_peaks = get_promoter_peaks(species_allcon_peaks[[sp]],sp_annotation[[sp]])
  promoter_peaks_names = paste(seqnames(promoter_peaks),start(promoter_peaks),end(promoter_peaks),sep = '-')
  #Get dreamlet peaks
  dreamlet_peaks = c()
  for (celltype in cicero_celltypes){
    celltype_peaks = all_peaks_list[[celltype]][[con]]$ID
    dreamlet_peaks = c(dreamlet_peaks,celltype_peaks)
  }
  dreamlet_peaks = unique(dreamlet_peaks)
  dreamlet_peaks_gr = species_allcon_peaks[[sp]][dreamlet_peaks]
  dreamlet_peaks_names = paste(seqnames(dreamlet_peaks_gr),start(dreamlet_peaks_gr),end(dreamlet_peaks_gr),sep = '-')
  #Get peaks in >= 1% of DA neurons
  sp_integrated <- subset(multi_integrated, subset = species == species_names[[sp]] & cell_type %in% cicero_celltypes)
  counts <- GetAssayData(sp_integrated, assay = "peaks_consensus", slot = "counts")
  rownames(counts) = total_peaks_df[[sp]]$coords_name #put name for correct species because peaks in sp_integrated were named by human
  binary_sparse_matrix <- counts
  binary_sparse_matrix@x <- as.numeric(binary_sparse_matrix@x > 0)
  gene_expression_percentage <- rowSums(binary_sparse_matrix) / ncol(binary_sparse_matrix) * 100
  peaks_high_exp_names <- names(which(gene_expression_percentage > 1))
  sp_peaks_to_keep_names = unique(c(promoter_peaks_names, dreamlet_peaks_names,peaks_high_exp_names))
  peaks_to_keep_names_con[[sp]] = total_peaks_df[[sp]]$name[total_peaks_df[[sp]]$coords_name %in% sp_peaks_to_keep_names]
}
unique_peaks_to_keep_con = unique(unlist(peaks_to_keep_names_con))
saveRDS(peaks_to_keep_names_con, paste0(folder,'peaks_to_keep_names_con.rds'))
#Get coords names in each species
peaks_to_keep_names_coords = c()
for (sp in cicero_species){
  peaks_to_keep_names_coords[[sp]] = total_peaks_df[[sp]]$coords_name[total_peaks_df[[sp]]$name %in% unique_peaks_to_keep_con]
}
#Links in DA lineage
for (sp in cicero_species){
  sp_integrated <- subset(multi_integrated, subset = species == species_names[[sp]] & cell_type %in% cicero_celltypes)
  #Put back correct annotation and rename with peak coords for correct species (integrated object had human)
  DefaultAssay(sp_integrated) <- 'peaks_consensus'
  Annotation(sp_integrated) = sp_annotation[[sp]]
  sp_integrated_cds <- as.cell_data_set(x = sp_integrated)
  rownames(sp_integrated_cds) <- total_peaks_df[[sp]]$coords_name
  sp_integrated_cds <- sp_integrated_cds[rownames(sp_integrated_cds) %in% peaks_to_keep_names_coords[[sp]], ]
  cicero_cds[[sp]] <- make_cicero_cds(sp_integrated_cds, reduced_coordinates = reducedDims(sp_integrated_cds)$UMAP)
  # get the chromosome lengths
  chrom_lengths <- sapply(levels(seqnames(sp_annotation[[sp]])), function(chr) max(end(sp_annotation[[sp]][seqnames(sp_annotation[[sp]]) == chr]))) #estimation based on gtf
  # convert chromosome lengths to a dataframe
  chrom_lengths_df <- data.frame("chr" = names(chrom_lengths), "length" = chrom_lengths)
  chrom_lengths_df = chrom_lengths_df[!is.infinite(chrom_lengths_df$length),] #some weird chromosomes like chrM are messed up in primate so remove
  # run cicero
  sp_conns[[sp]] <- run_cicero(cicero_cds[[sp]], genomic_coords = chrom_lengths_df, sample_num = 100)
}
qs::qsave(cicero_cds,paste0(folder,'cicero_cds.rds'))
qs::qsave(sp_conns,paste0(folder,'sp_conns.rds'))

###Linking peaks to genes----
all_peaks_coi <- all_peaks_list$DA_neurons$human_vs_chimp
all_peaks_coi$peak_name = all_peaks_coi$ID
all_genes_coi = all_genes_list$D40_100$DA_neurons$human_vs_chimp
all_genes_coi$gene = all_genes_coi$ID

cicero_links <- list()
for (i in seq_along(coaccess_filters)){
  cf <- coaccess_filters[i]
  cicero_links[[i]] <- list()
  for (sp in cicero_species){
    tss <- GetTSSPositions(sp_annotation[[sp]])
    gene_gr <- sp_annotation[[sp]][sp_annotation[[sp]]$type %in% c("gene", "exon", "UTR"), ]
    # Summarize the full range for each gene by grouping by gene_id
    gene_ranges <- gene_gr %>%
      as.data.frame() %>%
      group_by(gene_name) %>%
      summarize(
        start = min(start),
        end = max(end),
        seqnames = unique(seqnames),
        strand = unique(strand)
      )
    gene_ranges_gr <- makeGRangesFromDataFrame(gene_ranges, keep.extra.columns = TRUE)
    gene_ranges_with_promoter = expandRange(gene_ranges_gr, upstream = 1000, downstream = 0)
    gene_ranges_with_promoter_df <- as.data.frame(gene_ranges_with_promoter) %>%
      dplyr::rename(gene = gene_name)
    # Annotate cds with peaks falling anywhere in gene body
    cds_annotated <- annotate_cds_by_site(cicero_cds[[sp]], gene_ranges_with_promoter_df)
    gene_peaks <- rownames(fData(cds_annotated))[!is.na(fData(cds_annotated)$gene)]
    # Filter the connections to keep only those involving at least one promoter peak
    all_gene_connections <- subset(sp_conns[[sp]], Peak1 %in% gene_peaks | Peak2 %in% gene_peaks)
    all_gene_connections <- all_gene_connections[!is.na(all_gene_connections$coaccess),]
    #Identify all peaks linked to peaks within gene bodies
    # Create a data frame of peak-to-gene mappings
    gene_peaks_df <- fData(cds_annotated) %>%
      as.data.frame() %>%
      rownames_to_column(var = "peak") %>%
      dplyr::filter(!is.na(gene)) %>%  # Keep only peaks associated with genes (gene peaks)
      dplyr::select(peak, gene)
    # Find all peaks connected to gene peaks
    connections_with_genes <- all_gene_connections %>%
      left_join(gene_peaks_df, by = c("Peak1" = "peak")) %>%
      dplyr::rename(gene1 = gene) %>%
      left_join(gene_peaks_df, by = c("Peak2" = "peak")) %>%
      dplyr::rename(gene2 = gene)
    # #Filter out connections between two different gene peaks
    # peaks_linked_to_genes <- connections_with_genes %>%
    #   dplyr::filter(is.na(gene1) | is.na(gene2) | gene1 == gene2)
    peaks_linked_to_genes <- connections_with_genes[connections_with_genes$coaccess >= cf,]
    # #Filter to only genes that met expression cutoff in that species
    # sp_peak_names_coords = total_peaks_df[[sp]]$coords_name[total_peaks_df[[sp]]$name %in% peaks_to_keep_names_con[[sp]]]
    # peaks_linked_to_genes <- peaks_linked_to_genes %>%
    #   dplyr::filter(Peak1 %in% sp_peak_names_coords & Peak2 %in% sp_peak_names_coords)
    #Add information about DA peaks and DE genes
    cicero_links[[i]][[sp]] <- peaks_linked_to_genes %>%
      dplyr::mutate(gene = coalesce(gene1, gene2)) %>%
      dplyr::select(-coaccess) %>% # Remove the coaccess column
      pivot_longer(cols = c(Peak1, Peak2), names_to = "PeakType", values_to = "coords_name") %>%
      distinct(gene, coords_name) %>%
      left_join(total_peaks_df[[sp]], by = 'coords_name') %>%
      dplyr::mutate(peak_name = name) %>%
      dplyr::select(gene, coords_name, peak_name) %>%
      left_join(allrs_df[[coi]][[sp]], by = 'coords_name', suffix = c('','')) %>%
      dplyr::select(gene, coords_name, peak_name, logFC, z.std, adj.P.Val) %>%
      left_join(all_genes_coi, by = 'gene', suffix = c('_peak','_gene')) %>%
      dplyr::select(gene, coords_name,peak_name, logFC_peak, z.std_peak,adj.P.Val_peak, logFC_gene, z.std_gene,adj.P.Val_gene) %>%
      mutate(concordant = ifelse(sign(logFC_peak) == sign(logFC_gene), TRUE, FALSE)) %>%
      mutate(both_sig = ifelse(adj.P.Val_gene < 0.05 & adj.P.Val_peak < p_thresh, TRUE, FALSE))
  }
}
saveRDS(cicero_links,paste0(folder,'cicero_links.rds'))

#Load processed data(for skipping calculations when complete)----
annotation_categories <- readRDS(paste0(folder,'annotation_categories.rds'))
sp_annotation <- qread(paste0(folder,'sp_annotation.rds'))
species_allcon_peaks <- readRDS(paste0(peaks_folder,'species_allcon_peaks.rds'))
species_only_peaks <- readRDS(paste0(peaks_folder,'species_only_peaks.rds'))
res_dl_da <- readRDS(paste0(dreamlet_folders$DA_neurons,'res_dl.rds'))
dars_dreamlet <- readRDS(paste0(folder,'dars_dreamlet.rds'))
dars_gr <- readRDS(paste0(folder,'dars_gr.rds'))
dars_df <- readRDS(paste0(folder,'dars_df.rds'))
dars_and_spec_gr <- readRDS(paste0(folder,'dars_and_spec_gr.rds'))
spec_peaks_gr <- readRDS(paste0(folder,'spec_peaks_gr.rds'))
allrs_gr <- readRDS(paste0(folder,'allrs_gr.rds'))
allrs_df <- readRDS(paste0(folder,'allrs_df.rds'))
total_peaks_df <- readRDS(paste0(folder,'total_peaks_df.rds'))
spec_res_proc <- readRDS(paste0(folder,'spec_res_proc.rds'))
all_peaks_list <-readRDS(paste0(folder,'all_peaks_list.rds'))
de_genes_list <- readRDS(paste0(rna_folder,'de_genes.rds'))
all_genes_list <- readRDS(paste0(rna_folder,'all_genes.rds'))
cicero_links <- readRDS(paste0(folder,'cicero_links.rds'))
peaks_overlap_tracks_human <- readRDS(paste0(folder,'peaks_overlap_tracks_human.rds'))
peaks_overlap_tracks_human <- readRDS(paste0(folder,'peaks_overlap_tracks_chimp.rds'))
tracks_overlap_peaks_human <- readRDS(paste0(folder,'tracks_overlap_peaks_human.rds'))
tracks_overlap_peaks_chimp <- readRDS(paste0(folder,'tracks_overlap_peaks_chimp.rds'))

#Main panels----
##Panel B: Individual analysis for each species using peaks_celltypes----
multi_species <- qread(paste0(object_folder,'multi_species.rds'))
plot_summary <- data.frame()
for (sp in species) {
  atac_peaks <- multi_species[[sp]]@assays$peaks_celltypes@ranges
  plot_summary_sp <- assign_peaks_to_annotation_categories(atac_peaks, annotation_categories[[sp]], peak_width, category_types)
  plot_summary_sp <- as.data.frame(plot_summary_sp)
  plot_summary_sp$species <- sp
  # Combine with overall plot_summary
  plot_summary <- rbind(plot_summary, plot_summary_sp)
}
# Plot the stacked bar chart
plot_summary <- plot_summary %>% mutate(species_labels = str_to_title(species))
plot_summary$species_labels <- factor(plot_summary$species_labels, levels = rev(species_labels))
plot_summary$category <- factor(plot_summary$category, levels = category_types)
gray_palette <- scales::grey_pal()(length(unique(plot_summary$category))) 
panelB <- ggplot(plot_summary, aes(x = species_labels,y = count, fill = category)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_y_continuous(labels = scales::percent_format(), expand = c(0,0)) +
  scale_fill_manual(values = gray_palette)+
  labs(x = NULL, y = "Percent of peaks") +
  theme_basic_smallest()+
  theme(axis.ticks.x = element_blank(),
        plot.margin = unit(c(0,0,0,0),'in'),
        legend.position = 'bottom', 
        legend.key.size = unit(c(0.1),'in'),
        legend.text = element_text(size = 6)) +
  coord_flip() +
  guides(fill = guide_legend(reverse = TRUE))

##Panel D: Conserved markers----
idents_plot = c("Ventral_FB_MB_progenitors","DA_STN_neurons_immature","DA_neurons")
celltypes_colors = c(tab20_colors[19],tab20_colors[2], tab20_colors[1])
extend_up = 500
extend_down = 500
genes = c('SOX2','EN1','TH')
scale_factor = 1e7
ymaxes = c(210,250,200,150)

sp_gene_plots = c()
for (gene in genes){
  index = which(genes==gene)
  for (sp in species){
    Idents(multi_species[[sp]]) <- multi_species[[sp]]$cell_type
    obj = multi_species[[sp]]
    p <- CoveragePlot(
      object = obj,
      assays = 'peaks_consensus',
      region = gene,
      peaks = FALSE,
      idents = idents_plot,
      extend.upstream = extend_up,
      extend.downstream = extend_down,
      scale.factor = 1e7,
      ymax = ymaxes[index]
    )
    if (sp == 'human'){
      sp_gene_plots[[sp]][[gene]] <- p & scale_fill_manual(values = celltypes_colors) &
        theme_basic_smallest() &
        theme(legend.position = 'none')
    } else {
      sp_gene_plots[[sp]][[gene]] <- p & scale_fill_manual(values = celltypes_colors) &
        theme_basic_smallest() &
        theme(axis.title.y = element_blank(),
              axis.text.y = element_blank(),
              axis.ticks.y = element_blank(),
              axis.line.y = element_blank(),
              strip.text.y.left = element_blank(),
              strip.background = element_blank(),
              legend.position = 'none')
    }
  }
}

panelD <- sp_gene_plots$human$SOX2 + sp_gene_plots$chimp$SOX2 + sp_gene_plots$rhesus$SOX2 +
  sp_gene_plots$human$LMX1A + sp_gene_plots$chimp$LMX1A + sp_gene_plots$rhesus$LMX1A + 
  sp_gene_plots$human$EN1 + sp_gene_plots$chimp$EN1 + sp_gene_plots$rhesus$EN1 +
  sp_gene_plots$human$TH + sp_gene_plots$chimp$TH + sp_gene_plots$rhesus$TH +
  plot_layout(ncol = 3, nrow = 4)
ggsave(filename = paste0(folder,'panelD.pdf'), panelD, width = 7, height = 7, units = 'in', dpi = 300)
rm(multi_species)

##Panel C: UMAP----
multi_integrated <- qread(paste0(folder,'multi_integrated.rds'))
panelC <- DimPlot(object = multi_integrated, label = FALSE, cols = celltype_colors, group.by = 'cell_type',pt.size = 1.5, stroke.size = 0, alpha = 1, shuffle = FALSE, raster = TRUE, raster.dpi = c(600,600)) +
  labs(x = 'UMAP1',y = 'UMAP2')+
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 0.25),
        axis.text = element_blank(),
        axis.title = element_text(size = 6),
        axis.ticks = element_blank(),
        plot.title = element_blank(),
        legend.position = 'none',
        plot.margin = unit(c(0,0,0,0),'in')
  ) 
panelC
ggsave(paste0(folder,'PanelC.pdf'), width = 2.4, height = 1.7,  units = 'in', dpi = 600)
rm(multi_integrated)

##Panel E: DARs summary----
#Locations
#Consensus regions (use human annotation)
plot_summary_con <- assign_peaks_to_annotation_categories(dars_gr[[coi]]$human, annotation_categories$human, peak_width, category_types)
plot_summary_con$category <- factor(plot_summary_con$category, levels = category_types)
plot_summary_con <- plot_summary_con[order(plot_summary_con$category), ]
#Human only
plot_summary_human <- assign_peaks_to_annotation_categories(spec_peaks_gr$human, annotation_categories$human, peak_width, category_types)
plot_summary_human$category <- factor(plot_summary_human$category, levels = category_types)
plot_summary_human <- plot_summary_human[order(plot_summary_human$category), ]
#Chimp only
plot_summary_chimp <- assign_peaks_to_annotation_categories(spec_peaks_gr$chimp, annotation_categories$chimp, peak_width, category_types)
plot_summary_chimp$category <- factor(plot_summary_chimp$category, levels = category_types)
plot_summary_chimp <- plot_summary_chimp[order(plot_summary_chimp$category), ]
#Combine
plot_summary = data.frame(category = factor(category_types, levels = category_types),
                          number = plot_summary_con$count + plot_summary_human$count + plot_summary_chimp$count)
#Plotting
gray_palette <- scales::grey_pal()(length(category_types))  # Adjust the number (3) to the number of subcategories
categories_bar <- ggplot(plot_summary, aes(x = "", y = number, fill = category)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_y_continuous(labels = scales::percent_format(), expand = c(0,0)) +
  scale_fill_manual(values = gray_palette)+
  labs(x = NULL, y = "Percent of DARs", fill = "Annotation") +
  theme_basic_smallest()+
  theme(plot.title = element_text(size = 7, hjust = 0.5),
        axis.ticks.y = element_blank(),
        legend.key.size = unit(c(0.1),'in'),
        legend.text = element_text(size = 6),
        legend.position = 'none',
        plot.margin = unit(c(0.2,0,0,0),'in')
  ) + coord_flip()

#Numbers
#up, down, human-spec, chimp-spec
num_up = sum(dars_dreamlet[[coi]]$logFC > 0)
num_down = sum(dars_dreamlet[[coi]]$logFC < 0)
num_human_spec = length(spec_peaks_gr$human)
num_chimp_spec = length(spec_peaks_gr$chimp)
direction_df = data.frame(
  category = c("Human up", "Chimp up", "Human-\nonly", "Chimp-\nonly"),
  number = c(num_up, num_down, num_human_spec, num_chimp_spec),
  species = c('human', 'chimp', 'human', 'chimp')
)
direction_df$category = factor(direction_df$category, levels = c('Human up', 'Chimp up',"Human-\nonly", "Chimp-\nonly"))
numbers_bar <- ggplot(direction_df, aes(x = category, y = number)) +
  geom_bar(stat = 'identity', aes(fill = species)) +
  scale_fill_manual(values = colors_species) +
  scale_y_continuous(expand = expansion(mult = c(0, 0))) +
  labs(y = 'Number of DARs') +
  theme_basic_smallest() +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_text(angle = 60, hjust = 1),
    legend.position = c(0.8, 0.5),
    legend.key.size = unit(c(0.1), 'in'),
    legend.text = element_text(size = 6),
    plot.margin = unit(c(0.2, 0, 0, 0), 'in')
  )
panelBE <- panelB / categories_bar /numbers_bar + plot_layout(heights = c(2,0.75,3.6))

##Panel H: Cicero summary plot----
cicero_species = c('human','chimp')
cf_index = which(coaccess_filters==cf_filter_to_plot)
human_links <- cicero_links[[cf_index]]$human %>%
  dplyr::select(gene, peak_name)
chimp_links <- cicero_links[[cf_index]]$chimp %>%
  dplyr::select(gene, peak_name)
common_peaks <- intersect(human_links$peak_name, chimp_links$peak_name)
# Classify the peaks
peak_classification <- human_links %>%
  dplyr::filter(peak_name %in% common_peaks) %>%
  inner_join(chimp_links, by = "peak_name", suffix = c("_human", "_chimp")) %>%
  mutate(classification = if_else(gene_human == gene_chimp, "Same gene", "Different genes"))
# Peaks Human only
human_only_peaks <- human_links %>%
  dplyr::filter(!peak_name %in% chimp_links$peak_name) %>%
  mutate(gene_human = gene,
         classification = "Human only")
# Peaks Chimp only
chimp_only_peaks <- chimp_links %>%
  dplyr::filter(!peak_name %in% human_links$peak_name) %>%
  mutate(gene_chimp = gene,
         classification = "Chimp only")
# Combine all categories
peak_summary <- bind_rows(peak_classification, human_only_peaks, chimp_only_peaks)
# Consolidate peak classifications by peak_name
# If any row for a given peak_name has "Same gene", classify the peak as "Same gene"
consolidated_peaks <- peak_summary %>%
  group_by(peak_name) %>%
  summarise(
    gene_human_present = any(!is.na(gene_human)),
    gene_chimp_present = any(!is.na(gene_chimp)),
    same_gene_in_both = any(classification == "Same gene"),
    different_genes = all(classification == "Different genes")
  ) %>%
  mutate(
    final_classification = case_when(
      same_gene_in_both ~ "Same gene",
      different_genes ~ "Different genes",
      gene_human_present & !gene_chimp_present ~ "Human only",
      !gene_human_present & gene_chimp_present ~ "Chimp only"
    )
  )
# Count the occurrences of each classification
classification_counts <- consolidated_peaks %>%
  group_by(final_classification) %>%
  summarise(count = n())
classification_counts <- classification_counts %>%
  mutate(category_percent = count / sum(count) )
classification_counts$final_classification <- factor(classification_counts$final_classification, levels = rev(c('Same gene', 'Different genes', 'Human only', 'Chimp only')))

gray_palette <- scales::grey_pal()(length(classification_counts$final_classification)) 
panelH <- ggplot(classification_counts, aes(x = "", y = category_percent, fill = final_classification)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_y_continuous(labels = scales::percent_format(), expand = c(0,0)) +
  scale_fill_manual(values = gray_palette)+
  labs(y = "Percent of linked peaks", x = NULL, fill = "Classification") +
  theme_basic_smallest() +
  theme(axis.ticks.x = element_blank(),
        legend.position = 'bottom', 
        legend.key.size = unit(c(0.1),'in'),
        legend.text = element_text(size = 6),
        plot.margin = unit(c(0.1,0,0,0),'in')
  )+
  coord_flip() +
  guides(fill = guide_legend(reverse = TRUE))

total_peaks = union(cicero_links[[cf_index]]$human$peak_name, cicero_links[[cf_index]]$chimp$peak_name)
if (length(total_peaks) != sum(classification_counts$count)){
  stop('Number of peaks in plot is not equal to union of human and chimp linked peaks.')
}

##Panel I: Cicero plot across thresholds----
de_genes_up = de_genes_list$D40_100$DA_neurons$human_vs_chimp$ID[de_genes_list$D40_100$DA_neurons$human_vs_chimp$logFC > 0]
de_genes_down = de_genes_list$D40_100$DA_neurons$human_vs_chimp$ID[de_genes_list$D40_100$DA_neurons$human_vs_chimp$logFC < 0]
de_genes_all =  de_genes_list$D40_100$DA_neurons$human_vs_chimp$ID
all_genes_up = all_genes_list$D40_100$DA_neurons$human_vs_chimp$ID[all_genes_list$D40_100$DA_neurons$human_vs_chimp$logFC > 0]
all_genes_down = all_genes_list$D40_100$DA_neurons$human_vs_chimp$ID[all_genes_list$D40_100$DA_neurons$human_vs_chimp$logFC < 0]
non_de_genes_up = all_genes_up[all_genes_up %!in% de_genes_all]
non_de_genes_down = all_genes_down[all_genes_down %!in% de_genes_all]

#Summarize concordance for each group of DE genes
groups = c('DE up', 'DE down', 'Non-DE') #note all categories are from the perspective of human
groups_genes = list('DE up' = de_genes_up, 'DE down' = de_genes_down, 'Non-DE' = non_de_genes_down)
species_percentages = list()
species_numbers = list()
species_genes = list()
for (i in seq_along(coaccess_filters)){
  cf = coaccess_filters[i]
  # Initialize lists for each coaccess_filter
  species_percentages[[i]] = list()
  species_numbers[[i]] = list()
  species_genes[[i]] = list()
  for (sp in cicero_species){
    sp_numbers = list()
    sp_percentages = list()
    sp_genes = list()
    for (j in seq_along(groups)){
      group = groups[j]
      #Filter and merge data for up-regulated DE genes
      group_peaks <- cicero_links[[i]][[sp]] %>%
        dplyr::filter(gene %in% groups_genes[[group]]) 
      scores_df <- dars_df[[coi]][[sp]] %>%
        dplyr::filter(name %in% group_peaks$peak_name) %>%
        dplyr::select(z.std, name,category) %>%
        mutate(Group = group)
      total_peaks <- length(unique(scores_df$name))
      genes_df <- group_peaks %>%
        dplyr::filter(peak_name %in% scores_df$name)
      total_genes = length(unique(genes_df$gene))
      group_genes = data.frame(
        Group = group,
        num_genes = total_genes
      )
      sp_genes = rbind(sp_genes,group_genes)
      positive_peaks <- sum(scores_df$z.std > 0)
      negative_peaks <- sum(scores_df$z.std < 0)
      group_numbers = data.frame(
        Group = group,
        Number_Positive = positive_peaks,
        Number_Negative = negative_peaks
      )
      sp_numbers = rbind(sp_numbers, group_numbers)
      group_percentages = data.frame(
        Group = group,
        Percent_Positive = (positive_peaks / total_peaks) * 100,
        Percent_Negative = (negative_peaks / total_peaks) * 100
      )
      sp_percentages = rbind(sp_percentages, group_percentages)
    }
    species_percentages[[i]][[sp]] = sp_percentages
    species_numbers[[i]][[sp]] = sp_numbers
    species_genes[[i]][[sp]]= sp_genes
  }
}
saveRDS(species_percentages,paste0(folder,'cicero_species_percentages.rds'))
saveRDS(species_numbers,paste0(folder,'cicero_species_numbers.rds'))
saveRDS(species_genes,paste0(folder,'cicero_species_genes.rds'))

#Concordance across coaccess thresholds
percent_de_linked = c()
percent_concordant = c()
for (i in seq_along(coaccess_filters)){
  cf <- coaccess_filters[i]
  #Percent of DE genes with linked DA peaks
  de_genes_with_links = c()
  de_genes_with_links_both_sp = c()
  for (sp in cicero_species){
    de_genes_cicero_links <- cicero_links[[i]][[sp]] %>%
      dplyr::filter(gene %in% de_genes_all) %>%
      dplyr::filter(both_sig == TRUE) #%>%
    #dplyr::filter(concordant == TRUE)
    de_genes_with_links[[sp]] = unique(de_genes_cicero_links$gene)
    de_genes_with_links_both_sp = c(de_genes_with_links_both_sp,de_genes_with_links[[sp]])
  }
  de_genes_with_links_both_sp = unique(de_genes_with_links_both_sp)
  percent_de_linked[i] = length(de_genes_with_links_both_sp)/length(de_genes_all)*100
  #Percent of linked DARs that are concordant with DE up genes (across species)
  #Note that DE categories and z scores are from perspective of human so need to flip it for chimp
  sp_pos = 0
  sp_neg = 0
  #Human
  num_pos = species_numbers[[i]]$human[species_numbers[[i]][[sp]]$Group == "DE up", "Number_Positive"]
  num_neg = species_numbers[[i]]$human[species_numbers[[i]][[sp]]$Group == "DE up", "Number_Negative"]
  sp_pos = sp_pos + num_pos
  sp_neg = sp_neg + num_neg
  #Chimp
  num_pos = species_numbers[[i]]$chimp[species_numbers[[i]][[sp]]$Group == "DE down", "Number_Negative"] 
  num_neg = species_numbers[[i]]$chimp[species_numbers[[i]][[sp]]$Group == "DE down", "Number_Positive"]
  sp_pos = sp_pos + num_pos
  sp_neg = sp_neg + num_neg
  percent_concordant[i] = sp_pos/(sp_pos + sp_neg)*100
}

df_panelI <- data.frame(coaccess_filters, percent_concordant, percent_de_linked)
df_long_panelI <- df_panelI %>%
  pivot_longer(cols = c(percent_concordant, percent_de_linked),
               names_to = "metric", values_to = "value")
panelI <- ggplot(df_long_panelI, aes(x = coaccess_filters, y = value, color = metric)) +
  geom_line() +
  geom_point() +
  labs(x = "Co-accessibility score threshold", y = "Percent", color = "Metric") +
  scale_color_manual(values = c("percent_concordant" = "darkred", "percent_de_linked" = "black"),
                     labels = c("percent_concordant" = "Percent concordant DARs",
                                "percent_de_linked" = "Percent of DE genes \nwith linked DARs")) +
  scale_y_continuous(limits = c(0,80),expand = c(0,5))+
  scale_x_continuous(limits = c(0,0.3))+
  theme_basic_smallest()+
  theme(legend.position = c(0.5,0.5),
        plot.margin = unit(c(0,0,0,0),'in'),
        legend.key.size = unit(c(0.1),'in'),
        legend.text = element_text(size = 6)) 

##PanelJ: Cicero concordance barplot----
cf_index = which(coaccess_filters==cf_filter_to_plot)
human_percents <- tidyr::gather(species_percentages[[cf_index]]$human, key = "Score_Type", value = "Percent", -Group)
human_percents$Score_Type = factor(human_percents$Score_Type, levels = c('Percent_Positive','Percent_Negative'))
human_percents_up = human_percents[human_percents$Group == 'DE up',]
human_percents_up$Group = 'Human \nDE up'
chimp_percents <- tidyr::gather(species_percentages[[cf_index]]$chimp, key = "Score_Type", value = "Percent", -Group)
chimp_percents$Score_Type = factor(chimp_percents$Score_Type, levels = c('Percent_Positive','Percent_Negative'))
chimp_percents_up = chimp_percents[chimp_percents$Group == 'DE down',] #DE is from perspective of human so need DE down for chimp-up genes
chimp_percents_up$Group = 'Chimp \nDE up'
chimp_percents_up$Score_Type <- ifelse(chimp_percents_up$Score_Type == "Percent_Positive", #Switch to be point of view of chimp
                                       "Percent_Negative", 
                                       "Percent_Positive")
plot_data <- rbind(human_percents_up, chimp_percents_up)
plot_data$Group <- factor(plot_data$Group, levels = c('Human \nDE up','Chimp \nDE up'))
panelJ <- ggplot(plot_data, aes(x = Group, y = Percent, fill = Score_Type)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_basic() +
  labs(y = "Percent of linked DARs") +
  scale_fill_manual(values = c("Percent_Positive" = "firebrick", "Percent_Negative" = "steelblue"), labels = c('DAR Up','DAR Down')) +
  scale_y_continuous(expand = c(0, 0)) +
  theme_basic_smallest()+
  theme(legend.title = element_blank(),
        axis.title.x = element_blank(),
        legend.position  = c(0.8,0.6), 
        legend.key.size = unit(c(0.1),'in'),
        legend.text = element_text(size = 6),
        plot.margin = unit(c(0.1,0,0,0), 'in')
  )
panelHIJ <- panelH / panelI / panelJ + plot_layout(heights = c(1,4,4))
panelHIJ
ggsave(paste0(folder,'panelHIJ.pdf'), width = 2, height = 5)

panelBEHIJ <- panelB / categories_bar / numbers_bar / panelH / panelI / panelJ + plot_layout(heights = c(2,0.75,3.6,0.8,4.8,4.8))
panelBEHIJ
ggsave(paste0(folder,'panelBEHIJ.pdf'), height = 7, width = 2.1, units = 'in')

#Load(for genome browser plots)----
multi_species <- qread(paste0(object_folder,'multi_species.rds'))
res_procs <- readRDS(paste0(rna_folder,'res_procs.rds'))
degs_polarized <- readRDS(paste0(rna_folder,'degs_polarized.rds'))
hs_tracks <- readRDS(paste0(folder,'hs_tracks.rds'))
cs_tracks <- readRDS(paste0(folder,'cs_tracks.rds'))

##Panel G: enrichment of evolutionary signatures----
# Calculate odds ratios and confidence intervals for each signature
results_df <- data.frame(Track = character(), OddsRatio = numeric(), LowerCI = numeric(), UpperCI = numeric())
for (track in names(contingency_tables)) {
  res <- OddsRatio(contingency_tables[[track]], conf.level = 0.95)
  results_df <- rbind(results_df, data.frame(Track = track, OddsRatio = res[1], LowerCI = res[2], UpperCI = res[3]))
}
results_df$PVal <- fisher_test_result
results_df$FDR <- p.adjust(results_df$PVal, method = "BH")
results_df$Significance <- ifelse(results_df$FDR < 0.05, 'YES', 'NO')
results_df$Significance <- factor(results_df$Significance, levels = c('YES','NO'))
results_df <- results_df[order(results_df$OddsRatio), ]
results_df$Track[results_df$Track == 'hCONDELs_indelsized'] = 'hCONDELs\n_indelsized'
results_df$Track <- factor(results_df$Track, levels = results_df$Track)
panelG <- ggplot(results_df, aes(x = OddsRatio, y = Track, color = Significance)) +
  geom_point() +
  geom_errorbarh(aes(xmin = LowerCI, xmax = UpperCI), height = 0.2) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  labs(x = "Enrichment (odds ratio)", y = NULL) +
  scale_color_manual(
    values = c('YES' = "darkred", 'NO' = "gray50"),
    labels = c('YES' = bquote("FDR" ~ "\n<" ~ alpha), 'NO' = 'ns')
  ) +
  theme_basic_smallest()+
  theme(legend.text = element_text(size = 6),
        plot.margin = unit(c(0,0,0,0),'in'),
        legend.position = 'left')

##PanelK: GABRG3 example----
peaks_to_label = c()
gabrg3_peak_index = 10
species_genomes = c('hg38','Chimp','Rhesus'); names(species_genomes) = species
peak_name = peaks_overlap_tracks_human$hIns$name[gabrg3_peak_index]
peaks_to_label = c(peaks_to_label, peak_name)
insertion_length = tracks_overlap_peaks_human$hIns$Length[gabrg3_peak_index]
extend_up = 1000
extend_down = 2600

sp_plots = c()
for (sp in species){
  Idents(multi_species[[sp]]) <- multi_species[[sp]]$cell_type
  DefaultAssay(multi_species[[sp]]) <- 'peaks_consensus'
  peak_granges = species_allcon_peaks[[sp]][peak_name]
  peak_string = paste0(seqnames(peak_granges),'-',start(peak_granges),'-',end(peak_granges))
  genome(peak_granges) = species_genomes[[sp]]
  multi_species[[sp]]@assays$peaks_consensus@meta.features$interest = FALSE
  multi_species[[sp]]@assays$peaks_consensus@meta.features[peak_string,'interest'] = TRUE
  if (sp == 'human'){
    sp_plots[[sp]] <- CoveragePlot(
      object = multi_species[[sp]],
      assay = 'peaks_consensus',
      region = peak_granges,
      idents = c('DA_neurons'),
      extend.upstream = extend_up ,
      extend.downstream = extend_down + insertion_length,
      peaks.group.by = 'interest',
      ranges = hs_tracks$hIns,
      ranges.title = 'Insertions',
      scale.factor = 1e7,
      ymax = 50,
      annotation = TRUE
    ) & scale_fill_manual(values = celltype_colors[coi]) &
      theme(legend.position = 'none',
            plot.margin = unit(c(0,0,0,0),'in'),
            axis.title = element_text(size = 6),
            axis.text = element_text(size = 6),
            axis.title.y = element_text(angle = 0),
            strip.text.y.left = element_blank(),
            strip.background = element_blank())
  } else sp_plots[[sp]] <- p1 <- CoveragePlot(
    object = multi_species[[sp]],
    assay = 'peaks_consensus',
    region = peak_granges,
    idents = c('DA_neurons'),
    extend.upstream = extend_down,
    extend.downstream = extend_up,
    peaks.group.by = 'interest',
    scale.factor = 1e7,
    ymax = 50,
    annotation = TRUE
  ) & scale_fill_manual(values = celltype_colors[coi]) &
    theme(legend.position = 'none',
          plot.margin = unit(c(0,0,0,0),'in'),
          axis.title = element_text(size = 6),
          axis.text = element_text(size = 6),
          axis.title.y = element_text(angle = 0),
          strip.text.y.left = element_blank(),
          strip.background = element_blank())
}
range_others = extend_up+extend_down+width(peak_granges)
range_human = range_others + insertion_length
ratio = range_others/range_human
p1 <- sp_plots$human + plot_spacer() + plot_layout(widths = c(1,0))
p2 <- sp_plots$chimp + plot_spacer() + plot_layout(widths = c(ratio,1-ratio))
p3 <- sp_plots$rhesus + plot_spacer() + plot_layout(widths = c(ratio,1-ratio))
panelK_atac <- p1 / p2 / p3
ggsave(paste0(folder,'panelK_atac.pdf'), height = 2.75, width = 4, units = 'in')

#Gene plot
genes_sig = c('GABRG3')
data = extractData(res_procs$D40_100,coi)
filtered_data <- data %>% 
  dplyr::filter(species != 'orangutan') %>%
  droplevels()
da <- plotSelectedGenesList_withPolarize(genes_sig, filtered_data, coi, degs_polarized$D40_100, colors_species,colors_polarize)
panelK_rna <- da$GABRG3 + theme(axis.title.x = element_text(size = 6))
ggsave(paste0(folder,'panelK_rna.pdf'), panelK_rna,height = 2, width = 1, units = 'in')

##PanelL: NRN1 example----
species = c('human','chimp','rhesus')
species_genomes = c('hg38','Chimp','Rhesus'); names(species_genomes) = species
extend_up = 2000
extend_down = 2000
gene = 'NRN1'

sp_plots = list()
extended_range = c()
for (sp in species){
  Idents(multi_species[[sp]]) <- multi_species[[sp]]$cell_type
  DefaultAssay(multi_species[[sp]]) <- 'peaks_consensus'
  #Get gene range
  gene_range <- subset(sp_annotation[[sp]], gene_name == gene)
  extended_range[[sp]] <- GRanges(
    seqnames = unique(seqnames(gene_range)),
    ranges = IRanges(
      start = min(start(gene_range)) - extend_up,
      end = max(end(gene_range)) + extend_down
    ),
    strand = unique(strand(gene_range)))
  #Find DARs that overlap
  result = findOverlaps(dars_gr[[coi]][[sp]],extended_range[[sp]])
  dars_in_range = dars_gr[[coi]][[sp]][queryHits(result)]
  nrn1_peak_names = dars_in_range$name
  peaks_to_label = c(peaks_to_label, nrn1_peak_names)
  #Get coords for overlapping DARs
  multi_species[[sp]]@assays$peaks_consensus@meta.features$interest = FALSE
  if (length(dars_in_range) > 0){ 
    peak_strings = paste0(seqnames(dars_in_range),'-',start(dars_in_range),'-',end(dars_in_range))
    multi_species[[sp]]@assays$peaks_consensus@meta.features[peak_strings,'interest'] = TRUE
  } else {print('No DARs in range.')}
  
  if (sp == 'human'){
    sp_plots[[sp]] <- CoveragePlot(
      object = multi_species[[sp]],
      assay = 'peaks_consensus',
      region = gene,
      idents = c('DA_neurons'),
      extend.upstream = extend_up,
      extend.downstream = extend_down,
      peaks.group.by = 'interest',
      # ranges = hs_tracks$hIns,
      # ranges.title = 'Insertions',
      scale.factor = 1e7,
      ymax = 100
    ) & scale_fill_manual(values = celltype_colors[coi]) &
      theme(legend.position = 'none',
            plot.margin = unit(c(0,0,0,0),'in'),
            axis.title = element_text(size = 6),
            axis.text = element_text(size = 6),
            axis.title.y = element_text(angle = 0),
            strip.text.y.left = element_blank(),
            strip.background = element_blank())
  } else {
    sp_plots[[sp]] <- p1 <- CoveragePlot(
      object = multi_species[[sp]],
      assay = 'peaks_consensus',
      region = gene,
      idents = c('DA_neurons'),
      extend.upstream = extend_up,
      extend.downstream = extend_down,
      peaks.group.by = 'interest',
      scale.factor = 1e7,
      ymax = 100
    ) & scale_fill_manual(values = celltype_colors[coi]) &
      theme(legend.position = 'none',
            plot.margin = unit(c(0,0,0,0),'in'),
            axis.title = element_text(size = 6),
            axis.text = element_text(size = 6),
            axis.title.y = element_text(angle = 0),
            strip.text.y.left = element_blank(),
            strip.background = element_blank())
  }
}
p1 <- sp_plots$human 
p2 <- sp_plots$chimp
p3 <- sp_plots$rhesus 
panelL_atac <- p1 / p2 /p3
panelL_atac
ggsave(paste0(folder,'panelL_atac.pdf'), height = 2.75, width = 4, units = 'in')

#Gene plot
genes_sig = c('NRN1')
data = extractData(res_procs$D40_100,coi)
filtered_data <- data %>% 
  dplyr::filter(species != 'orangutan') %>%
  droplevels()
da <- plotSelectedGenesList_withPolarize(genes_sig, filtered_data, coi, degs_polarized$D40_100, colors_species,colors_polarize)
panelL_rna <- da$NRN1 + theme(axis.title.x = element_text(size = 6))
ggsave(paste0(folder,'panelL_rna.pdf'), panelL_rna,height = 2, width = 1, units = 'in')

rm(multi_species)

##Panel F: volcano----
peaks_to_label = unique(peaks_to_label)
panelF <- plotMyVolcano(res_dl_da, assay = coi, coef = con, pt.size = 0.5, label_genes = peaks_to_label, cutoff = 0.1)  +
  scale_color_manual(values = c("grey", "darkred")) +
  theme_basic_smallest()+
  theme(plot.title = element_blank(),
        legend.position = 'none',
        plot.margin = unit(c(0,0,0,0),'in'))
panelFG <- panelF + panelG
panelFG
ggsave(paste0(folder,'panelFG.pdf'), width = 5, height = 2, units = 'in')

#Table S5----
file_path = paste0(folder,'TableS5.xlsx')
wb <- createWorkbook()
table_legend = paste0("Dreamlet results for human vs chimp contrast for all peaks meeting expression cutoffs in ",
                      coi, " including peak coordinates for human, chimp, and macaque,CrossPeak categories (in the case of multiple categories the first is for human lifted to chimp and the second is for human lifted to macaque), and overlaps with evolutionary signatures on hg38 and panTro6.")
#DA neurons
peaks_categories <- data.frame(name = character(), category = character(), stringsAsFactors = FALSE)
for (track in names(peaks_overlap_tracks_human)) { #Summarize peak categories for DA neurons to add to table
  gr <- peaks_overlap_tracks_human[[track]]
  gr_df <- as.data.frame(gr, row.names = NULL)
  gr_df <- gr_df %>%
    dplyr::select(name, category)
  gr_df$signature <- track
  peaks_categories <- bind_rows(peaks_categories, gr_df)
}
peaks_categories_summary_hg38 <- peaks_categories %>%
  group_by(name) %>%
  summarize(
    signature_overlap_hg38 = paste(unique(signature[!is.na(signature)]), collapse = ";"),
    .groups = 'drop'
  )
peaks_categories <- data.frame(name = character(), category = character(), stringsAsFactors = FALSE)
for (track in names(peaks_overlap_tracks_chimp)) {
  gr <- peaks_overlap_tracks_chimp[[track]]
  gr_df <- as.data.frame(gr, row.names = NULL)
  gr_df <- gr_df %>%
    dplyr::select(name, category)
  gr_df$signature <- track
  peaks_categories <- bind_rows(peaks_categories, gr_df)
}
peaks_categories_summary_panTro6 <- peaks_categories %>%
  group_by(name) %>%
  summarize(
    signature_overlap_panTro6 = paste(unique(signature[!is.na(signature)]), collapse = ";"),
    .groups = 'drop'
  )
coi_tab <- allrs_df[[coi]]$human %>%
  mutate(cell_type = assay) %>%
  mutate(CrossPeak_category = category,seqnames_human = seqnames, start_human = start, end_human = end) %>%
  dplyr::select(-assay, -orig_name, -coords_name, -width, -strand, -seqnames, -start, -end) %>%
  left_join(allrs_df[[coi]]$chimp %>%
              dplyr::select(name, seqnames, start, end) %>%
              mutate(seqnames_chimp = seqnames, start_chimp = start, end_chimp = end) %>%
              dplyr::select(-seqnames, -start, -end), by = "name") %>%
  left_join(allrs_df[[coi]]$rhesus %>%
              dplyr::select(name, seqnames, start, end) %>%
              mutate(seqnames_macaque = seqnames, start_macaque = start, end_macaque = end) %>%
              dplyr::select(-seqnames, -start, -end), by = "name") %>%
  dplyr::select(-name, -category_orig, -category, -pass, -summit_diff) %>%
  left_join(peaks_categories_summary_hg38, by = c("ID" = "name")) %>%
  left_join(peaks_categories_summary_panTro6, by = c("ID" = "name")) %>%
  mutate(signature_overlap_hg38 = if_else(adj.P.Val > 0.1, "not tested", signature_overlap_hg38)) %>%
  mutate(signature_overlap_panTro6 = if_else(adj.P.Val > 0.1, "not tested", signature_overlap_panTro6))
addWorksheet(wb, coi)
writeData(wb, coi, table_legend, startRow = 1)  
writeData(wb, coi, coi_tab, startRow = 2) 

celltypes = c('DA_STN_neurons_immature','Ventral_FB_MB_progenitors')
for (celltype in celltypes){
  table_legend = paste0("Dreamlet results for human vs chimp contrast for all peaks meeting expression cutoffs in ",
                      celltype, " including peak coordinates for human, chimp, and macaque and CrossPeak categories (in the case of multiple categories the first is for human lifted to chimp and the second is for human lifted to macaque).")
  celltype_tab <- allrs_df[[celltype]]$human %>%
    mutate(cell_type = assay) %>%
    mutate(CrossPeak_category = category,seqnames_human = seqnames, start_human = start, end_human = end) %>%
    dplyr::select(-assay, -orig_name, -coords_name, -width, -strand, -seqnames, -start, -end) %>%
    left_join(allrs_df[[celltype]]$chimp %>%
                dplyr::select(name, seqnames, start, end) %>%
                mutate(seqnames_chimp = seqnames, start_chimp = start, end_chimp = end) %>%
                dplyr::select(-seqnames, -start, -end), by = "name") %>%
    left_join(allrs_df[[celltype]]$rhesus %>%
                dplyr::select(name, seqnames, start, end) %>%
                mutate(seqnames_macaque = seqnames, start_macaque = start, end_macaque = end) %>%
                dplyr::select(-seqnames, -start, -end), by = "name") %>%
    dplyr::select(-name, -category_orig, -category, -pass, -summit_diff)
  addWorksheet(wb, celltype)
  writeData(wb, celltype, table_legend, startRow = 1)  
  writeData(wb, celltype, celltype_tab, startRow = 2) 
}
#DA neurons species-only peaks
for (sp in c('human','chimp')){
  table_legend = paste0("Information for species-only peaks meeting expression cutoffs in DA neurons in ",
                        sp, " including CrossPeak categories (in the case of multiple categories the first is for human lifted to chimp and the second is for human lifted to macaque).")
sp_tab <- spec_peaks_df[[sp]] %>%
  mutate(CrossPeak_category = category,!!paste0('seqnames_', sp) := seqnames,!!paste0('start_', sp) := start,!!paste0('end_', sp) := end) %>%
  dplyr::select(-orig_name, -coords_name, -width, -strand, -seqnames, -start, -end,-category) %>%
  left_join(peaks_categories_summary_hg38, by = c("name")) %>%
  left_join(peaks_categories_summary_panTro6, by = c("name"))
tab_title = paste0(str_to_title(sp),'-only')
addWorksheet(wb, tab_title)
writeData(wb, tab_title, table_legend, startRow = 1)  
writeData(wb, tab_title, sp_tab, startRow = 2) 
}
saveWorkbook(wb, file_path, overwrite = TRUE)

#Print for text----
print(paste0('Total cells: ',dim(multi_integrated)[2]))
print(paste0('Human =  ',sum(multi_integrated$species=='human')))
print(paste0('Chimp =  ',sum(multi_integrated$species=='chimp')))
print(paste0('Macaque =  ',sum(multi_integrated$species=='macaque')))
print(paste0('Orangutan =  ',sum(multi_integrated$species=='orangutan')))

#Panel I percents
print(paste0('Percent concordant range: ',df_panelI$percent_concordant[1], ' to ', df_panelI$percent_concordant[length(df_panelI$percent_concordant)]))
print(paste0('Percent DE linked range: ',df_panelI$percent_de_linked[1], ' to ', df_panelI$percent_de_linked[length(df_panelI$percent_de_linked)]))

#Number of genes for panelJ legend (first category is DE up)
print(paste0('Number of DE up genes for human: ',species_genes[[cf_index]]$human$num_genes[1])) 
print(paste0('Number of linked DARs for human: ',species_numbers[[cf_index]]$human$Number_Positive[1] + species_numbers[[cf_index]]$human$Number_Negative[1]))
print(paste0('Number of DE up genes for chimp: ',species_genes[[cf_index]]$chimp$num_genes[1])) 
print(paste0('Number of linked DARs for chimp: ',species_numbers[[cf_index]]$chimp$Number_Positive[1] + species_numbers[[cf_index]]$chimp$Number_Negative[1]))

#DARs overlapping promoters - use refseq annotations for highest accuracy here (not used above because need fair species comparison)
df <- as.data.frame(refseq_annotations$human)
df$gene_biotype <- as.character(df$gene_biotype)
# Group by gene_name and assign 'protein_coding' to gene_biotype if any row has type == 'exon'
df <- df %>%
  group_by(gene_name) %>%
  mutate(gene_biotype = ifelse(any(type == 'CDS'), 'protein_coding', gene_biotype)) %>%
  ungroup()
human_annotation <- makeGRangesFromDataFrame(df, keep.extra.columns = TRUE)
tss <- GetTSSPositions(human_annotation)
promoter_regions = expandRange(tss, upstream = 1000, downstream = 0)
result = findOverlaps(dars_gr[[coi]]$human, promoter_regions, minoverlap = peak_width/2)
gene_name = promoter_regions[subjectHits(result)]
peak_name = dars_gr[[coi]]$human[queryHits(result)]
result_df <- as.data.frame(result)
result_df <- result_df %>%
  mutate(gene_name = gene_name$gene_name) %>%
  mutate(gene_up = case_when(
    gene_name %in% de_genes_up ~ TRUE,
    gene_name %in% de_genes_down ~ FALSE,
    TRUE ~ NA
  )) %>%
  mutate(name = peak_name$name) %>%
  left_join(dars_df[[coi]]$human, by = 'name') %>%
  mutate(peak_up = ifelse(logFC > 0, TRUE, FALSE)) %>%
  mutate(concordant = ifelse(gene_up == peak_up, TRUE, FALSE))
concordant_promoters <- result_df %>%
  dplyr::filter(concordant==TRUE)
saveRDS(concordant_promoters, paste0(folder,'concordant_promoters.rds'))
num_genes =  length(unique(concordant_promoters$gene_name))
num_con_promoters = sum(result_df$concordant==TRUE, na.rm = TRUE)
num_noncon_promoters = sum(result_df$concordant==FALSE, na.rm = TRUE)
num_total_promoters = num_con_promoters + num_noncon_promoters
print(paste0('Number of concordant peak-promoter pairs: ', num_con_promoters))
print(paste0('Number of DE gene/DA peak peak-promoter pairs: ', num_total_promoters))
print(paste0('Number of genes represented by concordant peak-promoter pairs: ', num_genes))

#Concordance ranking
cf_index = which(coaccess_filters==cf_filter_to_plot)
current_links =  cicero_links[[cf_index]]$human
both_sig <- current_links %>%
  dplyr::filter(both_sig == TRUE)
summary_table <- both_sig %>%
  dplyr::group_by(gene) %>%
  dplyr::summarise(
    num_concordant = sum(concordant == TRUE),
    num_non_concordant = sum(concordant == FALSE)
  ) %>%
  mutate(more_concordant = num_concordant - num_non_concordant) %>%
  dplyr::filter(more_concordant>0) %>%
  dplyr::arrange(desc(more_concordant)) %>%
  dplyr::mutate(rank = dplyr::min_rank(dplyr::desc(more_concordant)))
nrn1_rank <- summary_table %>%
  dplyr::filter(gene == "NRN1") %>%
  dplyr::select(rank)
print(paste0('NRN1 rank: ',nrn1_rank))

#p values for examples
print('GABRG3 peak: ')
dars_dreamlet[[coi]][dars_dreamlet[[coi]]$ID == peaks_overlap_tracks_human$hIns[gabrg3_peak_index]$name,]
print('GABRG3 gene: ')
de_genes_list$D40_100$DA_neurons$human_vs_chimp[de_genes_list$D40_100$DA_neurons$human_vs_chimp$ID == 'GABRG3',]

print('EPHA10 gene: ')
de_genes_list$D40_100$DA_neurons$human_vs_chimp[de_genes_list$D40_100$DA_neurons$human_vs_chimp$ID == 'EPHA10',]
print('NRN1 gene: ')
de_genes_list$D40_100$DA_neurons$human_vs_chimp[de_genes_list$D40_100$DA_neurons$human_vs_chimp$ID == 'NRN1',]
print('NTNG2 gene: ')
de_genes_list$D40_100$DA_neurons$human_vs_chimp[de_genes_list$D40_100$DA_neurons$human_vs_chimp$ID == 'NTNG2',]
print('NRN1 peaks: ')
dars_dreamlet[[coi]][dars_dreamlet[[coi]]$ID %in% nrn1_peak_names, ]

#Percent of original peaks preserved
peaks <- readRDS(paste0(peaks_folder, "peaks.rds"))
species_allcon_peaks <- readRDS(paste0(peaks_folder,'species_allcon_peaks.rds'))
species_only_peaks <- readRDS(paste0(peaks_folder,'species_only_peaks.rds'))
species_prefixes = c('Hu','Ch','Rh'); names(species_prefixes) = species
percentage_retained <- c()
for (sp in species){
  original_peaks <- mcols(peaks[[sp]])$name
  species_peaks_orig_names <- mcols(species_allcon_peaks[[sp]])$orig_name
  split_orig_names <- strsplit(species_peaks_orig_names, ";")
  peaks_in_species_allcon <- unique(unlist(lapply(split_orig_names, function(x) {
    grep(paste0("^", species_prefixes[[sp]],"_peak_"), x, value = TRUE)
  })))
  peaks_in_species_only <- species_only_peaks[[sp]]$orig_name
  num_original_peaks <- length(original_peaks)
  num_retained_peaks_con <- sum(original_peaks %in% peaks_in_species_allcon)
  num_retained_peaks_only <- sum(original_peaks %in% peaks_in_species_only)
  percentage_retained[[sp]] <- ((num_retained_peaks_con + num_retained_peaks_only) / num_original_peaks) * 100
}
print('Percent peaks retained in each species: ')
percentage_retained