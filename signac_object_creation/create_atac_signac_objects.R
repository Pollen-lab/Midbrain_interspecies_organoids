rm(list = ls()); gc()
folder = 'Midbrain/Object_creation/V5/'; dir.create(folder)
species_peak_folder = 'Midbrain/Consensus_peaks/'
con_peak_folder = paste0(species_peak_folder,'/D40_100/V8/')
rna_folder = '/media/jenelle/4TB_disk/Dropbox/Analysis/Dreamlet_R_4.4.0/Midbrain/Ancestral_genome/D40_D100_D80/V22/'
source("/media/jenelle/4TB_disk/Dropbox/R_analysis/MyFunctions.R")
source("/media/jenelle/4TB_disk/Dropbox/R_analysis/MyPlottingFunctions.R")
source("/media/jenelle/4TB_disk/Dropbox/R_analysis/MySeuratFunctions.R")
source("/media/jenelle/4TB_disk/Dropbox/R_analysis/MyGenomicRangesFunctions.R")
library(Seurat)
library(Signac)
library(dplyr)
library(GenomicRanges)
library(SingleCellExperiment)
library(Matrix)
library(tidyverse)
library(EnsDb.Hsapiens.v86)
library(BSgenome.Hsapiens.UCSC.hg38)
library(biovizBase)
library(stringr)
library(future)
library(qs)
library(patchwork)
figure_folder = paste0(folder,'Figures/'); dir.create(figure_folder)
info <- capture.output(sessionInfo()); writeLines(info, paste0(folder, "session_info.txt"))
fragment_dir = 'Midbrain/ATAC_fragment_files/D40_100/Fragment_files_prefix/'
human_macs2_temp = 'Midbrain/Human_Macs2_temp/'; dir.create(human_macs2_temp)
chimp_macs2_temp = 'Midbrain/Chimp_Macs2_temp/'; dir.create(chimp_macs2_temp)
rhesus_macs2_temp = 'Midbrain/Rhesus_Macs2_temp/'; dir.create(rhesus_macs2_temp)
human_annotation_file = 'Genome_annotations/Human_gencodev33.gtf'
chimp_annotation_file = 'Genome_annotations/Chimp_gencodev33.gtf'
rhesus_annotation_file = 'Genome_annotations/Rhesus_gencodev33.gtf'
annotation_files = list(human = human_annotation_file, chimp = chimp_annotation_file, rhesus = rhesus_annotation_file)
macs2_path = "/home/jenelle/anaconda3/envs/signac_macs2/bin/macs2"
half_width = 250 #fixed width peaks - added 250 on either side of the summit for total 501 bp peak
species = c("human","chimp","rhesus")
species_names = c(human = "human",chimp = "chimp",rhesus = "macaque")
colors_species = c(human = '#F59121',chimp = '#3957A6',rhesus = '#7E2859')
#only works when running in terminal, not in rstudio (comment out for rstudio)
plan("multicore", workers = 14)
options(future.globals.maxSize = 120 * 1024 ^ 3) # for 120 out of 128 Gb RAM

#Load files
rna_sce <- readRDS(paste0(rna_folder,'sce.rds')) #use the one from dreamlet since it has sexes, etc
rna <- as.Seurat(rna_sce)
species_allcon_peaks <- readRDS(paste0(con_peak_folder,'species_allcon_peaks.rds'))
species_only_peaks <- readRDS(paste0(con_peak_folder,'species_only_peaks.rds'))
macs2_temp_folders = list(human = human_macs2_temp,chimp = chimp_macs2_temp, rhesus = rhesus_macs2_temp)

#Loop through species and make objects
for (sp in species){
    species_rna <- subset(rna, species == species_names[[sp]])
    macs2_temp = macs2_temp_folders[[sp]]
    if (file.exists(paste0(folder,sp,"_annotation.rds"))){
      annotation <- readRDS(paste0(folder,sp,"_annotation.rds"))
    } else {
      annotation <- rtracklayer::import(annotation_files[[sp]])
      if (is.null(annotation$gene_biotype)){annotation$gene_biotype = annotation$gene_type}
      annotation <- annotation[!is.na(annotation$gene_biotype)] #NAs here cause errors in TSSenrichment. Ok to remove since they are on chr we don't care about like chrM
      annotation$tx_id <- annotation$transcript_id #needed for CoveragePlot
      # mcols(annotation) <- mcols(annotation)[, c("tx_id","gene_name","gene_id","gene_biotype","type")] #select only columns that are needed as extras might cause errors
      saveRDS(annotation, paste0(folder,sp,"_annotation.rds"))
    }
  #Make fragment files
  all_files = dir(fragment_dir,'*.gz')
  frag_files = all_files[!grepl("\\.tsv\\.gz\\.tbi$", all_files)] #exclude index files
  species_files <- frag_files[grepl(str_to_upper(sp), frag_files, ignore.case = TRUE)]
  species_files = basename(species_files)
  barcode_prefixes = str_replace(species_files, "_fragments.*", "")
  if (file.exists(paste0(folder,sp,"_frags_list.rds"))){
    frags_list <- readRDS(paste0(folder,sp,"_frags_list.rds"))
    fragpath_list <- readRDS(paste0(folder,sp,"_fragpath_list.rds"))
  } else {
    frags_list = list()
    indices_without_cells = c()
    for (f in seq_along(barcode_prefixes)){
      if (sum(species_rna$orig.ident==barcode_prefixes[f]) != 0){ #make sure there are cells - some fragment files had all cells excluded
        cells = colnames(subset(species_rna, subset = orig.ident== barcode_prefixes[f]))
        frags_list[f] <- CreateFragmentObject(path = paste0(fragment_dir,species_files[f]), cells = cells, max.lines = NULL, tolerance = 0)
      } else {
        indices_without_cells = c(indices_without_cells, f)
      }
    }
    #Remove objects that were null (happens when a file has no cells)
    if (!is.null(indices_without_cells)){
      frags_list <- frags_list[-indices_without_cells]
    }
    saveRDS(frags_list, paste0(folder,sp,"_frags_list.rds"))
    fragpath = dir(fragment_dir,"*gz$",full.names=TRUE)
    fragpath_list = list()
    for (num in seq_along(barcode_prefixes)){
      fragpath_list[num] = fragpath[num]
    }
    if (!is.null(indices_without_cells)){
      fragpath_list = fragpath_list[-indices_without_cells]
    }
    saveRDS(fragpath_list, paste0(folder,sp,"_fragpath_list.rds"))
  }
  #Build chromatin assay for individual species peaks
  if (file.exists(paste0(folder, sp,'_multi.rds'))){
    multi <- readRDS(file = paste0(folder, sp,'_multi.rds'))
  } else {
    species_peaks <- import.bed(paste0(species_peak_folder,sp,'_peaks.bed'))
    multi = species_rna
    cells = colnames(multi)
    macs2_counts <- FeatureMatrix(fragments = frags_list,features = species_peaks,cells = cells);
    multi[["peaks_celltypes"]] <- CreateChromatinAssay(counts = macs2_counts,fragments = frags_list, annotation = annotation,min.cells=-1,min.features = -1, cells = cells)
    saveRDS(multi,file = paste0(folder, sp, '_multi.rds'));
  }
  #Quality metrics
  if (file.exists(paste0(folder, sp,'_multi2.rds'))){
    multi2 <- readRDS(paste0(folder, sp,'_multi2.rds'))
  } else{
    multi2 <- multi
    rm(multi); gc()
    DefaultAssay(multi2) <- "peaks_celltypes"
    multi2 <- NucleosomeSignal(multi2)
    multi2 <- TSSEnrichment(multi2)
    frag_counts <- CountFragments(fragments = fragpath_list, cells = colnames(multi2))
    multi2$nCount_frags <- frag_counts$frequency_count #works since above line returns barcodes in same order
    multi2$pct_reads_in_peaks <- multi2$nCount_peaks_celltypes/multi2$nCount_frags
    saveRDS(multi2,file = paste0(folder, sp,'_multi2.rds'));
  }
  #Build chromatin assay for consensus peaks
  if (file.exists(paste0(folder, sp,'_multi3.rds'))){
    multi3 <- readRDS(file = paste0(folder, sp,'_multi3.rds'))
  } else {
    multi3 = multi2
    rm(multi2); gc()
    cells = colnames(multi3)
    all_peaks = c(species_allcon_peaks[[sp]],species_only_peaks[[sp]])
    macs2_counts <- FeatureMatrix(fragments = frags_list,features = all_peaks,cells = cells)
    if (length(rownames(macs2_counts))!=length(all_peaks)){stop('Some peaks were excluded from FeatureMatrix - check for non standard chromosomes or duplicated peaks.')}
    multi3[["peaks_consensus"]] <- CreateChromatinAssay(counts = macs2_counts,fragments = frags_list, annotation = annotation,min.cells=-1,min.features = -1, cells = cells)
    saveRDS(multi3,file = paste0(folder, sp, '_multi3.rds'));
  }
  rm(multi); rm(multi2);rm(multi3); rm(macs2_counts); gc()
}

#Quality control----
#For each species using peaks_celltypes
multi_species <- vector(mode = "list", length = length(species)); names(multi_species) <- species
species_qplots <- vector(mode = "list", length = length(species)); names(species_qplots) <- species
for (sp in species){
  annotation_file = annotation_files[[sp]]
  multi3 <- readRDS(paste0(folder,sp,'_multi3.rds'))
  DefaultAssay(multi3) <- 'peaks_celltypes'
  multi3$pct_reads_in_peaks = multi3$pct_reads_in_peaks*100

  #Quality control
  cells_to_keep = !is.na(multi3$nCount_peaks_celltypes)
  cells_to_keep_names <- names(cells_to_keep)[cells_to_keep]
  multi3 <- subset(multi3, cells = cells_to_keep_names) #exclude cells with NA values (not sure how they got there)
  DensityScatter(multi3, x = 'nCount_peaks_celltypes', y = 'TSS.enrichment', log_x = TRUE, quantiles = TRUE)
  ggsave(paste0(figure_folder,sp, '_quality_density_scatter.png'))
  p1 <- VlnPlot(object = multi3, features = 'nCount_peaks_celltypes',pt.size = 0, cols = colors_species[[sp]],y.max = 60000, log = TRUE) +
    labs(y = 'Number of fragments\nin peaks')+
    theme_basic_smallest() +
    theme(plot.title = element_blank(),
          axis.title.x = element_blank(),
          axis.text.x = element_blank(),
          legend.position = 'none')
  p2 <- VlnPlot(object = multi3, features = 'TSS.enrichment',pt.size = 0, cols = colors_species[[sp]],y.max = 10) +
    labs(y = 'TSS enrichment')+
    theme_basic_smallest() +
    theme(plot.title = element_blank(),
          axis.title.x = element_blank(),
          axis.text.x = element_blank(),
          legend.position = 'none')
  p3 <- VlnPlot(object = multi3, features = 'nucleosome_signal',pt.size = 0, cols = colors_species[[sp]],y.max = 1.2) +
    labs(y = 'Nucleosome signal')+
    theme_basic_smallest() +
    theme(plot.title = element_blank(),
          axis.title.x = element_blank(),
          axis.text.x = element_blank(),
          legend.position = 'none')
  p4 <- VlnPlot(object = multi3, features = 'pct_reads_in_peaks',pt.size = 0, cols = colors_species[[sp]],y.max = 80) +
    labs(y = 'Percent reads in peaks')+
    theme_basic_smallest() +
    theme(plot.title = element_blank(),
          axis.title.x = element_blank(),
          axis.text.x = element_blank(),
          legend.position = 'none')

  species_qplots[[sp]] <- p1 + p2 + p3 + p4
  multi4 <- subset(multi3, subset = TSS.enrichment > 2 & nCount_peaks_celltypes > 1000)
  rm(multi3); gc()
  multi_species[[sp]] <- multi4
}
qs::qsave(multi_species, paste0(folder,'multi_species.rds'))
saveRDS(species_qplots, paste0(figure_folder, 'species_qplots.rds'))
#Quality plots for all species
quality_plot_all <- species_qplots$human[[1]] + species_qplots$chimp[[1]] + species_qplots$rhesus[[1]] +
  species_qplots$human[[2]] + species_qplots$chimp[[2]] + species_qplots$rhesus[[2]] +
  species_qplots$human[[3]] + species_qplots$chimp[[3]] + species_qplots$rhesus[[3]] +
  species_qplots$human[[4]] + species_qplots$chimp[[4]] + species_qplots$rhesus[[4]] + plot_layout(nrow = 4)
quality_plot_all
ggsave(paste0(figure_folder,'quality_plots.pdf'), height = 8, width = 6, units = 'in')
ggsave(paste0(figure_folder,'quality_plots.png'), height = 8, width = 6, units = 'in')

#Make sce objects----
#Consensus peaks
num_con_peaks = length(species_allcon_peaks[[1]])
con_peaks_names <- species_allcon_peaks[[1]]$name

sp_meta = c()
sp_counts_con = c()
for (sp in species){
  sp_meta[[sp]] = multi_species[[sp]]@meta.data
  sp_counts = multi_species[[sp]]@assays$peaks_consensus@counts
  #Consensus
  sp_counts_con[[sp]] = sp_counts[1:num_con_peaks,]
  rownames(sp_counts_con[[sp]]) = con_peaks_names
  #Species only
  sp_counts_only = sp_counts[(num_con_peaks+1):dim(sp_counts)[1],]
  if (dim(sp_counts_only)[2] != dim(sp_meta[[sp]])[1]){#if number of cells is not the same between counts and metadata - not sure why
    warning('Metadata and counts matrices have different numbers of cells.')
    cells_in_both = intersect(rownames(sp_meta[[sp]]),colnames(sp_counts_only))
    sp_counts_only = sp_counts_only[,cells_in_both]
    sp_meta[[sp]] = sp_meta[[sp]][cells_in_both,]
  }
  sp_only_peaks_names = species_only_peaks[[sp]]$name
  rownames(sp_counts_only) = sp_only_peaks_names
  atac_spec_sce <- SingleCellExperiment(assays = list(counts = sp_counts_only), colData = sp_meta[[sp]])
  qs::qsave(atac_spec_sce,paste0(folder,sp,'_atac_spec_sce.rds'))
  }
#Combine consensus
counts_combined = Reduce(cbind,sp_counts_con)
meta_combined <- Reduce(rbind, sp_meta)
atac_consensus_sce <- SingleCellExperiment(assays = list(counts = counts_combined), colData = meta_combined)
qs::qsave(atac_consensus_sce,paste0(folder,'atac_consensus_sce.rds'))
rm(atac_consensus_sce); gc()
