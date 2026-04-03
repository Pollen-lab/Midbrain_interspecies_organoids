rm(list = ls()); gc()  ## remove any variable to start clean
coi = c('DA_neurons')
species = 'chimp'
source('/media/jenelle/4TB_disk/Dropbox/R_analysis/MyFunctions.R')
source('/media/jenelle/4TB_disk/Dropbox/R_analysis/MySceFunctions.R')
base_folder = "/media/jenelle/4TB_disk/Dropbox/Analysis/Dreamlet_R_4.4.0/Midbrain/ATAC/"
folder = paste0(base_folder,"D40_D100_D80/V26/",coi,"/",species,"_only/"); dir.create(folder) #folder to save results from this version of the model
figure_folder = paste0(folder,"Figures/"); dir.create(figure_folder)
con_model_folder = paste0(base_folder,"D40_D100_D80/V26/",coi,"/")
sce_folder = "/media/jenelle/4TB_disk/Dropbox/Analysis/Signac/Midbrain/Object_creation/V5/"
num_cores <- 14
library(data.table)
library(SingleCellExperiment)
library(stringr)
library(GSEABase)
library(dreamlet)
library(scater)
library(zenith)
library(knitr)
library(kableExtra)
library(scattermore)
library(cowplot)
library(ggplot2)
library(qvalue)
library(tidyverse)
library(RColorBrewer)
library(BiocParallel)
library(DelayedArray)
library(Seurat)
library(qs)
setAutoBlockSize(1e9)
info <- capture.output(sessionInfo()); writeLines(info, paste0(folder, "session_info.txt")) #save sessionInfo in case there are any errors due to versions of packages
min_cells = 30
set.seed(12345)

#Load data and adjust metadata----
sce1 = qs::qread(paste0(sce_folder, "chimp_atac_spec_sce.rds"))
cells_to_keep <- colData(sce1)$cell_type %in% coi; sce1 <- sce1[, cells_to_keep]
cells_to_keep <-  colData(sce1)$species %in% species; sce1 <- sce1[, cells_to_keep]
sce_con = qs::qread(paste0(con_model_folder,'sce.rds')) #Load downsampled object fron consensus model and use those same cells for this model
sce_con_cells = colnames(sce_con)
cells_to_keep <- intersect(sce_con_cells, colnames(sce1)); sce1 <- sce1[,cells_to_keep]
unique_levels <- sort(unique(sce1$cell_type))
sce1$cell_type <- factor(sce1$cell_type, levels = unique_levels) #creating a factor (cell types will be in alphabetical order or you can reorder them however you like)
sce1$stage <- ifelse(sce1$time_point == 'D40', 'early', 'late')

#Setting up sce1 object
sce1$sample <- paste(sce1$species, sce1$indiv, sce1$sex, sce1$stage, sep = "_")  #sample column determines which variables to use - needs to match equation
sce <- sce1
qs::qsave(sce, paste0(folder, 'sce.rds'))

# Process pseudobulk data to estimate precision weights----
pb <- aggregateToPseudoBulk(sce,
                            assay = "counts",
                            cluster_id = "cell_type",
                            sample_id = "sample")

reordered_levels = c('human','chimp','orangutan','macaque') #setting order of levels (can keep what they are by default - alphabetical or reset to your choice)
pb$species <- factor(pb$species, levels = reordered_levels) #creating a factor
unique_levels <- sort(unique(pb$indiv))#setting order of levels
pb$indiv <- factor(pb$indiv, levels = unique_levels)#creating a factor
unique_levels <- sort(unique(pb$sex))#setting order of levels
pb$sex <- factor(pb$sex, levels = unique_levels)#creating a factor
stage_levels = c('early','late')
pb$stage <- factor(pb$stage, levels = stage_levels)

#Add metadata for num_peaks_detected
detected_peaks_per_sample <-  colSums(assay(pb, coi)>0)
log_num_peaks = log2(detected_peaks_per_sample)
samples = colnames(pb)
df <- data.frame(sample = names(detected_peaks_per_sample), num_peaks = log_num_peaks, stringsAsFactors = FALSE)
pb$log_num_peaks = log_num_peaks

saveRDS(pb,paste0(folder,'pb.rds'))

# Normalize and apply voom/voomWithDreamWeights----
form = ~ (1|species) + (1|indiv) + (1|sex) + (1|stage) + log_num_peaks
res_proc = processAssays( pb,
                          form,
                          min.count=2, #will drop genes with fewer pseudobulk counts
                          min.cells = min_cells, #will drop samples with fewer cells
                          min.prop = 0.25, #chosen based on number of species
                          norm.method = 'RLE',
                          BPPARAM = SnowParam(num_cores, type = "SOCK"),
                          quiet = FALSE)
details(res_proc)
saveRDS(res_proc, paste0(folder,"res_proc.rds"))

#Show voom-style mean-variance trends----
plotVoom( res_proc, ncol=6)
savePlot("png",paste0(figure_folder,'voom_plot.png'), height=7, width=11, units = "in", res = 300)
# 
# #Variance partitioning analysis----
# vp_form = ~  (1|species)  + (1|indiv) + (1|sex)  + (1|stage) + log_num_peaks
# vp_lst = fitVarPart(res_proc, vp_form,BPPARAM = SnowParam(num_cores, type = "SOCK"))
# saveRDS(vp_lst, paste0(folder,"vp_lst.rds"))
# plotVarPart(sortCols(vp_lst), label.angle=60, ncol = 6)
# savePlot("png",paste0(figure_folder,'vp_plot.png'), height=7, width=11, units = "in", res = 300)
# 
# 
# # Dreamlet analysis----
# d_form = ~ 0 + species + (1|indiv)+ (1|sex)  + (1|stage) + log_num_peaks
# contrasts = c(human_vs_chimp = 'specieshuman - specieschimp',
#               human_vs_macaque = 'specieshuman - speciesmacaque',
#               chimp_vs_macaque = 'specieschimp - speciesmacaque',
#               human_specific = 'specieshuman - specieschimp/2 - speciesmacaque/2',
#               chimp_specific = 'specieschimp - specieshuman/2 - speciesmacaque/2',
#               macaque_specific = 'speciesmacaque - specieschimp/2 - specieshuman/2',
#               hominid_specific = 'specieshuman + specieschimp - speciesmacaque - speciesorangutan',
#               human_specific_vs_all = 'specieshuman - specieschimp/3 - speciesmacaque/3 - speciesorangutan/3',
#               chimp_specific_vs_all = 'specieschimp - specieshuman/3 - speciesmacaque/3- speciesorangutan/3',
#               macaque_specific_vs_all = 'speciesmacaque - specieschimp/3 - specieshuman/3- speciesorangutan/3')
# 
# # dreamlet
# res_dl = dreamlet(res_proc, 
#                   d_form, 
#                   contrasts=contrasts, 
#                   BPPARAM = SnowParam(num_cores, type = "SOCK"))
# saveRDS(res_dl, paste0(folder,"res_dl.rds"))
# 
