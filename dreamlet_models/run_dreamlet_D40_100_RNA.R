rm(list = ls()); gc()  ## remove any variable to start clean
base_folder = "/media/jenelle/4TB_disk/Dropbox/Analysis/Dreamlet_v1.3/Midbrain/Ancestral_genome/"
folder = paste0(base_folder,"D40_D100_D80/V22/"); dir.create(folder) #folder to save results from this version of the model
figure_folder = paste0(folder,"Figures/"); dir.create(figure_folder)
num_cores <- 6
library(data.table)
library(SingleCellExperiment)
library(zellkonverter)
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
setAutoBlockSize(1e9)
info <- capture.output(sessionInfo()); writeLines(info, paste0(folder, "session_info.txt")) #save sessionInfo in case there are any errors due to versions of packages

#coi = 'DA_neurons'

#Load data and adjust metadata----
sce_orig = readRDS(paste0(base_folder, "sce_new_40.rds"))
sce1 = sce_orig
sce1$indiv = sce1$individual
#Exclude doublets and unknown indiv
exclude_indiv = NA
cells_to_keep <- !colData(sce1)$indiv %in% exclude_indiv; sce2 <- sce1[, cells_to_keep]
#Add sex
df = data.frame(indiv = c("H28126","H23555","H20961","H21792","H29089","H28834","H21194","H9",
                          "C8861","C3624","C40670","C3651","C40210","C40300","C4933",
                          "O11045-4593","ZH26-HS16","ES_Lyon","ZG15-M11-10"),
                sex = c("M","M","M","F","F","F","F","F",
                        "M","M","M","F","F","F","F",
                        "F","M","F","F"))
sex_vector <- df$sex[match(colData(sce2)$indiv, df$indiv)]
colData(sce2)$sex <- sex_vector

#Exclude celltypes
unique_celltypes = unique(sce2$cell_type)
exclude_celltypes = c('Low_quality_technical')
cells_to_keep <- !colData(sce2)$cell_type %in% exclude_celltypes; sce <- sce2[, cells_to_keep]
#cells_to_keep <- colData(sce2)$cell_type %in% coi; sce <- sce2[, cells_to_keep]
unique_levels <- sort(unique(sce$cell_type))
sce$cell_type <- factor(sce$cell_type, levels = unique_levels) #creating a factor (cell types will be in alphabetical order or you can reorder them however you like)

#Setting up sce object
#head(colData(sce)) #can uncomment this line to view first 10 lines of sce metadata matrix (can check if you have columns for all the variables you want)
#make timepoint continuous
sce$sample <- paste(sce$species, sce$experiment, sce$indiv, sce$sex, sep = "_")  #sample column determines which variables to use - needs to match equation
sce$day <- sce$time_point; sce$day <- as.integer(gsub("[^0-9]", "", sce$day)); sce$day <- as.integer(sce$day) #for continuous variable
saveRDS(sce, paste0(folder, 'sce.rds'))

# Process pseudobulk data to estimate precision weights----
pb <- aggregateToPseudoBulk(sce,
                            assay = "counts",
                            cluster_id = "cell_type",
                            sample_id = "sample")

reordered_levels = c('human','chimp','orangutan','macaque') #setting order of levels (can keep what they are by default - alphabetical or reset to your choice)
pb$species <- factor(pb$species, levels = reordered_levels) #creating a factor
unique_levels <- sort(unique(pb$indiv))#setting order of levels
pb$indiv <- factor(pb$indiv, levels = unique_levels)#creating a factor
reordered_levels = c('first_experiment','replicate_experiment','rotenone_experiment', 'outgroup_experiment') #setting order of levels (can keep what they are by default - alphabetical or reset to your choice)
pb$experiment <- factor(pb$experiment, levels = reordered_levels) #creating a factor
unique_levels <- sort(unique(pb$sex))#setting order of levels
pb$sex <- factor(pb$sex, levels = unique_levels)#creating a factor
saveRDS(pb,paste0(folder,'pb.rds'))

# Normalize and apply voom/voomWithDreamWeights----
form = ~ species  + (1|experiment) + day  + (1|indiv) + (1|sex)
res_proc = processAssays( pb,
                          form,
                          min.count=5, #will drop genes with fewer pseudobulk counts
                          min.cells = 5, #will drop samples with fewer cells
                          min.prop = 0.25, #chosen based on number of species
                          norm.method = 'RLE',
                          BPPARAM = SnowParam(num_cores, type = "SOCK"),
                          quiet = FALSE)
details(res_proc)
saveRDS(res_proc, paste0(folder,"res_proc.rds"))

#Show voom-style mean-variance trends----
plotVoom( res_proc, ncol=4)
savePlot("png",paste0(figure_folder,'voom_plot.png'))

#Variance partitioning analysis----
vp_form = ~  (1|species) + (1|experiment) + day + (1|indiv)+ (1|sex)
vp_lst = fitVarPart(res_proc, vp_form,BPPARAM = SnowParam(num_cores, type = "SOCK"))
saveRDS(vp_lst, paste0(folder,"vp_lst.rds"))
plotVarPart(sortCols(vp_lst), label.angle=60, ncol = 4)
savePlot("png",paste0(figure_folder,'vp_plot.png'), height=7, width=11, units = "in", res = 300)


# Dreamlet analysis----
d_form = ~ 0 + species + (1|experiment) + day + (1|indiv)+ (1|sex)
contrasts = c(human_vs_chimp = 'specieshuman - specieschimp',
              human_vs_macaque = 'specieshuman - speciesmacaque',
              chimp_vs_macaque = 'specieschimp - speciesmacaque',
              human_specific = 'specieshuman - specieschimp/2 - speciesmacaque/2',
              chimp_specific = 'specieschimp - specieshuman/2 - speciesmacaque/2',
              macaque_specific = 'speciesmacaque - specieschimp/2 - specieshuman/2',
              hominid_specific = 'specieshuman + specieschimp - speciesmacaque - speciesorangutan',
              human_specific_vs_all = 'specieshuman - specieschimp/3 - speciesmacaque/3 - speciesorangutan/3',
              chimp_specific_vs_all = 'specieschimp - specieshuman/3 - speciesmacaque/3- speciesorangutan/3',
              macaque_specific_vs_all = 'speciesmacaque - specieschimp/3 - specieshuman/3- speciesorangutan/3')

# dreamlet
res_dl = dreamlet(res_proc, 
                  d_form, 
                  contrasts=contrasts, 
                  BPPARAM = SnowParam(num_cores, type = "SOCK"))
saveRDS(res_dl, paste0(folder,"res_dl.rds"))

