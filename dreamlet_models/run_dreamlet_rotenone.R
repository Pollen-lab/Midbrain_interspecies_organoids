rm(list = ls()); gc()  ## remove any variable to start clean
source('/media/jenelle/4TB_disk/Dropbox/R_analysis/MyFunctions.R')
source('/media/jenelle/4TB_disk/Dropbox/R_analysis/MySceFunctions.R')
base_folder = "Midbrain/Ancestral_genome/"
indiv_folder = paste0(base_folder,"Rotenone/")
folder = paste0(base_folder,"Rotenone/V10/"); dir.create(folder) #folder to save results from this version of the model
figure_folder = paste0(folder,"Figures/"); dir.create(figure_folder)
num_cores <- 8
library(data.table)
library(SingleCellExperiment)
#library(zellkonverter)
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
sce_orig = readRDS(paste0(base_folder, "sce_new_rotenone.rds"))
sce1 = sce_orig
sce1$species[sce1$species=='Chimp'] = 'chimp'
sce1$species[sce1$species=='Human'] = 'human'
sce1$species_condition = paste(sce1$species, sce1$condition, sep = '_')

#Add individuals
chimp_indiv <- read.csv(paste0(indiv_folder,'Chimp_all_assignments2.csv'), sep = '\t', header = FALSE, col.names = c('Barcode','Assignment','SorD','LogLikelihoodRatio'))
human_indiv <- read.csv(paste0(indiv_folder,'Human_all_assignments2.csv'), sep = '\t', header = FALSE, col.names = c('Barcode','Assignment','SorD','LogLikelihoodRatio'))
chimp_indiv$Indiv <- ifelse(chimp_indiv$SorD == 'S', chimp_indiv$Assignment, 'doublet')
human_indiv$Indiv <- ifelse(human_indiv$SorD == 'S', human_indiv$Assignment, 'doublet')
indiv <- rbind(chimp_indiv,human_indiv)
sce1 <- addIndividualToSce(sce1, indiv)

#Exclude doublets and unknown indiv
exclude_indiv = c(NA, 'doublet')
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
sce <- sce2
sce$cell_type = 'DA_neurons'
unique_levels <- sort(unique(sce$cell_type))
sce$cell_type <- factor(sce$cell_type, levels = unique_levels) #creating a factor (cell types will be in alphabetical order or you can reorder them however you like)

#Setting up sce object
sce$sample <- paste(sce$species_condition, sce$indiv, sce$sex, sep = "_")  #sample column determines which variables to use - needs to match equation
saveRDS(sce, paste0(folder, 'sce.rds'))

# Process pseudobulk data to estimate precision weights----
pb <- aggregateToPseudoBulk(sce,
                            assay = "counts",
                            cluster_id = "cell_type",
                            sample_id = "sample")

reordered_levels = c('human_CNTRL','human_24H', 'human_72H','chimp_CNTRL','chimp_24H','chimp_72H') #setting order of levels (can keep what they are by default - alphabetical or reset to your choice)
pb$species_condition<- factor(pb$species_condition, levels = reordered_levels) #creating a factor
cond_levels = c('CNTRL','24H','72H')
pb$condition = factor(pb$condition, levels = cond_levels)
unique_levels <- sort(unique(pb$indiv))#setting order of levels
pb$indiv <- factor(pb$indiv, levels = unique_levels)#creating a factor
unique_levels <- sort(unique(pb$sex))#setting order of levels
pb$sex <- factor(pb$sex, levels = unique_levels)#creating a factor
saveRDS(pb,paste0(folder,'pb.rds'))

# Normalize and apply voom/voomWithDreamWeights----
form = ~ (1|species_condition)  + (1|indiv) + (1|sex)
res_proc = processAssays( pb,
                          form,
                          min.count=5, #will drop genes with fewer pseudobulk counts
                          min.cells = 5, #will drop samples with fewer cells
                          min.prop = 0.15, #chosen based on number of species_conditions
                          norm.method = 'RLE',
                          BPPARAM = SnowParam(num_cores, type = "SOCK"),
                          quiet = FALSE)
print(details(res_proc))
saveRDS(res_proc, paste0(folder,"res_proc.rds"))

#Show voom-style mean-variance trends----
plotVoom(res_proc, ncol=4)
savePlot("png",paste0(figure_folder,'voom_plot.png'))

#Variance partitioning analysis----
vp_form = ~  (1|species_condition) + (1|indiv) + (1|sex)
vp_lst = fitVarPart(res_proc, vp_form,BPPARAM = SnowParam(num_cores, type = "SOCK"))
saveRDS(vp_lst, paste0(folder,"vp_lst.rds"))
plotVarPart(sortCols(vp_lst), label.angle=60, ncol = 4)
savePlot("png",paste0(figure_folder,'vp_plot.png'), height=7, width=11, units = "in", res = 300)

# Dreamlet analysis----
d_form = ~ 0 + species_condition  + (1|indiv) + (1|sex)
contrasts = c(Species = '(species_conditionhuman_CNTRL - species_conditionchimp_CNTRL 
                        + species_conditionhuman_24H - species_conditionchimp_24H
                        + species_conditionhuman_72H - species_conditionchimp_72H)/3',
              Condition24H = '((species_conditionhuman_24H - species_conditionhuman_CNTRL) + (species_conditionchimp_24H - species_conditionchimp_CNTRL))/2',
              Condition72H = '((species_conditionhuman_72H - species_conditionhuman_CNTRL) + (species_conditionchimp_72H - species_conditionchimp_CNTRL))/2',
              Human24H = 'species_conditionhuman_24H - species_conditionhuman_CNTRL',
              Human72H = 'species_conditionhuman_72H - species_conditionhuman_CNTRL',
              Chimp24H = 'species_conditionchimp_24H - species_conditionchimp_CNTRL',
              Chimp72H = 'species_conditionchimp_72H - species_conditionchimp_CNTRL',
              Human_vs_Chimp_CNTRL = 'species_conditionhuman_CNTRL - species_conditionchimp_CNTRL',
              Human_vs_Chimp_24H =  'species_conditionhuman_24H - species_conditionchimp_24H',
              Human_vs_Chimp_72H = 'species_conditionhuman_72H - species_conditionchimp_72H',
              SpeciesCNTRL = 'species_conditionhuman_CNTRL - species_conditionchimp_CNTRL',
              Species24H = '((species_conditionhuman_24H - species_conditionhuman_CNTRL) - (species_conditionchimp_24H - species_conditionchimp_CNTRL))',
              Species72H = '((species_conditionhuman_72H - species_conditionhuman_CNTRL) - (species_conditionchimp_72H - species_conditionchimp_CNTRL))',
              SpeciesRot = '(species_conditionhuman_24H + species_conditionhuman_72H)/2 - species_conditionhuman_CNTRL - 
                            ((species_conditionchimp_24H + species_conditionchimp_72H)/2 - species_conditionchimp_CNTRL)')

# dreamlet
res_dl = dreamlet(res_proc, 
                  d_form, 
                  contrasts=contrasts, 
                  BPPARAM = SnowParam(num_cores, type = "SOCK"))
saveRDS(res_dl, paste0(folder,"res_dl.rds"))

