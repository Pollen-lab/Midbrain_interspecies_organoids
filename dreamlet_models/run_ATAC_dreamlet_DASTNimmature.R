rm(list = ls()); gc()  ## remove any variable to start clean
coi = c('DA_STN_neurons_immature')
source('/media/jenelle/4TB_disk/Dropbox/R_analysis/MyFunctions.R')
source('/media/jenelle/4TB_disk/Dropbox/R_analysis/MySceFunctions.R')
base_folder = "/media/jenelle/4TB_disk/Dropbox/Analysis/Dreamlet_R_4.4.0/Midbrain/ATAC/"
folder = paste0(base_folder,"D40_D100_D80/V26/",coi,"/"); dir.create(folder) #folder to save results from this version of the model
figure_folder = paste0(folder,"Figures/"); dir.create(figure_folder)
sce1_folder = "/media/jenelle/4TB_disk/Dropbox/Analysis/Signac/Midbrain/Object_creation/V5/"
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
species = c('human','chimp','macaque')
seq_days = c('Run_20211206','Run_20230126') #exclude sequencing days for D80 and D100 because <5 cells per species
set.seed(12345)

#Load data and adjust metadata----
sce1 = qs::qread(paste0(sce1_folder, "atac_consensus_sce.rds"))
cells_to_keep <- colData(sce1)$cell_type %in% coi; sce1 <- sce1[, cells_to_keep]
cells_to_keep <-  colData(sce1)$species %in% species; sce1 <- sce1[, cells_to_keep]
cells_to_keep <- colData(sce1)$day_10x %in% seq_days; sce1 <- sce1[, cells_to_keep]
unique_levels <- sort(unique(sce1$cell_type))
sce1$cell_type <- factor(sce1$cell_type, levels = unique_levels) #creating a factor (cell types will be in alphabetical order or you can reorder them however you like)

#Setting up sce1 object
sce1$sample <- paste(sce1$species, sce1$indiv, sce1$sex, sce1$day_10x, sep = "_")  #sample column determines which variables to use - needs to match equation

#Downsampling to make number of cells match in human and chimp samples
num_cells_table = table(sce1$sample)
num_cells_df = as.data.frame(num_cells_table)
num_cells_df <- num_cells_df %>%
  dplyr::rename(sample = Var1)  %>%
  dplyr::rename(num_cells = Freq) 
df <- num_cells_df %>%
  dplyr::filter(num_cells>min_cells)
df$species <- str_extract(df$sample, "human|chimp|macaque|orangutan")
df$day_10x <- str_extract(df$sample, "Run_20211206|Run_20230126")
df$indiv <- str_extract(df$sample, paste(unique(sce1$indiv), collapse = '|'))

# Calculate downsampling
human_samples <- df %>% dplyr::filter(species == "human")
chimp_samples <- df %>% dplyr::filter(species == "chimp")
macaque_samples <- df %>% dplyr::filter(species == "macaque")
# Sort human samples by num_cells
human_samples <- human_samples %>% arrange(num_cells)
chimp_samples <- chimp_samples %>% arrange(num_cells)  # Arrange in ascending order to match human
macaque_samples <- macaque_samples %>% arrange(num_cells)  # Arrange in ascending order to match human
# Function to match the entire range
match_full_range <- function(species_samples, human_samples) {
  num_species_samples <- nrow(species_samples)
  num_human_samples <- nrow(human_samples)
  # Create a linear interpolation function based on human samples
  interp_func <- approxfun(seq_len(num_species_samples), 
                           human_samples$num_cells[seq(1, num_human_samples, length.out = num_species_samples)], 
                           rule = 2) # rule = 2 ensures extrapolation
  # Use the function to determine the downsampled cell numbers
  species_samples <- species_samples %>%
    mutate(num_cells_ds = pmin(num_cells, interp_func(seq_len(num_species_samples))))
  return(species_samples)
}
chimp_samples <- match_full_range(chimp_samples, human_samples)
macaque_samples <- match_full_range(macaque_samples, human_samples)
# Combine all samples
ds_df <- bind_rows(human_samples %>% mutate(num_cells_ds = num_cells),
                   chimp_samples,
                   macaque_samples)
saveRDS(ds_df, paste0(folder,'ds_df.rds'))
#Perform downsampling
sce2 = sce1
samples = as.character(ds_df$sample)
for (s in samples){
  num_cells_sample = ds_df$num_cells_ds[ds_df$sample==s]
  sample_cells <- which(colData(sce2)$sample == s)
  ds_cells <- sample(sample_cells, size = num_cells_sample, replace = FALSE)
  other_cells <- which(colData(sce2)$sample != s)
  sce2 <- sce2[, c(ds_cells, other_cells)]
}
#Check
num_cells_table_ds = table(sce2$sample)
num_cells_ds_df = as.data.frame(num_cells_table_ds)
num_cells_ds_df <- num_cells_ds_df %>%
  dplyr::rename(sample = Var1)  %>%
  dplyr::rename(num_cells = Freq) %>%
  dplyr::arrange(desc(num_cells))

sce = sce2
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
seqday_levels <- sort(unique(pb$day_10x))
pb$day_10x <- factor(pb$day_10x, levels = seqday_levels) #creating a factor

#Add metadata for num_peaks_detected
detected_peaks_per_sample <-  colSums(assay(pb, coi)>0)
log_num_peaks = log2(detected_peaks_per_sample)
samples = colnames(pb)
df <- data.frame(sample = names(detected_peaks_per_sample), num_peaks = log_num_peaks, stringsAsFactors = FALSE)
pb$log_num_peaks = log_num_peaks

saveRDS(pb,paste0(folder,'pb.rds'))

# Normalize and apply voom/voomWithDreamWeights----
form = ~ (1|species) + (1|indiv) + (1|sex) + log_num_peaks + (1|day_10x)
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

#Variance partitioning analysis----
vp_form = ~  (1|species)  + (1|indiv) + (1|sex)   + log_num_peaks + (1|day_10x)
vp_lst = fitVarPart(res_proc, vp_form,BPPARAM = SnowParam(num_cores, type = "SOCK"))
saveRDS(vp_lst, paste0(folder,"vp_lst.rds"))
plotVarPart(sortCols(vp_lst), label.angle=60, ncol = 6)
savePlot("png",paste0(figure_folder,'vp_plot.png'), height=7, width=11, units = "in", res = 300)


# Dreamlet analysis----
d_form = ~ 0 + species + (1|indiv)+ (1|sex)  + log_num_peaks + (1|day_10x)
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

