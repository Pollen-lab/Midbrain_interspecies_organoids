#Setup----
rm(list = ls()); gc()
folder = 'Midbrain/Figure5/V9/'
object_folder = 'Midbrain/Object_creation/V5/'
peaks_folder = 'Midbrain/Consensus_peaks/D40_100/V8/'
dreamlet_folder <- '/media/jenelle/4TB_disk/Dropbox/Analysis/Dreamlet_R_4.4.0/Midbrain/ATAC/D40_D100_D80/V26/DA_neurons/'
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
source("/media/jenelle/4TB_disk/Dropbox/Analysis/Dreamlet_R_4.4.0/plotVarPart_mod.R")
library(Seurat)
source("/media/jenelle/4TB_disk/Dropbox/R_analysis/DimPlot_mod.R", echo=TRUE) #overwriting DimPlot to add stroke.size
library(Signac)
library(dplyr)
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
library(JASPAR2020)
library(TFBSTools)
library(rtracklayer)
library(dreamlet)
library(extrafont)
library(rlang)
library(cowplot)
library(data.table)
library(ggrepel)
library(DescTools)
library(scattermore)
library(eulerr)
library(ggplot2)
library(ggforce)
loadfonts(device = "pdf")
info <- capture.output(sessionInfo()); writeLines(info, paste0(folder, "session_info.txt")) #save sessionInfo in case there are any errors due to versions of packages
coi = 'DA_neurons'
con = 'human_vs_chimp'
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

#Calculations----
##Metadata for quality control----
all_metadata <- data.frame()
for (sp in species) {
  sp_name = species_names[sp]
  multi3 <- readRDS(paste0(object_folder, sp, '_multi3.rds'))
  DefaultAssay(multi3) <- 'peaks_celltypes'
  multi3$pct_reads_in_peaks = multi3$pct_reads_in_peaks * 100
  data <- as.data.frame(multi3@meta.data)
  all_metadata <- bind_rows(all_metadata, data)
}
rm(multi3)

# Initialize an empty data frame to store all metadata
all_metadata <- data.frame()

# Loop to extract metadata from each species and combine it into one big data frame
for (sp in species) {
  sp_name = species_names[sp]
  multi3 <- readRDS(paste0(object_folder, sp, '_multi3.rds'))
  DefaultAssay(multi3) <- 'peaks_celltypes'
  multi3$pct_reads_in_peaks = multi3$pct_reads_in_peaks * 100
  
  # Extract metadata and add species information
  data <- as.data.frame(multi3@meta.data)
  data$species <- sp_name
  
  # Combine into the overall metadata data frame
  all_metadata <- bind_rows(all_metadata, data)
}
saveRDS(all_metadata,paste0(folder,'all_metadata.rds'))

#Load (for skipping calculations when complete)----
all_metadata <- readRDS(paste0(folder,'all_metadata.rds'))
annotation_categories <- readRDS(paste0(folder,'annotation_categories.rds'))
sp_annotation <- qread(paste0(folder,'sp_annotation.rds'))
species_allcon_peaks <- readRDS(paste0(peaks_folder,'species_allcon_peaks.rds'))
species_only_peaks <- readRDS(paste0(peaks_folder,'species_only_peaks.rds'))
res_dl <- readRDS(paste0(dreamlet_folder,'res_dl.rds'))
dars_dreamlet <- readRDS(paste0(folder,'dars_dreamlet.rds'))
dars_dreamlet_immature <- readRDS(paste0(folder,'dars_dreamlet_immature.rds'))
dars_dreamlet_prog <- readRDS(paste0(folder,'dars_dreamlet_prog.rds'))
dars_gr <- readRDS(paste0(folder,'dars_gr.rds'))
dars_df <- readRDS(paste0(folder,'dars_df.rds'))
dars_and_spec_gr <- readRDS(paste0(folder,'dars_and_spec_gr.rds'))
spec_peaks_gr <- readRDS(paste0(folder,'spec_peaks_gr.rds'))
allrs_gr <- readRDS(paste0(folder,'allrs_gr.rds'))
allrs_df <- readRDS(paste0(folder,'allrs_df.rds'))
total_peaks_df <- readRDS(paste0(folder,'total_peaks_df.rds'))
all_peaks_list <-readRDS(paste0(folder,'all_peaks_list.rds'))
de_genes_list <- readRDS(paste0(rna_folder,'de_genes.rds'))
all_genes_list <- readRDS(paste0(rna_folder,'all_genes.rds'))
cicero_links <- readRDS(paste0(folder,'cicero_links.rds'))
res_procs <- readRDS(paste0(rna_folder,'res_procs.rds'))
degs_polarized <- readRDS(paste0(rna_folder,'degs_polarized.rds'))

#Supp panels----
##Supp Panel A: Quality plots----
sp_frag_plots = list()
sp_tss_plots = list()
sp_nuc_plots = list()
sp_reads_plots = list()
for (sp in species) {
  sp_name = species_names[sp]
  # Filter the combined data frame for the current species
  species_data <- all_metadata %>% dplyr::filter(species == sp_name)
  # Fragment count plot
  sp_frag_plots[[sp]] <- ggplot(species_data, aes(x = factor(1), y = nCount_peaks_celltypes, fill = species))+
    geom_violin() +
    scale_y_log10(limits = c(500, 1e5)) +
    scale_fill_manual(values = colors_species) +
    labs(y = '# of frags\nin peaks') +
    theme_basic_smallest() +
    theme(plot.title = element_blank(), 
          axis.title.x = element_blank(), 
          axis.text.x = element_blank(),
          legend.position = 'none',
          plot.margin = unit(c(0,0,0,0),'in'))
  # TSS enrichment plot
  sp_tss_plots[[sp]] <- ggplot(species_data, aes(x = factor(1), y = TSS.enrichment, fill = species))+
    geom_violin() +
    scale_y_continuous(limits = c(0, 10)) +
    scale_fill_manual(values = colors_species) +
    labs(y = 'TSS \nenrichment') +
    theme_basic_smallest() +
    theme(plot.title = element_blank(), 
          axis.title.x = element_blank(), 
          axis.text.x = element_blank(),
          legend.position = 'none',
          plot.margin = unit(c(0,0,0,0),'in'))
  # Nucleosome signal plot
  sp_nuc_plots[[sp]] <- ggplot(species_data, aes(x = factor(1), y = nucleosome_signal, fill = species))+
    geom_violin() +
    scale_y_continuous(limits = c(0, 1.25)) +
    scale_fill_manual(values = colors_species) +
    labs(y = 'Nucleosome \nsignal') +
    theme_basic_smallest() +
    theme(plot.title = element_blank(), 
          axis.title.x = element_blank(), 
          axis.text.x = element_blank(),
          legend.position = 'none',
          plot.margin = unit(c(0,0,0,0),'in'))
  # Percent reads in peaks plot
  sp_reads_plots[[sp]] <- ggplot(species_data, aes(x = factor(1), y = pct_reads_in_peaks, fill = species))+
    geom_violin() +
    scale_y_continuous(limits = c(20, 80)) +
    scale_fill_manual(values = colors_species) +
    labs(y = '% reads \nin peaks') +
    theme_basic_smallest() +
    theme(plot.title = element_blank(), 
          axis.title.x = element_blank(), 
          axis.text.x = element_blank(),
          legend.position = 'none',
          plot.margin = unit(c(0,0,0,0),'in'))
}

row1 <- wrap_plots(sp_frag_plots) + plot_layout(nrow = 1, axes = 'collect')
row2 <- wrap_plots(sp_tss_plots) + plot_layout(nrow = 1, axes = 'collect')
row3 <- wrap_plots(sp_nuc_plots) + plot_layout(nrow = 1, axes = 'collect')
row4 <- wrap_plots(sp_reads_plots) + plot_layout(nrow = 1, axes = 'collect')
supp_panelA <- (row1 / row2 / row3 / row4) + plot_layout(axes = 'collect')

#Load for UMAPs----
multi_integrated <- qread(paste0(folder,'multi_integrated.rds'))
multi_integrated$species <- factor(multi_integrated$species, levels = c('human','chimp','macaque'))
multi_integrated$time_point <- factor(multi_integrated$time_point, levels = c('D40','D80','D100'))

##Supp Panel B and C: UMAPs with labels----
supp_panelB <- DimPlot(object = multi_integrated, label = FALSE, group.by = 'species', shuffle = TRUE, cols = colors_species,
                       pt.size = 1.5, stroke.size = 0, alpha = 1, raster = TRUE, raster.dpi = c(600,600)) +
  labs(x = 'UMAP1',y = 'UMAP2')+
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 0.25),
        axis.text = element_blank(),
        axis.title = element_text(size = 6),
        axis.ticks = element_blank(),
        plot.title = element_blank(),
        legend.position = c(0.04,0.85),
        legend.key.size = unit(c(0.1),'in'),
        legend.text = element_text(size = 6),
        plot.margin = unit(c(0,0,0,0),'in')
  )
supp_panelC <- DimPlot(object = multi_integrated, label = FALSE, group.by = 'time_point', shuffle = TRUE,
                       pt.size = 1.5, stroke.size = 0, alpha = 1, raster = TRUE, raster.dpi = c(600,600)) +
  labs(x = 'UMAP1',y = 'UMAP2')+
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 0.25),
        axis.text = element_blank(),
        axis.title = element_text(size = 6),
        axis.ticks = element_blank(),
        plot.title = element_blank(),
        legend.position = c(0.05,0.9),
        legend.key.size = unit(c(0.1),'in'),
        legend.text = element_text(size = 6),
        plot.margin = unit(c(0,0,0,0),'in')
  )
rm(multi_integrated); gc()

##Supp panel D: Venn diagram----
A <- unique(dars_dreamlet_prog$ID)
B <- unique(dars_dreamlet_immature$ID)
C <- unique(dars_dreamlet$ID)
nA <- length(setdiff(A, union(B, C)))
nB <- length(setdiff(B, union(A, C)))
nC <- length(setdiff(C, union(A, B)))
nAB <- length(setdiff(intersect(A, B), C))
nAC <- length(setdiff(intersect(A, C), B))
nBC <- length(setdiff(intersect(B, C), A))
nABC <- length(intersect(intersect(A, B), C))
counts <- c(
  "A" = nA,
  "B" = nB,
  "C" = nC,
  "A&B" = nAB,
  "A&C" = nAC,
  "B&C" = nBC,
  "A&B&C" = nABC
)
fit <- euler(counts, shape = "ellipse")
p1 <- plot(fit,
           fills = list(fill = c(celltype_colors['Ventral_FB_MB_progenitors'],
                                 celltype_colors['DA_STN_neurons_immature'], celltype_colors['DA_neurons']), alpha = 0.5),
           edges = list(lwd = 1),
           labels = list(font = 4),
           quantities = TRUE)
supp_panelD <- p1 %>% as.ggplot()

##Supp Panel E-H: Cicero histogram for number of peaks and genes linked----
cicero_species = c('human','chimp')
cf_index = which(coaccess_filters==cf_filter_to_plot)
sp_peaks_plots = list()
sp_genes_plots = list()
sp_peaks_medians = c()
sp_genes_medians = c()
for (sp in cicero_species){
  # Count the number of peaks linked to each gene
  peaks_per_gene <- cicero_links[[cf_index]][[sp]] %>%
    group_by(gene) %>%
    summarise(n_peaks = n())
  sp_peaks_medians[[sp]] = median(peaks_per_gene$n_peaks)
  
  # Count the number of genes linked to each peak
  genes_per_peak <- cicero_links[[cf_index]][[sp]] %>%
    group_by(peak_name) %>%
    summarise(n_genes = n())
  sp_genes_medians[[sp]] = median(genes_per_peak$n_genes)
  
  # Histogram for the number of peaks linked to each gene
  sp_peaks_plots[[sp]] <- ggplot(peaks_per_gene, aes(x = n_peaks)) +
    geom_histogram(binwidth = 5, fill = colors_species[[sp]]) +
    labs(x = "Number of Peaks",
         y = "Count")+
    theme_basic_smallest()+
    theme(plot.margin = unit(c(0,0.01,0,0),'in'))
  
  # Histogram for the number of genes linked to each peak
  sp_genes_plots[[sp]] <- ggplot(genes_per_peak, aes(x = n_genes)) +
    geom_histogram(binwidth = 1, fill = colors_species[[sp]]) +
    labs(x = "Number of Genes",
         y = "Count")+
    theme_basic_smallest()+
    theme(plot.margin = unit(c(0,0.01,0,0),'in'))
}

##Supp panel G and H: Bars for each species in all categories----
groups = c('DE up', 'DE down', 'Non-DE')
species_percentages <- readRDS(paste0(folder,'cicero_species_percentages.rds'))
species_numbers <- readRDS(paste0(folder,'cicero_species_numbers.rds'))
species_genes <- readRDS(paste0(folder,'cicero_species_genes.rds'))
sp_bars = list()
for (sp in cicero_species){
  cf_index = which(coaccess_filters==cf_filter_to_plot)
  plot_data <- tidyr::gather(species_percentages[[cf_index]][[sp]], key = "Score_Type", value = "Percent", -Group)
  if (sp == 'chimp'){
    plot_data$Group <- ifelse(plot_data$Group == "DE up", "DE down", 
                              ifelse(plot_data$Group == "DE down", "DE up", plot_data$Group)) #Change DE to be perspective of chimp
    plot_data$Score_Type <- ifelse(plot_data$Score_Type == "Percent_Positive", #Switch to be point of view of chimp
           "Percent_Negative", 
           "Percent_Positive")
  }
  plot_data$Group = factor(plot_data$Group, levels = groups)
  plot_data$Score_Type = factor(plot_data$Score_Type, levels = c('Percent_Positive','Percent_Negative'))
  sp_bars[[sp]] <- ggplot(plot_data, aes(x = Group, y = Percent, fill = Score_Type)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_basic() +
    labs(y = "Percent of linked DARs") +
    scale_fill_manual(values = c("Percent_Positive" = "firebrick", "Percent_Negative" = "steelblue"), labels = c('DAR Up','DAR Down')) +
    scale_y_continuous(expand = c(0, 0)) +
    theme_basic_smallest()+
    theme(legend.title = element_blank(),
          axis.title.x = element_blank(),
          legend.key.size = unit(c(0.1),'in'),
          legend.text = element_text(size = 6)
    )
}
supp_panelG <- sp_bars$human + theme(legend.position  = c(0.5,0.9))
supp_panelH <- sp_bars$chimp + theme(legend.position  = 'none')

##Supp panel I: GREAT concordance barplot in human----
great_table <- read.csv(paste0(folder,'GREAT_gene_region_associations.txt'), sep = '\t', skip = 1, col.names = c('gene', 'peak_name'))
de_genes_up = de_genes_list$D40_100$DA_neurons$human_vs_chimp$ID[de_genes_list$D40_100$DA_neurons$human_vs_chimp$logFC > 0]
de_genes_down = de_genes_list$D40_100$DA_neurons$human_vs_chimp$ID[de_genes_list$D40_100$DA_neurons$human_vs_chimp$logFC < 0]
de_genes_all =  de_genes_list$D40_100$DA_neurons$human_vs_chimp$ID
all_genes_up = all_genes_list$D40_100$DA_neurons$human_vs_chimp$ID[all_genes_list$D40_100$DA_neurons$human_vs_chimp$logFC > 0]
all_genes_down = all_genes_list$D40_100$DA_neurons$human_vs_chimp$ID[all_genes_list$D40_100$DA_neurons$human_vs_chimp$logFC < 0]
non_de_genes_up = all_genes_up[all_genes_up %!in% de_genes_all]
non_de_genes_down = all_genes_down[all_genes_down %!in% de_genes_all]
all_peaks_coi <- all_peaks_list$DA_neurons$human_vs_chimp
all_peaks_coi$peak_name = all_peaks_coi$ID
all_genes_coi = all_genes_list$D40_100$DA_neurons$human_vs_chimp
all_genes_coi$gene = all_genes_coi$ID

great_links <- great_table %>%
  separate_rows(peak_name, sep = ", ") %>%  # Separate multiple peaks into rows
  mutate(
    distance = str_extract(peak_name, "\\(.*\\)"),        # Extract the distance in parentheses
    distance = str_replace_all(distance, "[()]", ""), # Remove parentheses
    distance = as.numeric(distance),                  # Convert distance to numeric
    peak_name = str_replace(peak_name, " \\(.*\\)", "")       # Remove the distance part from peaks column
  )  %>% left_join(all_peaks_coi, by = 'peak_name') %>%
  left_join(all_genes_coi, by = 'gene', suffix = c('_peak','_gene')) %>%
  dplyr::select(gene, peak_name, distance, logFC_peak, z.std_peak,adj.P.Val_peak, logFC_gene, z.std_gene,adj.P.Val_gene) %>%
  mutate(concordant = ifelse(sign(logFC_peak) == sign(logFC_gene), TRUE, FALSE)) %>%
  mutate(both_sig = ifelse(adj.P.Val_gene < 0.05, TRUE, FALSE))

#Summarize concordance for each group of DE genes
groups = c('DE up', 'DE down', 'Non-DE') #note all categories are from the perspective of human
groups_genes = list('DE up' = de_genes_up, 'DE down' = de_genes_down, 'Non-DE' = non_de_genes_down)
species_percentages_great = list()
species_numbers_great = list()
species_genes_great = list()
cf_index = which(coaccess_filters==cf_filter_to_plot)
cf = coaccess_filters[cf_index]
sp = 'human'
sp_numbers = list()
sp_percentages = list()
sp_genes = list()
for (j in seq_along(groups)){
  group = groups[j]
  #Filter and merge data for up-regulated DE genes
  group_peaks <- great_links %>%
    dplyr::filter(gene %in% groups_genes[[group]]) 
  scores_df <- dars_df[[sp]] %>%
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
species_percentages_great[[sp]] = sp_percentages
species_numbers_great[[sp]] = sp_numbers
species_genes_great[[sp]]= sp_genes

#Bars for each species in all categories
plot_data <- tidyr::gather(species_percentages_great[[sp]], key = "Score_Type", value = "Percent", -Group)
plot_data$Group = factor(plot_data$Group, levels = groups)
plot_data$Score_Type = factor(plot_data$Score_Type, levels = c('Percent_Positive','Percent_Negative'))
supp_panelI <- ggplot(plot_data, aes(x = Group, y = Percent, fill = Score_Type)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_basic() +
  labs(y = "Percent of linked DARs") +
  scale_fill_manual(values = c("Percent_Positive" = "firebrick", "Percent_Negative" = "steelblue"), labels = c('DAR Up','DAR Down')) +
  scale_y_continuous(expand = c(0, 0)) +
  theme_basic_smallest()+
  theme(legend.title = element_blank(),
        axis.title.x = element_blank(),
        legend.position  = 'none', 
        legend.key.size = unit(c(0.1),'in'),
        legend.text = element_text(size = 6)
  )

#Percent DE genes with DA peak
de_genes_great_links <-great_links %>%
  dplyr::filter(gene %in% de_genes_all)
de_genes_with_links = unique(de_genes_great_links$gene)

##Supp Panel J: GREAT terms----
great_terms = read.csv(paste0(folder,'greatExportAll.tsv'), sep = '\t', skip = 3)
great_terms <- great_terms %>%
  dplyr::rename(Ontology = X..Ontology) %>%
  dplyr::filter(Ontology %in% c('GO Biological Process')) %>%
  dplyr::mutate(PropGenes = NumFgGenesHit/TotalGenes) %>%
  dplyr::arrange(HyperFdrQ) %>%
  dplyr::select(Desc, Rank, HyperFdrQ, PropGenes, NumFgGenesHit)
great_terms_toplot <- great_terms[1:10,]
great_terms_toplot <- great_terms_toplot %>% arrange(desc(great_terms_toplot$PropGenes))
great_terms_toplot$Rank <- seq_along(great_terms_toplot$PropGenes) #originally were ranked by raw p value but rank by proportion sig genes
great_terms_toplot$Desc_wrapped <- str_wrap(great_terms_toplot$Desc, width = 40)

supp_panelJ <-ggplot(great_terms_toplot, aes(x = 0, y = reorder(Desc_wrapped, -Rank), size = PropGenes, color = HyperFdrQ)) +
  geom_point()+
  scale_size_continuous(range = c(1,3), limits = c(0.25,0.75)) +  # Adjust size range as needed
  scale_color_gradient(low = 'black', high = 'gray80') +
  labs(size = 'Proportion \n sig. genes',color = "FDR") +
  theme_basic_smallest() +
  theme(
    axis.text.x = element_blank(),
    axis.text.y = element_text(lineheight =0.8),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    axis.title = element_blank(),
    legend.title = element_text(size = 6),
    legend.text = element_text(size = 6),
    legend.key.size = unit(0.08, "in"),
  )


#Putting together figure----
supp_figure_row1 <- (supp_panelA | supp_panelB | supp_panelC| supp_panelD) + plot_layout(widths = c(1,1.5,1.5,0.8))
supp_panelEF <- (sp_peaks_plots$human + sp_genes_plots$human + sp_peaks_plots$chimp + sp_genes_plots$chimp + 
                  plot_layout(nrow = 2, axes = 'collect'))
supp_figure_row2 <-  supp_panelEF | supp_panelG | supp_panelH | supp_panelI +
  plot_layout(nrow = 1)
supp_figure_rows12 = (supp_figure_row1 / supp_figure_row2) + plot_layout(heights = c(1,0.8))
ggsave(paste0(folder,'Supp_rows1and2.pdf'), supp_figure_rows12, width = 8, height = 3.8)

#Load for genome browser plots----
concordant_promoters <- readRDS(paste0(folder,'concordant_promoters.rds'))
refseq_annotations <- readRDS(paste0(folder,'refseq_annotations.rds'))
hs_tracks <- readRDS(paste0(folder,'hs_tracks.rds'))
cs_tracks <- readRDS(paste0(folder,'cs_tracks.rds'))
multi_species <- qread(paste0(object_folder,'multi_species.rds'))

##Supp panelK: GO term example----
#Analyze peaks in GO term
go_term_links <- read.csv(paste0(folder,'GO_0030804_gene_region.txt'),sep = '\t',skip = 1, col.names = c('gene', 'peak_names'))

go_term_links_long <- go_term_links %>%
  separate_rows(peak_names, sep = ",")
go_term_links_long$peak_names <- str_trim(go_term_links_long$peak_names)
go_term_links_long$peak_name <- str_extract(go_term_links_long$peak_names, "Peak_\\d+")
merged_df <- merge(go_term_links_long, dars_df$human, by.x = 'peak_name', by.y = 'name', all.x = TRUE)
merged_df$logFC_sign <- ifelse(merged_df$logFC > 0, '+',
                               ifelse(merged_df$logFC < 0, '-', NA))
merged_df$peak_names_with_sign <- paste0(merged_df$peak_names, ' (', merged_df$logFC_sign, ')')
result_df <- merged_df %>%
  group_by(gene) %>%
  summarise(peak_names = paste(peak_names_with_sign, collapse = ', '))
go_term_links_updated <- go_term_links %>%
  dplyr::select(-peak_names) %>%
  left_join(result_df, by = 'gene')
#Add gene info
human_specific_genes <- degs_polarized$D40_100$DA_neurons$human_specific
human_specific_genes$category = 'human_specific'
chimp_specific_genes <- degs_polarized$D40_100$DA_neurons$chimp_specific
chimp_specific_genes$category = 'chimp_specific'
divergent_genes <- degs_polarized$D40_100$DA_neurons$divergent
divergent_genes$category = 'divergent'
other_genes <- degs_polarized$D40_100$DA_neurons$other
other_genes$category = 'other'
polarized_genes = c(human_specific_genes$ID, chimp_specific_genes$ID, divergent_genes$ID, other_genes$ID)
tested_genes = all_genes_list$D40_100$DA_neurons$human_vs_chimp$ID
tested_genes = setdiff(tested_genes, polarized_genes)
tested_genes_df <- data.frame(ID = tested_genes, sigp_hvc = FALSE, sign_hvc = NA, sigp_hvm = NA, sign_hvm = NA, sigp_cvm = NA, sign_cvm = NA, category = 'tested')
gene_category_df <- rbind(human_specific_genes,chimp_specific_genes,divergent_genes,other_genes, tested_genes_df)
degs_info <- gene_category_df %>%
  dplyr::select(ID, sign_hvc, category)
go_term_links_final <- go_term_links_updated %>%
  left_join(degs_info, by = c("gene" = "ID"))
write.csv(go_term_links_final, paste0(folder,'go_term_links_final.csv'))

# Calculate percentages
total_term_peaks <- nrow(merged_df) - sum(is.na(merged_df$logFC_sign))
positive_peaks <- sum(merged_df$logFC_sign == '+', na.rm = TRUE)
negative_peaks <- sum(merged_df$logFC_sign == '-', na.rm = TRUE)
percent_positive <- (positive_peaks / total_term_peaks) * 100
percent_negative <- (negative_peaks / total_term_peaks) * 100

#ADGRD1 linked peaks
species = c('human','chimp','rhesus')
species_genomes = c('hg38','Chimp','Rhesus'); names(species_genomes) = species
extend_up = 500
extend_down = 500
linked_peaks = go_term_links$peak_names[go_term_links$gene=='ADGRD1']
peak_names <- str_extract_all(linked_peaks, "Peak_\\d+")[[1]]
gene_da_gr = c()
for (sp in species){
  gene_da_gr[[sp]] <- species_allcon_peaks[[sp]][peak_names]
}
peaks_order = sort(gene_da_gr$human); peaks_order = peaks_order$name #sort 5' to 3'
gene_peaks_plots <- c()
for (peak_name in peaks_order){
  peak_index = which(peaks_order==peak_name)
  for (sp in species){
    sp_multi = multi_species[[sp]]
    Annotation(sp_multi) <- refseq_annotations[[sp]]
    Idents(sp_multi)<- sp_multi$cell_type
    gene_peaks_plots[[sp]][[peak_name]] <- CoveragePlot(
      sp_multi,
      region = gene_da_gr[[sp]][peak_name],
      peaks = FALSE,
      idents = coi,
      extend.upstream =  500,
      extend.downstream = 500,
      ranges = gene_da_gr[[sp]][peak_name],
      ranges.title = NULL,
      ymax = 60,
      scale = 1e7
    ) & scale_fill_manual(values = celltype_colors[coi]) &
      theme(legend.position = 'none',
            plot.margin = unit(c(0,0,0,0),'in'),
            axis.title = element_text(size = 6),
            axis.text = element_text(size = 6),
            axis.title.y = element_text(angle = 0),
            axis.line = element_line(linewidth = 0.25),
            strip.text.y.left = element_blank(),
            strip.background = element_blank()) &
      labs(y = '')
    if (peak_index == 1){
      gene_peaks_plots[[sp]][[peak_name]] <- gene_peaks_plots[[sp]][[peak_name]] &
        labs(y = c('Norm \nsignal'))
    }
  }
}
supp_panelK <- wrap_plots(gene_peaks_plots$human, nrow = 1)/wrap_plots(gene_peaks_plots$chimp, nrow = 1)/wrap_plots(gene_peaks_plots$rhesus,nrow = 1)
print(paste0('panelK titles (in order): ',paste(peaks_order, collapse = " ")))

##Supp panel L: KCNJ16 example----
kcnj16_peaks = concordant_promoters$name[concordant_promoters$gene_name=='KCNJ16']
kcnj16_peak = kcnj16_peaks[1] #pick first if more than one
extend_up = 1000
extend_down = 1000
#Human
sp = 'human'
#Get genomic region and liftover to chimp and rhesus on UCSC
peak_range <- species_allcon_peaks$human[kcnj16_peak]
extended_range <- GRanges(
  seqnames = unique(seqnames(peak_range)),
  ranges = IRanges(
    start = min(start(peak_range)) - extend_up,
    end = max(end(peak_range)) + extend_down
  ),
  strand = unique(strand(peak_range)))
export.bed(extended_range, paste0(folder,'hg38_KCNJ16_promoter_coords.bed'))
kcnj16_promoter_ranges = list()
kcnj16_promoter_ranges$human = extended_range
kcnj16_promoter_ranges$chimp = import(paste0(folder,'panTro6_KCNJ16_promoter_coords.bed'))
kcnj16_promoter_ranges$rhesus = import(paste0(folder,'rheMac10_KCNJ16_promoter_coords.bed'))
sp_plots = list()
for (sp in species){
  sp_multi = multi_species[[sp]]
  Annotation(sp_multi) <- refseq_annotations[[sp]]
  Idents(sp_multi)<- sp_multi$cell_type
  sp_plots[[sp]] <- CoveragePlot(
    sp_multi,
    region = kcnj16_promoter_ranges[[sp]],
    peaks = FALSE,
    idents = coi,
    extend.upstream =  500,
    extend.downstream = 500,
    ranges = dars_gr[[sp]],
    ranges.title = 'DARs',
    ymax = 50,
    scale = 1e7
  ) & scale_fill_manual(values = celltype_colors[coi]) &
    theme(legend.position = 'none',
          plot.margin = unit(c(0,0,0,0),'in'),
          axis.title = element_text(size = 6),
          axis.text = element_text(size = 6),
          axis.title.y = element_text(angle = 0),
          axis.line = element_line(linewidth = 0.25),
          strip.text.y.left = element_blank(),
          strip.background = element_blank()) &
    labs(y = c('Norm \nsignal'))
}
supp_panelL <- sp_plots$human / sp_plots$chimp / sp_plots$rhesus + plot_layout(heights = c(1,1,1))
print(paste0('panelL title: ',paste(kcnj16_peaks, collapse = " and ")))

#Putting together figure----
ggsave(paste0(folder,'Supp_panelJ.pdf'), supp_panelJ, width = 2.6, height = 1.8)
ggsave(paste0(folder,'Supp_panelK.pdf'), supp_panelK, width = 5, height = 3)
ggsave(paste0(folder,'Supp_panelL.pdf'), supp_panelL, width = 2, height = 3)

#Print for legend/text----
print('Histogram medians:')
sp_peaks_medians
sp_genes_medians

print('Cicero barplots:')
print(paste0('Number of DE up genes for human: ',species_genes[[cf_index]]$human$num_genes[1])) 
print(paste0('Number of linked DARs for human: ',species_numbers[[cf_index]]$human$Number_Positive[1] + species_numbers[[cf_index]]$human$Number_Negative[1]))
print(paste0('Number of DE down genes for human: ',species_genes[[cf_index]]$human$num_genes[2])) 
print(paste0('Number of linked DARs for human: ',species_numbers[[cf_index]]$human$Number_Positive[2] + species_numbers[[cf_index]]$human$Number_Negative[2]))
print(paste0('Number of non-DE down genes for human: ',species_genes[[cf_index]]$human$num_genes[3])) 
print(paste0('Number of linked DARs for human: ',species_numbers[[cf_index]]$human$Number_Positive[3] + species_numbers[[cf_index]]$human$Number_Negative[3]))

print(paste0('Number of DE up genes for chimp: ',species_genes[[cf_index]]$chimp$num_genes[1])) 
print(paste0('Number of linked DARs for chimp: ',species_numbers[[cf_index]]$chimp$Number_Positive[1] + species_numbers[[cf_index]]$chimp$Number_Negative[1]))
print(paste0('Number of DE down genes for chimp: ',species_genes[[cf_index]]$chimp$num_genes[2])) 
print(paste0('Number of linked DARs for chimp: ',species_numbers[[cf_index]]$chimp$Number_Positive[2] + species_numbers[[cf_index]]$chimp$Number_Negative[2]))
print(paste0('Number of non-DE down genes for chimp: ',species_genes[[cf_index]]$chimp$num_genes[3])) 
print(paste0('Number of linked DARs for chimp: ',species_numbers[[cf_index]]$chimp$Number_Positive[3] + species_numbers[[cf_index]]$chimp$Number_Negative[3]))

print('GREAT barplot:')
print(paste0('Number of DE up genes for human: ',species_genes_great$human$num_genes[1])) 
print(paste0('Number of linked DARs for human: ',species_numbers_great$human$Number_Positive[1] + species_numbers_great$human$Number_Negative[1]))
print(paste0('Number of DE down genes for human: ',species_genes_great$human$num_genes[2])) 
print(paste0('Number of linked DARs for human: ',species_numbers_great$human$Number_Positive[2] + species_numbers_great$human$Number_Negative[2]))
print(paste0('Number of non-DE down genes for human: ',species_genes_great$human$num_genes[3])) 
print(paste0('Number of linked DARs for human: ',species_numbers_great$human$Number_Positive[3] + species_numbers_great$human$Number_Negative[3]))

print(paste0('Genes in top GREAT term: ', dim(go_term_links_final)[1]))
print(paste0('Number of GO term genes in dreamlet model: ', sum(!is.na(go_term_links_final$category))))
print(paste0('Number of human-up GO term DEGs: ', sum(go_term_links_final$sign_hvc == 1, na.rm = TRUE)))
print(paste0('Number of chimp-up GO term DEGs: ', sum(go_term_links_final$sign_hvc == -1, na.rm = TRUE)))
print(paste0('Total peaks: ', total_term_peaks))
cat(sprintf("Percent of peaks with positive logFC: %.2f%%\n", percent_positive))
cat(sprintf("Percent of peaks with negative logFC: %.2f%%\n", percent_negative))
print(paste0('Percent DE genes: ', sum(!is.na(go_term_links_final$sign_hvc))/length(go_term_links_final$sign_hvc)*100))
