rm(list = ls()); gc()  ## remove any variable to start clean
source('/media/jenelle/4TB_disk/Dropbox/R_analysis/MyFunctions.R')
source('/media/jenelle/4TB_disk/Dropbox/R_analysis/MySceFunctions.R')
source('/media/jenelle/4TB_disk/Dropbox/R_analysis/MyPlottingFunctions.R')
source("/media/jenelle/4TB_disk/Dropbox/R_analysis/MyDreamletFunctions.R")
source("/media/jenelle/4TB_disk/Dropbox/R_analysis/MyUpsetFunctions.R")
source("plotStratify_mod.R")
source("plotMyVolcano.R")
rot_folder = "Midbrain/Ancestral_genome/Rotenone/V10/"
figure_folder = "Midbrain/Ancestral_genome/Figure7/Versions_main/V4/"; dir.create(figure_folder)
figure4_folder = "Midbrain/Ancestral_genome/Figure4/Versions_main/V8/"
library(muscat)
library(SingleCellExperiment)
library(dreamlet)
library(scattermore)
library(cowplot)
library(ggplot2)
library(qvalue)
library(purrr)
library(tidyverse)
library(dplyr)
library(zenith)
library(cowplot)
library(ggrepel)
library(EnrichmentBrowser)
library(GO.db)
library(GSEABase)
library(ComplexHeatmap)
library(patchwork)
library(reshape2)
library(circlize)
library(extrafont)
library(ggupset)
library(colorspace)
library(ggraph)
library(igraph)
library(data.table)
loadfonts(device = "pdf")
info <- capture.output(sessionInfo()); writeLines(info, paste0(figure_folder, "session_info.txt")) #save sessionInfo in case there are any errors due to versions of packages

#Setup and constants----
celltype = 'DA_neurons'
p_thresh_cond = 0.05
p_thresh_sp_cond = 0.1
fc_thresh = 0
species_names = c('human','chimp','macaque','orangutan')
colors_species = c(human = '#F59121',chimp = '#3957A6',macaque = '#7E2859')
colors_species_cond = c(human_CNTRL = '#F59121',chimp_CNTRL = '#3957A6',human_24H = '#F59121', chimp_24H = '#3957A6', human_72H = '#F59121', chimp_72H = '#3957A6')
colors_species_dark = c(human = rgb(208/255,109/255,0/255), chimp = rgb(18/255,48/255,128/255))
colors_4species = c(human = '#F59121',chimp = '#3957A6',orangutan = '#754D27',macaque = '#7E2859')
colors_polarize = c('human_specific' = '#F59121','chimp_specific' = '#3957A6',
                    'divergent' = '#079655', 'other' = 'black')
base_coef = c('species_conditionhuman_CNTRL','species_conditionhuman_24H','species_conditionhuman_72H','species_conditionchimp_CNTRL','species_conditionchimp_24H','species_conditionchimp_72H')

#Load raw data----
res_dl <- readRDS(paste0(rot_folder,'res_dl.rds'))
res_proc <- readRDS(paste0(rot_folder,'res_proc.rds'))
go_human <- readRDS(paste0(figure_folder,'go_human.rds'))
sce <- readRDS(paste0(rot_folder,'sce.rds'))

#Calculations - only need to run once ----
#Removing ribosomal genes
n <- names(go_human); rib_terms <- grep("ribosom", n, value = TRUE)
rib_genes <- list()
for (term in rib_terms){
  genes = go_human[[term]]
  rib_genes = append(rib_genes,genes@geneIds)
}
rib_genes = unlist(rib_genes)

res_dl_norib <- res_dl
a = res_dl_norib[[celltype]]
rows_to_keep <- !rownames(a) %in% rib_genes
b <- a[rows_to_keep, ]
res_dl_norib[[celltype]] = b
saveRDS(res_dl_norib,paste0(figure_folder,'res_dl_norib.rds'))
res_dl = res_dl_norib
saveRDS(res_dl,paste0(figure_folder,'res_dl.rds'))

contrasts = coefNames(res_dl);
contrasts <- contrasts[!sapply(contrasts, function(x) any(x == base_coef))]
de_genes = list()
all_genes = list()
celltype_list = list()
celltype_list_all = list()
for (con in contrasts) {
  df_con <- as.data.frame(topTable(res_dl, coef = con, number = Inf)) 
  df_celltype <- df_con[df_con$assay == celltype & df_con$adj.P.Val < p_thresh_cond, ]
  df_celltype_all <- df_con[df_con$assay == celltype,]
  df_celltype$gene_sign = paste0(sign(df_celltype$logFC),df_celltype$ID)
  celltype_list[[con]] = df_celltype
  celltype_list_all[[con]] = df_celltype_all
}
de_genes = celltype_list
all_genes = celltype_list_all
saveRDS(de_genes,paste0(figure_folder,'de_genes.rds'))
saveRDS(all_genes,paste0(figure_folder,'all_genes.rds'))

res_zenith_24hr = zenith_gsa(res_dl, coef = 'Condition24H', go_human)
res_zenith_species_24hr = zenith_gsa(res_dl, coef = 'Species24H', go_human)
saveRDS(res_zenith_24hr,paste0(figure_folder,'res_zenith_24hr.rds'))
saveRDS(res_zenith_species_24hr,paste0(figure_folder,'res_zenith_species_24hr.rds'))

#Condition DEGs
res_zenith_24hr_up = res_zenith_24hr[res_zenith_24hr$Direction=='Up',]
res_zenith_24hr_down = res_zenith_24hr[res_zenith_24hr$Direction=='Down',]
go_terms_up = res_zenith_24hr_up$Geneset[1:5]
go_terms_down = res_zenith_24hr_down$Geneset[1:5]
go_genes_up = c()
for (term in go_terms_up){
  go_genes_up = c(go_genes_up,geneIds(go_human[[term]]))
}
go_genes_down = c()
for (term in go_terms_down){
  go_genes_down = c(go_genes_down,geneIds(go_human[[term]]))
}
saveRDS(go_genes_down, paste0(figure_folder,'go_genes_down.rds'))
saveRDS(go_genes_up, paste0(figure_folder,'go_genes_up.rds'))

#Load processed data----
res_dl <- readRDS(paste0(figure_folder,'res_dl.rds'))
res_proc <- readRDS(paste0(rot_folder,'res_proc.rds'))
de_genes <- readRDS(paste0(figure_folder,'de_genes.rds'))
all_genes <- readRDS(paste0(figure_folder,'all_genes.rds'))
go_human <- readRDS(paste0(figure_folder,'go_human.rds'))
res_zenith_24hr <- readRDS(paste0(figure_folder,'res_zenith_24hr.rds'))
res_zenith_species_24hr <- readRDS(paste0(figure_folder,'res_zenith_species_24hr.rds'))
all_genes_d40 <- readRDS(paste0(figure4_folder,'all_genes.rds'))
go_genes_down <- readRDS(paste0(figure_folder,'go_genes_down.rds'))
go_genes_up <- readRDS(paste0(figure_folder,'go_genes_up.rds'))

#Main figure
##PanelE: Volcano plot for 24 hr----
genes_up = c('FOS','HSPH1','UQCR11')
genes_down = c('TUBB3','EPHA5','GRIK1')
selected_genes <- c(genes_up,genes_down)
con = 'Condition24H'
data = de_genes[[con]]
selected_data <- data %>% filter(ID %in% selected_genes)
panelE <- plotMyVolcano(res_dl, assay = celltype, coef = con, pt.size = 0.4, nGenes = 1)  +
  geom_point(data = selected_data, aes(x = logFC, y = -log10(P.Value)),size = 2.5, color = 'black', shape = 18)+
  geom_text_repel(data = selected_data, aes(x = logFC, y = -log10(P.Value), label = ID),size = 2, segment.size = 0.5)+
  theme_basic_smallest()+
  theme(aspect.ratio = 1,
        plot.title = element_blank(),
        plot.margin = unit(c(0,0,0,0),'in'),
        legend.position = 'none')
panelE

##PanelF: GO terms dotplot----
panelF_up = plotGOtermsDotplots(res_zenith_24hr,num_terms = 7,'Condition24H', direction = 'Up', size_limits = c(0.5,0.75),all_genes,de_genes)
panelF_up = panelF_up + theme(axis.title.x = element_blank(), axis.text.x = element_blank())+ xlim(0,0)
panelF_down = plotGOtermsDotplots(res_zenith_24hr,num_terms = 7,'Condition24H', direction = 'Down',size_limits = c(0.5,0.75),all_genes,de_genes)
panelF_down = panelF_down + xlim(0,0)
panelF = panelF_up/panelF_down 
panelF

##PanelG: Example conserved genes----
data = extractData(res_proc, celltype)
genes_plots_up <- plotSelectedGenesList_speciesCondition(data,'CNTRL',genes_up,colors_species,xlabels = FALSE)
genes_plots_down <- plotSelectedGenesList_speciesCondition(data,'CNTRL',genes_down, colors_species,xlabels = TRUE)
panelG_top <- wrap_plots(genes_plots_up)
panelG_bottom <- wrap_plots(genes_plots_down)

##PanelH: LogFC heatmap for ScenicPlus TFs----
tfs = c('PBX1','POU3F2','BACH2','CREB5','SOX4')
contrasts = c('Condition24H','Condition72H')
logfc_data_ggplot <- prepare_logfc_heatmap_data_condition(tfs, all_genes, 'DA_neurons', contrasts)
logfc_data_ggplot$gene <- factor(logfc_data_ggplot$gene, levels = rev(tfs))
logfc_data_ggplot$condition <- factor(logfc_data_ggplot$condition, levels = contrasts)
panelH <- ggplot(logfc_data_ggplot, aes(x = condition, y = gene, fill = logFC)) +
  geom_tile() +
  geom_text(aes(label = asterisks), color = "black", size = 2, vjust = 0.5, hjust = 0.5) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0, na.value = "gray") +
  theme_minimal() +
  labs(
    fill = "logFC"
  ) +
  scale_x_discrete(labels = c('24H','72H'))+
  theme(
    aspect.ratio = length(unique(logfc_data_ggplot$gene))/length(unique(logfc_data_ggplot$condition)),
    axis.text = element_text(size = 6, color = 'black'),
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    legend.key.size = unit(0.1, "in"),
    legend.title = element_text(size = 6),
    legend.text = element_text(size = 6)
  )

##PanelJ: Scatterplot for human vs chimp for control vs rotenone----
genes <- c('CAT','PRDX2','PXDN','PRDX4','PRDX3','NNT','PRDX5','DUOX1')
colors <- c('All' = 'gray30', 'GO' = 'black')
color_df <- data.frame(ID = all_genes$Species$ID, Celltype_category = 'All')
color_df <- color_df %>%
  dplyr::mutate(Celltype_category = ifelse(ID %in% genes, 'GO', 'All'))
result <- plotLogfcCorrelationContrasts(
  all_genes, con1 = 'Human_vs_Chimp_CNTRL', con2 = 'Human_vs_Chimp_24H', 'Human vs chimp logFC CNTRL', 'Human vs chimp logFC 24H',
  p_thresh = 1, labels = genes, label_top_n = NULL, filter_points = FALSE, na_to_zero = TRUE,
  color_by = color_df, colors = colors)
p <- result$plot + ggtitle("Species difference") + theme(plot.title = element_text(size = 6, hjust = 0.5))
panelJ <- p + xlim(-4,4) + ylim(-4,4) + geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") + theme(legend.position = 'none')
panelJ

##PanelL: Species*condition scatterplot----
#Scatterplot of activity fold change across species in one activity condition
genes <- c('BDNF','MCU')
coi = 'DA_neurons'; aoi = '24H'
degs_all = all_genes$Condition24H$ID#[all_genes$Condition24H$adj.P.Val<p_thresh_cond]
degs_species_cond = all_genes$Species24H$ID[all_genes$Species24H$adj.P.Val<p_thresh_sp_cond]
genes_to_label <- c('BDNF','MCU')
data = extractData(res_proc,coi)
human_data = data %>% filter(species=='human'); 
human_control = human_data %>% filter(condition=='CNTRL');human_control = as.data.frame(dplyr::select(human_control,degs_all))
human_24H =  human_data %>% filter(condition==aoi); human_24H = as.data.frame(dplyr::select(human_24H,degs_all))
human_log_condition = human_24H - human_control
human_log_condition_mean = apply(human_log_condition,2,mean)
chimp_data = data %>% filter(species=='chimp'); 
chimp_control = chimp_data %>% filter(condition=='CNTRL');chimp_control = as.data.frame(dplyr::select(chimp_control,degs_all))
chimp_24H =  chimp_data %>% filter(condition==aoi); chimp_24H = as.data.frame(dplyr::select(chimp_24H,degs_all))
chimp_log_condition = chimp_24H - chimp_control
chimp_log_condition_mean = apply(chimp_log_condition,2,mean)
hc_data <- data.frame(human = human_log_condition_mean, chimp = chimp_log_condition_mean)
min_lim = min(hc_data$human,hc_data$chimp); max_lim = max(hc_data$human,hc_data$chimp)
hc_data$Gene = rownames(hc_data)
hc_data$IsInterest <- ifelse(hc_data$Gene %in% genes_to_label, "Interest", "Not Interest")
hc_data$IsGO <- ifelse(hc_data$Gene %in% go_genes_up, 
                       "GO_up", 
                       ifelse(hc_data$Gene %in% go_genes_down, "GO_down", "Not GO"))
hc_data$IsDEG <- ifelse(hc_data$Gene %in% degs_species_cond, "Sig", "Not sig")
panelL <- ggplot(hc_data, aes(x = human, y = chimp)) +
  # Plot all points with smaller size and black color by default
  geom_point(alpha = 0.5, size = 0.5, shape = 16, color = "gray30") +
  #Plot different shape for species*cond DEGs
  geom_point(data = subset(hc_data, IsDEG == "Sig"), 
             aes(fill = IsGO), stroke = NA, size = 1, shape = 24, alpha = 0.7) +
  # Plot GO_up and GO_down points with larger size and specific colors on top
  geom_point(data = subset(hc_data, IsGO == "GO_down"), 
             aes(color = IsGO), size = 0.5, shape = 16, alpha = 0.7) +
  geom_point(data = subset(hc_data, IsGO == "GO_up"), 
             aes(color = IsGO), size = 0.5, shape = 16, alpha = 0.7) +
  #Plot genes of interest
  geom_point(data = subset(hc_data, IsInterest == "Interest"),aes(fill = IsGO),size = 2,shape = 24,stroke = 1)+
  geom_label(data = hc_data[hc_data$Gene == 'MCU',], aes(x = -1.5, y = 2, label = Gene),size = 2) +
  geom_label(data = hc_data[hc_data$Gene == 'BDNF',], aes(x = 2, y = -1, label = Gene),size = 2) +
  # Reference line
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  labs(x = "Human 24H hr vs CNTRL", y = "Chimp 24H hr vs CNTRL") +
  # Custom colors for GO_up and GO_down
  scale_color_manual(values = c("GO_up" = "red", "GO_down" = "blue"))+
  scale_fill_manual(values = c("GO_up" = "red", "GO_down" = "blue", 'Not GO' = "black"))+
  coord_fixed() +
  theme_basic_smallest() +
  ggtitle("Species-condition difference") +
  theme(aspect.ratio = 1,
    plot.title = element_text(size = 6, hjust = 0.5),
    plot.margin = unit(c(0, 0, 0, 0), 'in'),
    legend.position = 'none'
  ) +
  xlim(-4, 4) +
  ylim(-4, 4)
panelL

##PanelK: Genes from Fig 4 GO term----
genes <- c('CAT','PRDX2','PXDN','PRDX4','PRDX3','NNT','PRDX5','DUOX1')
data = extractData(res_proc, celltype)
plot_list_box <- plotSelectedGenesListCondition(genes, '~ species_condition', data, celltype, colors_species_cond, xlabels = TRUE)
df1 <- all_genes_d40$D40_100$DA_neurons$human_vs_chimp %>%
  dplyr::filter(ID %in% genes) %>%
  dplyr::mutate(dataset = "Dev", order = 1, col = "black")
df2 <- all_genes$Human_vs_Chimp_CNTRL %>%
  dplyr::filter(ID %in% genes) %>%
  dplyr::mutate(dataset = "CNTRL", order = 2, col = "red")
df3 <- all_genes$Human_vs_Chimp_24H %>%
  dplyr::filter(ID %in% genes) %>%
  dplyr::mutate(dataset = "24H", order = 3, col = "darkgreen")
df4 <- all_genes$Human_vs_Chimp_72H %>%
  dplyr::filter(ID %in% genes) %>%
  dplyr::mutate(dataset = "72H", order = 4, col = "blue")

dataset_lookup <- tibble::tibble(
  dataset = c("Dev","CNTRL","24H","72H"),
  order   = c(1, 2, 3, 4),
  col     = c("gray","red","darkgreen","blue")  # use "green" if you prefer
)

comb <- bind_rows(df1, df2, df3, df4) %>%
  dplyr::select(ID, dataset, adj.P.Val, logFC) %>%
  dplyr::mutate(ID = as.character(ID)) %>%
  tidyr::complete(ID = genes, dataset = dataset_lookup$dataset,
                  fill = list(adj.P.Val = NA_real_)) %>%
  dplyr::left_join(dataset_lookup, by = "dataset") %>%
  dplyr::arrange(ID, order)

# Fill if logfc positive, otherwise white with colored border
comb <- comb %>%
  mutate(
    is_fill  = sign(logFC) > 0,
    fill_col = ifelse(is_fill, col, "white"),
    border   = col,
    # Asterisk labels
    stars = case_when(
      !is.na(adj.P.Val) & adj.P.Val < 0.01 ~ "***",
      !is.na(adj.P.Val) & adj.P.Val < 0.05  ~ "**",
      !is.na(adj.P.Val) & adj.P.Val < 0.1  ~ "*",
      TRUE ~ ""
    )
  )

# Plotting helper for one gene
plot_one_gene <- function(g) {
  dat <- filter(comb, ID == g)
  ggplot(dat, aes(x = order, y = 1)) +
    geom_tile(aes(fill = fill_col, color = border), linewidth = 0.5) +
    geom_text(aes(label = stars), vjust = 0.1, size = 2) +
    scale_fill_identity() +
    scale_color_identity() +
    scale_x_continuous(breaks = dat$order, labels = dat$dataset, expand = expansion(mult = c(0.05, 0.05))) +
    #coord_fixed(ratio = 1) +
    labs(x = NULL, y = NULL) +
    theme_basic_smallest()+
    theme(
      axis.line = element_blank(),
      axis.text.y = element_blank(),
      axis.ticks = element_blank(),
      plot.title = element_text(face = "bold", hjust = 0.5),
      axis.text.x = element_blank(),
      plot.margin = unit(c(0,0,0,0.05),'in')
    )
}
plot_list_colors <- genes %>%
  unique() %>%
  set_names() %>%
  map(plot_one_gene)

panelK <- (wrap_plots(plot_list_box, nrow = 1) & theme(aspect.ratio = 1.7) & scale_x_discrete(breaks = c('human_CNTRL','human_24H','human_72H'),
                                                                       labels = c('CNTRL','24H','72H'))) /
  wrap_plots(plot_list_colors, nrow = 1) + plot_layout(heights = c(4,1))

##PanelMN: BDNF and MCU----
bdnf_plot <- plotSelectedGenesList_speciesCondition(data,'CNTRL','BDNF',colors_species,xlabels = FALSE, type = 'LFC')
bdnf_plot = bdnf_plot$BDNF + theme(aspect.ratio = 2) + theme(plot.margin = unit(c(0,0,0.08,0),'in'))
mcu_plot <-  plotSelectedGenesList_speciesCondition(data,'CNTRL','MCU',colors_species,xlabels = TRUE, type = 'LFC')
mcu_plot = mcu_plot$MCU + theme(aspect.ratio = 2)

##Putting together main figure----
figure7_EFG <- plot_spacer() / panelE | panelF | panelG_top / panelG_bottom + theme(plot.margin = unit(c(0,0,0,0),'in')) + plot_layout(widths = c(1,2,6))
figure7_EFG
ggsave(paste0(figure_folder,'figure7_EFG.pdf'), figure7_EFG, height = 3.5, width = 10, units = 'in')
ggsave(paste0(figure_folder,'panelH.pdf'), panelH, height = 1.4, width = 1.8, units = 'in')

figure7_JL = panelJ / panelL
figure7_JL
ggsave(paste0(figure_folder,'panelJL.pdf'), figure7_JL, height = 6, width = 2, units = 'in')

panelK
ggsave(paste0(figure_folder,'panelK.pdf'), panelK, height = 3.5, width = 5, units = 'in')

panelMN <- bdnf_plot + mcu_plot
panelMN
ggsave(paste0(figure_folder,'panelMN.pdf'),panelMN, height = 1.75, width = 2)

#Supplementary----
##PanelI: Scatterplot of fold change between 24 and 72 hr----
all_degs = unique(c(de_genes$Condition24$ID, de_genes$Condition72H$ID))
time24 = all_genes$Condition24H[all_genes$Condition24H$ID %in% all_degs,]
time72 = all_genes$Condition72H[all_genes$Condition72H$ID %in% all_degs,]
cond_data = inner_join(time24,time72, by = 'ID', suffix = c('_24','_72'))
cond_data$IsGO <- ifelse(cond_data$ID %in% go_genes_up, 
                       "GO_up", 
                       ifelse(cond_data$ID %in% go_genes_down, "GO_down", "Not GO"))
min_value = min(cond_data$logFC_24,cond_data$logFC_72); max_value = max(cond_data$logFC_24,cond_data$logFC_72)
supp_panelI <- ggplot(cond_data, aes(x = logFC_24, y = logFC_72)) +
  geom_point(color = 'black', alpha = 0.5, size = 0.5, shape = 16) +  # Plot all points with some transparency
  # Plot GO_up and GO_down points with larger size and specific colors on top
  geom_point(data = subset(cond_data, IsGO == "GO_down"), 
             aes(color = IsGO), size = 1.2, shape = 16, alpha = 0.8) +
  geom_point(data = subset(cond_data, IsGO == "GO_up"), 
             aes(color = IsGO), size = 1.2, shape = 16, alpha = 0.8) +
  scale_color_manual(values = c("GO_up" = "blue", "GO_down" = "red"))+
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  labs(x = 'Rotenone 24H logFC', y = 'Rotenone 72H logFC')+
  coord_fixed()+
  theme_basic_smallest()+
  ylim(min_value,max_value)+
  xlim(min_value,max_value)+
  theme(aspect.ratio = 1,
        plot.margin = unit(c(0,0,0,0),'in'),
        legend.position = 'none')
correlation_test <- cor.test(cond_data$logFC_24, cond_data$logFC_72)

#Example genes for GO categories
data = extractData(res_proc,celltype)
gene1_plot <- plotSelectedGenesList('UQCR11', '~ species_condition', data, celltype, colors_species_cond, xlabels = TRUE)
gene1_plot = gene1_plot$UQCR11  + scale_x_discrete(labels = c('CNTRL','24H','72H','CNTRL','24H','72H')) + ggtitle('UQCR11')
gene2_plot <- plotSelectedGenesList('COX8A', '~ species_condition', data, celltype, colors_species_cond, xlabels = TRUE)
gene2_plot = gene2_plot$COX8A  + scale_x_discrete(labels = c('CNTRL','24H','72H','CNTRL','24H','72H')) + ggtitle('COX8A')
gene1_plot + gene2_plot
ggsave(paste0(figure_folder,'Reviewer_gene_plots_upreg.pdf'), height = 3, width = 6)

gene1_plot <- plotSelectedGenesList('TUBB3', '~ species_condition', data, celltype, colors_species_cond, xlabels = TRUE)
gene1_plot = gene1_plot$TUBB3  + scale_x_discrete(labels = c('CNTRL','24H','72H','CNTRL','24H','72H')) + ggtitle('TUBB3')
gene2_plot <- plotSelectedGenesList('GRIN2B', '~ species_condition', data, celltype, colors_species_cond, xlabels = TRUE)
gene2_plot = gene2_plot$GRIN2B  + scale_x_discrete(labels = c('CNTRL','24H','72H','CNTRL','24H','72H')) + ggtitle('GRIN2B')
gene1_plot + gene2_plot
ggsave(paste0(figure_folder,'Reviewer_gene_plots_downreg.pdf'), height = 3, width = 6)

##PanelJ: scatterplot----
#Scatterplot of activity fold change across species in one activity condition
coi = 'DA_neurons'; aoi = '24H'
degs = all_genes$Condition24H$ID[all_genes$Condition24H$adj.P.Val<p_thresh_cond]
genes_to_label <- c('BDNF','MCU')
data = extractData(res_proc,coi)
human_data = data %>% filter(species=='human'); 
human_control = human_data %>% filter(condition=='CNTRL');human_control = as.data.frame(dplyr::select(human_control,degs))
human_24H =  human_data %>% filter(condition==aoi); human_24H = as.data.frame(dplyr::select(human_24H,degs))
human_log_condition = human_24H - human_control
human_log_condition_mean = apply(human_log_condition,2,mean)
chimp_data = data %>% filter(species=='chimp'); 
chimp_control = chimp_data %>% filter(condition=='CNTRL');chimp_control = as.data.frame(dplyr::select(chimp_control,degs))
chimp_24H =  chimp_data %>% filter(condition==aoi); chimp_24H = as.data.frame(dplyr::select(chimp_24H,degs))
chimp_log_condition = chimp_24H - chimp_control
chimp_log_condition_mean = apply(chimp_log_condition,2,mean)
hc_data <- data.frame(human = human_log_condition_mean, chimp = chimp_log_condition_mean)
min_lim = min(hc_data$human,hc_data$chimp); max_lim = max(hc_data$human,hc_data$chimp)
hc_data$Gene = rownames(hc_data)
hc_data$IsInterest <- ifelse(hc_data$Gene %in% genes_to_label, "Interest", "Not Interest")
hc_data$IsGO <- ifelse(hc_data$Gene %in% go_genes_up, 
                       "GO_up", 
                       ifelse(hc_data$Gene %in% go_genes_down, "GO_down", "Not GO"))
supp_panelJ <- ggplot(hc_data, aes(x = human, y = chimp)) +
  # Plot all points with smaller size and black color by default
  geom_point(alpha = 0.5, size = 0.5, shape = 16, color = "black") +  
  # Plot GO_up and GO_down points with larger size and specific colors on top
  geom_point(data = subset(hc_data, IsGO == "GO_down"), 
             aes(color = IsGO), size = 1.2, shape = 16, alpha = 0.8) +
  geom_point(data = subset(hc_data, IsGO == "GO_up"), 
             aes(color = IsGO), size = 1.2, shape = 16, alpha = 0.8) +
  # Reference line
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  labs(x = "Human 24H hr vs CNTRL", y = "Chimp 24H hr vs CNTRL") +
  # Custom colors for GO_up and GO_down
  scale_color_manual(values = c("GO_up" = "blue", "GO_down" = "red"))+
  coord_fixed() +
  xlim(min_lim, max_lim) +
  ylim(min_lim, max_lim) +
  theme_basic_smallest() +
  theme(
    aspect.ratio = 1,
    plot.title = element_blank(),
    plot.margin = unit(c(0, 0, 0, 0), 'in'),
    legend.position = 'none'
  )
ggsave(paste0(figure_folder,'Reviewer_rotenone_scatterplot.pdf'), supp_panelJ,height = 3, width = 3)

##PanelK-N: BDNF and MCU plots----
data = extractData(res_proc,celltype)
bdnf_plot <- plotSelectedGenesList('BDNF', '~ species_condition', data, celltype, colors_species_cond, xlabels = TRUE)
bdnf_plot = bdnf_plot$BDNF  + scale_x_discrete(labels = c('CNTRL','24H','72H','CNTRL','24H','72H')) + ggtitle('BDNF (Rotenone)')
mcu_plot <- plotSelectedGenesList('MCU', '~ species_condition', data, celltype, colors_species_cond, xlabels = TRUE)
mcu_plot = mcu_plot$MCU + scale_x_discrete(labels = c('CNTRL','24H','72H','CNTRL','24H','72H')) + ggtitle('MCU (Rotenone)')

res_procs <- readRDS(paste0(figure4_folder,'res_procs.rds'))
degs_polarized <- readRDS(paste0(figure4_folder,'degs_polarized.rds'))
data = extractData(res_procs$D40_100,celltype)
bdnf_allspecies_plot <- plotSelectedGenesList_withPolarize('BDNF', data, celltype, degs_polarized$D40_100, colors_4species,colors_polarize)
bdnf_allspecies_plot = bdnf_allspecies_plot$BDNF + ggtitle('BDNF (D40-100)')
mcu_allspecies_plot <- plotSelectedGenesList_withPolarize('MCU', data, celltype, degs_polarized$D40_100, colors_4species,colors_polarize)
mcu_allspecies_plot = mcu_allspecies_plot$MCU + ggtitle('MCU (D40-100)')

##Panel O species*condition GO terms dotplots----
supp_panelJ_up = plotGOtermsDotplots_nosize(res_zenith_species_24hr,num_terms = 7,'Species24H',
                                            direction = 'Up', all_genes,de_genes, color_low = 'skyblue', color_high = 'blue')
supp_panelJ_up = supp_panelJ_up + theme(axis.title.x = element_blank(), axis.text.x = element_blank())+ xlim(0,0)
supp_panelJ_down = plotGOtermsDotplots_nosize(res_zenith_species_24hr,num_terms = 7,'Species24H',
                                              direction = 'Down',all_genes,de_genes,color_low = 'pink', color_high = 'red')
supp_panelJ_down = supp_panelJ_down + xlim(0,0)
supp_panelJ = supp_panelJ_up/supp_panelJ_down 
supp_panelJ

ggsave(paste0(figure_folder,'supp_panelJ.pdf'), height = 2.5, width = 6, units = 'in')


##Putting together supp----
figure7supp <- supp_panelI + supp_panelJ + mcu_allspecies_plot + (mcu_plot + theme(axis.title.y = element_blank())) + 
                                                                (bdnf_allspecies_plot + theme(axis.title.y = element_blank())) + 
                                                                (bdnf_plot + theme(axis.title.y = element_blank())) +
                                                                   plot_layout(nrow = 1)
figure7supp
ggsave(paste0(figure_folder,'figure7supp.pdf'), height = 3.2, width = 8, units = 'in')

#Table S7----
file_path = paste0(figure_folder,'TableS7.xlsx')
wb <- createWorkbook()
# Rotenone 24 hour vs control contrast (average across both species)
tab_legend = paste0("Dreamlet results for all genes meeting expression cutoffs for rotenone 24 hour vs control contrast (average across both species).")
con = 'Condition24H'; tab <- all_genes[[con]]
addWorksheet(wb, con); writeData(wb, con, tab_legend, startRow = 1); writeData(wb, con, tab, startRow = 2)

# Rotenone 24 hour vs control contrast in human
tab_legend = paste0("Dreamlet results for all genes meeting expression cutoffs for rotenone 24 hour vs control contrast in human.")
con = 'Human24H'; tab <- all_genes[[con]]
addWorksheet(wb, con); writeData(wb, con, tab_legend, startRow = 1); writeData(wb, con, tab, startRow = 2)

# Rotenone 24 hour vs control contrast in chimp
tab_legend = paste0("Dreamlet results for all genes meeting expression cutoffs for rotenone 24 hour vs control contrast in chimp.")
con = 'Chimp24H'; tab <- all_genes[[con]]
addWorksheet(wb, con); writeData(wb, con, tab_legend, startRow = 1); writeData(wb, con, tab, startRow = 2)

# Species:condition contrast at 24 hours
tab_legend = paste0("Dreamlet results for all genes meeting expression cutoffs for species:condition contrast at 24 hours.")
con = 'Species24H'; tab <- all_genes[[con]]
addWorksheet(wb, con); writeData(wb, con, tab_legend, startRow = 1); writeData(wb, con, tab, startRow = 2)

# Rotenone 72 hour vs control contrast (average across both species)
tab_legend = paste0("Dreamlet results for all genes meeting expression cutoffs for rotenone 72 hour vs control contrast (average across both species).")
con = 'Condition72H'; tab <- all_genes[[con]]
addWorksheet(wb, con); writeData(wb, con, tab_legend, startRow = 1); writeData(wb, con, tab, startRow = 2)

# Rotenone 72 hour vs control contrast in human
tab_legend = paste0("Dreamlet results for all genes meeting expression cutoffs for rotenone 72 hour vs control contrast in human.")
con = 'Human72H'; tab <- all_genes[[con]]
addWorksheet(wb, con); writeData(wb, con, tab_legend, startRow = 1); writeData(wb, con, tab, startRow = 2)

# Rotenone 72 hour vs control contrast in chimp
tab_legend = paste0("Dreamlet results for all genes meeting expression cutoffs for rotenone 72 hour vs control contrast in chimp.")
con = 'Chimp72H'; tab <- all_genes[[con]]
addWorksheet(wb, con); writeData(wb, con, tab_legend, startRow = 1); writeData(wb, con, tab, startRow = 2)

# Species:condition contrast at 72 hours
tab_legend = paste0("Dreamlet results for all genes meeting expression cutoffs for species:condition contrast at 72 hours.")
con = 'Species72H'; tab <- all_genes[[con]]
addWorksheet(wb, con); writeData(wb, con, tab_legend, startRow = 1); writeData(wb, con, tab, startRow = 2)

# Gene set enrichment analysis with Zenith for rotenone 24 hour vs control contrast
tab_legend = 'Results of gene set enrichment analysis with Zenith for rotenone 24 hour vs control contrast.'
con = 'Zenith24HR'; tab <- res_zenith_24hr
addWorksheet(wb, con); writeData(wb, con, tab_legend, startRow = 1); writeData(wb, con, tab, startRow = 2)

# Gene set enrichment analysis with Zenith for species*condition contrast at 24 hours
tab_legend = 'Results of gene set enrichment analysis with Zenith for species:condition contrast at 24 hours.'
con = 'ZenithSpecies24HR'; tab <- res_zenith_species_24hr
addWorksheet(wb, con); writeData(wb, con, tab_legend, startRow = 1); writeData(wb, con, tab, startRow = 2)
saveWorkbook(wb, file_path, overwrite = TRUE)

#Printing for text----
print(correlation_test)

#cell proportions by species for reviewer
prop_df <- as.data.frame(colData(sce)) %>%
  dplyr::group_by(condition, species) %>%
  dplyr::summarize(n = dplyr::n(), .groups = "drop_last") %>%
  dplyr::group_by(condition) %>%
  dplyr::mutate(proportion = n / sum(n))
print(prop_df)

#Relationship between pseudotime and transcriptional response
df <- read.csv(paste0(rot_folder,'pseudotime_3d_dist_vals.csv')) %>%
  dplyr::mutate(species = stringr::str_to_lower(species))

#Set up factors
df$species <- factor(df_24h$species, levels = c('chimp','human'))
df$individual <- as.factor(df$individual)
df$time <- factor(df$time, levels = c('CNTRL', '24H', '72H'))

# #full model
# model <- lmerTest::lmer(
#   X3d_euclidian_distance ~ scale(mean_pseudotime) + species + time + (1 | individual),
#   data = df
# )
# summary(model)

#24 hour only
df_24h <- df[df$time == "24H", ]
model_24h <- lm(
  X3d_euclidian_distance ~ scale(mean_pseudotime) + species,
  data = df_24h
)
summary(model_24h)

#72 hour only
df_72h <- df[df$time == "72H", ]
model_72h <- lm(
  X3d_euclidian_distance ~ scale(mean_pseudotime) + species,
  data = df_72h
)
summary(model_72h)

# helper function to add predictions
make_pred_df <- function(df, model) {
  newdata <- expand.grid(
    mean_pseudotime = seq(min(df$mean_pseudotime), max(df$mean_pseudotime), length.out = 100),
    species = levels(df$species)
  )
  
  newdata$pred <- predict(model, newdata = newdata)
  newdata
}

df_24h$species <- factor(df_24h$species)
df_72h$species <- factor(df_72h$species)

pred_24h <- make_pred_df(df_24h, model_24h)
pred_72h <- make_pred_df(df_72h, model_72h)

##PanelQ----
p1 <- ggplot(df_24h, aes(x = mean_pseudotime, y = X3d_euclidian_distance, color = species)) +
  geom_point(size = 1, shape = 16) +
  geom_line(data = pred_24h, aes(y = pred), linewidth = 0.4) +
  scale_color_manual(values = colors_species) +
  ylim(6,13)+
  labs(x = 'Mean pseudotime', y = 'Distance to control')+
  theme_basic_smallest() +
  theme(
        legend.position = 'none')
p2 <- ggplot(df_72h, aes(x = mean_pseudotime, y = X3d_euclidian_distance, color = species)) +
  geom_point(size = 1, shape = 16) +
  geom_line(data = pred_72h, aes(y = pred), linewidth = 0.4) +
  scale_color_manual(values = colors_species) +
  labs(x = 'Mean pseudotime', y = 'Distance to control')+
  ylim(6,13)+
  theme_basic_smallest() +
  theme(
        legend.position = 'none')
supp_panelq <- p1 | p2 + plot_layout(axes = 'collect_y')
supp_panelq
ggsave(paste0(figure_folder,'supp_panelq.pdf'), height = 1.8, width = 3)