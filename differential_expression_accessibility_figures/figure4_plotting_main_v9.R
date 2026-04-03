rm(list = ls()); gc()  ## remove any variable to start clean
source('/media/jenelle/4TB_disk/Dropbox/R_analysis/MyFunctions.R')
source('/media/jenelle/4TB_disk/Dropbox/R_analysis/MySceFunctions.R')
source('/media/jenelle/4TB_disk/Dropbox/R_analysis/MyPlottingFunctions.R')
source("/media/jenelle/4TB_disk/Dropbox/R_analysis/MyDreamletFunctions.R")
source("/media/jenelle/4TB_disk/Dropbox/R_analysis/MyUpsetFunctions.R")
source("plotStratify_mod.R")
d16_folder = "Midbrain/Ancestral_genome/D16/V8/"
d40_folder = "Midbrain/Ancestral_genome/D40_D100_D80/V22/"
figure_folder = "Midbrain/Ancestral_genome/Figure4/Versions_main/V8/"; dir.create(figure_folder)
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
library(openxlsx)
loadfonts(device = "pdf")
info <- capture.output(sessionInfo()); writeLines(info, paste0(figure_folder, "session_info.txt")) #save sessionInfo in case there are any errors due to versions of packages

#Setup and constants----
datasets = c('D16','D40_100'); num_datasets = length(datasets)
pval = 0.05
fc_thresh = 0
thresh_num_cells = 200
num_iter = 1000
title_size = 7
base_coef = c('specieshuman','specieschimp','speciesmacaque','speciesorangutan')
species_names = c('human','chimp','macaque','orangutan')
colors_species = c(human = '#F59121',chimp = '#3957A6',macaque = '#7E2859')
colors_polarize = c('human_specific' = '#F59121','chimp_specific' = '#3957A6',
                    'divergent' = '#079655', 'other' = 'black')
colors_polarize_upd = c('human_specific_Up' = '#F59121', 'human_specific_Down' = '#ffbb78',
                        'chimp_specific_Up' = '#3957A6', 'chimp_specific_Down' = '#aec7e8',
                        'divergent_Up' = '#079655', 'divergent_Down' = '#AFE8CD',
                        'other_Up' = 'gray50', 'other_Down' = 'gray75')
colors_species_ld <- list(
  human = c('#FFD7B5', '#B35900'), # Very light and very dark versions of #F59121
  chimp = c('#D1D7FF', '#1E3A73')  # Very light and very dark versions of #3957A6
)

#Order of celltypes - matching Figs 1 and 2
ctorder_16 = c("vFB_progenitors","vFB_progenitors_cycling_G1_S_phase","vFB_progenitors_cycling_M_G2_phase","Rostral_vMB_Caudal_vFB_progenitors",
               "vFB_vMB_progenitors_cycling","vMB_progenitors","vMB_progenitors_cycling_G1_S_phase","vMB_progenitors_cycling_M_G2_phase",
               "Caudal_vMB_progenitors","vMB_BP_progenitors","vMB_vHB_progenitors","Rostral_vHB_progenitors","vHB_progenitors_cycling",
               "Progenitors_high_ECM_Actin_regulation","Progenitors_subtype_unknown","Immature_neurons_and_motor_neurons","Glutamatergic_neurons")
ctorder_abbr_16 = c("vFB prog.", "vFB prog. (G1/S)", "vFB prog. (M/G2)","vFB/vMB prog.","vFB/vMB prog. (cyc)",
                    "vMB prog.","vMB prog. (G1/S)","vMB prog. (M/G2)", "Caudal vMB prog.","vMB/BP prog.","vMB/vHB prog.","Rostral vHB prog.",
                    "vHB prog. (cyc)","Prog. (high EMC/actin-reg)","Prog. (unknown)","Immature/motor neurons","Glut. neurons")
short_abbr_16 = c("vFB", "vFB (G1/S)", "vFB (M/G2)","vFB/vMB","vFB/vMB (cyc)",
                  "vMB","vMB (G1/S)","vMB (M/G2)", "Caudal vMB","vMB/BP","vMB/vHB","Rostral vHB",
                  "vHB (cyc)","Prog. (high EMC/actin-reg)","Prog. (unknown)","Immature/motor neurons","Glut. neurons")
name_map_16 = setNames(ctorder_abbr_16, ctorder_16)
ctorder_40 = c("DA_neurons","DA_STN_neurons_immature","STN_neurons","MB_glutamatergic_neurons","MB_HB_glutamatergic_neurons",
               "MB_HB_glutamatergic_neurons_immature","MB_HB_neurons_LHX1","Oculomotor_neurons","MB_GABAergic_neurons",
               "Hypothalamic_neurons","Ventral_FB_MB_progenitors","Ventral_FB_MB_progenitors_cycling","FB_progenitors",
               "Lateral_MB_progenitors","Lateral_MB_progenitors_cycling","Progenitors_cycling","MB_HB_FP_cells",
               "Glial_progenitors_astrocytes")
ctorder_abbr_40 = c("DA neurons","DA/STN immature","STN neurons","MB glut. neurons","MB/HB glut. neurons","MB/HB glut. immature",
                    "MB/HB neurons LHX1","Oculo neurons","MB GABA neurons", "Hypothal neurons","Ventral FB/MB prog.",
                    "Ventral FB/MB (cyc)","FB prog.","Lateral MB prog.","Lateral MB prog. (cyc)", "Prog. (cyc)",
                    "MB/HB FP cells","Glial prog./astrocytes")
short_abbr_40 = c("DA","DA/STN imm.","STN","MB glut.","MB/HB glut.","MB/HB glut. imm.",
                  "MB/HB LHX1","Oculo","MB GABA", "Hypothal","Ventral FB/MB prog.",
                  "Ventral FB/MB (cyc)","FB prog.","Lateral MB prog.","Lateral MB prog. (cyc)", "Prog. (cyc)",
                  "MB/HB FP cells","Glial prog./astrocytes")
name_map_df = data.frame(cell_type = c(ctorder_16,ctorder_40),abbr = c(ctorder_abbr_16,ctorder_abbr_40), short_abbr = c(short_abbr_16, short_abbr_40))
rownames(name_map_df) = name_map_df$cell_type
celltypes_upset_d16 = c('vMB_progenitors', 'vFB_progenitors', 'Caudal_vMB_progenitors', 'Rostral_vHB_progenitors')
celltypes_upset_d40 = c('MB_HB_neurons_LHX1', 'DA_neurons','MB_HB_glutamatergic_neurons','DA_STN_neurons_immature','STN_neurons')
celltypes_upset = list(celltypes_upset_d16,celltypes_upset_d40); names(celltypes_upset) = datasets
celltypes_da = c('DA_neurons','DA_STN_neurons_immature','vMB_progenitors')

#Calculations (only need once then load results)----
#Load data
folders = c(d16_folder, d40_folder); names(folders) = datasets
res_procs = vector(mode = 'list', length= num_datasets); names(res_procs) = datasets 
res_dls_orig = vector(mode = 'list', length= num_datasets); names(res_dls_orig) = datasets 
sces = vector(mode = 'list', length= num_datasets); names(sces) = datasets 

for (dataset in datasets){
  res_procs[[dataset]] = readRDS(paste0(folders[[dataset]],"res_proc.rds"))
  res_dls_orig[[dataset]] = readRDS(paste0(folders[[dataset]],"res_dl.rds"))
  sces[[dataset]] = readRDS(paste0(folders[[dataset]],"sce.rds"))
}
saveRDS(res_procs, paste0(figure_folder,'res_procs.rds'))

#Filter celltypes based on number of cells for human and chimp
species = c('human','chimp')
df_cell_nums = vector(mode = "list", length = num_datasets); names(df_cell_nums) = datasets
for (dataset in datasets){
  metadata <- as.data.frame(colData(sces[[dataset]]))
  cell_counts <- metadata %>%
    group_by(cell_type, species) %>%
    summarise(num_cells = n())
  exclude_celltypes = remove_celltypes_below_threshold(cell_counts, species, thresh_num_cells)
  df_cell_nums[[dataset]] <- cell_counts %>%
    pivot_wider(names_from = species, values_from = num_cells, names_prefix = "num_cells_")  %>%
    filter(cell_type %!in% exclude_celltypes)
  res_dl = res_dls_orig[[dataset]]
  res_dls_orig[[dataset]] = res_dl[names(res_dl) %!in% exclude_celltypes]
}
df_summary_cell_nums = bind_rows(df_cell_nums[['D16']],df_cell_nums[['D40_100']])
df_summary_cell_nums <- df_summary_cell_nums %>% left_join(name_map_df, by = "cell_type")
saveRDS(df_summary_cell_nums, paste0(figure_folder,'df_summary_cell_nums.rds'))
saveRDS(res_dls_orig,paste0(figure_folder,'res_dls_orig.rds'))

#Removing ribosomal genes
#go_human = get_GeneOntology(to="SYMBOL", org = 'hsa') #using human annotations
#saveRDS(go_human, paste0(figure_folder,'go_human.rds'))
res_dls = vector(mode = 'list', length= num_datasets); names(res_dls) = datasets 
go_human <- readRDS(paste0(figure_folder,'go_human.rds'))
n <- names(go_human); rib_terms <- grep("ribosom", n, value = TRUE)
rib_genes <- list()
for (term in rib_terms){
  genes = go_human[[term]]
  rib_genes = append(rib_genes,genes@geneIds)
}
rib_genes = unique(unlist(rib_genes))

for (dataset in datasets){
  res_dl_norib <- res_dls_orig[[dataset]]
  for (assay in names(res_dl_norib)){
    a = res_dl_norib[[assay]]
    rows_to_keep <- !rownames(a) %in% rib_genes
    b <- a[rows_to_keep, ]
    res_dl_norib[[assay]] = b
  }
  res_dls[[dataset]] = res_dl_norib
}
saveRDS(res_dls,paste0(figure_folder,'res_dls.rds'))

#DEGs without ribosomal genes
de_genes = vector(mode = 'list', length = num_datasets); names(de_genes) = datasets
all_genes = vector(mode = 'list', length = num_datasets); names(all_genes) = datasets
for (dataset in datasets){
  celltypes = names(res_dls[[dataset]])
  contrasts = coefNames(res_dls[[dataset]]); 
  contrasts <- contrasts[contrasts %!in% base_coef]
  for (celltype in celltypes) {
    celltype_list_de = list()
    celltype_list_all = list()
    for (con in contrasts) {
      df_con <- as.data.frame(topTable(res_dls[[dataset]], coef = con, number = Inf))
      df_celltype_all <- df_con[df_con$assay == celltype,]
      df_celltype <- df_celltype_all[df_celltype_all$adj.P.Val < pval, ]
      df_celltype$gene_sign = paste0(sign(df_celltype$logFC),df_celltype$ID)
      celltype_list_de[[con]] = df_celltype
      celltype_list_all[[con]] = df_celltype_all
    }
    de_genes[[dataset]][[celltype]] = celltype_list_de
    all_genes[[dataset]][[celltype]] = celltype_list_all
  }
}
saveRDS(de_genes,paste0(figure_folder,'de_genes.rds'))
saveRDS(all_genes,paste0(figure_folder,'all_genes.rds'))

res_zeniths = vector(mode = 'list', length = num_datasets); names(res_zeniths) = datasets
for (dataset in datasets){
  res_zeniths[[dataset]] = zenith_gsa(res_dls[[dataset]], coef = 'human_vs_chimp', go_human)
}
saveRDS(res_zeniths,paste0(figure_folder,'res_zeniths.rds'))

#Polarization
degs_polarized = vector(mode = 'list', length = num_datasets); names(degs_polarized) = datasets
panelD_plots = vector(mode = 'list', length = num_datasets); names(panelD_plots) = datasets
celltypes_to_polarize = list(celltypes_upset_d16, celltypes_upset_d40); names(celltypes_to_polarize) = datasets
for (dataset in datasets){
  degs_polarized[[dataset]] <- polarizeThreeSpeciesDEGs(res_dls[[dataset]], celltypes_to_polarize[[dataset]], pval, fc_thresh)
}
saveRDS(degs_polarized,paste0(figure_folder,'degs_polarized.rds'))

#Load processed data (if skipping calculations after running previously) ----
df_summary_cell_nums<- readRDS(paste0(figure_folder,'df_summary_cell_nums.rds'))
go_human <- readRDS(paste0(figure_folder,'go_human.rds'))
res_procs <- readRDS(paste0(figure_folder,'res_procs.rds'))
res_dls <- readRDS(paste0(figure_folder,'res_dls.rds'))
de_genes <- readRDS(paste0(figure_folder,'de_genes.rds'))
all_genes <- readRDS(paste0(figure_folder,'all_genes.rds'))
res_zeniths <- readRDS(paste0(figure_folder,'res_zeniths.rds'))
degs_polarized <- readRDS(paste0(figure_folder,'degs_polarized.rds'))

#Panel A: summary heatmap with dotplot----
#Heatmap
con = 'human_vs_chimp'
#Get all degs across all celltypes in both datasets
df_list <- list()
for (dataset in datasets){
  #dataset = c('D16')
  for (celltype in names(de_genes[[dataset]])) {
    df <- de_genes[[dataset]][[celltype]][[con]]
    df_list[[celltype]] <- df
  }
}
de_genes_all <- bind_rows(df_list)
all_degs = unique(de_genes_all$ID)

#Combine all data frames for all degs and add score column
df_list <- list()
for (dataset in datasets){
  for (celltype in names(de_genes[[dataset]])) {
    df <- all_genes[[dataset]][[celltype]][[con]]
    df_list[[celltype]] <- df
  }
}
long_df <- bind_rows(df_list)
long_df <- long_df %>% 
  filter(ID %in% all_degs) %>%
  mutate(celltype = assay)
long_df$score = long_df$logFC * -log10(long_df$P.Value)
long_df <- long_df %>%
  dplyr::select(ID, celltype, score)  # Adjust the column names if necessary
df_wide <- long_df %>%
  pivot_wider(names_from = celltype, values_from = score, values_fill = NA)
# Calculate correlation matrix
cor_matrix <- cor(df_wide[,-1], use = "pairwise.complete.obs", method = "pearson")
col_fun = colorRamp2(c(0, 1), c("white", "black"))
col_fun(seq(-3, 3))
col_indices <- match(colnames(cor_matrix), name_map_df$cell_type)
colnames(cor_matrix) <- name_map_df$abbr[col_indices]
row_indices <- match(rownames(cor_matrix), name_map_df$cell_type)
rownames(cor_matrix) <- name_map_df$abbr[row_indices]
panelA_heatmap <- Heatmap(cor_matrix,
                          name = 'correlation',
                          col = col_fun,
                          clustering_distance_rows = "pearson",
                          clustering_distance_columns = "pearson",
                          clustering_method_rows  = "complete",
                          clustering_method_columns  = "complete",
                          column_names_side = "bottom",
                          row_names_gp = gpar(fontsize = 6, fontfamily = 'Arial'),
                          column_names_gp = gpar(fontsize = 6, fontfamily = 'Arial'),
                          show_row_dend = FALSE,
                          show_row_names = FALSE,
                          show_column_dend = FALSE,
                          heatmap_legend_param = list(title_gp = gpar(fontsize = 6, fontface = 'plain'),labels_gp = gpar(fontsize = 6), tick_length = unit(0.1,'in')),
                          width = unit(1.85,"in"),
                          height = unit(1.85,"in")
)
ht_size = calc_heatmap_size(panelA_heatmap);
ht_size_adj = ht_size*1.1 #add a little extra but keep aspect ratio bc otherwise it is partially cut off (not sure why)
pdf(paste0(figure_folder,'panelA_heatmap.pdf'), height = ht_size_adj[1], width = ht_size_adj[2],pointsize = 6)
panelA_heatmap
dev.off()
celltype_order_indices = column_order((panelA_heatmap))
celltype_order = colnames(cor_matrix)[celltype_order_indices]

#Dotplot
con = 'human_vs_chimp'
df_de = vector(mode = 'list', length = num_datasets); names(df_de) = datasets
for (dataset in datasets){
  df_de[[dataset]] = res_dls[[dataset]] %>%
    topTable(coef=con, number=Inf) %>%
    as_tibble %>% 
    group_by(assay) %>%
    dplyr::rename(cell_type = assay)  %>%
    summarise( 
      nGenes = length(  adj.P.Val), 
      nDE = sum(  adj.P.Val < pval),
      pi1 = 1 - qvalue(P.Value)$pi0) %>%
    left_join(name_map_df, by = "cell_type") 
}
df_de_both <- rbind(df_de[['D16']],df_de[['D40_100']])
df_summary <- inner_join(df_summary_cell_nums,df_de_both,by = 'cell_type', suffix = c('',''))
df_summary <-  df_summary %>% mutate(lcd = min(num_cells_chimp,num_cells_human))

max_de <- max(df_summary$nDE)
max_cells <- max(df_summary$lcd)

n_breaks <- 5; gene_breaks <- seq(0, max_de, length.out = n_breaks)
rounded_breaks <- floor(gene_breaks / 1000) * 1000 

#Set same order as heatmap
df_summary$abbr = factor(df_summary$abbr, levels = rev(celltype_order))
panelA_dotplot <- ggplot(df_summary, aes(x = 1, y = abbr)) +
  geom_tile(fill = "white") +
  geom_point(aes(color = nDE, size = lcd)) +
  scale_color_gradient(name = "# DEGs", low = "white", high = "black", lim = c(0, max_de)) +
  scale_size_continuous(name = "# cells", limits = c(0, max_cells), breaks = rounded_breaks,  range = c(0.5,2.5)) +
  theme_basic() +
  theme(aspect.ratio = as.numeric(nrow(df_de[[dataset]])),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.y = element_blank(),
        axis.line = element_blank()) +
  xlab('') + ylab('')+
  theme(
    legend.position = 'bottom',
    text = element_text(size = 6), # Base text size
    axis.text = element_text(size = 6, color= 'black'),
    legend.title = element_text(size = 6),
    legend.text = element_text(size = 6),
    legend.key.size = unit(0.1, "in"),
    legend.ticks = element_line(color = 'black', linewidth = 0.5),
    plot.margin = margin(0.01, -0.01, 0.01, 0.01, "in"))
panelA_dotplot
ggsave(paste0(figure_folder,'panelA_dotplot.pdf'), height = 2.55, width = 4, units = 'in')

#Panel B: upset plots----
con = 'human_vs_chimp'
titles = list('Progenitor cell types (D16)','Neuronal cell types (D40-100)'); names(titles) = datasets
upset_plot = vector(mode = 'list', length = num_datasets); names(upset_plot) = datasets
dev_plot = vector(mode = 'list', length = num_datasets); names(dev_plot) = datasets

#Add set size to cell type names
upset_names_df = name_map_df
upset_names_df$with_num = character(length(upset_names_df$cell_type))
for (celltype in celltypes_upset_d16){
  celltype_index = which(upset_names_df$cell_type==celltype)
  upset_names_df[celltype_index,"with_num"] = paste0(upset_names_df[celltype_index,'short_abbr'],'(',dim(de_genes$D16[[celltype]]$human_vs_chimp)[1],')')
}
for (celltype in celltypes_upset_d40){
  celltype_index = which(upset_names_df$cell_type==celltype)
  upset_names_df[celltype_index,"with_num"] = paste0(upset_names_df[celltype_index,'short_abbr'],'(',dim(de_genes$D40_100[[celltype]]$human_vs_chimp)[1],')')
}

con = 'human_vs_chimp'
for (dataset in datasets){
  abbrs = upset_names_df[celltypes_upset[[dataset]],]$with_num
  #Upset plot
  tidy_genes_df <- flatten_list_for_ggupset(de_genes, celltypes_upset[[dataset]],con)
  tidy_genes <-  inner_join(tidy_genes_df,upset_names_df, by = 'cell_type')  %>%
    group_by(gene_sign) %>%
    summarise(celltypes = list(unique(with_num)), .groups = 'drop')
  upset_plot[[dataset]] <- ggplot(tidy_genes,aes(x=celltypes)) +
    geom_bar(fill = 'black') +
    scale_x_upset(order_by = "degree", sets = abbrs) +
    theme_basic_smallest()+
    labs(y = 'Human-chimp DEGs \nintersection size', title = titles[[dataset]])+
    scale_y_continuous(expand = expansion(mult = c(0, 0))) +
    theme_combmatrix(combmatrix.panel.point.size = 1, 
                     combmatrix.panel.line.size = 0.2,
                     axis.title.x = element_blank(),
                     axis.text.y = element_text(size=6))+
    theme(plot.margin = unit(c(0, 0, 0, 0.02), "in"),
          axis.title.x = element_blank(),
          plot.title = element_text(size = title_size, hjust = 0.5))
  #Calculate deviation
  celltypes_degs_sign = list()
  for (celltype in celltypes_upset[[dataset]]) {
    degs = de_genes[[dataset]][[celltype]]$human_vs_chimp
    celltypes_degs_sign[[celltype]] = degs$gene_sign
  }
  intersect_matrix <- fromListWithNames(celltypes_degs_sign)
  intersect_matrix <- intersect_matrix[, rev(celltypes_upset[[dataset]])] #reordering according to celltype list to match
  intersect_matrix$Name = rownames(intersect_matrix); intersect_matrix <- intersect_matrix[, c(ncol(intersect_matrix), 1:(ncol(intersect_matrix)-1))]
  deviation <- calculateUpsetDeviation(intersect_matrix, num_iter, celltypes_upset[[dataset]])
  deviation_sorted <- deviation %>% #Sort to match plot
    arrange(Degree, desc(Count))
  deviation_sorted$Category <- factor(deviation_sorted$Category, levels = deviation_sorted$Category)
  #Deviation plot
  dev_plot[[dataset]] <- ggplot(deviation_sorted, aes(x = Category, y = Deviation/100)) + 
    geom_col(fill = 'black') +  # geom_col uses stat="identity" which is suited for plotting actual values
    geom_hline(yintercept = 0, linetype = "solid", color = "black") + 
    theme_basic_smallest() + # Using a minimal theme for better aesthetics
    theme(axis.text.x = element_blank(),  # Remove x-axis text
          axis.ticks.x = element_blank(),  # Remove x-axis ticks
          axis.line.x = element_line(colour = "white"),
          plot.margin = unit(c(0,0,0,0),'in'))+  
    labs(y = "Deviation", x = 'Intersection')
}
#Combine
panelB = upset_plot[['D16']] + upset_plot[['D40_100']]  + theme(axis.title.y = element_blank()) + 
  dev_plot[['D16']] + dev_plot[['D40_100']] + theme(axis.title.y = element_blank()) + plot_layout(heights = c(2,1), widths = c(1,2))
panelB
ggsave(paste0(figure_folder,'panelB.pdf'), height = 3, width = 4.2, units = 'in')

#Panel DE:  polarization scatterplots and barplots----
category_levels = c('human_specific','chimp_specific','divergent','other')
panelD_plots = vector(mode = 'list', length = num_datasets); names(panelD_plots) = datasets
cois = c('vMB_progenitors','DA_neurons'); names(cois) = datasets
genes_to_label = list(c(),c('KCNJ16','CAT','PRDX2')); names(genes_to_label) = datasets

for (dataset in datasets){
  coi = cois[[dataset]]
  result = summarizePolarizedDEGs(degs_polarized[[dataset]], colors_polarize_upd)
  degs_summary_upd <- result$degs_summary_upd %>%
    filter(CellType == coi)
  panelD_bar <- ggplot(degs_summary_upd, aes(x = CellType, y = Count, fill = SpecReg)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.9)) + 
    scale_fill_manual(values = colors_polarize_upd) +
    theme_basic_smallest() +
    labs(x = '', y = '') +
    theme(axis.text.x = element_blank(),
          legend.position = 'none') +
    scale_y_continuous(limits = c(0, NA), expand = c(0, 0))
  
  panelD_scatter <- plotPolarizedScatterplot(degs_polarized[[dataset]], coi, res_procs[[dataset]], genes_to_label[[dataset]], 'human', 'chimp')
  panelD_scatter <- panelD_scatter + theme(plot.margin = margin(0, 0, 0, 0, "in"))
  panelD_plots[[dataset]] <- panelD_scatter + inset_element(panelD_bar, left = 0.6, bottom = 0, right = 1, top = 0.4)
}
saveRDS(degs_polarized,paste0(figure_folder,'degs_polarized.rds'))

panelD = panelD_plots$D16 + panelD_plots$D40_100 
panelD
ggsave(paste0(figure_folder,'panelD.pdf'),height = 2.7, width = 5.4)

#Panel EF: Go terms----
coef = 'human_vs_chimp'
res_zenith_vmb = res_zeniths[['D16']][res_zeniths[['D16']]$assay=='vMB_progenitors',]
res_zenith_da_imm = res_zeniths[['D40_100']][res_zeniths[['D40_100']]$assay=='DA_STN_neurons_immature',]
res_zenith_da =  res_zeniths[['D40_100']][res_zeniths[['D40_100']]$assay=='DA_neurons',]

#Combine and rename assays 
celltypes_da_abbr = name_map_df$abbr[match(celltypes_da, name_map_df$cell_type)]
res_dl_da = res_dls$D40_100[names(res_dls$D40_100) %in% celltypes_da]
res_dl_da$vMB_progenitors = res_dls$D16$vMB_progenitors #so all assays can be on same heatmap (works because same contrasts)
names(res_dl_da)[names(res_dl_da) %in% celltypes_da] <- celltypes_da_abbr
saveRDS(res_dl_da, paste0(folder,'res_dl_da.rds'))

term = 'GO0019896: axonal transport of mitochondrion'
genes=geneIds(go_human[[term]])
panelE_heatmap <- plotGeneHeatmap(res_dl_da, coef=coef, assays = celltypes_da_abbr, genes= genes, transpose=TRUE, zmax = 6) + 
  labs(x = '', y = '', title = term)+
  theme(legend.position = "right",
        axis.text.x=element_text(size=6, angle=60, color = 'black'),
        axis.text.y=element_text(size=6, color = 'black'),
        plot.title = element_text(size = title_size,hjust = 0.5),
        legend.title = element_text(size=6),
        legend.ticks = element_line(colour = 'black', linewidth = 0.5),
        legend.text = element_text(size=6),
        legend.justification = 'center',
        legend.key.size = unit(0.1, "in"))
panelE_heatmap

term = 'GO0042744: hydrogen peroxide catabolic process'
genes=geneIds(go_human[[term]])
panelF_heatmap <- plotGeneHeatmap(res_dl_da, coef=coef, assays = celltypes_da_abbr, genes= genes, transpose=TRUE) + 
  labs(x = '', y = '', title = term)+
  theme(legend.position = "right",
        axis.text.x=element_text(size=6, angle=60, color = 'black'),
        axis.text.y=element_blank(),
        plot.title = element_text(size = title_size,hjust = 0.5),
        legend.title = element_text(size=6),
        legend.ticks = element_line(colour = 'black', linewidth = 0.5),
        legend.text = element_text(size=6),
        legend.justification = 'center',
        legend.key.size = unit(0.1, "in"))
panelF_heatmap

panelEF_heatmap = panelE_heatmap + panelF_heatmap + plot_layout(guides = 'collect')
panelEF_heatmap
ggsave(paste0(figure_folder,'panelEF_heatmap.pdf'),panelEF_heatmap, height = 1.5, width = 6.85)


coi = 'DA_STN_neurons_immature'
genes_sig = c('MAPT','NEFL','UCHL1','TRAK1','AGTPBP1','SYBU')
data = extractData(res_procs$D40_100,coi)
filtered_data <- data %>% 
  filter(species != 'orangutan') %>%
  droplevels()
panelE_list <- plotSelectedGenesList_withPolarize(genes_sig, filtered_data, coi, degs_polarized$D40_100, colors_species,colors_polarize)

coi = 'DA_neurons'
genes_sig = c('CAT','PRDX2','PXDN','PRDX4','PRDX3','NNT','PRDX5','DUOX1')
data = extractData(res_procs$D40_100,coi)
filtered_data <- data %>% filter(species != 'orangutan') %>%
  droplevels()
panelF_list <- plotSelectedGenesList_withPolarize(genes_sig, filtered_data, coi, degs_polarized$D40_100, colors_species,colors_polarize)

panelEF_list = c(panelE_list,list(plot_spacer()),panelF_list)
panelEF_genes = wrap_plots(panelEF_list, nrow = 1) 
panelEF_genes
ggsave(paste0(figure_folder,'panelEF_genes.pdf'),panelEF_genes, height = 2, width = 6.85)

# PanelG: KCNJ16 cell type specificity----
pb <- readRDS(paste0(d40_folder, 'pb.rds'))
celltype_pol_spec = list()
da_lin_celltypes = c('DA_neurons', 'STN_neurons','DA_STN_neurons_immature','Ventral_FB_MB_progenitors','Ventral_FB_MB_progenitors_cycling')
cois = c('DA_STN_neurons_immature','DA_neurons')
for (coi in cois){
  df_pol <- formatDEGsPolarizedLong(degs_polarized$D40_100, coi)
  df_pol <- df_pol %>%
    dplyr::filter(specificity %in% c('human_specific','divergent'))
  df_spec <- cellTypeSpecificity(pb, method = 'RLE')
  df_spec <- as.data.frame(df_spec) %>%
    mutate(da_lineage = rowSums(across(all_of(da_lin_celltypes)), na.rm = TRUE)) %>%
    rownames_to_column(var = "ID") %>%
    dplyr::select('da_lineage','ID')
  df_pol_spec <- inner_join(df_pol, df_spec, by = 'ID') %>%
    arrange(desc(da_lineage))
  celltype_pol_spec[[coi]] <- df_pol_spec
}

#Heatmap
species = c('human','chimp','macaque')
con = 'human_vs_chimp'
celltypes = c('Ventral_FB_MB_progenitors','DA_STN_neurons_immature','STN_neurons','DA_neurons','MB_HB_FP_cells')
panelG <- plotGeneCelltypeSpecHeatmap('KCNJ16',species,celltypes,res_procs$D40_100,all_genes$D40_100, con,name_map_df, 'black','gray90', sort_by = 'input')
panelG + theme(legend.position = 'bottom')
ggsave(paste0(figure_folder,'panelG.pdf'),height = 1.5, width = 3,units = 'in')

#Panels for other figures----
##Figure 2 cell type proportions----
#Make color map
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
celltype_colormap <- color_mapping[color_mapping != "skip"]
#Adjust to make it in the order cell types are listed in Fig 2F
celltype_order_fig2 <- c("DA_neurons","DA_STN_neurons_immature", "STN_neurons", "MB_glutamatergic_neurons", "MB_HB_glutamatergic_neurons","MB_HB_glutamatergic_neurons_immature",
                         "MB_HB_glutamatergic_neurons_immature","MB_HB_neurons_LHX1","Oculomotor_neurons","MB_GABAergic_neurons",
                         "Hypothalamic_neurons","Ventral_FB_MB_progenitors","Ventral_FB_MB_progenitors_cycling","FB_progenitors",
                         "Lateral_MB_progenitors","Lateral_MB_progenitors_cycling","Progenitors_cycling","MB_HB_FP_cells","Glial_progenitors_astrocytes")
celltype_colors <- celltype_colormap[celltype_order_fig2]
indiv_order <- c('H20961', 'H29089', 'H21792', 'H28834', 'H21194', 'H23555', 'H28126', 'H9',
                 'C8861', 'C4933', 'C3651', 'C3624', 'C40210', 'C40670', 'C40300', 'ES_Lyon')
sce <- readRDS(paste0(d40_folder,'sce.rds'))

df <- as.data.frame(colData(sce)) %>%
  dplyr::filter(time_point == 'D40') %>%
  dplyr::select(indiv, cell_type, pool_type) %>%
  dplyr::filter(indiv %in% indiv_order) %>%
  dplyr::filter(!is.na(indiv)) %>%
  dplyr::filter(!is.na(pool_type))
celltype_levels <- rev(unique(names(celltype_colors)))
pooltype_levels <- c('Intraspecies', 'Interspecies')
plot_df <- df %>%
  mutate(
    indiv      = factor(indiv, levels = indiv_order),
    cell_type  = factor(cell_type, levels = celltype_levels),
    pool_type  = factor(pool_type, levels = pooltype_levels),
    x = paste(indiv, pool_type, sep = " • ")
  )
x_levels <- expand.grid(indiv = indiv_order, pool_type = levels(plot_df$pool_type)) %>%
  arrange(indiv, pool_type) %>%
  transmute(x = paste(indiv, pool_type, sep = " • ")) %>%
  pull(x)
plot_df$x <- factor(plot_df$x, levels = x_levels)
fig2_panelI <- ggplot(plot_df, aes(x = x, fill = cell_type)) +
  geom_bar(position = "fill", width = 0.6) +
  scale_fill_manual(values = celltype_colors, drop = FALSE) +
  scale_y_continuous(breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0), expand = expansion(mult = c(0, 0.02))) +
  scale_x_discrete(labels = rep(indiv_order, each = 2)) +
  labs(x = NULL, y = "Proportion", fill = "Cell type") +
  theme_basic_smallest() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.23),
        legend.position = 'none'
  )
fig2_panelI
ggsave(paste0(figure_folder,'Fig2_panelI.pdf'), fig2_panelI, height = 1.7, width = 2.85, units = 'in')

##Figure 2 heatmap----
res_proc_combined = res_procs$D40_100
res_proc_combined$vMB_progenitors = res_procs$D16$vMB_progenitors
res_proc_combined$vFB_progenitors = res_procs$D16$vFB_progenitors
res_proc_combined$Caudal_vMB_progenitors = res_procs$D16$Caudal_vMB_progenitors
res_proc_combined$Rostral_vHB_progenitors = res_procs$D16$Rostral_vHB_progenitors
celltypes = c('MB_HB_neurons_LHX1','MB_HB_glutamatergic_neurons','STN_neurons','DA_STN_neurons_immature','DA_neurons',
              'vMB_progenitors','Caudal_vMB_progenitors','vFB_progenitors','Rostral_vHB_progenitors')
genes = c('SOX2','GAP43','PAX6','PAX5','GPC3','EN1','TH','NR4A2','LMX1A','PITX2','SLC17A6','LHX1')
heatmaps_plots <- plotHeatmapMarkerGenes(res_proc_combined, celltypes,genes,c('human','chimp'), colors_species_ld, name_map_df)
heatmaps_plots[['human']] + theme(axis.text.y = element_blank())+ heatmaps_plots[['chimp']]  + plot_layout(guides = 'collect') &   theme(legend.position = 'bottom')
ggsave(paste0(figure_folder,'Figure2_heatmap.pdf'),height = 3, width = 5, units = 'in')

heatmaps_plots <- plotHeatmapMarkerGenes(res_proc_combined, celltypes,genes,c('human','chimp'), colors_species_ld,name_map_df)
heatmaps_plots[['human']] + theme(axis.text.y = element_blank())+ heatmaps_plots[['chimp']]
ggsave(paste0(figure_folder,'Figure2_heatmap.pdf'),height = 3, width = 5, units = 'in')

##Figure 5 gene plots----
coi = 'DA_neurons'
genes_sig = c('GABRG3')
data = extractData(res_procs$D40_100,coi)
filtered_data <- data %>% 
  dplyr::filter(species != 'orangutan') %>%
  droplevels()
da <- plotSelectedGenesList_withPolarize(genes_sig, filtered_data, coi, degs_polarized$D40_100, colors_species,colors_polarize)
da$GABRG3
ggsave(paste0(figure_folder,genes_sig,'_',coi,'.pdf'), height = 2, width = 1, units = 'in')

coi = 'DA_neurons'
genes_sig = c('NRN1')
data = extractData(res_procs$D40_100,coi)
filtered_data <- data %>% 
  dplyr::filter(species != 'orangutan') %>%
  droplevels()
da <- plotSelectedGenesList_withPolarize(genes_sig, filtered_data, coi, degs_polarized$D40_100, colors_species,colors_polarize)
da$NRN1
ggsave(paste0(figure_folder,genes_sig,'_',coi,'.pdf'), height = 2, width = 1, units = 'in')

##Figure 6 plots----
#LogFC heatmap
tfs = c("ZFHX3", "POU2F2", "PBX1", "UNCX", "OTX2", "FOXP1", "PRRX1", "ATF6", "CREB5", "RFX3")
# res_proc_da = res_procs$D40_100[names(res_procs$D40_100) %in% celltypes_da]
# res_proc_da$vMB_progenitors = res_procs$D16$vMB_progenitors
# all_genes_da = all_genes$D40_100
# all_genes_da$vMB_progenitors = all_genes$D16$vMB_progenitors
# degs_polarized_da = degs_polarized$D40_100
# degs_polarized_da$vMB_progenitors = degs_polarized$D16$vMB_progenitors
celltypes = c('Ventral_FB_MB_progenitors','DA_STN_neurons_immature','DA_neurons')
logfc_data_ggplot <- prepare_logfc_heatmap_data(res_procs$D40_100, tfs, all_genes$D40_100, celltypes)
logfc_data_ggplot$gene <- factor(logfc_data_ggplot$gene, levels = rev(tfs))
logfc_data_ggplot$cell_type <- factor(logfc_data_ggplot$cell_type, levels = celltypes)
ggplot(logfc_data_ggplot, aes(x = cell_type, y = gene, fill = logFC)) +
  geom_tile() +
  geom_text(aes(label = asterisks), color = "black", size = 2, vjust = 0.5, hjust = 0.5) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0, na.value = "gray") +
  theme_minimal() +
  labs(
    fill = "logFC \n(Human vs Chimp)"
  ) +
  theme(
    aspect.ratio = length(unique(logfc_data_ggplot$gene))/length(unique(logfc_data_ggplot$cell_type)),
    axis.text.x = element_blank(),
    axis.text.y = element_text(size = 6, color = 'black'),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    legend.key.size = unit(0.1, "in"),
    legend.title = element_text(size = 6),
    legend.text = element_text(size = 6)
  )
ggsave(paste0(figure_folder,'selected_GRNs_logFC_heatmap.pdf'), width = 2, height = 2, units = 'in')

coi = 'DA_STN_neurons_immature'
genes_sig = c('ZFHX3')
data = extractData(res_procs$D40_100,coi)
filtered_data <- data %>% 
  dplyr::filter(species != 'orangutan') %>%
  droplevels()
da <- plotSelectedGenesList_withPolarize(genes_sig, filtered_data, coi, degs_polarized$D40_100, colors_species,colors_polarize)
da$ZFHX3
ggsave(paste0(figure_folder,genes_sig,'_',coi,'.pdf'), height = 2, width = 1, units = 'in')

coi = 'DA_STN_neurons_immature'
genes_sig = c('UNCX')
data = extractData(res_procs$D40_100,coi)
filtered_data <- data %>% 
  dplyr::filter(species != 'orangutan') %>%
  droplevels()
da <- plotSelectedGenesList_withPolarize(genes_sig, filtered_data, coi, degs_polarized$D40_100, colors_species,colors_polarize)
uncx_imm <- da$UNCX
coi = 'DA_neurons'
genes_sig = c('UNCX')
data = extractData(res_procs$D40_100,coi)
filtered_data <- data %>% 
  dplyr::filter(species != 'orangutan') %>%
  droplevels()
da <- plotSelectedGenesList_withPolarize(genes_sig, filtered_data, coi, degs_polarized$D40_100, colors_species,colors_polarize)
uncx_da <-da$UNCX
supp_panelP <- uncx_imm + uncx_da
ggsave(paste0(figure_folder,genes_sig,'.pdf'), height = 2, width = 2, units = 'in')

#Expression of TFs
plot_list = c()
for (celltype in celltypes_da){
  data = extractData(res_proc_da,celltype)
  filtered_data <- data %>% 
    filter(species != 'orangutan') %>%
    droplevels()
  plot_list[[celltype]] <- plotSelectedGenesList_withPolarize(tfs, filtered_data, celltype, degs_polarized_da, colors_species,colors_polarize)
  wrap_plots(plot_list[[celltype]], ncol = 3) + plot_annotation(title = celltype)
  ggsave(paste0(figure_folder,'tfs_expression_polarization_',celltype,'.pdf'), height = 6, width = 3, units = 'in')
}

#Table S4----
file_path = paste0(figure_folder,'TableS4.xlsx')
wb <- createWorkbook()
table_legend = paste0("Dreamlet results for all genes meeting expression cutoffs (hvc = human vs chimp contrast, hvm = human vs macaque contrast, cvm = chimp vs macaque contrast).")
cois = c('vMB_progenitors','vFB_progenitors','Caudal_vMB_progenitors','Rostral_vHB_progenitors',
         'MB_HB_neurons_LHX1','MB_HB_glutamatergic_neurons','STN_neurons')
cois_dataset = c('D16','D16','D16','D16','D40_100','D40_100','D40_100')
names(cois_dataset) = cois
for (coi in cois){
  degs_polarized_long <- formatDEGsPolarizedLong(degs_polarized[[cois_dataset[[coi]]]],coi)
  coi_tab <- all_genes[[cois_dataset[[coi]]]][[coi]]$human_vs_chimp %>%
    left_join(all_genes[[cois_dataset[[coi]]]][[coi]]$human_vs_macaque, by = "ID", suffix = c('_hvc','_hvm')) %>%
    left_join(all_genes[[cois_dataset[[coi]]]][[coi]]$chimp_vs_macaque %>%
                rename_with(~paste0(.x, "_cvm"), -ID), by = "ID", suffix = c('','_cvm')) %>%
    left_join(degs_polarized_long %>% dplyr::select(ID, specificity), by = 'ID') %>%
    mutate(polarization_category = specificity)
  addWorksheet(wb, coi)
  writeData(wb, coi, table_legend, startRow = 1)  
  writeData(wb, coi, coi_tab, startRow = 2) 
}
cois = c('DA_STN_neurons_immature','DA_neurons')
cois_dataset = c('D40_100','D40_100'); names(cois_dataset) = cois
for (coi in cois){
  degs_polarized_long <- formatDEGsPolarizedLong(degs_polarized[[cois_dataset[[coi]]]],coi)
  coi_tab <- all_genes[[cois_dataset[[coi]]]][[coi]]$human_vs_chimp %>%
    left_join(all_genes[[cois_dataset[[coi]]]][[coi]]$human_vs_macaque, by = "ID", suffix = c('_hvc','_hvm')) %>%
    left_join(all_genes[[cois_dataset[[coi]]]][[coi]]$chimp_vs_macaque %>%
                rename_with(~paste0(.x, "_cvm"), -ID), by = "ID", suffix = c('','_cvm')) %>%
    left_join(degs_polarized_long %>% dplyr::select(ID, specificity), by = 'ID') %>%
    mutate(polarization_category = specificity) %>%
    left_join(celltype_pol_spec[[coi]] %>% dplyr::select(ID,da_lineage), by = 'ID') %>%
    mutate(da_lineage_specificity_score = da_lineage)
  addWorksheet(wb, coi)
  writeData(wb, coi, table_legend, startRow = 1)  
  writeData(wb, coi, coi_tab, startRow = 2) 
}
#Tab for zenith results
table_legend = 'Results of gene set enrichment analysis with Zenith for DA neurons and immature DA/STN neurons for human vs chimp contrast.'
zenith_tab <- res_zeniths$D40_100[res_zeniths$D40_100$coef == 'human_vs_chimp' & res_zeniths$D40_100$assay %in% c('DA_neurons','DA_STN_neurons_immature'),] %>%
  arrange((PValue))
addWorksheet(wb, 'Zenith_results')
writeData(wb, 'Zenith_results', table_legend, startRow = 1)  
writeData(wb, 'Zenith_results', zenith_tab, startRow = 2) 
saveWorkbook(wb, file_path, overwrite = TRUE)

#Values for text----
#GO term ranks
res_zenith_da_imm_up = res_zenith_da_imm[res_zenith_da_imm$Direction=='Up',]
print(paste0('Rank for axonal transport of mitochondrion term in DA/STN neurons immature: ',which(res_zenith_da_imm_up$Geneset=='GO0019896: axonal transport of mitochondrion')))

res_zenith_da_up = res_zenith_da[res_zenith_da$Direction=='Up',]
print(paste0('Rank for hydrogen peroxide catabolic process term in DA neurons: ',which(res_zenith_da_up$Geneset=='GO0042744: hydrogen peroxide catabolic process')))

gois = c('KCNJ16','CAT')
cois = c('DA_STN_neurons_immature','DA_neurons')
contrasts = c('human_vs_chimp','human_vs_macaque','chimp_vs_macaque')
for (coi in cois){
  logfc_gois = vector(mode = 'list', length = length(contrasts)); names(logfc_gois) = contrasts
  pvals_gois = vector(mode = 'list', length = length(contrasts)); names(pvals_gois) = contrasts
  for (con in contrasts){
    for (goi in gois){
      goi_index = which(gois == goi)
      if (dim(de_genes$D40_100[[coi]][[con]][de_genes$D40_100[[coi]][[con]]$ID==goi,])[1] ==1){
        logfc_gois[[con]][goi_index] = de_genes$D40_100[[coi]][[con]][de_genes$D40_100[[coi]][[con]]$ID==goi,]$logFC
        pvals_gois[[con]][goi_index] = de_genes$D40_100[[coi]][[con]][de_genes$D40_100[[coi]][[con]]$ID==goi,]$  adj.P.Val
      } else {
        logfc_gois[[con]][goi_index] = NA
        pvals_gois[[con]][goi_index] = NA
      }
    }
  }
  summary_gois <- data.frame(gene = gois, logfc_human_vs_chimp = logfc_gois[['human_vs_chimp']], pval_human_vs_chimp = pvals_gois[['human_vs_chimp']],
                             logfc_human_vs_macaque = logfc_gois[['human_vs_macaque']], pval_human_vs_macaque = pvals_gois[['human_vs_macaque']],
                             logfc_chimp_vs_macaque = logfc_gois[['chimp_vs_macaque']], pval_chimp_vs_macaque = pvals_gois[['chimp_vs_macaque']])
  print(paste0(coi,':'))
  print(summary_gois)
}

#Cell type specificity ranks
print(paste0('Rank for KCNJ16 DA lineage specificity in DA neurons: ', which(celltype_pol_spec$DA_neurons$ID=='KCNJ16')))
print(paste0('Rank for KCNJ16 DA lineage specificity in immature DA neurons: ', which(celltype_pol_spec$DA_STN_neurons_immature$ID=='KCNJ16')))

#Inter- and intra-species pools
sce <- readRDS(paste0(d40_folder,'sce.rds'))
cells_to_keep <- colData(sce)$cell_type %in% 'DA_neurons'; sce_da <- sce[, cells_to_keep]
da_table <- table(sce_da$pool_type, sce_da$species)
print(paste0('Chimp interpsecies proportion :', da_table[1,1]/sum(da_table[1,1],da_table[2,1])))
print(paste0('Human interpsecies proportion :', da_table[1,2]/sum(da_table[1,2],da_table[2,2])))

#Response to reviewers----
term = 'GO0014059: regulation of dopamine secretion'
genes=geneIds(go_human[[term]])
plotGeneHeatmap(res_dl_da, coef=coef, assays = celltypes_da_abbr, genes= genes, transpose=TRUE, zmax = 6) + 
  labs(x = '', y = '', title = term)+
  theme(legend.position = "right",
        axis.text.x=element_text(size=6, angle=60, color = 'black'),
        axis.text.y=element_text(size=6, color = 'black'),
        plot.title = element_text(size = title_size,hjust = 0.5),
        legend.title = element_text(size=6),
        legend.ticks = element_line(colour = 'black', linewidth = 0.5),
        legend.text = element_text(size=6),
        legend.justification = 'center',
        legend.key.size = unit(0.1, "in"))

gene = 'CACNG3'
celltypes = names(res_dls$D40_100)
species = c('human','chimp','macaque')
con = 'human_vs_chimp'
#celltypes = c('Ventral_FB_MB_progenitors','DA_STN_neurons_immature','STN_neurons','DA_neurons','MB_HB_FP_cells')
p <- plotGeneCelltypeSpecHeatmap(gene,species,celltypes,res_procs$D40_100,all_genes$D40_100, con,name_map_df, 'black','pink', sort_by = 'input')
p + theme(legend.position = 'bottom') + theme(axis.text.x = element_text(size = 6))
ggsave(paste0(figure_folder,paste0(gene,'_specificity.png')),height = 3, width = 5,units = 'in')
