rm(list = ls()); gc()  ## remove any variable to start clean
source('/media/jenelle/4TB_disk/Dropbox/R_analysis/MyFunctions.R')
source('/media/jenelle/4TB_disk/Dropbox/R_analysis/MySceFunctions.R')
source('/media/jenelle/4TB_disk/Dropbox/R_analysis/MyPlottingFunctions.R')
source("/media/jenelle/4TB_disk/Dropbox/R_analysis/MyDreamletFunctions.R")
source("/media/jenelle/4TB_disk/Dropbox/R_analysis/MyUpsetFunctions.R")
source("plotMyVoom.R")
source("plotMyVolcano.R")
source("plotStratify_mod.R")
d16_folder = "Midbrain/Ancestral_genome/D16/V8/"
d40_folder = "Midbrain/Ancestral_genome/D40_D100_D80/V22/"
d40_alt_folder = "Midbrain/Ancestral_genome/D40_D100_D80/V29/"
d16_vp_folder = "Midbrain/Ancestral_genome/D16/V9/"
d40_vp_folder = "Midbrain/Ancestral_genome/D40_D100_D80/V28/"
base_folder = "Midbrain/Ancestral_genome/Figure4/";
main_folder = paste0(base_folder,"/Versions_main/V8/")
folder = paste0(base_folder,"/Versions_supp/V10/"); dir.create(folder)
library(muscat)
library(SingleCellExperiment)
library(dreamlet)
library(scattermore)
library(cowplot)
library(ggplot2)
library(qvalue)
library(tidyverse)
library(dplyr)
library(zenith)
library(cowplot)
library(ggrepel)
library(GO.db)
library(GSEABase)
library(ComplexHeatmap)
library(patchwork)
library(reshape2)
library(circlize)
library(extrafont)
library(ggupset)
library(data.table)
library(ggVennDiagram)
loadfonts(device = "pdf")
info <- capture.output(sessionInfo()); writeLines(info, paste0(folder, "session_info.txt")) #save sessionInfo in case there are any errors due to versions of packages

#Setup----
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
short_abbr_40 = c("DA","DA/STN imm.","STN","MB glut.","MB/HB glut.","MB/HB glut. immature",
                  "MB/HB LHX1","Oculo","MB GABA", "Hypothal","Ventral FB/MB prog.",
                  "Ventral FB/MB (cyc)","FB prog.","Lateral MB prog.","Lateral MB prog. (cyc)", "Prog. (cyc)",
                  "MB/HB FP cells","Glial prog./astrocytes")
name_map_df = data.frame(cell_type = c(ctorder_16,ctorder_40),abbr = c(ctorder_abbr_16,ctorder_abbr_40), short_abbr = c(short_abbr_16, short_abbr_40))
rownames(name_map_df) = name_map_df$cell_type
datasets = c('D16','D40_100'); num_datasets = length(datasets)
pval = 0.05
species_names = c('human','chimp','macaque','orangutan')
colors_species = c(human = '#F59121',chimp = '#3957A6',orangutan = '#754D27',macaque = '#7E2859')
colors_polarize = c('human_specific' = '#F59121','chimp_specific' = '#3957A6',
                    'divergent' = '#079655', 'other' = 'black')
base_coef = c('specieshuman','specieschimp','speciesmacaque','speciesorangutan')
con = 'human_vs_chimp'
vp_order_16 = c('assay','gene','species','indiv','experiment','sex','pool_type','lane','Residuals')
vp_order_40 = c('assay','gene','species','indiv','experiment','sex','day','pool_type','lane','Residuals')
vp_order_40_min = c('assay','gene','species','indiv','experiment','sex','day','Residuals')
vp_order_40_pseudo = c('assay','gene','species','indiv','experiment','sex','pseudotime','Residuals')
vp_colors = c(scales::hue_pal()(7), 'gray90') #exclude color for pool type
vp_colors_short = vp_colors[c(1:5,8)]

#Load processed data (calculated in main figure script)----
go_human <- readRDS(paste0(main_folder,'go_human.rds'))
res_dls <- readRDS(paste0(main_folder,'res_dls.rds'))
de_genes <- readRDS(paste0(main_folder,'de_genes.rds'))
all_genes <- readRDS(paste0(main_folder,'all_genes.rds'))
res_zeniths <- readRDS(paste0(main_folder,'res_zeniths.rds'))
res_dl_da <- readRDS(paste0(main_folder,'res_dl_da.rds'))
res_procs <- readRDS(paste0(main_folder,'res_procs.rds'))
degs_polarized <- readRDS(paste0(main_folder,'degs_polarized.rds'))

#Load data (not used in main figure)
folders = c(d16_vp_folder, d40_vp_folder); names(folders) = datasets
#Full vp for violin plots
vps_full = vector(mode = 'list', length = num_datasets); names(vps_full) = datasets
for (dataset in datasets){
  vps_full[[dataset]] = readRDS(paste0(folders[[dataset]],"vp_lst.rds"))
}
#Vp for current model with barplot
folders = c(d16_folder, d40_folder); names(folders) = datasets
vps = vector(mode = 'list', length = num_datasets); names(vps) = datasets
for (dataset in datasets){
  vps[[dataset]] = readRDS(paste0(folders[[dataset]],"vp_lst.rds"))
}
vp_pseudo <- readRDS(paste0(d40_alt_folder, 'vp_lst.rds'))
#Alternate D40 dreamlet model for DA lineage with pseudotime
res_dl_alt <- readRDS(paste0(d40_alt_folder, 'res_dl.rds'))

#Paneld: Voom plots----
#Combine into one object for plotting
#Combine into one object for plotting
num_d16 = length(ctorder_16)
res_proc_zoom_16 = res_procs$D16
names(res_proc_zoom_16) = name_map_df$short_abbr[1:num_d16]
num_d40 = length(ctorder_40)
res_proc_zoom_40 = res_procs$D40_100
names(res_proc_zoom_40) = name_map_df$short_abbr[(num_d16+1):(num_d16+num_d40)]
res_proc_zoom = res_proc_zoom_16
for (celltype in names(res_proc_zoom_40)){
  res_proc_zoom[[celltype]] = res_proc_zoom_40[[celltype]]
}

new_res_proc_zoom = res_proc_zoom
cois = c('vMB','vFB','Caudal vMB','Rostral vHB','MB/HB LHX1','DA','MB/HB glut.','DA/STN imm.','STN')
for (celltype in cois) {
  new_res_proc_zoom[[celltype]] <- res_proc_zoom[[celltype]]
}
paneld = plotMyVoom(new_res_proc_zoom,ncol = 9, assays = cois)+
  theme(plot.margin = unit(c(0,0.05,0,0),'in'))

#Panel e-h: Variance parition and volcano----
#D16 Variance partition and volcano
coi = 'vMB_progenitors'
vp_filter = vps_full$D16[vps_full$D16$assay == coi,]
vp_filter <- vp_filter[, vp_order_16]
colnames(vp_filter)[colnames(vp_filter)=='experiment']='expt'
panele <- plotVarPart(vp_filter) + 
  theme_basic_smallest()+
  labs(y = 'Variance (%)', title = name_map_df$abbr[name_map_df$cell_type == coi])+
  theme(legend.position = 'none',
        plot.title = element_text(size = 6,hjust = 0.5),
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 30, vjust = 0.6),
        plot.margin = unit(c(0,0.05,0,0),'in'))
coi = 'DA_STN_neurons_immature'
vp_filter = vps_full$D40_100[vps_full$D40_100$assay == coi,]
vp_filter <- vp_filter[, vp_order_40]
if (is.na(vp_filter$day[1])){
  vp_filter$day = vector(mode = 'double', length = length(vp_filter$day)) #set day to 0 instead of NA so it doesn't mess up plotting colors
}
colnames(vp_filter)[colnames(vp_filter)=='experiment']='expt'
panelf <- plotVarPart(vp_filter, outlier.size = 0.5) + 
  theme_basic_smallest()+
  labs(y = 'Variance (%)',title = name_map_df$abbr[name_map_df$cell_type == coi])+
  theme(legend.position = 'none',
        plot.title = element_text(size = 6,hjust = 0.5),
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 30, vjust = 0.6),
        plot.margin = unit(c(0,0.05,0,0),'in'))
coi = 'DA_neurons'
vp_filter = vps_full$D40_100[vps_full$D40_100$assay == coi,]
vp_filter <- vp_filter[, vp_order_40]
colnames(vp_filter)[colnames(vp_filter)=='experiment']='expt'
panelg <- plotVarPart(vp_filter, outlier.size = 0.5) + 
  theme_basic_smallest()+
  labs(y = 'Variance (%)',title = name_map_df$abbr[name_map_df$cell_type == coi])+
  theme(legend.position = 'none',
        plot.title = element_text(size = 6,hjust = 0.5),
        axis.text.x = element_text(angle = 30, vjust = 0.6),
        axis.title.x = element_blank(),
        plot.margin = unit(c(0,0.05,0,0),'in'))
coi = 'DA_neurons'
vp_filter = vp_pseudo[vp_pseudo$assay == coi,]
vp_filter <- vp_filter[, vp_order_40_pseudo]
panelh <- plotVarPart(vp_filter, outlier.size = 0.5) + 
  theme_basic_smallest()+
  labs(y = 'Variance (%)',title = name_map_df$abbr[name_map_df$cell_type == coi])+
  theme(legend.position = 'none',
        plot.title = element_text(size = 6,hjust = 0.5),
        axis.text.x = element_text(angle = 30, vjust = 0.6),
        axis.title.x = element_blank(),
        plot.margin = unit(c(0,0.05,0,0),'in'))
panelh
panelefgh <- panele |panelf | panelg |panelh

#Panel i-j: Variance partition for genes from main figure----
vp_lst = vps$D40_100
vp_lst <- vp_lst[, vp_order_40_min]

coi = 'DA_STN_neurons_immature'
genes_sig = c('MAPT','NEFL','UCHL1','TRAK1','AGTPBP1','SYBU')
paneli <- plotPercentBars(vp_lst[vp_lst$assay == coi & vp_lst$gene %in% genes_sig,],)+
  theme_basic_smallest()+
  scale_fill_manual(values = vp_colors_short)+
  theme(plot.title = element_text(size = 6, hjust = 0.5),
        axis.ticks.y = element_blank(),
        axis.line.y = element_blank(),
        axis.title.y = element_blank(),
        legend.position = 'none',
        aspect.ratio = 1,
        plot.margin = unit(c(0,0,0,0),'in'))

coi = 'DA_neurons'
genes_sig = c('CAT','PRDX2','PXDN','PRDX4','PRDX3','NNT','PRDX5','DUOX1')
panelj <- plotPercentBars(vp_lst[vp_lst$assay ==coi & vp_lst$gene %in% genes_sig,],)+
  theme_basic_smallest()+
  scale_fill_manual(values = vp_colors_short)+
  theme(plot.title = element_text(size = 6, hjust = 0.5),
        axis.ticks.y = element_blank(),
        axis.line.y = element_blank(),
        axis.title.y = element_blank(),
        legend.position = 'none',
        aspect.ratio = 1,
        plot.margin = unit(c(0,0,0,0),'in'))

#Panelk-m: volcano plots----
coi = 'vMB_progenitors'
panelk <- plotMyVolcano(res_dls$D16, assay = coi, coef = con, pt.size = 0.5, label_genes = '')  +
  scale_y_continuous(limits = c(0,35), expand = c(0,0), breaks = seq(0, 30, by = 10)) +
  theme_basic_smallest()+
  theme(plot.title = element_blank(),
        plot.margin = unit(c(0,0,0,0),'in'),
        legend.position = 'none')
coi = 'DA_STN_neurons_immature'
panell <- plotMyVolcano(res_dls$D40_100, assay = coi, coef = con, pt.size = 0.5, label_genes = c('MAPT','NEFL','UCHL1','TRAK1','AGTPBP1','SYBU'))+
  scale_y_continuous(limits = c(0,35), expand = c(0,0), breaks = seq(0, 30, by = 10)) +
  theme_basic_smallest()+
  theme(plot.title = element_blank(),
        plot.margin = unit(c(0,0,0,0),'in'),
        legend.position = 'none')
coi = 'DA_neurons'
panelm <- plotMyVolcano(res_dls$D40_100, assay = coi, coef = con, pt.size = 0.5, label_genes = c('KCNJ16','CAT','PRDX2'))+
  scale_y_continuous(limits = c(0,35), expand = c(0,0), breaks = seq(0, 30, by = 10)) +
  theme_basic_smallest()+
  theme(plot.title = element_blank(),
        plot.margin = unit(c(0,0,0,0),'in'),
        legend.position = 'none')

#Panel n-q Scatterplots for human vs macaque vMB and DA neurons----
category_levels = c('human_specific','chimp_specific','divergent','other')
panelno_plots = vector(mode = 'list', length = num_datasets); names(panelno_plots) = datasets
cois = c('vMB_progenitors','DA_neurons'); names(cois) = datasets
genes_to_label = list(c(),c('KCNJ16','CAT','PRDX2')); names(genes_to_label) = datasets

for (dataset in datasets){
  coi = cois[[dataset]]
  p <- plotPolarizedScatterplot(degs_polarized[[dataset]], coi, res_procs[[dataset]], genes_to_label[[dataset]], 'human', 'macaque')
  panelno_scatter <- p + theme(plot.margin = margin(0, 0.05, 0, 0, "in"), plot.title = element_text(size = 6, hjust = 0.5))
  panelno_plots[[dataset]]= panelno_scatter
}
paneln = panelno_plots$D16
panelo = panelno_plots$D40_100 

#Panel pq: scatterplots for DA neurons immature (human vs chimp and human vs macaque)----
dataset = 'D40_100'
panelpq_plots = vector(mode = 'list', length = num_datasets); names(panelpq_plots) = dataset
coi = 'DA_STN_neurons_immature'
genes_to_label = list(c('MAPT','NEFL','UCHL1','TRAK1','AGTPBP1','SYBU'),c('MAPT','NEFL','UCHL1','TRAK1','AGTPBP1','SYBU')); names(genes_to_label) = dataset

panelp <- plotPolarizedScatterplot(degs_polarized[[dataset]], coi, res_procs[[dataset]], genes_to_label[[dataset]], 'human', 'chimp')
panelp <- panelp + theme(plot.margin = margin(0, 0.05, 0, 0, "in"), plot.title = element_text(size = 6, hjust = 0.5))
panelq <- plotPolarizedScatterplot(degs_polarized[[dataset]], coi, res_procs[[dataset]], genes_to_label[[dataset]], 'human', 'macaque')
panelq <- panelq + theme(plot.margin = margin(0, 0.05, 0, 0, "in"), plot.title = element_text(size = 6, hjust = 0.5))

#Composing figure----
row1 <- paneld
ggsave(paste0(folder,'figure4_supp_d.pdf'), row1, height = 1, width = 6.85)
row2 <- panele + panelf + panelg + panelh + plot_layout(nrow = 1, axes = 'collect_y') 
row3 <- paneli + panelj + panelk + panell + panelm + plot_layout(nrow = 1, axes = 'collect_y') 
figure4_suppe_l = row2 / row3 + plot_layout(heights = c(0.8,1))
ggsave(paste0(folder,'figure4_supp_e_l.pdf'), figure4_suppe_l, height = 4.5, width = 7.5)

figure4_supp_nopq <- paneln  + panelo + panelp + panelq + plot_layout(nrow = 1)
ggsave(paste0(folder,'figure4_supp_nopq.pdf'), figure4_supp_nopq,height = 2, width = 7.5)

#Panel NO: GO terms genes with orangutan----
coi = 'DA_STN_neurons_immature'
genes_sig = c('MAPT','NEFL','UCHL1','TRAK1','AGTPBP1','SYBU')
data = extractData(res_procs$D40_100,coi)
panelN_list <- plotSelectedGenesList_withPolarize(genes_sig, data, coi, degs_polarized$D40_100, colors_species,colors_polarize)

coi = 'DA_neurons'
genes_sig = c('CAT','PRDX2','PXDN','PRDX4','PRDX3','NNT','PRDX5','DUOX1')
data = extractData(res_procs$D40_100,coi)
panelO_list <- plotSelectedGenesList_withPolarize(genes_sig, data, coi, degs_polarized$D40_100, colors_species,colors_polarize)

panelNO_list = c(panelN_list,list(plot_spacer()),panelO_list)
panelNO_genes = wrap_plots(panelNO_list, nrow = 1) 
ggsave(paste0(folder,'figure4_suppNO.pdf'),panelNO_genes, height = 2, width = 6.85)

#Comparing GO terms when using pseudotime in model----
#original V22
res_zenith_da_imm = res_zeniths[['D40_100']][res_zeniths[['D40_100']]$assay=='DA_STN_neurons_immature',]
res_zenith_da =  res_zeniths[['D40_100']][res_zeniths[['D40_100']]$assay=='DA_neurons',]
res_zenith_da_imm_up = res_zenith_da_imm[res_zenith_da_imm$Direction=='Up',]
res_zenith_da_up = res_zenith_da[res_zenith_da$Direction=='Up',]
p1 <- plotGOtermsDotplots(res_zenith_da_imm_up,num_terms = 3,'human_vs_chimp', direction = 'Up', size_limits = c(0.1,0.75),all_genes$D40_100$DA_STN_neurons_immature,de_genes$D40_100$DA_STN_neurons_immature, sig_col = 'p.greater', color_limits = c(0.0005, 0.005))
p2 <- plotGOtermsDotplots(res_zenith_da_up,num_terms = 3,'human_vs_chimp', direction = 'Up', size_limits = c(0.1,0.75),all_genes$D40_100$DA_STN_neurons_immature,de_genes$D40_100$DA_STN_neurons_immature, sig_col = 'p.greater', color_limits = c(0.0005, 0.005))
p1 + ggtitle('Immature DA neurons') + theme(plot.title = element_text(size = 6)) + p2 + ggtitle('DA neurons')+ theme(plot.title = element_text(size = 6)) + plot_layout(guides = 'collect')
ggsave(paste0(folder,'panelP_v22_go_dotplot.pdf'), height = 2, width = 7, units = 'in')
print(paste0('Rank for axonal transport of mitochondrion term in DA/STN neurons immature: ',which(res_zenith_da_imm_up$Geneset=='GO0019896: axonal transport of mitochondrion')))
print(paste0('Rank for hydrogen peroxide catabolic process term in DA neurons: ',which(res_zenith_da_up$Geneset=='GO0042744: hydrogen peroxide catabolic process')))

#new V29 with pseudotime
n <- names(go_human); rib_terms <- grep("ribosom", n, value = TRUE)
rib_genes <- list()
for (term in rib_terms){
  genes = go_human[[term]]
  rib_genes = append(rib_genes,genes@geneIds)
}
rib_genes = unique(unlist(rib_genes))
res_dl_alt_norib <- res_dl_alt
for (assay in names(res_dl_alt_norib)){
  a = res_dl_alt_norib[[assay]]
  rows_to_keep <- !rownames(a) %in% rib_genes
  b <- a[rows_to_keep, ]
  res_dl_alt_norib[[assay]] = b
}
saveRDS(res_dl_alt_norib,paste0(folder,'res_dl_alt.rds'))
res_dl_alt <- res_dl_alt_norib
de_genes_alt = vector(mode = 'list', length = num_datasets); names(de_genes_alt) = datasets
all_genes_alt = vector(mode = 'list', length = num_datasets); names(all_genes_alt) = datasets
celltypes = names(res_dl_alt)
contrasts = coefNames(res_dl_alt); 
contrasts <- contrasts[contrasts %!in% base_coef]
for (celltype in celltypes) {
  celltype_list_de = list()
  celltype_list_all = list()
  for (con in contrasts) {
    df_con <- as.data.frame(topTable(res_dl_alt, coef = con, number = Inf))
    df_celltype_all <- df_con[df_con$assay == celltype,]
    df_celltype <- df_celltype_all[df_celltype_all$adj.P.Val < pval, ]
    df_celltype$gene_sign = paste0(sign(df_celltype$logFC),df_celltype$ID)
    celltype_list_de[[con]] = df_celltype
    celltype_list_all[[con]] = df_celltype_all
  }
  de_genes_alt[[celltype]] = celltype_list_de
  all_genes_alt[[celltype]] = celltype_list_all
}
saveRDS(de_genes_alt,paste0(folder,'de_genes_alt.rds'))
saveRDS(all_genes_alt,paste0(folder,'all_genes_alt.rds'))

res_zenith_alt = zenith_gsa(res_dl_alt, coef = 'human_vs_chimp', go_human)
res_zenith_alt_da_imm = res_zenith_alt[res_zenith_alt$assay=='DA_STN_neurons_immature',]
res_zenith_alt_da =  res_zenith_alt[res_zenith_alt$assay=='DA_neurons',]
res_zenith_alt_da_imm_up = res_zenith_alt_da_imm[res_zenith_alt_da_imm$Direction=='Up',]
res_zenith_alt_da_up = res_zenith_alt_da[res_zenith_alt_da$Direction=='Up',]
p1 <- plotGOtermsDotplots(res_zenith_alt_da_imm_up,num_terms = 3,'human_vs_chimp', direction = 'Up', size_limits = c(0.1,0.75),all_genes$D40_100$DA_STN_neurons_immature,de_genes$D40_100$DA_STN_neurons_immature, sig_col = 'p.greater', color_limits = c(0.0005, 0.005))
p2 <- plotGOtermsDotplots(res_zenith_alt_da_up,num_terms = 3,'human_vs_chimp', direction = 'Up', size_limits = c(0.1,0.75),all_genes$D40_100$DA_STN_neurons_immature,de_genes$D40_100$DA_STN_neurons_immature, sig_col = 'p.greater', color_limits = c(0.0005, 0.005))
p1 + ggtitle('Immature DA neurons') + theme(plot.title = element_text(size = 6)) + p2 + ggtitle('DA neurons')+ theme(plot.title = element_text(size = 6)) + plot_layout(guides = 'collect')
ggsave(paste0(folder,'panelP_v29_go_dotplot.pdf'), height = 2, width = 7, units = 'in')
print(paste0('Rank for axonal transport of mitochondrion term in DA/STN neurons immature: ',which(res_zenith_alt_da_imm_up$Geneset=='GO0019896: axonal transport of mitochondrion')))
print(paste0('Rank for hydrogen peroxide catabolic process term in DA neurons: ',which(res_zenith_alt_da_up$Geneset=='GO0042744: hydrogen peroxide catabolic process')))

#KCNJ16 across ages----
#Need to use different model so timepoint is categorical
res_proc_v21 <- readRDS("/media/jenelle/4TB_disk/Dropbox/Analysis_midbrain/Old/Dreamlet/Midbrain/Ancestral_genome/D40_D100_D80/V21/res_proc.rds")
gene <- 'KCNJ16'
data <- extractData(res_proc_v21, 'DA_neurons')
plots <- plotStratifyBy2(data, gene, 'time_point', 'species',  colors_species, var1_levels = c('D40','D80','D100'), var2_levels = c('human','chimp','orangutan','macaque'))
panelQ <- plots[[1]] +
  (plots[[2]] + theme(axis.title.y = element_blank())) +
  (plots[[3]] + theme(axis.title.y = element_blank())) &
  theme(aspect.ratio=1.8,
        plot.margin = unit(c(0,0,0,0.05),'in'),
        plot.title = element_text(size = 6),
        legend.position = 'none',
        axis.line = element_line(linewidth = 0.23),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.ticks.length = unit(0.01, "in"),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.23))
panelQ
ggsave(paste0(folder,'figure4_suppQ.pdf'),panelQ, height = 0.8, width = 3.43, units = 'in')

#Cell stress Extended data Fig5----
all_genes_da <- c()
all_genes_da$DA_neurons <- all_genes$D40_100$DA_neurons
all_genes_da$DA_STN_neurons_immature <- all_genes$D40_100$DA_STN_neurons_immature
all_genes_da$vMB_progenitors <- all_genes$D16$vMB_progenitors

bhaduri_genes <- c('PGK1', 'ARCN1', 'GORASP2')
coi = 'DA_STN_neurons_immature'
data = extractData(res_procs$D40_100,coi)
dai_plots <- plotSelectedGenesList(bhaduri_genes, '~species', data, 'DA_neurons', colors_species, xlabels = FALSE)
coi = 'DA_neurons'
data = extractData(res_procs$D40_100,coi)
da_plots <- plotSelectedGenesList(bhaduri_genes, '~species', data, 'DA_neurons', colors_species, xlabels = FALSE)
supp5_panele <- dai_plots$PGK1 + dai_plots$ARCN1 + dai_plots$GORASP2 + da_plots$PGK1 + da_plots$ARCN1 + da_plots$GORASP2 + plot_layout(nrow = 1)
supp5_panele
ggsave(paste0(folder,'supp5_panele.pdf'), height = 3, width = 4, units = 'in')

#Boxplots
cell_types <- c('DA_STN_neurons_immature', 'DA_neurons')
result_list <- list()
for (cell_type in cell_types) {
  result_list[[cell_type]] <- dplyr::bind_rows(
    all_genes_da[[cell_type]]$human_vs_chimp %>%
      dplyr::mutate(contrast = "human_vs_chimp"),
    all_genes_da[[cell_type]]$human_vs_macaque %>%
      dplyr::mutate(contrast = "human_vs_macaque"),
    all_genes_da[[cell_type]]$chimp_vs_macaque %>%
      dplyr::mutate(contrast = "chimp_vs_macaque")
  ) %>%
    dplyr::filter(ID %in% bhaduri_genes) %>%
    dplyr::select(ID, contrast, adj.P.Val) %>%
    dplyr::mutate(cell_type = cell_type)
}
summary_stress_genes <- dplyr::bind_rows(result_list) %>%
  tidyr::pivot_wider(
    names_from = c(cell_type, contrast),
    values_from = adj.P.Val
  ) %>%
  as.data.frame()
summary_stress_genes

# #Heatmaps
# coef = 'human_vs_chimp'
# res_zenith_vmb = res_zeniths[['D16']][res_zeniths[['D16']]$assay=='vMB_progenitors',]
# res_zenith_da_imm = res_zeniths[['D40_100']][res_zeniths[['D40_100']]$assay=='DA_STN_neurons_immature',]
# res_zenith_da =  res_zeniths[['D40_100']][res_zeniths[['D40_100']]$assay=='DA_neurons',]
# 
# term <- 'GO0061621: canonical glycolysis'
# genes=geneIds(go_human[[term]])
# plotGeneHeatmap(res_dl_da, coef=coef, assays = celltypes_da_abbr, genes= genes, transpose=TRUE, zmax = 8) + 
#   labs(x = '', y = '', title = term)+
#   theme(legend.position = "right",
#         axis.text.x=element_text(size=6, angle=60, color = 'black'),
#         axis.text.y=element_text(size=6, color = 'black'),
#         plot.title = element_text(size = 6,hjust = 0.5),
#         legend.title = element_text(size=6),
#         legend.ticks = element_line(colour = 'black', linewidth = 0.5),
#         legend.text = element_text(size=6),
#         legend.justification = 'center',
#         legend.key.size = unit(0.1, "in"))
# 
# res_zenith_vmb[res_zenith_vmb$Geneset == term,]
# res_zenith_da_imm[res_zenith_da_imm$Geneset == term,]
# res_zenith_da[res_zenith_da$Geneset == term,]
# 
# term <- 'GO0034976: response to endoplasmic reticulum stress'
# genes=geneIds(go_human[[term]])
# plotGeneHeatmap(res_dl_da, coef=coef, assays = celltypes_da_abbr, genes= genes, transpose=TRUE, zmax = 8) + 
#   labs(x = '', y = '', title = term)+
#   theme(legend.position = "right",
#         axis.text.x=element_text(size=6, angle=60, color = 'black'),
#         axis.text.y=element_text(size=6, color = 'black'),
#         plot.title = element_text(size = 6,hjust = 0.5),
#         legend.title = element_text(size=6),
#         legend.ticks = element_line(colour = 'black', linewidth = 0.5),
#         legend.text = element_text(size=6),
#         legend.justification = 'center',
#         legend.key.size = unit(0.1, "in"))
# 
# res_zenith_vmb[res_zenith_vmb$Geneset == term,]
# res_zenith_da_imm[res_zenith_da_imm$Geneset == term,]
# res_zenith_da[res_zenith_da$Geneset == term,]

#Violin plot summary across terms
plot_GO_terms_violin <- function(all_genes_da, cell_type, terms_to_plot_nonsig, terms_to_plot_sig, term_labels){
  terms_to_plot <- c(terms_to_plot_nonsig,terms_to_plot_sig)
  genes_union <- unique(unlist(lapply(terms_to_plot, function(term) {
    geneIds(go_human[[term]])
  })))
  fill_vals <- c(
    Background = "gray30",
    setNames(rep("gray75", length(terms_to_plot_nonsig)), terms_to_plot_nonsig),
    setNames(rep("firebrick3", length(terms_to_plot_sig)), terms_to_plot_sig)
  )
  term_labels <- c("Background", term_labels)
  
df <- all_genes_da[[cell_type]]$human_vs_chimp
plot_df_sets <- purrr::map_dfr(terms_to_plot, function(term) {
  genes_in_set <- geneIds(go_human[[term]])
  df %>%
    filter(ID %in% genes_in_set) %>%
    mutate(Geneset = term)
})
plot_df_bg <- df %>%
  filter(!ID %in% genes_union) %>%
  mutate(Geneset = "Background")
plot_df <- bind_rows(plot_df_bg, plot_df_sets) %>%
  mutate(Geneset = factor(Geneset, levels = c("Background", terms_to_plot)))
plot <- ggplot(plot_df, aes(x = Geneset, y = logFC, fill = Geneset, color = Geneset)) +
  geom_violin(trim = FALSE, alpha = 0.6) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.8, color = "gray40") +
  geom_jitter(
    data = plot_df_sets,
    shape = 16,
    width = 0.1,
    alpha = 0.2,
    size = 0.5,
    color = "black"
  ) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    x = NULL,
    y = "logFC (human vs chimp)",
    title = cell_type
  ) +
  theme_basic_smallest() +
  scale_x_discrete(labels = term_labels) +
  scale_color_manual(values = fill_vals) +
  scale_fill_manual(values = fill_vals) +
  ylim(-4,4)+
  theme(
    plot.title = element_text(size = 7, hjust = 0.5),
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )
}

terms_to_plot_nonsig <- c(
  "GO0061621: canonical glycolysis",
  "GO0006110: regulation of glycolytic process",
  "GO0034976: response to endoplasmic reticulum stress",
  "GO0006915: apoptotic process",
  "GO0090398: cellular senescence"
)
#DA/STN immature
terms_to_plot_sig <- c(
  'GO0019896: axonal transport of mitochondrion',
  'GO0042744: hydrogen peroxide catabolic process'
)
term_labels <- c(
  "GO0061621: \ncanonical glycolysis",
  "GO0006110: \nregulation of glycolytic process",
  "GO0034976: response to \nendoplasmic reticulum stress",
  "GO0006915: \napoptotic process",
  "GO0090398: \ncellular senescence",
  'GO0019896: axonal \ntransport of mitochondrion',
  'GO0042744: hydrogen \nperoxide catabolic process'
)
p1 <- plot_GO_terms_violin(all_genes_da, 'DA_STN_neurons_immature', terms_to_plot_nonsig, terms_to_plot_sig, term_labels) + ggtitle('Immature DA/STN neurons')

#DA neurons
terms_to_plot_sig <- c(
  'GO0042744: hydrogen peroxide catabolic process'
)
term_labels <- c(
  "GO0061621: \ncanonical glycolysis",
  "GO0006110: \nregulation of glycolytic process",
  "GO0034976: response to \nendoplasmic reticulum stress",
  "GO0006915: \napoptotic process",
  "GO0090398: \ncellular senescence",
  'GO0042744: hydrogen \nperoxide catabolic process'
)
p2 <- plot_GO_terms_violin(all_genes_da, 'DA_neurons', terms_to_plot_nonsig, terms_to_plot_sig, term_labels)
supp5_panelfg <- p1 + p2
supp5_panelfg
ggsave(paste0(folder,'supp5_panelfg.pdf'), height = 3.2, width = 6, units = 'in')

#Summary of p values (not FDR per term) for human vs chimp
summary_df <- data.frame(term = terms_to_plot)
summary_df$p_value_imm <- res_zenith_da_imm$PValue[
  match(summary_df$term, res_zenith_da_imm$Geneset)
]
summary_df$p_value_da <- res_zenith_da$PValue[
  match(summary_df$term, res_zenith_da$Geneset)
]
summary_df

#Get numbers for text----
d16_celltypes = names(res_dls$D16)
df = as.data.frame(vps_full$D16[vps_full$D16$assay %in% d16_celltypes,])
vars = colnames(df); vars = vars[vars %!in% c('assay','gene')]
df <- df %>% dplyr::select(vars)
vars_means_16 = colMeans(df)
print('D16 variance partition means:')
print(vars_means_16)

d40_celltypes = names(res_dls$D40_100)
df = as.data.frame(vps_full$D40_100[vps_full$D40_100$assay %in% d40_celltypes,])
vars = colnames(df); vars = vars[vars %!in% c('assay','gene')]
df <- df %>% dplyr::select(vars)
vars_means_40 = colMeans(df, na.rm = TRUE)
print('D40-100 variance partition means:')
print(vars_means_40)

df = as.data.frame(vp_pseudo)
vars = colnames(df); vars = vars[vars %!in% c('assay','gene')]
df <- df %>% dplyr::select(vars)
vars_means_40_pseudo = colMeans(df, na.rm = TRUE)
print('D40-100 variance partition means for pseudotime:')
print(vars_means_40_pseudo)





