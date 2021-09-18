# ------------ import packages --------------
rm(list = ls())
options(stringsAsFactors = F)
library(magrittr)
library(Seurat)
library(scPred)
library(cowplot)
# library(scCancer)
setwd("E:\\bio\\scCancer_MicroEnv")
# -------------------------------------------

# ------------ data processing --------------
data.count <- readRDS("./data/T-cell/colorectal/liver_cell2017_Zemin_Count.rds")
na.indexes <- which(is.na(data.count$symbol))
data.count <- data.count[-na.indexes,]
genes <- data.count[,1]
genes <- chartr(old="_", new="-", x=genes)
# for (i in 1:length(genes)){
#     if (is.na(genes[i])){
#         genes[i] <- paste0("NA", i)
#     }
#     genes[i] <- chartr(old="_", new="-", x=genes[i])
# }
row.names(data.count) <- genes

data.gene <- readRDS("./data/T-cell/liver/liver_cell2017_Zemin_gene_info.rds")
data.cell <- readRDS("./data/T-cell/liver/liver_cell2017_Zemin_cell_info.rds")
anno.ref <- subset(data.cell, select = c("UniqueCell_ID", "PatientID", "majorCluster"))
anno.ref$UniqueCell_ID <- chartr(old="-", new=".", x=anno.ref$UniqueCell_ID)

# for (i in 1:length(anno.ref$UniqueCell_ID)){
#     anno.ref$UniqueCell_ID[i] <- chartr(old="-", new=".", x=anno.ref$UniqueCell_ID[i])
# }

temp1 <- matrix(data = colnames(data.count)[2:length(colnames(data.count))])
colnames(temp1) <- c("UniqueCell_ID")
temp2 <- matrix(nrow = length(temp1), ncol = 1)
colnames(temp2) <- c("majorCluster")
temp3 <- matrix(nrow = length(temp1), ncol = 1)
colnames(temp3) <- c("PatientID")

for (i in 1:length(temp1)){
  for (j in 1:length(temp1)){
    if (anno.ref$UniqueCell_ID[j] == temp1[i]){
      temp2[i] <- anno.ref$majorCluster[j]
      temp3[i] <- anno.ref$PatientID[j]
      break
    }
  }
}
cellType <- data.frame(cbind(temp1, temp3, temp2))
raw.data <- data.count[, 2:dim(data.count)[2]]
saveRDS(raw.data, file="./data/T-cell/liver/data_for_scPred.rds")
saveRDS(cellType, file="./data/T-cell/liver/anno_for_scPred.rds")
# -------------------------------------------