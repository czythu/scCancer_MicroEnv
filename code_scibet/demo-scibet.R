rm(list = ls())
options(stringsAsFactors = F)
suppressMessages(library(ggplot2))
suppressMessages(library(tidyverse))
suppressMessages(library(viridis))
suppressMessages(library(ggsci))
suppressMessages(library(edgeR))

source("E:/FinalProject/T-cell/code_scibet/SciBet_modified.R")
source("E:/FinalProject/T-cell/code_scibet/Batch_correction.R")

data.lung <- "E:/FinalProject/T-cell/lung/data_transpose_count.rds"
anno.lung <- "E:/FinalProject/T-cell/lung/anno.rds"
name.lung <- c('CD4_C1-CCR7', 'CD4_C2-ANXA1', 'CD4_C3-GNLY', 'CD4_C4-CD69', 'CD4_C5-EOMES','CD4_C6-GZMA',
               'CD4_C7-CXCL13', 'CD4_C8-FOXP3', 'CD4_C9-CTLA4', 'CD8_C1-LEF1', 'CD8_C2-CD28', 'CD8_C3-CX3CR1',
               'CD8_C4-GZMK', 'CD8_C5-ZNF683', 'CD8_C6-LAYN', 'CD8_C7-SLC4A10')
model.lung <- "E:/FinalProject/T-cell/lung/T-cell_lung_model.csv"
data.colorectal <- "E:/FinalProject/T-cell/colorectal/data_transpose_count.rds"
anno.colorectal <- "E:/FinalProject/T-cell/colorectal/anno.rds"
name.colorectal <- c('CD4_C01-CCR7', 'CD4_C02-ANXA1', 'CD4_C03-GNLY', 'CD4_C04-TCF7', 'CD4_C05-CXCR6', 'CD4_C06-CXCR5', 'CD4_C07-GZMK',
                'CD4_C08-IL23R', 'CD4_C09-CXCL13', 'CD4_C10-FOXP3', 'CD4_C11-IL10', 'CD4_C12-CTLA4', 'CD8_C01-LEF1', 'CD8_C02-GPR183',
                'CD8_C03-CX3CR1', 'CD8_C04-GZMK', 'CD8_C05-CD6', 'CD8_C06-CD160', 'CD8_C07-LAYN', 'CD8_C08-SLC4A10')
model.colorectal <- "E:/FinalProject/T-cell/colorectal/T-cell_colorectal_model.csv"
data.liver <- "E:/FinalProject/T-cell/liver/data_transpose_count.rds"
anno.liver <- "E:/FinalProject/T-cell/liver/anno.rds"
name.liver <- c('C01_CD8-LEF1', 'C02_CD8-CX3CR1', 'C03_CD8-SLC4A10', 'C04_CD8-LAYN', 'C05_CD8-GZMK',
                'C06_CD4-CCR7', 'C07_CD4-FOXP3', 'C08_CD4-CTLA4', 'C09_CD4-GZMA', 'C10_CD4-CXCL13', 'C11_CD4-GNLY')
model.liver <- "E:/FinalProject/T-cell/liver/T-cell_liver_model.csv"

#Part1.1 single dataset: lung
# Read data
data_transpose_count <- readRDS(data.lung)
label <- readRDS(anno.lung)
data_transpose_count$label <- label$majorCluster
etest_gene <- SelectGene_R(data_transpose_count, k = 50)
etest_gene
Marker_heatmap(data_transpose_count, etest_gene)
index_remain <- which(data_transpose_count$label == "CD4_C1-CCR7" 
                      | data_transpose_count$label == "CD4_C2-ANXA1"
                      | data_transpose_count$label == "CD4_C3-GNLY"
                      | data_transpose_count$label == "CD4_C4-CD69"
                      | data_transpose_count$label == "CD4_C5-EOMES"
                      | data_transpose_count$label == "CD4_C6-GZMA"
                      | data_transpose_count$label == "CD4_C7-CXCL13"
                      | data_transpose_count$label == "CD4_C8-FOXP3"
                      | data_transpose_count$label == "CD4_C9-CTLA4"
                      | data_transpose_count$label == "CD8_C1-LEF1"
                      | data_transpose_count$label == "CD8_C2-CD28"
                      | data_transpose_count$label == "CD8_C3-CX3CR1"
                      | data_transpose_count$label == "CD8_C4-GZMK"
                      | data_transpose_count$label == "CD8_C5-ZNF683"
                      | data_transpose_count$label == "CD8_C6-LAYN"
                      | data_transpose_count$label == "CD8_C7-SLC4A10")
length(index_remain)
data_CD4_CD8 <- data_transpose_count[index_remain,]
label_CD4_CD8 <- data_CD4_CD8$label 

tibble(
  ID = 1:nrow(data_CD4_CD8),
  label = data_CD4_CD8$label
) %>%
  dplyr::sample_frac(0.7) %>%
  dplyr::pull(ID) -> ID

train_set <- data_CD4_CD8[ID,]      #construct reference set
test_set <- data_CD4_CD8[-ID,]      #construct query set
prob <- Train(train_set, k = 2000)
model.save <- data.frame(t(prob))
predict <- Test(prob, test_set)
Confusion_heatmap_Value(name.lung, name.lung, test_set$label, predict, 'Reference','Prediction',T)
Confusion_heatmap_Value(name.lung, name.lung, test_set$label, predict, 'Reference','Prediction',F)
write.csv(model.save, model.lung)
# ------------------------------------------------------------


#Part1.2 single dataset: colorectal
# Read data
data_transpose_count <- readRDS(data.colorectal)
label <- readRDS(anno.colorectal)
data_transpose_count$label <- label$majorCluster
etest_gene <- SelectGene_R(data_transpose_count, k = 50)
etest_gene
Marker_heatmap(data_transpose_count, etest_gene)
index_remain <- which(data_transpose_count$label == "CD4_C01-CCR7" 
                      | data_transpose_count$label == "CD4_C02-ANXA1"
                      | data_transpose_count$label == "CD4_C03-GNLY"
                      | data_transpose_count$label == "CD4_C04-TCF7"
                      | data_transpose_count$label == "CD4_C05-CXCR6"
                      | data_transpose_count$label == "CD4_C06-CXCR5"
                      | data_transpose_count$label == "CD4_C07-GZMK"
                      | data_transpose_count$label == "CD4_C08-IL23R"
                      | data_transpose_count$label == "CD4_C09-CXCL13"
                      | data_transpose_count$label == "CD4_C10-FOXP3"
                      | data_transpose_count$label == "CD4_C11-IL10"
                      | data_transpose_count$label == "CD4_C12-CTLA4"
                      | data_transpose_count$label == "CD8_C01-LEF1"
                      | data_transpose_count$label == "CD8_C02-GPR183"
                      | data_transpose_count$label == "CD8_C03-CX3CR1"
                      | data_transpose_count$label == "CD8_C04-GZMK"
                      | data_transpose_count$label == "CD8_C05-CD6"
                      | data_transpose_count$label == "CD8_C06-CD160"
                      | data_transpose_count$label == "CD8_C07-LAYN"
                      | data_transpose_count$label == "CD8_C08-SLC4A10")
length(index_remain)
data_CD4_CD8 <- data_transpose_count[index_remain,]
label_CD4_CD8 <- data_CD4_CD8$label 

tibble(
  ID = 1:nrow(data_CD4_CD8),
  label = data_CD4_CD8$label
) %>%
  dplyr::sample_frac(0.7) %>%
  dplyr::pull(ID) -> ID

train_set <- data_CD4_CD8[ID,]      #construct reference set
test_set <- data_CD4_CD8[-ID,]      #construct query set
prob <- Train(train_set, k = 2000)
model.save <- data.frame(t(prob))
predict <- Test(prob, test_set)
Confusion_heatmap_Value(name.colorectal, name.colorectal, test_set$label, predict, 'Reference','Prediction',T)
Confusion_heatmap_Value(name.colorectal, name.colorectal, test_set$label, predict, 'Reference','Prediction',F)
write.csv(model.save, model.colorectal)
# ------------------------------------------------------------


#Part1.3 single dataset: liver(test combat for batch effect)
# Read data
data_transpose_count <- readRDS(data.liver)
print(dim(data_transpose_count))
label <- readRDS(anno.liver)
data_transpose_count$label <- label$majorCluster
# etest_gene <- SelectGene_R(data_transpose_count, k = 50)
# a NA exists in liver cancer data
for (i in 1:length(label$majorCluster)){
  if (is.na(label$majorCluster[i])){
    data_transpose_count$label[i] <- "unknown" 
    label$majorCluster[i] <- "unknown"
    print(i)
  }
}

data_transpose_count$batch <- label$PatientID
batch.info <- label$PatientID
print(length(batch.info))
for (i in 1:length(batch.info)){
  if (is.na(batch.info[i])){
    batch.info[i] <- "unknown"
    data_transpose_count$batch[i] <- "unknown"
    print(i)
  }
}
# delete unknown cells
data_delete_unknown <- data_transpose_count[-which(data_transpose_count$label == "unknown"),]
meta.data <- data_delete_unknown[,(dim(data_delete_unknown)[2]-1):dim(data_delete_unknown)[2]]
# Feature seclection by E-test
temp <- data_delete_unknown[,1:(dim(data_delete_unknown)[2]-1)]
etest_gene <- SelectGene_R(temp, k = 2000)
index_gene <- c()
for (i in 1:length(etest_gene)){
  index_gene <- c(index_gene, which(colnames(temp) == etest_gene[i]))
}
index_gene <- sort(index_gene, decreasing = FALSE)
data_delete_unknown <- temp[, index_gene]
data_delete_unknown

# input for Combat_seq(data as matrix, batch as factor)
data_delete_unknown <- data.frame(t(data_delete_unknown))
data_delete_unknown <- as.matrix(data_delete_unknown)
batch.info <- meta.data$batch
batch.info <- as.factor(batch.info)
data.new <- ComBat_seq(data_delete_unknown, batch=batch.info, group=NULL)
data.new

# input for Scibet, cell-gene, final col = label
# if combat
# data_delete_unknown <- data.frame(t(data.new))
data_delete_unknown$label <- meta.data$label

# test feature selection
etest_gene <- SelectGene_R(data_delete_unknown, k = 50)
etest_gene
Marker_heatmap(data_delete_unknown, etest_gene)
tibble(
  ID = 1:nrow(data_delete_unknown),
  label = data_delete_unknown$label
) %>%
  dplyr::sample_frac(0.7) %>%
  dplyr::pull(ID) -> ID
# train and test
train_set <- data_delete_unknown[ID,]      #construct reference set
test_set <- data_delete_unknown[-ID,]      #construct query set
prob <- Train(train_set, k = 1000)
model.save <- data.frame(t(prob))
predict <- Test(prob, test_set)
# visualization
Confusion_heatmap_Value(name.liver, name.liver, test_set$label, predict, 'Reference','Prediction',T)
Confusion_heatmap_Value(name.liver, name.liver, test_set$label, predict, 'Reference','Prediction',F)
# model saving
write.csv(model.save, model.liver)


# Part2.1 CrossTest: liver, lung
model.lung <- readr::read_csv(model.lung) 
model.lung <- pro.core(model.lung)
model.colorectal <- readr::read_csv(model.colorectal) 
model.colorectal <- pro.core(model.colorectal)
model.liver <- readr::read_csv(model.liver) 
model.liver <- pro.core(model.liver)
