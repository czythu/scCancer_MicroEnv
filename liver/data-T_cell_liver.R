# Type: T-cell liver

# ------------ source -----------------------
# data: Landscape of Infiltrating T Cells in Liver Cancer Revealed by Single-Cell Sequencing
# package: scPred: accurate supervised method for cell-type classification from single-cell RNA-seq data
# -------------------------------------------

# ------------ installation -----------------
# devtools::install_github("powellgenomicslab/scPred")
# install.packages('e1071') # for SVM
# -------------------------------------------

# ------------ instruction ------------------
# https://powellgenomicslab.github.io/scPred/articles/introduction.html
# -------------------------------------------

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
data.count <- readRDS("./data/T-cell/liver/liver_cell2017_Zemin_Count.rds")
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

# ------------ dataset ----------------------
data <- readRDS("./data/T-cell/liver/data_for_scPred.rds")
label <- readRDS("./data/T-cell/liver/anno_for_scPred.rds")
write.csv(data, "./data/T-cell/liver/data_for_scPred.csv")
write.csv(label, "./data/T-cell/liver/anno_for_scPred.csv")
for (i in 1:length(label$majorCluster)){
    if (is.na(label$majorCluster[i])){
        label$majorCluster[i] <- "unknown"
        print(i)
    }
}
set.seed(1234)
train.ratio=0.8
train.index <- sample(1:ncol(data),round(ncol(data)*train.ratio))
data_train <- data[, train.index]
label_train <- label[train.index, ]
data_test <- data[, -train.index]
label_test <- label[-train.index, ]

reference <- CreateSeuratObject(count = data_train)
reference$cell_type <- label_train$majorCluster
query <- CreateSeuratObject(count = data_test)
query$cell_type <-label_test$majorCluster
# -------------------------------------------

# -------------- train ----------------------
reference <- reference %>%
    NormalizeData(verbose = TRUE) %>%
    FindVariableFeatures(selection.method = "vst",verbose = TRUE) %>%
    ScaleData() %>%
    RunPCA() %>%
    RunUMAP(dims = 1:30)

DimPlot(reference, group.by = "cell_type", repel = TRUE)
feature <- getFeatureSpace(reference, "cell_type")
# model = "nnet", "svmLinearWeights", "adaboost"
# model <- trainModel(feature, model="svmLinearWeights")
model <- trainModel(feature)
get_probabilities(model) %>% head()
get_scpred(model)
plot_probabilities(model)
saveRDS(model, file="./data/T-cell/liver/model_SVM_liver.rds")
# -------------------------------------------

# -------------- test -----------------------
# predict
query <- NormalizeData(query)
result <- scPredict(query, model)
DimPlot(result, group.by = "scpred_prediction", reduction = "scpred")
result <- RunUMAP(result, reduction = "scpred", dims = 1:30)
# compare
p1 <- DimPlot(result, group.by = "scpred_prediction", repel = TRUE)
p2 <- DimPlot(result, group.by = "cell_type", repel = TRUE)
# plot_grid(p1, p2)
plot_grid(p1)
plot_grid(p2)
predict <- result@meta.data[["scpred_prediction"]]
correct <- 0
for (i in 1:length(predict)){
    if (predict[i] == label_test$majorCluster[i]){
        correct <- correct + 1
    }
}
print(correct / length(predict))
# -------------------------------------------

# -------------- note -----------------------
# label中出现NA
# Error in if (nrow(spmodel@features[[positiveClass]]) == 0){:参数长度为零
# -------------------------------------------
