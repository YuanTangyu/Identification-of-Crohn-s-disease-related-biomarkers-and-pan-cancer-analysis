
library(limma)
library(reshape2)
library(ggpubr)

expFile="geneExp.txt"       
scoreFile="TMEscores.txt"     
setwd("")   

exp=read.table(expFile, header=T, sep="\t", check.names=F, row.names=1)
gene=colnames(exp)[1]
exp=exp[exp$Type=="Tumor",1,drop=F]
exp$Type=ifelse(exp[,gene]>median(exp[,gene]), "High", "Low")

score=read.table(scoreFile, header=T, sep="\t", check.names=F, row.names=1)
score=score[,1:3]

sameSample=intersect(row.names(exp), row.names(score))
exp=exp[sameSample,"Type",drop=F]
score=score[sameSample,,drop=F]
rt=cbind(score, exp)
rt$Type=factor(rt$Type, levels=c("Low", "High"))

data=melt(rt, id.vars=c("Type"))
colnames(data)=c("Type", "scoreType", "Score")

p=ggviolin(data, x="scoreType", y="Score", fill = "Type",
	     xlab="",
	     ylab="TME score",
	     legend.title=gene,
	     add = "boxplot", add.params = list(color="white"),
	     palette = c("blue","red"), width=1)
p=p+rotate_x_text(45)
p1=p+stat_compare_means(aes(group=Type),
	      method="wilcox.test",
	      symnum.args=list(cutpoints = c(0, 0.001, 0.01, 0.05, 1), symbols = c("***", "**", "*", " ")),
	      label = "p.signif")

pdf(file="vioplot.pdf", width=6, height=5)
print(p1)
dev.off()

