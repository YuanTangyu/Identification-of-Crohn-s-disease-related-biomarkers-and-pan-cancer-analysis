
library(survival)
library(survminer)
library(timeROC)

inputFile="expTime.txt"     
setwd(" ")    

rt=read.table(inputFile, header=T, sep="\t", check.names=F, row.names=1)
gene=colnames(rt)[3]

ROC_rt=timeROC(T=rt$futime, delta=rt$fustat,
	           marker=rt[,gene], cause=1,
	           weighting='aalen',
	           times=c(1,3,5), ROC=TRUE)
pdf(file="ROC.pdf", width=5, height=5)
plot(ROC_rt,time=1,col='green',title=FALSE,lwd=2)
plot(ROC_rt,time=3,col='blue',add=TRUE,title=FALSE,lwd=2)
plot(ROC_rt,time=5,col='red',add=TRUE,title=FALSE,lwd=2)
legend('bottomright',
	   c(paste0('AUC at 1 years: ',sprintf("%.03f",ROC_rt$AUC[1])),
	     paste0('AUC at 3 years: ',sprintf("%.03f",ROC_rt$AUC[2])),
	     paste0('AUC at 5 years: ',sprintf("%.03f",ROC_rt$AUC[3]))),
	     col=c("green",'blue','red'),lwd=2,bty = 'n')
dev.off()


