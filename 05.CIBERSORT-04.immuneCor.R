
library(limma)
library(reshape2)
library(ggpubr)
library(ggExtra)

gene="S100P"                    
expFile="normalize.txt"            
immFile="CIBERSORT-Results.txt"     
setwd("")    

rt=read.table(expFile, header=T, sep="\t", check.names=F)
rt=as.matrix(rt)
rownames(rt)=rt[,1]
exp=rt[,2:ncol(rt)]
dimnames=list(rownames(exp),colnames(exp))
data=matrix(as.numeric(as.matrix(exp)),nrow=nrow(exp),dimnames=dimnames)
data=avereps(data)

data=t(data[gene,,drop=F])
data=as.data.frame(data)

immune=read.table(immFile, header=T, sep="\t", check.names=F, row.names=1)

sameSample=intersect(row.names(immune), row.names(data))
rt=cbind(immune[sameSample,,drop=F], data[sameSample,,drop=F])

outTab=data.frame()

for(i in colnames(rt)[1:(ncol(rt)-1)]){
  x=as.numeric(rt[,gene])
  y=as.numeric(rt[,i])
  if(sd(y)==0){y[1]=0.00001}
  cor=cor.test(x, y, method="spearma")
  
  outVector=cbind(Gene=gene, Cell=i, cor=cor$estimate, pvalue=cor$p.value)
  outTab=rbind(outTab,outVector)
  
  if(cor$p.value<0.05){
    outFile=paste0("cor.", i, ".pdf")
    df1=as.data.frame(cbind(x,y))
    p1=ggplot(df1, aes(x, y)) + 
      xlab(paste0(gene, " expression")) + ylab(i)+
      geom_point() + geom_smooth(method="lm",formula = y ~ x) + theme_bw()+
      stat_cor(method = 'spearman', aes(x =x, y =y))
    p2=ggMarginal(p1, type="density", xparams=list(fill = "orange"), yparams=list(fill = "blue"))
    pdf(file=outFile, width=5.2, height=5)
    print(p2)
    dev.off()
  }
}

write.table(outTab,file="cor.result.txt",sep="\t",row.names=F,quote=F)


