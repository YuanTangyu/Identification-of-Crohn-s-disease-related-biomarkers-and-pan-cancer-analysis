CoreAlg <- function(X, y){
  svn_itor <- 3
  
  res <- function(i){
    nus <- 0 
    if(i == 1) {nus <- 0.25}
    if(i == 2) {nus <- 0.5}
    if(i == 3) {nus <- 0.75}
    model <- svm(X, y, type="nu-regression", kernel="linear", nu=nus, scale=F)
    return(model)
  }
  
  if(Sys.info()['sysname'] == 'Windows') {
    out <- mclapply(1:svn_itor, res, mc.cores=1) 
  } else {
    out <- mclapply(1:svn_itor, res, mc.cores=svn_itor)
  }
  
  nusvm <- rep(0, svn_itor)
  corrv <- rep(0, svn_itor)
  t <- 1
  while(t <= svn_itor) {
    weights = t(out[[t]]$coefs) %*% out[[t]]$SV
    weights[which(weights < 0)] <- 0
    w <- weights / sum(weights)
    
    u <- sweep(X, MARGIN=2, w, '*')
    k <- apply(u, 1, sum)
    
    nusvm[t] <- sqrt((mean((k - y)^2)))
    corrv[t] <- cor(k, y)
    t <- t + 1
  }
  rmses <- nusvm
  mn <- which.min(rmses)
  model <- out[[mn]]
  q <- t(model$coefs) %*% model$SV
  q[which(q < 0)] <- 0
  w <- (q / sum(q))
  mix_rmse <- rmses[mn]
  mix_r <- corrv[mn]
  return(list("w" = w, "mix_rmse" = mix_rmse, "mix_r" = mix_r))
}

doPerm <- function(perm, X, Y){
  itor <- 1
  Ylist <- as.list(data.matrix(Y))
  dist <- matrix()
  while(itor <= perm){
    yr <- as.numeric(Ylist[sample(length(Ylist),dim(X)[1])])
    yr <- (yr - mean(yr)) / sd(yr)
    
    result <- CoreAlg(X, yr)
    mix_r <- result$mix_r
    
    if(itor == 1) {
      dist <- mix_r
    } else {
      dist <- rbind(dist, mix_r)
    }
    itor <- itor + 1
  }
  return(list("dist" = dist))
}

CIBERSORT <- function(sig_matrix, mixture_file, perm=0, QN=TRUE){

  library(e1071)
  library(parallel)
  library(preprocessCore)
  
  X <- read.table(sig_matrix,header=T,sep="\t",row.names=1,check.names=F)
  Y <- read.table(mixture_file, header=T, sep="\t", row.names=1,check.names=F)
  
  X <- data.matrix(X)
  Y <- data.matrix(Y)
  X <- X[order(rownames(X)),]
  Y <- Y[order(rownames(Y)),]
  
  if(max(Y) < 50) {Y <- 2^Y}
  
  if(QN == TRUE){
    tmpc <- colnames(Y)
    tmpr <- rownames(Y)
    Y <- normalize.quantiles(Y)
    colnames(Y) <- tmpc
    rownames(Y) <- tmpr
  }
  
  Xgns <- row.names(X)
  Ygns <- row.names(Y)
  YintX <- Ygns %in% Xgns
  Y <- Y[YintX,]
  XintY <- Xgns %in% row.names(Y)
  X <- X[XintY,]
  
  X <- (X - mean(X)) / sd(as.vector(X))
  
  if(perm > 0) {
    nulldist <- sort(doPerm(perm, X, Y)$dist)
  }
  
  header <- c('Mixture',colnames(X),"P-value","Correlation","RMSE")
  output <- matrix()
  itor <- 1
  mixtures <- dim(Y)[2]
  pval <- 9999
  while(itor <= mixtures){
    y <- Y[,itor]
    y <- (y - mean(y)) / sd(y)

    result <- CoreAlg(X, y)
    w <- result$w
    mix_r <- result$mix_r
    mix_rmse <- result$mix_rmse

    if(perm > 0) {
      pval <- 1 - (which.min(abs(nulldist - mix_r)) / length(nulldist))
    }
    
    out <- c(colnames(Y)[itor],w,pval,mix_r,mix_rmse)
    if(itor == 1) {
      output <- out
    } else {
      output <- rbind(output, out)
    }
    itor <- itor + 1
  }

  write.table(rbind(header,output), file="CIBERSORT-Results.txt", sep="\t", row.names=F, col.names=F, quote=F)
  obj <- rbind(header,output)
  obj <- obj[,-1]
  obj <- obj[-1,]
  obj <- matrix(as.numeric(unlist(obj)),nrow=nrow(obj))
  rownames(obj) <- colnames(Y)
  colnames(obj) <- c(colnames(X),"P-value","Correlation","RMSE")
  return(obj)
}

setwd(" ")

inputFile = "normalize.txt"

outTab <- CIBERSORT("ref.txt", inputFile, perm=1000, QN=TRUE)

outTab <- outTab[outTab[,"P-value"] < 0.05,]

outTab <- as.matrix(outTab[, 1:(ncol(outTab) - 3)])
outTab <- rbind(colnames(outTab), outTab) 
write.table(outTab, file="CIBERSORT-Results.txt", sep="\t", quote=F, col.names=F)
