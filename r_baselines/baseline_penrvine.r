library('penRvine')

bernstein <- function (v, x, n){
  return((choose(n, v) * x^v * (1 - x)^(n - v)) * (n + 1))
}

int.my.bspline <- function(help.env) {
  
  stand.num <- c()
  len.k <- length(get("knots.val",help.env)$val)
  if(is.vector(get("base.den",help.env))) base.den <- matrix(base.den,1,length(get("base.den",help.env)))
  
  len.b <- dim(get("base.den",help.env))[2]
  q <- get("q",help.env)#-1 #q ist als order hinterlegt, brauche hier den grad!
  
  knots.val <- get("knots.val",help.env)
  
  #piecewise polynomial calculation
  
  len.k <- length(knots.val$val)   
  
  #generate help-sequences for calculation
  
  y.all.help <- c()
  for(j in 1:(len.k-1)) {
    help.seq <-  seq(knots.val$val[j],knots.val$val[j+1],length=(q))
    assign(paste("y.help",j,sep=""),help.seq,envir=help.env)
    y.all.help <- c(y.all.help,help.seq)
  }
  
  y.all.help <- unique(y.all.help)
  
  base.help <- bsplineS(y.all.help,breaks=knots.val$val,norder=get("q",help.env))
  
  for(j in 1:(len.k-1)) {
    list <- which(get("y",help.env)>=knots.val$val[j] & get("y",help.env)<=knots.val$val[j+1])
    assign(paste("y.list",j,sep=""),list,envir=help.env)
    assign(paste("y.part",j,sep=""),get("y",help.env)[list],envir=help.env)
    for(i in 1:(dim(get("base.den",help.env))[2])) { 
      assign(paste("base.part",j,i,sep=""),get("base.den",help.env)[list,i],envir=help.env)
    }
  }  
  
  #for (i in 1:(len.k-(q-1))) {
  for(i in 1:(len.k-1)) {
    compare <- get(paste("y.help",i,sep=""),envir=help.env)
    list <- which(y.all.help%in%compare)
    for(j in 1:(dim(base.help)[2])) {
      assign(paste("y.base.help",i,j,sep=""),base.help[list,j],envir=help.env)
      assign(paste("y.list.help",i,j,sep=""),list,envir=help.env)
    }
  }
  
  #search the relevant points for calculations und calculate the polynomial-coefficients
  
  q <- q-1 
  for(i in 1:(len.k-1)) {
    y.vec <- c()
    for(j in 1:(dim(base.help)[2])) {
      if(q>=0) y.vec <- c(knots.val$val[i])
      if(q>=1) y.vec <- c(y.vec,knots.val$val[i+1])
      if(q>=2) y.vec <- seq(y.vec[1],y.vec[2],length=3)
      if(q>=3) y.vec <- seq(y.vec[1],y.vec[3],length=4)
      if(q>=4) y.vec <- seq(y.vec[1],y.vec[4],length=5)
      assign(paste("y.vec",i,sep=""),y.vec,envir=help.env)
      assign(paste("coef",i,".",j,sep=""),(solve(outer(y.vec,0:q,"^"))%*%(get(paste("y.base.help",i,j,sep=""),envir=help.env))),envir=help.env)
    }
  }
  #calculate the integrals and coefficients for standardisation of the splines at the borders
  INT <- matrix(0,dim(base.help)[2],len.k-1)
  
  for(i in 1:(len.k-1)) {
    for(j in 1:(dim(base.help)[2])) {
      y2 <- knots.val$val[i+1]
      y1 <- knots.val$val[i]
      coef <- get(paste("coef",i,".",j,sep=""),envir=help.env)
      y2 <- 1/(1:(q+1))*y2^(1:(q+1))
      y1 <- 1/(1:(q+1))*y1^(1:(q+1))
      INT[j,i] <- sum(coef*y2)-sum(coef*y1)
    }
  }
  assign("INT",INT,help.env)
  INT.help <- 1/rowSums(INT)
  assign("stand.num",INT.help,help.env)
}

my.bspline <- function(y,K,q,margin.normal=FALSE,kn=NULL) {
  knots <- seq(0,1,length=K) 
  if(margin.normal) {
    knots <- qnorm(knots)
    knots[1]<-qnorm(0.0000001)
    knots[length(knots)]<-qnorm(1-0.0000001)
  }
  
  len.k <- length(knots)
  base.den <- bsplineS(y,breaks=knots,norder=q)
  len.b <- dim(base.den)[2]
  
  knots.val <- list()
  knots.val$val <- knots
  
  #integration
  help.env <- new.env()
  assign("base.den",base.den,help.env)
  assign("knots.val",knots.val,help.env)
  assign("y",y,help.env)
  assign("q",q,help.env)
  int.my.bspline(help.env)
  stand.num <- get("stand.num",help.env)
  INT <- get("INT",help.env)
  for(j in 1:len.b) base.den[,j] <- base.den[,j]*stand.num[j]
  
  return(list(base.den=base.den,stand.num=stand.num,knots.val=knots.val,K=K,INT=INT))
}

eval.paircopula <- function(x,val=NULL){
  p <- get("p",x)
  d <- get("d",x)
  Index.basis.D <- get("Index.basis.D",x)
  ck <- get("ck.val",x)
  base <- get("base",x)
  index.b <- matrix(0:get("dd",x))
  
  if(!is.matrix(val)) {
    if(is.data.frame(val)) val <- as.matrix(val) else stop("val has to be a data.frame or a matrix")
  }
  tilde.Psi.d <-  array(NA, dim=c((length(val)/p),get("ddb",x),p))
  val <- matrix(val,(length(val)/p),p)
  if(base=="Bernstein"){
    tilde.Psi.d[,,1] <-  apply(index.b,1,bernstein,x=val[,1],n=get("dd",x))
    tilde.Psi.d[,,2] <-  apply(index.b,1,bernstein,x=val[,2],n=get("dd",x))  
  }
  if(base=="B-spline"){
    tilde.Psi.d[,,1] <- my.bspline(y=val[,1],K=get("K",x),q=get("q",x),kn=get("knots1",x))$base.den
    tilde.Psi.d[,,2] <- my.bspline(y=val[,2],K=get("K",x),q=get("q",x),kn=get("knots2",x))$base.den
  }
  tilde.PSI.d.D <- tilde.Psi.d[,Index.basis.D[,1],1]
  tilde.PSI.d.D <- tilde.PSI.d.D * tilde.Psi.d[,Index.basis.D[,2],2]
  
  val2<-tilde.PSI.d.D%*%ck
  ind<-which(val2<1e-12)
  datafr <- data.frame(val,val2)
  colnames(datafr)[p+1] <- "fit"
  return(datafr)
}

# baseline <- 'pbern'
baselines <- c('pbern', 'pspl1', 'pspl2')
# dss <- c('boston', 'intcmsft', 'googfb')
dss <- c('gauss1', 'gauss5', 'gauss9', 
         'clayton1', 'clayton5', 'clayton10', 
         'frank1', 'frank5', 'frank10')

# read
# ds <- 'boston'
for (ds in dss){
  trn <- read.csv(sprintf('data/%s/trn.csv', ds), header = FALSE)
  tst <- read.csv(sprintf('data/%s/tst.csv', ds), header = FALSE)
  
  for (baseline in baselines){
    print(c(ds, baseline))
    
    # train
    if (baseline == 'pbern') model <- paircopula(data = trn, base = 'Bernstein', K = 14)
    if (baseline == 'pspl1') model <- paircopula(data = trn, base = 'B-spline', K = 14, q = 2)
    if (baseline == 'pspl2') model <- paircopula(data = trn, base = 'B-spline', K = 10, q = 3)
    
    # eval
    yhat <- eval.paircopula(model, val = tst)[, 3]
    
    # export
    write.table(yhat, sprintf('data/%s/%s_yhat.csv', ds, baseline), row.names = FALSE, col.names = FALSE)
  }
}

