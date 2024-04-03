
rm(list = ls())

library('kdecopula')

baselines <- c('bern', 'T', 'TLL1', 'TLL2', 'TLL2nn', 'MR', 'beta')
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
    model <- kdecop(trn, method = baseline)
    # summary(model)
    
    # eval
    yhat <- predict(model, as.matrix(tst))
    
    # export
    write.table(yhat, sprintf('data/%s/%s_yhat.csv', ds, baseline), row.names = FALSE, col.names = FALSE)
  }
}
