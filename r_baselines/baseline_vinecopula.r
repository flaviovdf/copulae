library('VineCopula')

baseline <- 'par'
# uncomment for real datasets
# dss <- c('boston', 'intcmsft', 'googfb')

# synthetic datasets
dss <- c('gauss1', 'gauss5', 'gauss9',
         'clayton1', 'clayton5', 'clayton10',
         'frank1', 'frank5', 'frank10')

# read
for (ds in dss){
  trn <- read.csv(sprintf('data/%s/trn.csv', ds), header = FALSE)
  tst <- read.csv(sprintf('data/%s/tst.csv', ds), header = FALSE)

  # train
  model <- BiCopSelect(trn[, 1], trn[, 2], familyset = 1:10)
  summary(model)

  # eval
  yhat <- BiCopPDF(tst[, 1], tst[, 2], model)

  # export
  # write.table(yhat, sprintf('data/%s/%s_yhat.csv', ds, baseline), row.names = FALSE, col.names = FALSE)
}
