library(forecast)
library(tseries)

#Loading bitcoin-usd data
Data<-read.csv("/home/ggcarvalho/Documents/CIn - UFPE/2019.2/Time Series/Second Project/BTC-USDw.csv")
btc.close<-Data$Close
btc.ts<-ts(btc.close,)
btc.ts.log<-log(btc.ts)
plot(btc.ts.log)
btc.diff<-diff(btc.ts.log)
plot(btc.diff)
ft <- as.numeric(btc.diff)
ft <- ft[!is.na(ft)]
ftfinal.aic <- Inf
ftfinal.order <- c(0,0,0)
for (p in 1:4) for (d in 0:1) for (q in 1:4) {
  ftcurrent.aic <- AIC(arima(ft, order=c(p, d, q)))
  if (ftcurrent.aic < ftfinal.aic) {
    ftfinal.aic <- ftcurrent.aic
    ftfinal.order <- c(p, d, q)
    ftfinal.arima <- arima(ft, order=ftfinal.order)
  }
}
ftfinal.order
acf(resid(ftfinal.arima), main="ARMA(1,1) res. ACF")
acf(resid(ftfinal.arima)^2, main ="ARMA(1,1) squared res. ACF")
ft.garch <- garch(ft, trace=F)
ft.res <- ft.garch$res[-1]
acf(ft.res, main="GARCH res. ACF")
acf(ft.res^2, main="GARCH squared res. ACF")
