#!/usr/bin/env Rscript

library(argparser)
library(rhdf5)
library(dplyr)
library(coda)
library(duckdb)
library(dplyr)
library(itertools)
library(foreach)
library(progress)
library(arrow)

psr <- arg_parser("gelman")
psr <- add_argument(psr, "--ipt", "FSMP inputs")
psr <- add_argument(psr, "--opt", "Gelman-Rubin output")
args <- parse_args(psr)
con <- dbConnect(duckdb::duckdb(), ":memory:")

# 读取数据
file_path <- args$ipt
recon_data <- h5read(file_path, "/data")
recon_df <- as.data.frame(recon_data)

convergence_result <- recon_df %>%
  group_by(EventID) %>%
  do(
    convergence_v = {
      lapply(seq(2500, 25000, by = 100), function(i) {
        gelman.diag(
          mcmc(list(
            mcmc(cbind(.data$x[1:i], .data$y[1:i], .data$z[1:i])), mcmc(cbind(.data$x_2[1:i], .data$y_2[1:i], .data$z_2[1:i])),
            mcmc(cbind(.data$x_3[1:i], .data$y_3[1:i], .data$z_3[1:i])), mcmc(cbind(.data$x_4[1:i], .data$y_4[1:i], .data$z_4[1:i])),
            mcmc(cbind(.data$x_5[1:i], .data$y_5[1:i], .data$z_5[1:i])), mcmc(cbind(.data$x_6[1:i], .data$y_6[1:i], .data$z_6[1:i])),
            mcmc(cbind(.data$x_7[1:i], .data$y_7[1:i], .data$z_7[1:i])), mcmc(cbind(.data$x_8[1:i], .data$y_8[1:i], .data$z_8[1:i])),
            mcmc(cbind(.data$x_9[1:i], .data$y_9[1:i], .data$z_9[1:i])), mcmc(cbind(.data$x_10[1:i], .data$y_10[1:i], .data$z_10[1:i]))
          )),confidence = 0.999,
          autoburnin = FALSE,
          multivariate = TRUE
        )$psrf
      })
    },
    convergence_E = {
      lapply(seq(2500, 25000, by = 100), function(i) {
        gelman.diag(
          mcmc(list(mcmc(.data$E[1:i]), mcmc(.data$E_2[1:i]), mcmc(.data$E_3[1:i]), mcmc(.data$E_4[1:i]), mcmc(.data$E_5[1:i]), mcmc(.data$E_6[1:i]), mcmc(.data$E_7[1:i]), mcmc(.data$E_8[1:i]), mcmc(.data$E_9[1:i]), mcmc(.data$E_10[1:i]))),
          confidence = 0.999,
          autoburnin = FALSE,
          multivariate = FALSE
        )$psrf
      })  
    },
    convergence_t = {
      lapply(seq(2500, 25000, by = 100), function(i) {
        gelman.diag(
          mcmc(list(mcmc(.data$t[1:i]), mcmc(.data$t_2[1:i]), mcmc(.data$t_3[1:i]), mcmc(.data$t_4[1:i]), mcmc(.data$t_5[1:i]), mcmc(.data$t_6[1:i]), mcmc(.data$t_7[1:i]), mcmc(.data$t_8[1:i]), mcmc(.data$t_9[1:i]), mcmc(.data$t_10[1:i]))),
          confidence = 0.999,
          autoburnin = FALSE,
          multivariate = FALSE
        )$psrf
      })
    }
  )

convergence_result <- arrow_table(convergence_result)
write_parquet(convergence_result, args$opt, compression = "ZSTD")

dbDisconnect(con, shutdown = TRUE)