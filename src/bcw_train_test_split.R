library(tidyverse)
library(readxl)
library(here)


set.seed(42)


# 
# import_data <- read.delim(file=here("src","data","bcw","wdbc.data"), sep=",", header=FALSE)
# names(import_data) <- c("id","diagnosis",
#                         "mean_radius","mean_texture","mean_perimeter","mean_area","mean_smoothness","mean_compactness","mean_concavity",
#                         "mean_concave_points","mean_symmetry", "mean_fractal_dimension",
#                         "se_radius","se_texture","se_perimeter","se_area","se_smoothness","se_compactness","se_concavity","se_concave_points",
#                         "se_symmetry", "se_fractal_dimension",
#                         "worst_radius","worst_texture","worst_perimeter","worst_area","worst_smoothness","worst_compactness","worst_concavity",
#                         "worst_concave_points","worst_symmetry", "worst_fractal_dimension")
# import_data$id <- as.factor(import_data$id)
# import_data$diagnosis <- as.factor(import_data$diagnosis)
# 
# n <- nrow(import_data)

enriched_split <- function(df,
                           id_col = "id",
                           y_col  = "diagnosis",
                           rare_filter,          # expression like quote(pM < 0.05)
                           dev_frac = 0.70,
                           seed = 42) {
  set.seed(seed)
  
  rare_idx <- with(df, eval(rare_filter))
  rare_df  <- df[rare_idx, , drop = FALSE]
  main_df  <- df[!rare_idx, , drop = FALSE]
  
  strat_split_idx <- function(d, frac, y_col) {
    y <- d[[y_col]]
    idx_dev <- integer(0)
    
    for (cls in unique(y)) {
      cls_idx <- which(y == cls)
      n_dev <- floor(length(cls_idx) * frac)
      idx_dev <- c(idx_dev, sample(cls_idx, n_dev))
    }
    idx_dev
  }
  
  rare_dev_local <- strat_split_idx(rare_df, dev_frac, y_col)
  rare_dev <- rare_df[rare_dev_local, , drop = FALSE]
  rare_test <- rare_df[-rare_dev_local, , drop = FALSE]
  
  main_dev_local <- strat_split_idx(main_df, dev_frac, y_col)
  main_dev <- main_df[main_dev_local, , drop = FALSE]
  main_test <- main_df[-main_dev_local, , drop = FALSE]
  
  dev  <- rbind(rare_dev, main_dev)
  test <- rbind(rare_test, main_test)
  
  dev  <- dev[sample(nrow(dev)), , drop = FALSE]
  test <- test[sample(nrow(test)), , drop = FALSE]
  
  counts <- list(
    rare = table(rare_df[[y_col]]),
    dev  = table(dev[[y_col]]),
    test = table(test[[y_col]]),
    rare_dev  = table(rare_dev[[y_col]]),
    rare_test = table(rare_test[[y_col]])
  )
  
  list(dev = dev, test = test, counts = counts)
}


# split <- enriched_split(
#   df = import_data,
#   rare_filter = quote(worst_perimeter >= 50 & worst_perimeter <= 105),
#   dev_frac = 0.65,
#   seed = 7
# )

# dev  <- split$dev
# test <- split$test
# print(split$counts)
# 
# 
# write.csv(dev, file=here("src","data","bcw","dev.csv"))
# write.csv(test, file=here("src","data","bcw","test.csv"))