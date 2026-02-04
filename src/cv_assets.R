library(MASS)
library(tidyverse)
library(readxl)
library(ggplot2)
library(patchwork)
library(ranger)
library(caret)
library(corpcor)
library(FNN)
library(mgcv)
library(Matrix)
library(xgboost)



set.seed(42)

#-----------------------------------------------------------------------------------------------------

clamp01 <- function(x, eps=1e-6) pmin(pmax(x, eps), 1 - eps)

winsorize_vec <- function(x, probs = c(0.01, 0.99)) {
  
  qs <- quantile(x, probs, na.rm=TRUE)
  pmin(pmax(x, qs[1]), qs[2])
}


#-----------------------------------------------------------------------------------------------------

get_X_from_df <- function(df, features) {
  X <- as.matrix(df[, features, drop = FALSE])
  storage.mode(X) <- "double"
  X
}


marginal_z_from_X <- function(X, mu, sd_marg, eps = 1e-8) {
  sd <- pmax(sd_marg, eps)
  sweep(sweep(X, 2, mu, "-"), 2, sd, "/")
}


MarginalZ <- function(df, ref) {
  X <- get_X_from_df(df, ref$features)
  Zm <- marginal_z_from_X(X, ref$mu, ref$sd_marg)
  colnames(Zm) <- paste0("mz_", ref$features)
  as.data.frame(Zm)
}


dot_score <- function(Z, v) {
  as.vector(Z %*% v)
}

proj_coef <- function(Z, u, eps = 1e-12) {
  denom <- sum(u * u)
  if (denom < eps) stop("u has near-zero norm; cannot project.")
  as.vector(Z %*% u) / denom
}


mahalanobis2_from_X <- function(X, mu, Sigma, ridge = 1e-8) {
  p <- ncol(X)
  Sigma_r <- Sigma + diag(ridge, p)
  R <- chol(Sigma_r)
  
  Xc <- sweep(X, 2, mu, "-")
  Y <- forwardsolve(t(R), t(Xc))
  colSums(Y^2)
}


fit_class_reference <- function(train_df, class_label,
                                diag_col="diagnosis", id_col="id") {
  df <- train_df %>%
    dplyr::filter(.data[[diag_col]] == class_label) %>%
    dplyr::select(-dplyr::all_of(c(diag_col, id_col)))
  
  X <- as.matrix(df)
  X <- apply(X, 2, winsorize_vec)
  mu <- colMeans(X)
  Sigma <- corpcor::cov.shrink(X, verbose=FALSE)
  sd_marg <- sqrt(pmax(1e-12, diag(Sigma)))
  
  list(mu = mu, Sigma = Sigma, sd_marg = sd_marg, features = colnames(df))
}

fit_references_BM <- function(train_df, safe="B", disease="M", diag_col="diagnosis", id_col="id") {
  ref_B <- fit_class_reference(train_df, safe, diag_col, id_col)
  ref_M <- fit_class_reference(train_df, disease, diag_col, id_col)
  
  if (!identical(ref_B$features, ref_M$features)) {
    stop("Feature mismatch between ref_B and ref_M. Ensure same columns/order.")
  }
  
  list(B = ref_B, M = ref_M)
}

fit_drift_directions <- function(train_df, refs, safe="B", disease="M", diag_col="diagnosis", id_col="id") {
  ref <- refs$B
  Zm <- MarginalZ(train_df, ref)
  
  y <- train_df[[diag_col]]
  if (!all(y %in% c(safe, disease))) stop("diag_col must contain safe and disease labels.")
  
  mu_m_B <- colMeans(apply(Zm[y == safe, , drop=FALSE], 2, winsorize_vec))
  mu_m_M <- colMeans(apply(Zm[y == disease, , drop=FALSE], 2, winsorize_vec))
  v_marg <- mu_m_M - mu_m_B
  
  norm2 <- function(v) sqrt(sum(v^2))
  v_marg_unit <- v_marg / pmax(1e-12, norm2(v_marg))
  
  list(
    v_marg = v_marg, v_marg_unit = v_marg_unit,
    mz_names = names(v_marg)
  )
}

augment_with_z_and_projections <- function(df, refs, dirs,
                                           diag_col="diagnosis", id_col="id", ridge=1e-8) {
  
  ref_B <- refs$B
  ref_M <- refs$M
  
  X <- get_X_from_df(df, ref_B$features)
  
  Zm <- marginal_z_from_X(X, ref_B$mu, ref_B$sd_marg)
  
  colnames(Zm) <- paste0("mz_", ref_B$features)
  
  Zm_mat <- Zm[, dirs$mz_names, drop = FALSE]
  
  proj_marg <- dot_score(Zm_mat, dirs$v_marg_unit)
  
  d2_B <- mahalanobis2_from_X(X, ref_B$mu, ref_B$Sigma, ridge = ridge)
  
  d2_M <- mahalanobis2_from_X(X, ref_M$mu, ref_M$Sigma, ridge = ridge)
  d_d2 <- d2_B - d2_M
  d_dist <- sqrt(pmax(d2_B,0)) - sqrt(pmax(d2_M,0))
  
  out <- dplyr::bind_cols(
    df,
    as.data.frame(Zm),
    data.frame(
      proj_marg = proj_marg,
      d_dist = d_dist
    )
  )
  
  out
}


strat_boot_idx <- function(y) {
  idx_B <- which(y == "B")
  idx_M <- which(y == "M")
  c(sample(idx_B, length(idx_B), replace = TRUE),
    sample(idx_M, length(idx_M), replace = TRUE))
}


bootstrap_proj_and_ddist <- function(train_df, eval_df,
                                     B = 300,
                                     diag_col="diagnosis", id_col="id",
                                     ridge = 1e-8,
                                     seed = 1) {
  set.seed(seed)
  
  y <- train_df[[diag_col]]
  n_eval <- nrow(eval_df)
  
  proj_mat <- matrix(NA_real_, nrow = B, ncol = n_eval)
  ddist_mat <- matrix(NA_real_, nrow = B, ncol = n_eval)
  pM_mat <- matrix(NA_real_, nrow = B, ncol = n_eval)
  
  for (b in seq_len(B)) {
    boot_idx <- strat_boot_idx(y)
    train_b <- train_df[boot_idx, , drop = FALSE]
    
    refs <- fit_references_BM(train_b, diag_col = diag_col, id_col = id_col)
    
    dirs <- fit_drift_directions(train_b, refs, diag_col = diag_col, id_col = id_col)
    
    sig <- augment_with_z_and_projections(eval_df, refs, dirs, diag_col = diag_col, id_col = id_col, ridge = ridge)
    
    rf_train <- ranger::ranger(
      stats::as.formula(paste0(diag_col, " ~ .")),
      data = train_b %>% dplyr::select(-dplyr::all_of(id_col)),
      mtry = 8,
      min.node.size = 5,
      keep.inbag = TRUE,
      probability = TRUE,
      num.trees = 500
    )
    
    rf_res <- RFPredict(eval_df, rf_train)
    
    proj_mat[b, ] <- sig$proj_marg
    ddist_mat[b, ] <- sig$d_dist
    pM_mat[b, ] <- rf_res$pM
  }
  
  colnames(proj_mat) <- if (id_col %in% names(eval_df)) as.character(eval_df[[id_col]]) else paste0("row", seq_len(n_eval))
  colnames(ddist_mat) <- colnames(proj_mat)
  colnames(pM_mat) <- colnames(proj_mat)
  
  list(proj_boot = proj_mat, ddist_boot = ddist_mat, pM_boot = pM_mat)
}



#-----------------------------------------------------------------------------------------------------

RFPredict <- function(data, randomforest) {
  
  pred_obj <- predict(randomforest, data%>%select(-diagnosis, -id), type="se", se.method="infjack")
  
  pM <- clamp01(pred_obj$predictions[, "M"], eps=1e-4)
  
  results_raw <- data.frame(
    id = data$id,
    pM = pM,
    pM_se = pred_obj$se[, "M"]
  )
  
  return(results_raw)
  
}


#-----------------------------------------------------------------------------------------------------

make_vars <- function(df,
                      p_col = "pM",
                      u_col = "local_pct_proj_marg",
                      y_col = "diagnosis") {
  df %>%
    mutate(
      p = clamp01(.data[[p_col]]),
      u = clamp01(.data[[u_col]]))%>%
        mutate(
      l = qlogis(p),
      y = ifelse(.data[[y_col]] == "M", 1L, 0L)
    )
}

#-----------------------------------------------------------------------------------------------------

make_strat_folds <- function(y, k = 5, seed = 42) {
  if (!requireNamespace("caret", quietly = TRUE)) {
    stop("Package 'caret' is required for make_strat_folds(). Install it or pass folds explicitly.")
  }
  set.seed(seed)
  caret::createFolds(y, k = k, list = TRUE)
}

#-----------------------------------------------------------------------------------------------------


rf_cv_tune_uncertainty <- function(
    dev,
    folds = NULL,
    k = 5,
    seed = 42,
    param_grid = expand.grid(
      mtry = c(4, 8, 16, 25),
      min.node.size = c(5, 15)
    ),
    id_col = "id",
    y_col  = "diagnosis",
    num_trees = 500,
    keep_inbag = TRUE,
    probability = TRUE,
    verbose = TRUE
) {
  
  if (is.null(folds)) {
    folds <- make_strat_folds(dev[[y_col]], k = k, seed = seed)
  } else {
    k <- length(folds)
  }
  
  if (!all(c(id_col, y_col) %in% names(dev))) {
    stop("dev must contain columns: ", id_col, " and ", y_col)
  }
  if (!is.data.frame(param_grid) || nrow(param_grid) < 1) {
    stop("param_grid must be a non-empty data.frame (e.g., expand.grid).")
  }
  
  if (verbose) message("Starting integrated CV tuning loop...")
  
  tuning_results <- vector("list", nrow(param_grid))
  
  for (j in seq_len(nrow(param_grid))) {
    
    current_params <- param_grid[j, , drop = FALSE]
    if (verbose) {
      message(
        "Testing: ",
        paste(names(current_params), unlist(current_params), sep = "=", collapse = " ")
      )
    }
    
    stack_df <- NULL
    
    for (i in seq_len(k)) {
      if (verbose) message("  Fold ", i, " / ", k)
      
      test_idx  <- folds[[i]]
      train_idx <- setdiff(seq_len(nrow(dev)), test_idx)
      
      d_train <- dev[train_idx, , drop = FALSE]
      d_test  <- dev[test_idx,  , drop = FALSE]
      
      model <- ranger::ranger(
        stats::as.formula(paste0(y_col, " ~ .")),
        data = d_train %>% dplyr::select(-dplyr::all_of(id_col)),
        mtry = current_params$mtry,
        min.node.size = current_params$min.node.size,
        keep.inbag = keep_inbag,
        probability = probability,
        num.trees = num_trees
      )
      
      fold_res <- RFPredict(d_test, model)
      
      fold_res <- fold_res %>%
        dplyr::mutate(
          mtry = current_params$mtry,
          min.node.size = current_params$min.node.size,
          fold = i
        )
      
      stack_df <- dplyr::bind_rows(stack_df, fold_res)
      
      se_mu <- mean(stack_df$pM_se)
      se_sd <- sd(stack_df$pM_se)
      stack_df <- stack_df %>%
        mutate(pM_se_scaled = (pM_se - se_mu) / se_sd)
    }
    
    tuning_results[[j]] <- stack_df
  }
  
  master_results <- dplyr::bind_rows(tuning_results)
  
  # join labels
  master_results <- master_results %>%
    dplyr::left_join(
      dev %>% dplyr::select(dplyr::all_of(c(id_col, y_col))),
      by = stats::setNames(id_col, id_col)
    ) %>%
    dplyr::mutate(y = as.integer(.data[[y_col]] == "M"))
  
  if (!requireNamespace("ModelMetrics", quietly = TRUE)) {
    stop("Package 'ModelMetrics' is required for AUC calculation here.")
  }
  
  metrics <- master_results %>%
    dplyr::group_by(mtry, min.node.size) %>%
    dplyr::summarise(
      correlation = stats::cor(pM, pM_se),
      mid_se_sd = stats::sd(pM_se[pM > 0.4 & pM < 0.6]),
      n_mid = sum(pM > 0.4 & pM < 0.6),
      auc = ModelMetrics::auc(y, pM),
      .groups = "drop"
    ) %>%
    dplyr::arrange(abs(correlation))
  
  list(
    folds = folds,
    param_grid = param_grid,
    master_results = master_results,
    metrics = metrics
  )
}

#-----------------------------------------------------------------------------------------------------

pick_top_bottom_combos <- function(metrics, n = 4) {
  list(
    top = head(metrics, n),
    bottom = tail(metrics, n)
  )
}

#-----------------------------------------------------------------------------------------------------


rf_cv_build_calibration_stack <- function(
    dev,
    folds,
    mtry = 8,
    min.node.size = 5,
    id_col = "id",
    y_col  = "diagnosis",
    num_trees = 500,
    keep_inbag = TRUE,
    probability = TRUE,
    seed = 42,
    verbose = TRUE
) {
  
  if (is.null(folds)) stop("rf_cv_build_calibration_stack() requires 'folds' (use make_strat_folds() or reuse from tuning output).")
  k <- length(folds)
  
  set.seed(seed)
  
  calibration_z_stack <- NULL
  
  for (i in seq_len(k)) {
    if (verbose) message("Fold ", i, " / ", k)
    
    test_idx  <- folds[[i]]
    train_idx <- setdiff(seq_len(nrow(dev)), test_idx)
    
    d_train <- dev[train_idx, , drop = FALSE]
    d_test  <- dev[test_idx,  , drop = FALSE]
    
    model <- ranger::ranger(
      stats::as.formula(paste0(y_col, " ~ .")),
      data = d_train %>% dplyr::select(-dplyr::all_of(id_col)),
      mtry = mtry,
      min.node.size = min.node.size,
      keep.inbag = keep_inbag,
      probability = probability,
      num.trees = num_trees
    )
    
    rf_res <- RFPredict(d_test, model)
    
    refs <- fit_references_BM(d_train)
    dirs <- fit_drift_directions(d_train, refs)
    
    test_aug <- augment_with_z_and_projections(d_test,  refs, dirs)

    fold_res <- dplyr::bind_cols(
      rf_res,
      test_aug %>% dplyr::select(-dplyr::all_of(id_col))
    ) %>%
      dplyr::mutate(fold = i)
    
    calibration_z_stack <- dplyr::bind_rows(calibration_z_stack, fold_res)
  }
  
  se_mu <- mean(calibration_z_stack$pM_se)
  se_sd <- sd(calibration_z_stack$pM_se)
  calibration_z_stack <- calibration_z_stack %>%
    mutate(pM_se_scaled = (pM_se - se_mu) / se_sd)
  
  calibration_z_stack
}


make_case_weights <- function(df,
                               perim_col="worst_perimeter",
                               y_col="diagnosis",
                               lo=65, hi=110,
                               w_zone_all=2,
                               w_zone_malignant=5) {
  w <- rep(1, nrow(df))
  in_zone <- df[[perim_col]] >= lo & df[[perim_col]] <= hi
  is_M <- df[[y_col]] == "M"
  
  w[in_zone] <- w[in_zone] * w_zone_all
  w[in_zone & is_M] <- w[in_zone & is_M] * w_zone_malignant
  w
}



rf_cv_build_calibration_stack_weighted <- function(
    dev,
    folds,
    weight_col="worst_perimeter",
    lo=65,
    hi=110,
    w_zone_all=2,
    w_zone_malignant=5,
    mtry = 8,
    min.node.size = 5,
    id_col = "id",
    y_col  = "diagnosis",
    num_trees = 500,
    keep_inbag = TRUE,
    probability = TRUE,
    seed = 42,
    verbose = TRUE
) {
  
  if (is.null(folds)) stop("rf_cv_build_calibration_stack_weighted() requires 'folds' (use make_strat_folds() or reuse from tuning output).")
  k <- length(folds)
  
  set.seed(seed)
  
  calibration_z_stack <- NULL
  
  for (i in seq_len(k)) {
    if (verbose) message("Fold ", i, " / ", k)
    
    test_idx  <- folds[[i]]
    train_idx <- setdiff(seq_len(nrow(dev)), test_idx)
    
    d_train <- dev[train_idx, , drop = FALSE]
    d_test  <- dev[test_idx,  , drop = FALSE]
    
    cw <- make_case_weights(d_train, perim_col = weight_col, lo=lo, hi=hi, w_zone_all=w_zone_all, w_zone_malignant=w_zone_malignant)
    
    # Fit RF
    model <- ranger::ranger(
      stats::as.formula(paste0(y_col, " ~ .")),
      data = d_train %>% dplyr::select(-dplyr::all_of(id_col)),
      mtry = mtry,
      min.node.size = min.node.size,
      keep.inbag = keep_inbag,
      probability = probability,
      num.trees = num_trees,
      case.weights = cw
    )
    
    rf_res <- RFPredict(d_test, model)
    
    refs <- fit_references_BM(d_train)
    dirs <- fit_drift_directions(d_train, refs)
    
    test_aug <- augment_with_z_and_projections(d_test,  refs, dirs)
    
    fold_res <- dplyr::bind_cols(
      rf_res,
      test_aug %>% dplyr::select(-dplyr::all_of(id_col))
    ) %>%
      dplyr::mutate(fold = i)
    
    calibration_z_stack <- dplyr::bind_rows(calibration_z_stack, fold_res)
  }
  
  se_mu <- mean(calibration_z_stack$pM_se)
  se_sd <- sd(calibration_z_stack$pM_se)
  calibration_z_stack <- calibration_z_stack %>%
    mutate(pM_se_scaled = (pM_se - se_mu) / se_sd)
  
  calibration_z_stack
}


make_xgb_matrix <- function(df, feature_cols, ref_levels = NULL) {

  xdf <- df[, feature_cols, drop = FALSE]
  if (!is.null(ref_levels)) {
    for (nm in names(xdf)) {
      if (is.factor(xdf[[nm]]) || is.character(xdf[[nm]])) {
        xdf[[nm]] <- factor(xdf[[nm]], levels = ref_levels[[nm]])
      }
    }
  } else {

    for (nm in names(xdf)) {
      if (is.character(xdf[[nm]])) xdf[[nm]] <- factor(xdf[[nm]])
    }
  }
  
  X <- sparse.model.matrix(~ . - 1, data = xdf)
  return(X)
}


xgb_oof_predict <- function(
    dev,
    folds=NULL,
    k = 5,
    seed = 42,
    nrounds = 500,
    early_stopping_rounds = 30,
    eta = 0.05,
    max_depth = 4,
    min_child_weight = 1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    gamma = 0,
    lambda = 1,
    alpha = 0,
    scale_pos_weight = NULL,
    case_weights = NULL,
    verbose = 0
) {
  set.seed(seed)
  
  stopifnot("diagnosis" %in% names(dev))
  stopifnot(all(dev$diagnosis %in% c("B", "M")))
  if ("id" %in% names(dev)) {
    # ok
  } else {
    dev$id <- seq_len(nrow(dev))
  }
  
  drop_cols <- intersect(names(dev), c("diagnosis", "id"))
  feature_cols <- setdiff(names(dev), drop_cols)
  
  ref_levels <- list()
  for (nm in feature_cols) {
    if (is.character(dev[[nm]])) dev[[nm]] <- factor(dev[[nm]])
    if (is.factor(dev[[nm]])) ref_levels[[nm]] <- levels(dev[[nm]])
  }
  
  if (is.null(folds)) {
    folds <- createFolds(dev$diagnosis, k = k, list = TRUE)
    oof <- vector("list", k)
  } else {
    oof <- vector("list", length(folds))
  }

  
  
  for (i in seq_len(k)) {
    message(sprintf("XGB fold %d / %d", i, k))
    test_idx <- folds[[i]]
    train_idx <- setdiff(seq_len(nrow(dev)), test_idx)
    
    d_train <- dev[train_idx, , drop = FALSE]
    d_test  <- dev[test_idx,  , drop = FALSE]
    
    y_train <- as.integer(d_train$diagnosis == "M")
    y_test  <- as.integer(d_test$diagnosis == "M")
    
    X_train <- make_xgb_matrix(d_train, feature_cols, ref_levels = ref_levels)
    X_test  <- make_xgb_matrix(d_test,  feature_cols, ref_levels = ref_levels)
    
    w_train <- NULL
    w_test  <- NULL
    if (!is.null(case_weights)) {
      w_train <- case_weights[train_idx]
      w_test  <- case_weights[test_idx]
    }
    
    dtrain <- xgb.DMatrix(data = X_train, label = y_train, weight = w_train)
    dtest  <- xgb.DMatrix(data = X_test,  label = y_test,  weight = w_test)
    
    params <- list(
      objective = "binary:logistic",
      eval_metric = "logloss",
      eta = eta,
      max_depth = max_depth,
      min_child_weight = min_child_weight,
      subsample = subsample,
      colsample_bytree = colsample_bytree,
      gamma = gamma,
      lambda = lambda,
      alpha = alpha
    )
    
    if (!is.null(scale_pos_weight)) params$scale_pos_weight <- scale_pos_weight
    
    watchlist <- list(train = dtrain, eval = dtest)
    
    fit <- xgb.train(
      params = params,
      data = dtrain,
      nrounds = nrounds,
      watchlist = watchlist,
      early_stopping_rounds = early_stopping_rounds,
      verbose = verbose
    )
    
    pM <- predict(fit, dtest)
    pM <- pmin(1 - 1e-6, pmax(1e-6, pM))
    
    oof[[i]] <- data.frame(
      id = d_test$id,
      fold = i,
      diagnosis = d_test$diagnosis,
      y = y_test,
      pM = pM,
      pred = ifelse(pM >= 0.5, "M", "B")
    )
  }
  
  bind_rows(oof) %>% arrange(id)
}


xgb_train_model <- function(
    train_df,
    feature_cols = NULL,
    label_col = "diagnosis",
    positive_label = "M",
    seed = 42,
    nrounds = 2000,
    early_stopping_rounds = 50,
    eta = 0.03,
    max_depth = 3,
    min_child_weight = 1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    gamma = 0,
    lambda = 1,
    alpha = 0,
    scale_pos_weight = NULL,
    case_weights = NULL,
    verbose = 0
) {
  set.seed(seed)
  
  stopifnot(label_col %in% names(train_df))
  
  # pick features if not provided
  if (is.null(feature_cols)) {
    drop_cols <- intersect(names(train_df), c(label_col, "id"))
    feature_cols <- setdiff(names(train_df), drop_cols)
  }
  
  # keep factor levels consistent between train/test
  ref_levels <- list()
  for (nm in feature_cols) {
    if (is.character(train_df[[nm]])) train_df[[nm]] <- factor(train_df[[nm]])
    if (is.factor(train_df[[nm]])) ref_levels[[nm]] <- levels(train_df[[nm]])
  }
  
  y_train <- as.integer(train_df[[label_col]] == positive_label)
  
  X_train <- make_xgb_matrix(train_df, feature_cols, ref_levels = ref_levels)
  
  w_train <- NULL
  if (!is.null(case_weights)) {
    stopifnot(length(case_weights) == nrow(train_df))
    w_train <- case_weights
  }
  
  dtrain <- xgb.DMatrix(data = X_train, label = y_train, weight = w_train)
  
  params <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    eta = eta,
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    gamma = gamma,
    lambda = lambda,
    alpha = alpha
  )
  if (!is.null(scale_pos_weight)) params$scale_pos_weight <- scale_pos_weight
  
  # If you want internal early stopping, pass eval_set via xgb_train_with_eval() below.
  fit <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    verbose = verbose
  )
  
  # return an object that knows how to featurize
  list(
    fit = fit,
    feature_cols = feature_cols,
    ref_levels = ref_levels,
    label_col = label_col,
    positive_label = positive_label
  )
}


xgb_predict_prob <- function(model, new_df, clip_eps = 1e-6) {
  missing_cols <- setdiff(model$feature_cols, names(new_df))
  if (length(missing_cols) > 0) {
    stop("Missing feature columns in new_df: ", paste(missing_cols, collapse = ", "))
  }
  
  X_new <- make_xgb_matrix(new_df, model$feature_cols, ref_levels = model$ref_levels)
  dnew <- xgb.DMatrix(data = X_new)
  
  pM <- predict(model$fit, dnew)
  pM <- pmin(1 - clip_eps, pmax(clip_eps, pM))
  pM
}


