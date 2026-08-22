# MERFISH spinal cord cell-type classification (Hackathon Summer 2026, team Biostat Ted).
# Rebuilds output/prediction.csv from the four organiser csv files. Every fit is seeded;
# with nthread = 5 the run reproduces candidate_seeded/prediction.csv byte for byte.
# usage: Rscript main.R   (run from this directory; sequential, several hours;
# finished members under work/models/ are skipped, so an interrupted run can be restarted)

suppressPackageStartupMessages({
  library(data.table); library(FNN); library(Matrix); library(glmnet); library(xgboost)
})

data_dir  <- "../Hackathon-Summer-2026/data"
work_dir  <- "work"
out_dir   <- "output"
fold_file <- "folds/folds_5_seed2.csv"
python    <- "D:/python/python.exe"
nthread   <- 5

feat_dir    <- file.path(work_dir, "features")
model_dir   <- file.path(work_dir, "models")
scratch_dir <- file.path(work_dir, "scratch")
for (d in c(feat_dir, model_dir, scratch_dir, out_dir)) dir.create(d, showWarnings = FALSE, recursive = TRUE)
for (f in c("counts_train.csv", "counts_test.csv", "meta_train.csv", "meta_test.csv"))
  stopifnot(file.exists(file.path(data_dir, f)))

t_start <- Sys.time()
say <- function(...) cat(sprintf("[%s +%4.0fm] %s\n", format(Sys.time(), "%H:%M:%S"),
                                 as.numeric(difftime(Sys.time(), t_start, units = "mins")), sprintf(...)))

rdc <- function(f) fread(file.path(data_dir, f), colClasses = list(character = 1))
ct <- rdc("counts_train.csv"); cs <- rdc("counts_test.csv")
mt <- rdc("meta_train.csv");   ms <- rdc("meta_test.csv")
for (d in list(ct, cs, mt, ms)) setnames(d, 1, "cell_id")
# accept a test file that writes missing values as empty strings, or lists the genes in another order
for (d in list(mt, ms)) for (j in names(d)) if (is.character(d[[j]])) set(d, which(!nzchar(d[[j]])), j, NA_character_)
if (!identical(names(ct), names(cs)) && setequal(names(ct), names(cs))) setcolorder(cs, names(ct))
stopifnot(identical(ct$cell_id, mt$cell_id), identical(cs$cell_id, ms$cell_id), identical(names(ct), names(cs)))

genes <- setdiff(names(ct), "cell_id")
X_raw <- rbind(as.matrix(ct[, ..genes]), as.matrix(cs[, ..genes]))
meta  <- rbind(mt, ms)
n_tr <- nrow(mt); n_te <- nrow(ms); N <- n_tr + n_te
tr_rows <- seq_len(n_tr); te_rows <- n_tr + seq_len(n_te)

classes <- sort(unique(mt$MERFISH_cell_type_annotation)); K <- length(classes)
y_tr <- match(mt$MERFISH_cell_type_annotation, classes) - 1L

# Excitatory_vs_Inhibitory is metadata (known for test cells too) and a deterministic
# function of the class, so it gives a hard 0/1 compatibility mask over the 60 classes
ei_of_class <- sapply(classes, function(cl) {
  u <- unique(mt$Excitatory_vs_Inhibitory[mt$MERFISH_cell_type_annotation == cl])
  stopifnot(length(u) == 1); ifelse(is.na(u), "NA", u) })
ei_cell <- ifelse(is.na(meta$Excitatory_vs_Inhibitory), "NA", meta$Excitatory_vs_Inhibitory)
mask <- outer(ei_cell, ei_of_class, "==") * 1

tot   <- rowSums(X_raw)
X_log <- log1p(X_raw / pmax(tot, 1) * median(tot))
colnames(X_raw) <- paste0("raw_", genes); colnames(X_log) <- paste0("log_", genes)

meta_feat <- as.matrix(meta[, .(
  log_volume = log(volume), total_cnt = tot, log_total = log1p(tot), density = tot / volume,
  center_x, center_y,
  dataset = as.integer(factor(Datasets)),
  region  = fifelse(is.na(Region), 0L, as.integer(Region)),
  ei_code = fifelse(is.na(Excitatory_vs_Inhibitory), 0L, fifelse(Excitatory_vs_Inhibitory == "excitatory", 1L, 2L)),
  segment = fifelse(is.na(Segment), 0L, as.integer(Segment)),
  female  = as.integer(Gender == "female"),
  mouse   = as.integer(factor(Mouse_ID)),
  ap_pos  = as.integer(AP_position),
  section = as.integer(factor(Section_ID)),
  level_C = as.integer(grepl("_C$", Section_ID)), level_T = as.integer(grepl("_T$", Section_ID)),
  level_L = as.integer(grepl("_L$", Section_ID)), level_S = as.integer(grepl("_S$", Section_ID)))])

sec_ids <- split(seq_len(N), meta$Section_ID)
coords  <- as.matrix(meta[, .(center_x, center_y)])

# niche = mean log expression of the 10 nearest cells in the same section, self excluded
niche <- matrix(0, N, length(genes), dimnames = list(NULL, paste0("niche_", genes)))
niche_dist <- numeric(N)
for (idx in sec_ids) {
  k <- min(10, length(idx) - 1); if (k < 1) next
  nn <- get.knn(coords[idx, , drop = FALSE], k = k)
  for (j in seq_along(idx)) niche[idx[j], ] <- colMeans(X_log[idx[nn$nn.index[j, ]], , drop = FALSE])
  niche_dist[idx] <- rowMeans(nn$nn.dist)
}
base_feat <- cbind(X_raw, X_log, meta_feat, niche, niche_dist = niche_dist)

# neighbour label props among a labelled pool, leave-self-out, per section
label_feats <- function(pool_idx, pool_lab, ks = c(10, 30)) {
  out <- matrix(0, N, K * length(ks) + 2 * length(ks))
  cn <- c(); for (k in ks) cn <- c(cn, paste0("nb", k, "_", classes), paste0("nb", k, "_meandist"), paste0("nb", k, "_maxdist"))
  colnames(out) <- cn
  pool_set <- rep(FALSE, N); pool_set[pool_idx] <- TRUE
  lab_of <- rep(NA_integer_, N); lab_of[pool_idx] <- pool_lab
  for (idx in sec_ids) {
    pidx <- idx[pool_set[idx]]; if (length(pidx) < 2) next
    nn <- get.knnx(coords[pidx, , drop = FALSE], coords[idx, , drop = FALSE], k = min(max(ks) + 1, length(pidx)))
    for (j in seq_along(idx)) {
      nbi <- nn$nn.index[j, ]; nbd <- nn$nn.dist[j, ]
      self <- which(pidx[nbi] == idx[j]); if (length(self)) { nbi <- nbi[-self]; nbd <- nbd[-self] }
      off <- 0
      for (k in ks) {
        kk <- min(k, length(nbi)); labs <- lab_of[pidx[nbi[seq_len(kk)]]]
        out[idx[j], off + 1:K] <- tabulate(labs + 1L, nbins = K) / kk
        out[idx[j], off + K + 1] <- mean(nbd[seq_len(kk)]); out[idx[j], off + K + 2] <- max(nbd[seq_len(kk)])
        off <- off + K + 2
      }
    }
  }
  out
}

fd <- fread(fold_file, colClasses = list(character = "cell_id"))
stopifnot(identical(fd$cell_id, mt$cell_id))
folds <- split(seq_len(n_tr), fd$fold)

sub_rows <- function(P, rows) if (nrow(P) != length(rows)) P[rows, , drop = FALSE] else P
apply_mask <- function(P, rows) { P <- sub_rows(P, rows); Pm <- P * mask[rows, , drop = FALSE]; Pm / pmax(rowSums(Pm), 1e-12) }
acc <- function(P, rows) { P <- apply_mask(P, rows); mean(max.col(P) - 1 == y_tr[rows]) }
group_acc <- function(P, rows = tr_rows) { p <- max.col(apply_mask(P, rows)) - 1
  tapply(p == y_tr[rows], ei_cell[rows], mean) }

save_result <- function(name, oof_prob, test_prob, notes = "") {
  stopifnot(dim(oof_prob) == c(n_tr, K), dim(test_prob) == c(n_te, K))
  d <- file.path(model_dir, name); dir.create(d, showWarnings = FALSE, recursive = TRUE)
  colnames(oof_prob) <- classes; colnames(test_prob) <- classes
  saveRDS(oof_prob,  file.path(d, "oof_prob.rds"))
  saveRDS(test_prob, file.path(d, "test_prob.rds"))
  g <- group_acc(oof_prob)
  txt <- sprintf("[%s] OOF acc masked = %.4f | exc %.4f inh %.4f NA %.4f | %s",
                 name, acc(oof_prob, tr_rows), g["excitatory"], g["inhibitory"], g["NA"], notes)
  writeLines(txt, file.path(d, "summary.txt")); say("%s", txt)
}
done <- function(nm) file.exists(file.path(model_dir, nm, "test_prob.rds"))

xgb_par <- list(objective = "multi:softprob", num_class = K, eval_metric = "mlogloss",
                eta = 0.08, max_depth = 6, subsample = 0.8, colsample_bytree = 0.4,
                min_child_weight = 1, lambda = 1, nthread = nthread, tree_method = "hist")
say("data loaded: %d train / %d test, %d genes, %d classes, folds %s", n_tr, n_te, length(genes), K, paste(lengths(folds), collapse = "/"))

## spatial feature builders -------------------------------------------------

# per-section bandwidth = median 1-nn distance among all cells of the section
sec_bw <- sapply(sec_ids, function(idx) { if (length(idx) < 3) return(NA_real_)
  median(get.knn(coords[idx, , drop = FALSE], k = 1)$nn.dist[, 1]) })
sec_bw[is.na(sec_bw)] <- median(sec_bw, na.rm = TRUE)

# knn over any cells within section, self excluded
knn_any <- function(kmax) {
  lapply(seq_along(sec_ids), function(s) {
    idx <- sec_ids[[s]]; ni <- length(idx); k <- min(kmax, ni - 1)
    if (k < 1) return(NULL)
    nn <- get.knn(coords[idx, , drop = FALSE], k = k)
    list(idx = idx, NBI = matrix(idx[nn$nn.index], ni, k), NBD = nn$nn.dist)
  })
}

# label-free spatial block: within-section coordinates, expression pc niches, density, E/I + region composition
unsup_feats <- function(npc = 30, ks_niche = c(5, 25, 50), ks_ei = c(10, 30), rad = c(150, 400)) {
  geo <- matrix(0, N, 8, dimnames = list(NULL, c("dx", "dy", "rad", "theta", "pc1", "pc2", "abs_pc2", "log_secn")))
  for (idx in sec_ids) {
    co <- coords[idx, , drop = FALSE]; ce <- colMeans(co); d <- sweep(co, 2, ce)
    geo[idx, 1:2] <- d; geo[idx, 3] <- sqrt(rowSums(d^2)); geo[idx, 4] <- atan2(d[, 2], d[, 1])
    if (length(idx) >= 3) { pr <- prcomp(d, center = FALSE); sc <- pr$x
      geo[idx, 5] <- sc[, 1]; geo[idx, 6] <- sc[, 2]; geo[idx, 7] <- abs(sc[, 2]) }
    geo[idx, 8] <- log(length(idx))
  }
  pcs <- prcomp(X_log, center = TRUE, scale. = FALSE, rank. = npc)$x
  colnames(pcs) <- paste0("pc_", seq_len(npc))
  knn <- knn_any(max(ks_niche, ks_ei, 50))
  nich <- matrix(0, N, npc * length(ks_niche)); colnames(nich) <- paste0("nichepc", rep(ks_niche, each = npc), "_", seq_len(npc))
  dens <- matrix(0, N, length(ks_niche) + length(rad)); colnames(dens) <- c(paste0("dist_k", ks_niche), paste0("cnt_r", rad))
  regs <- sort(unique(na.omit(meta$Region))); ei_lv <- c("excitatory", "inhibitory", "NA")
  regcode <- match(meta$Region, regs); eicode <- match(ei_cell, ei_lv)
  eir <- matrix(0, N, length(ks_ei) * (3 + length(regs)))
  colnames(eir) <- as.vector(sapply(ks_ei, function(k) c(paste0("nbei", k, "_", ei_lv), paste0("nbreg", k, "_", regs))))
  for (s in seq_along(knn)) {
    o <- knn[[s]]; if (is.null(o)) next
    idx <- o$idx; ni <- length(idx); kav <- ncol(o$NBI)
    off <- 0
    for (k in ks_niche) { kk <- min(k, kav)
      W <- sparseMatrix(i = rep(seq_len(ni), kk), j = as.vector(o$NBI[, 1:kk]), x = 1 / kk, dims = c(ni, N))
      nich[idx, off + 1:npc] <- as.matrix(W %*% pcs); off <- off + npc }
    for (j in seq_along(ks_niche)) { kk <- min(ks_niche[j], kav); dens[idx, j] <- o$NBD[, kk] }
    for (j in seq_along(rad)) dens[idx, length(ks_niche) + j] <- rowSums(o$NBD <= rad[j])
    off <- 0
    for (k in ks_ei) { kk <- min(k, kav)
      E <- matrix(eicode[o$NBI[, 1:kk]], ni, kk); R <- matrix(regcode[o$NBI[, 1:kk]], ni, kk)
      for (l in 1:3) eir[idx, off + l] <- rowMeans(E == l)
      for (l in seq_along(regs)) eir[idx, off + 3 + l] <- rowSums(!is.na(R) & R == l) / kk
      off <- off + 3 + length(regs) }
  }
  cbind(geo, pcs, nich, dens, eir)
}

# label-derived spatial block from a labelled / pseudo-labelled pool (soft or one-hot pool_P),
# self always excluded: knn class props, gaussian-kernel props, distance to nearest cell of each
# class, section composition of the pool, dorsal-ventral axis from the pool centroids
lab_feats <- function(pool_idx, pool_P, ks = c(5, 15, 50), kmax = 50, bw_mult = 2.5, hard_lab = NULL) {
  stopifnot(nrow(pool_P) == length(pool_idx), ncol(pool_P) == K)
  BIG <- log1p(5000)
  cn <- c(as.vector(sapply(ks, function(k) c(paste0("nb", k, "_", classes), paste0("nb", k, "_meandist"), paste0("nb", k, "_maxdist")))),
          paste0("gau_", classes), "gau_logsumw", paste0("dmin_", classes), paste0("comp_", classes),
          "dv_proj", "dv_perp", "d_dh", "d_ven", "dv_ok")
  out <- matrix(0, N, length(cn), dimnames = list(NULL, cn))
  out[, paste0("dmin_", classes)] <- BIG
  pool_pos <- rep(NA_integer_, N); pool_pos[pool_idx] <- seq_along(pool_idx)
  hard <- if (is.null(hard_lab)) max.col(pool_P) - 1L else as.integer(hard_lab)
  is_dh <- grepl("^DH_", classes); is_ven <- grepl("^MV_|motoneuron$", classes)
  for (s in seq_along(sec_ids)) {
    idx <- sec_ids[[s]]; pidx <- idx[!is.na(pool_pos[idx])]
    np <- length(pidx); if (np < 2) next
    ni <- length(idx); kq <- min(kmax + 1, np)
    co_q <- coords[idx, , drop = FALSE]
    nn <- get.knnx(coords[pidx, , drop = FALSE], co_q, k = kq)
    NBI <- matrix(pidx[nn$nn.index], ni, kq); NBD <- nn$nn.dist
    is_self <- NBI == idx
    for (i in which(rowSums(is_self) > 0)) { keep <- !is_self[i, ]; NBI[i, ] <- c(NBI[i, keep], NA); NBD[i, ] <- c(NBD[i, keep], Inf) }
    Ppos <- matrix(pool_pos[NBI], ni, kq)
    kav <- kq - 1L
    off <- 0
    for (k in ks) { kk <- min(k, kav)
      W <- sparseMatrix(i = rep(seq_len(ni), kk), j = as.vector(Ppos[, 1:kk, drop = FALSE]), x = 1 / kk, dims = c(ni, length(pool_idx)))
      out[idx, off + 1:K] <- as.matrix(W %*% pool_P)
      out[idx, off + K + 1] <- rowMeans(NBD[, 1:kk, drop = FALSE]); out[idx, off + K + 2] <- NBD[, kk]
      off <- off + K + 2 }
    bw <- bw_mult * sec_bw[s]
    Wg <- exp(-(NBD[, 1:kav, drop = FALSE]^2) / (2 * bw^2)); sw <- rowSums(Wg)
    W <- sparseMatrix(i = rep(seq_len(ni), kav), j = as.vector(Ppos[, 1:kav, drop = FALSE]), x = as.vector(Wg / pmax(sw, 1e-12)), dims = c(ni, length(pool_idx)))
    out[idx, off + 1:K] <- as.matrix(W %*% pool_P); out[idx, off + K + 1] <- log1p(sw); off <- off + K + 1
    hl <- hard[pool_pos[pidx]]
    for (c in unique(hl)) { pc <- pidx[hl == c]; k2 <- min(2, length(pc))
      nnc <- get.knnx(coords[pc, , drop = FALSE], co_q, k = k2)
      d1 <- nnc$nn.dist[, 1]; selfhit <- pc[nnc$nn.index[, 1]] == idx
      if (k2 == 2) d1[selfhit] <- nnc$nn.dist[selfhit, 2] else d1[selfhit] <- Inf
      out[idx, off + c + 1] <- pmin(log1p(d1), BIG) }
    off <- off + K
    PP <- pool_P[pool_pos[pidx], , drop = FALSE]; sP <- colSums(PP)
    comp <- matrix(sP, ni, K, byrow = TRUE); inpool <- !is.na(pool_pos[idx])
    if (np > 1) comp[inpool, ] <- (comp[inpool, , drop = FALSE] - pool_P[pool_pos[idx[inpool]], , drop = FALSE]) / (np - 1)
    comp[!inpool, ] <- comp[!inpool, , drop = FALSE] / np
    out[idx, off + 1:K] <- comp; off <- off + K
    dh <- pidx[is_dh[hl + 1]]; ve <- pidx[is_ven[hl + 1]]
    if (length(dh) >= 1 && length(ve) >= 1) {
      sdh <- colSums(coords[dh, , drop = FALSE]); sve <- colSums(coords[ve, , drop = FALSE])
      ndh <- length(dh); nve <- length(ve)
      cdh <- matrix(sdh, ni, 2, byrow = TRUE); cve <- matrix(sve, ni, 2, byrow = TRUE)
      wdh <- rep(ndh, ni); wve <- rep(nve, ni)
      sd_ <- idx %in% dh; sv <- idx %in% ve
      cdh[sd_, ] <- cdh[sd_, , drop = FALSE] - co_q[sd_, , drop = FALSE]; wdh[sd_] <- ndh - 1
      cve[sv, ] <- cve[sv, , drop = FALSE] - co_q[sv, , drop = FALSE]; wve[sv] <- nve - 1
      ok <- wdh >= 1 & wve >= 1
      cdh <- cdh / pmax(wdh, 1); cve <- cve / pmax(wve, 1)
      ax <- cve - cdh; L <- sqrt(rowSums(ax^2)); u <- ax / pmax(L, 1e-9)
      v <- co_q - cdh
      proj <- rowSums(v * u) / pmax(L, 1e-9); perp <- (v[, 1] * u[, 2] - v[, 2] * u[, 1]) / pmax(L, 1e-9)
      out[idx[ok], off + 1] <- proj[ok]; out[idx[ok], off + 2] <- abs(perp[ok])
      out[idx[ok], off + 3] <- log1p(sqrt(rowSums((co_q - cdh)^2)))[ok]; out[idx[ok], off + 4] <- log1p(sqrt(rowSums((co_q - cve)^2)))[ok]
      out[idx[ok], off + 5] <- 1
    }
  }
  out
}

# slice-boundary block: distance to the section convex hull, neighbourhood-offset edge scores, radial rank
edge_feats <- function(ks = c(20, 50)) {
  cn <- c("d_hull", "log_d_hull", paste0("edge_off", ks), "rad_rank")
  out <- matrix(0, N, length(cn), dimnames = list(NULL, cn))
  knn <- knn_any(max(ks))
  for (s in seq_along(sec_ids)) {
    idx <- sec_ids[[s]]; ni <- length(idx); co <- coords[idx, , drop = FALSE]
    ce <- colMeans(co); rad <- sqrt(rowSums(sweep(co, 2, ce)^2)); out[idx, "rad_rank"] <- rank(rad) / ni
    if (ni >= 4) {
      h <- chull(co); H <- co[h, , drop = FALSE]; nh <- nrow(H); dmin <- rep(Inf, ni)
      for (e in seq_len(nh)) { A <- H[e, ]; B <- H[if (e == nh) 1 else e + 1, ]; AB <- B - A; L2 <- sum(AB^2)
        t <- if (L2 > 0) pmin(pmax(((co[, 1] - A[1]) * AB[1] + (co[, 2] - A[2]) * AB[2]) / L2, 0), 1) else rep(0, ni)
        px <- A[1] + t * AB[1]; py <- A[2] + t * AB[2]; dmin <- pmin(dmin, sqrt((co[, 1] - px)^2 + (co[, 2] - py)^2)) }
      out[idx, "d_hull"] <- dmin; out[idx, "log_d_hull"] <- log1p(dmin)
    }
    o <- knn[[s]]; if (is.null(o)) next; kav <- ncol(o$NBI)
    for (k in ks) { kk <- min(k, kav)
      mx <- rowMeans(matrix(coords[o$NBI[, 1:kk], 1], ni, kk)); my <- rowMeans(matrix(coords[o$NBI[, 1:kk], 2], ni, kk))
      md <- rowMeans(o$NBD[, 1:kk, drop = FALSE])
      out[idx, paste0("edge_off", k)] <- sqrt((mx - co[, 1])^2 + (my - co[, 2])^2) / pmax(md, 1e-9) }
  }
  out
}

onehot <- function(y) { M <- matrix(0, length(y), K); M[cbind(seq_along(y), y + 1L)] <- 1; M }

## stage 1: feature files ----------------------------------------------------

unsup_file <- file.path(feat_dir, "unsup_feats.rds")
edge_file  <- file.path(feat_dir, "edge_feats.rds")
reps_file  <- file.path(feat_dir, "expr_reps.rds")
nbll_file  <- file.path(feat_dir, "nb_ll_nb.rds")
lf_file <- function(ks, f) file.path(feat_dir, sprintf("lf_k%s_fold%d.rds", paste(ks, collapse = "-"), f))

# fold-wise feature matrix F (log counts + metadata + one-hot + niche + nb10/30 label props);
# fold f: label pool = training part of fold f, fold 0: pool = all 5000 (test model only)
if (!file.exists(file.path(feat_dir, "F_colnames.txt"))) {
  oh <- function(v, nm) { f <- factor(v); m <- model.matrix(~ f - 1); colnames(m) <- paste0(nm, "_", levels(f)); m }
  cat_oh <- cbind(oh(meta_feat[, "dataset"], "ds"), oh(meta_feat[, "region"], "reg"), oh(meta_feat[, "segment"], "seg"),
                  oh(meta_feat[, "mouse"], "mouse"), oh(meta_feat[, "ap_pos"], "ap"), oh(meta_feat[, "section"], "sec"))
  fold_id <- integer(n_tr); for (f in seq_along(folds)) fold_id[folds[[f]]] <- f
  fwrite(data.table(y = y_tr, fold = fold_id, ei = ei_cell[tr_rows]), file.path(feat_dir, "y_fold.csv"))
  for (f in 0:5) {
    pool <- if (f == 0) tr_rows else setdiff(tr_rows, folds[[f]])
    Fm <- cbind(X_log, meta_feat, cat_oh, niche, label_feats(pool, y_tr[pool]))
    saveRDS(Fm, file.path(feat_dir, sprintf("F_fold%d.rds", f)))
    con <- file(file.path(feat_dir, sprintf("F_fold%d.f64", f)), "wb"); writeBin(as.vector(t(Fm)), con, size = 8); close(con)
    say("F_fold%d: %d x %d", f, nrow(Fm), ncol(Fm))
  }
  writeLines(colnames(Fm), file.path(feat_dir, "F_colnames.txt"))
  rm(Fm, cat_oh)
}

# neighbour-label caches for the tree members
for (ks in list(c(10, 30), c(5, 15, 50))) for (f in 0:5) if (!file.exists(lf_file(ks, f))) {
  pool <- if (f == 0) tr_rows else setdiff(tr_rows, folds[[f]])
  saveRDS(label_feats(pool, y_tr[pool], ks), lf_file(ks, f))
}

if (!file.exists(unsup_file)) { saveRDS(unsup_feats(), unsup_file); say("unsup_feats written") }
if (!file.exists(edge_file))  { saveRDS(edge_feats(),  edge_file);  say("edge_feats written") }

# expression pcs + knn-smoothed expression (k = 15 + self, in pc space)
if (!file.exists(reps_file)) {
  pcs <- prcomp(scale(X_log), center = FALSE, scale. = FALSE, rank. = 30)$x[, 1:30]
  colnames(pcs) <- paste0("pc", 1:30)
  nn <- get.knn(pcs, k = 15)
  W <- sparseMatrix(i = c(rep(seq_len(N), each = 15), seq_len(N)), j = c(as.vector(t(nn$nn.index)), seq_len(N)),
                    x = 1 / 16, dims = c(N, N))
  sm <- as.matrix(W %*% X_log); colnames(sm) <- paste0("sm_", genes)
  saveRDS(list(pcs = pcs, sm = sm), reps_file)
  say("expr_reps written")
  rm(pcs, nn, W, sm)
}

# class-conditional count likelihoods, fold-wise (needed by the nb_glm members):
# negative-binomial log-posterior block, mu = total x class gene composition (Dirichlet
# prior, pseudo-count alpha picked on inner folds of the multinomial model), per-gene
# method-of-moments dispersion pooled over classes
if (!file.exists(nbll_file)) {
  alphas <- c(1, 5, 20, 50, 200); ng <- length(genes)
  class_counts <- function(idx) { Nk <- matrix(0, K, ng)
    for (k in 0:(K - 1)) { ii <- idx[y_tr[idx] == k]; if (length(ii)) Nk[k + 1, ] <- colSums(X_raw[ii, , drop = FALSE]) }
    Nk }
  logpi_of <- function(Nk, alpha) { pg <- colSums(Nk) / sum(Nk)
    log((Nk + alpha * matrix(pg, K, ng, byrow = TRUE)) / (rowSums(Nk) + alpha)) }
  ll_mult <- function(rows, logpi) X_raw[rows, , drop = FALSE] %*% t(logpi)
  logprior_of <- function(idx) log((tabulate(y_tr[idx] + 1L, K) + 0.5) / (length(idx) + 0.5 * K))
  lse <- function(M) { m <- apply(M, 1, max); m + log(rowSums(exp(M - m))) }
  logpost <- function(LL) pmax(LL - lse(LL), -20)
  softmax <- function(M) { P <- exp(M - apply(M, 1, max)); P / rowSums(P) }
  inner_folds <- function(idx, nf, seed) { set.seed(seed); fid <- integer(length(idx))
    for (cl in unique(y_tr[idx])) { ii <- which(y_tr[idx] == cl); fid[ii] <- sample(rep(seq_len(nf), length.out = length(ii))) }
    split(idx, fid) }
  theta_of <- function(idx, logpi) { pi <- exp(logpi); num <- numeric(ng); den <- numeric(ng)
    for (k in 0:(K - 1)) { ii <- idx[y_tr[idx] == k]; if (!length(ii)) next
      mu <- outer(tot[ii], pi[k + 1, ]); x <- X_raw[ii, , drop = FALSE]
      num <- num + colSums(mu^2); den <- den + colSums((x - mu)^2 - mu) }
    th <- num / pmax(den, 1e-8); th[den <= 0] <- 1e4; pmin(pmax(th, 0.05), 1e4) }
  ll_nb <- function(rows, logpi, theta) { pi <- exp(logpi); x <- X_raw[rows, , drop = FALSE]
    out <- matrix(0, length(rows), K)
    for (k in 1:K) { mu <- outer(tot[rows], pi[k, ]); th <- matrix(theta, length(rows), ng, byrow = TRUE)
      out[, k] <- rowSums(x * log(mu / (mu + th) + 1e-300) + th * log(th / (mu + th))) }
    out }

  LL_in <- vector("list", 6); LL_in_nb <- vector("list", 6)
  inner_acc <- matrix(NA_real_, 6, length(alphas))
  for (f in 0:5) {
    tr <- if (f == 0) tr_rows else setdiff(tr_rows, folds[[f]])
    ifs <- if (f == 0) folds else inner_folds(tr, 5, 500 + f)
    LL_in[[f + 1]] <- lapply(alphas, function(a) matrix(0, n_tr, K)); LL_in_nb[[f + 1]] <- matrix(0, n_tr, K)
    for (h in seq_along(ifs)) {
      ho <- ifs[[h]]; itr <- setdiff(tr, ho); Nk <- class_counts(itr)
      for (a in seq_along(alphas)) LL_in[[f + 1]][[a]][ho, ] <- ll_mult(ho, logpi_of(Nk, alphas[a]))
    }
    lp <- logprior_of(tr)
    for (a in seq_along(alphas)) inner_acc[f + 1, a] <- acc(softmax(sweep(LL_in[[f + 1]][[a]][tr, ], 2, lp, "+")), tr)
  }
  alpha_sel <- alphas[apply(inner_acc[2:6, ], 1, which.max)]
  alpha_te  <- alphas[which.max(inner_acc[1, ])]
  say("count model: alpha per fold %s, test alpha %g", paste(alpha_sel, collapse = "/"), alpha_te)

  for (f in 0:5) {
    tr <- if (f == 0) tr_rows else setdiff(tr_rows, folds[[f]]); a <- if (f == 0) alpha_te else alpha_sel[f]
    ifs <- if (f == 0) folds else inner_folds(tr, 5, 500 + f)
    for (h in seq_along(ifs)) { ho <- ifs[[h]]; itr <- setdiff(tr, ho); lpi <- logpi_of(class_counts(itr), a)
      LL_in_nb[[f + 1]][ho, ] <- ll_nb(ho, lpi, theta_of(itr, lpi)) }
  }
  LL_oof_nb <- matrix(0, n_tr, K)
  for (f in 1:5) { te <- folds[[f]]; tr <- setdiff(tr_rows, te)
    lpi <- logpi_of(class_counts(tr), alpha_sel[f]); LL_oof_nb[te, ] <- ll_nb(te, lpi, theta_of(tr, lpi)) }
  lpi <- logpi_of(class_counts(tr_rows), alpha_te)
  LL_te_nb <- ll_nb(te_rows, lpi, theta_of(tr_rows, lpi))
  # fold f block: training rows = inner-fold estimates, held-out rows = other-4-folds, test rows = all-train
  LL_fold <- lapply(1:5, function(f) { te <- folds[[f]]; tr <- setdiff(tr_rows, te)
    M <- matrix(0, N, K); M[tr, ] <- LL_in_nb[[f + 1]][tr, ]; M[te, ] <- LL_oof_nb[te, ]; M[te_rows, ] <- LL_te_nb
    logpost(M) })
  saveRDS(list(LL_fold = LL_fold, LL_fold0 = logpost(rbind(LL_oof_nb, LL_te_nb))), nbll_file)
  say("nb_ll_nb written")
  rm(LL_in, LL_in_nb, LL_fold, LL_oof_nb, LL_te_nb); gc(verbose = FALSE)
}

## stage 2: level-1 members --------------------------------------------------

zs <- function(M, tr) { mu <- colMeans(M[tr, , drop = FALSE]); s <- apply(M[tr, , drop = FALSE], 2, sd)
  s[is.na(s) | s < 1e-8] <- 1; sweep(sweep(M, 2, mu), 2, s, "/") }
mlogloss <- function(P, y) -mean(log(pmax(P[cbind(seq_along(y), y + 1L)], 1e-9)))
prob60 <- function(fit, X, s) { pr <- predict(fit, X, s = s, type = "response")[, , 1]
  P <- matrix(0, nrow(X), K); P[, as.integer(colnames(pr)) + 1L] <- pr; P }
glm_groups <- c("excitatory", "inhibitory", "NA")

# extra blocks shared by the glmnet-family members: edge 5 + smoothed expression 200 + expression pcs 30 + unsup 149
glm_extra <- function() {
  e <- readRDS(edge_file); colnames(e) <- paste0("edge_", colnames(e))
  r <- readRDS(reps_file); p <- r$pcs; colnames(p) <- paste0("epc_", seq_len(ncol(p)))
  u <- readRDS(unsup_file); colnames(u) <- paste0("un_", colnames(u))
  cbind(e, r$sm, p, u)
}

# multinomial grouped lasso on F + extra blocks (1083 cols, z-scored with training-fold stats);
# lambda per fold by one inner 80/20 split (mlogloss) or inner 5-fold cv.glmnet (deviance);
# grouped = one model per E/I/NA group (rows and classes restricted);
# test = refit on all 5000 at the geometric-mean lambda; ll adds the 60-col nb log-posterior block
glm_member <- function(tag, lmr, grouped = FALSE, inner = "split", ll = FALSE) {
  extra <- glm_extra()
  lls <- if (ll) readRDS(nbll_file)
  load_F <- function(f) {
    Fm <- cbind(readRDS(file.path(feat_dir, sprintf("F_fold%d.rds", f))), extra)
    if (ll) { M <- if (f == 0) lls$LL_fold0 else lls$LL_fold[[f]]; colnames(M) <- paste0("gll_", classes); Fm <- cbind(Fm, M) }
    Fm
  }
  fit_path <- function(X, y, lambda = NULL) {
    ok <- y %in% (which(tabulate(y + 1L, K) >= 2) - 1L)      # glmnet needs >= 2 obs per class
    glmnet(X[ok, , drop = FALSE], droplevels(factor(y[ok], levels = 0:(K - 1))), family = "multinomial", alpha = 1,
           type.multinomial = "grouped", nlambda = 40, lambda.min.ratio = lmr, lambda = lambda,
           standardize = FALSE, maxit = 1e5, thresh = 1e-4)
  }
  fit_predict <- function(Z, tr, pr, seed) {
    if (inner == "cv") {
      y <- y_tr[tr]; ok <- y %in% (which(tabulate(y + 1L, K) >= 5) - 1L)
      set.seed(seed); fid <- integer(sum(ok))
      for (cl in unique(y[ok])) { ii <- which(y[ok] == cl); fid[ii] <- sample(rep(1:5, length.out = length(ii))) }
      cvf <- cv.glmnet(Z[tr[ok], , drop = FALSE], droplevels(factor(y[ok], levels = 0:(K - 1))), family = "multinomial",
                       alpha = 1, type.multinomial = "grouped", nlambda = 40, lambda.min.ratio = lmr,
                       standardize = FALSE, maxit = 1e5, thresh = 1e-4, foldid = fid, type.measure = "deviance")
      i <- which.min(cvf$cvm); lam <- cvf$lambda[i]
      fit <- if (all(ok)) cvf$glmnet.fit else fit_path(Z[tr, ], y, lambda = cvf$lambda[seq_len(i)])
      return(list(P = prob60(fit, Z[pr, , drop = FALSE], lam), lam = lam, path = cvf$lambda))
    }
    set.seed(seed); va <- sample(tr, round(0.2 * length(tr))); itr <- setdiff(tr, va)
    fit_in <- fit_path(Z[itr, ], y_tr[itr])
    ll_va <- sapply(fit_in$lambda, function(s) mlogloss(prob60(fit_in, Z[va, ], s), y_tr[va]))
    i <- which.min(ll_va)
    fit <- fit_path(Z[tr, ], y_tr[tr], lambda = fit_in$lambda[seq_len(i)])
    list(P = prob60(fit, Z[pr, , drop = FALSE], fit_in$lambda[i]), lam = fit_in$lambda[i], path = fit_in$lambda)
  }
  refit_predict <- function(Z, tr, pr, path, lam) {
    fit <- fit_path(Z[tr, ], y_tr[tr], lambda = c(path[path > lam], lam))
    prob60(fit, Z[pr, , drop = FALSE], lam)
  }
  oof <- matrix(0, n_tr, K); paths <- list()
  lam_sel <- if (grouped) matrix(NA_real_, 5, 3, dimnames = list(NULL, glm_groups)) else numeric(5)
  for (f in 1:5) {
    te <- folds[[f]]; tr <- setdiff(tr_rows, te)
    Z <- zs(load_F(f), tr)
    if (!grouped) {
      r <- fit_predict(Z, tr, te, 100 + f); oof[te, ] <- r$P; lam_sel[f] <- r$lam; paths[[f]] <- r$path
    } else for (g in glm_groups) {
      trg <- tr[ei_cell[tr] == g]; teg <- te[ei_cell[te] == g]
      r <- fit_predict(Z, trg, teg, 100 + f); oof[teg, ] <- r$P; lam_sel[f, g] <- r$lam; paths[[paste(f, g)]] <- r$path
    }
    say("%s fold %d acc %.4f", tag, f, acc(oof, te))
    rm(Z); gc(verbose = FALSE)
  }
  Z <- zs(load_F(0), tr_rows)
  p_te <- matrix(0, n_te, K)
  if (!grouped) {
    p_te <- refit_predict(Z, tr_rows, te_rows, paths[[5]], exp(mean(log(lam_sel))))
  } else for (g in glm_groups) {
    trg <- tr_rows[ei_cell[tr_rows] == g]; teg <- te_rows[ei_cell[te_rows] == g]
    p_te[teg - n_tr, ] <- refit_predict(Z, trg, teg, paths[[paste(5, g)]], exp(mean(log(lam_sel[, g]))))
  }
  save_result(tag, oof, p_te, notes = sprintf("lambda %s", paste(signif(unlist(lam_sel), 3), collapse = ",")))
  rm(Z, extra); gc(verbose = FALSE)
}

# splice two members by cell group: exc/inh rows from the 60-class model, NA rows from the per-group model
splice_ei <- function(src_ei, src_na, tag) {
  rdm <- function(nm, w) readRDS(file.path(model_dir, nm, paste0(w, "_prob.rds")))
  oof <- rdm(src_ei, "oof"); p_te <- rdm(src_ei, "test")
  na_tr <- ei_cell[tr_rows] == "NA"; na_te <- ei_cell[te_rows] == "NA"
  oof[na_tr, ] <- rdm(src_na, "oof")[na_tr, ]; p_te[na_te, ] <- rdm(src_na, "test")[na_te, ]
  save_result(tag, oof, p_te, notes = sprintf("exc/inh rows from %s, NA rows from %s", src_ei, src_na))
}

if (!done("glm3_b")) glm_member("glm3_b", 5e-4, grouped = FALSE, inner = "split")
if (!done("glm3_f")) glm_member("glm3_f", 2e-4, grouped = TRUE,  inner = "cv")
if (!done("glm3_h3")) splice_ei("glm3_b", "glm3_f", "glm3_h3")

if (!done("nb_glm_b2")) glm_member("nb_glm_b2", 5e-4, grouped = FALSE, inner = "split", ll = TRUE)
if (!done("nb_glm_f2")) glm_member("nb_glm_f2", 2e-4, grouped = TRUE,  inner = "cv",    ll = TRUE)
if (!done("nb_glm2")) splice_ei("nb_glm_b2", "nb_glm_f2", "nb_glm2")

# elastic net on standardized F only
if (!done("diverse_glmnet")) {
  fit_en <- function(X, y, lambda = NULL) {
    ok <- y %in% (which(tabulate(y + 1L, K) >= 2) - 1L)
    glmnet(X[ok, ], droplevels(factor(y[ok], levels = 0:(K - 1))), family = "multinomial", alpha = 0.5,
           type.multinomial = "grouped", nlambda = 25, lambda.min.ratio = 0.02, lambda = lambda,
           standardize = FALSE, maxit = 2e4, thresh = 1e-5)
  }
  oof <- matrix(0, n_tr, K); lam_sel <- numeric(5)
  for (f in 1:5) {
    te <- folds[[f]]; tr <- setdiff(tr_rows, te)
    Z <- zs(readRDS(file.path(feat_dir, sprintf("F_fold%d.rds", f))), tr)
    set.seed(100 + f); va <- sample(tr, round(0.2 * length(tr))); itr <- setdiff(tr, va)
    fit_in <- fit_en(Z[itr, ], y_tr[itr])
    ll_va <- sapply(fit_in$lambda, function(s) mlogloss(prob60(fit_in, Z[va, ], s), y_tr[va]))
    i <- which.min(ll_va); lam_sel[f] <- fit_in$lambda[i]
    fit <- fit_en(Z[tr, ], y_tr[tr], lambda = fit_in$lambda[seq_len(i)])
    oof[te, ] <- prob60(fit, Z[te, ], lam_sel[f])
    say("diverse_glmnet fold %d acc %.4f", f, acc(oof, te))
  }
  Z <- zs(readRDS(file.path(feat_dir, "F_fold0.rds")), tr_rows)
  lam <- exp(mean(log(lam_sel)))
  fit <- fit_en(Z[tr_rows, ], y_tr, lambda = c(fit_in$lambda[fit_in$lambda > lam], lam))
  save_result("diverse_glmnet", oof, prob60(fit, Z[te_rows, ], lam),
              notes = sprintf("alpha 0.5, lambda %s -> %.4g", paste(signif(lam_sel, 3), collapse = ","), lam))
  rm(Z, fit, fit_in); gc(verbose = FALSE)
}

# two-level lasso following the annotation hierarchy: coarse group (12, per E/I group) x subtype | coarse
if (!done("hier2_a")) {
  coarse_of <- function(cl) {
    if (grepl("^DH_ex_|^DM_ex_", cl)) "dorsal_ex"
    else if (grepl("^DH_in_", cl)) "dorsal_in"
    else if (grepl("^M_ex_|^MV_ex_", cl)) "mv_ex"
    else if (grepl("^M_in_|^MV_in_|^VH_in_", cl)) "mv_in"
    else if (grepl("motoneuron$|^cholinergic", cl)) "cholinergic"
    else if (grepl("^oligodendrocyte", cl)) "oligo_lineage"
    else if (grepl("^astrocyte", cl)) "astro"
    else if (cl == "microglia") "microglia"
    else if (cl %in% c("endothelial", "pericyte")) "vascular"
    else if (grepl("^meninges", cl)) "meninges"
    else if (cl == "ependymal") "ependymal"
    else if (cl %in% c("Schwann_cell", "peripheral_glia")) "peripheral"
    else stop("unmapped class: ", cl)
  }
  coarse_classes <- c("dorsal_ex", "mv_ex", "dorsal_in", "mv_in", "cholinergic", "oligo_lineage", "astro",
                      "microglia", "vascular", "meninges", "ependymal", "peripheral")
  cls_coarse <- sapply(classes, coarse_of)
  C <- length(coarse_classes)
  g_of_class <- match(cls_coarse, coarse_classes) - 1L
  yc_tr <- g_of_class[y_tr + 1L]
  ei_of_coarse <- sapply(coarse_classes, function(cc) { u <- unique(ei_of_class[cls_coarse == cc])
    stopifnot(length(u) == 1); u })

  prob_C <- function(fit, X, s, Cc) { pr <- predict(fit, X, s = s, type = "response")
    pr <- if (length(dim(pr)) == 3) pr[, , 1, drop = FALSE] else pr
    pr <- matrix(pr, nrow(X), dimnames = list(NULL, colnames(pr)))
    P <- matrix(0, nrow(X), Cc); P[, as.integer(colnames(pr)) + 1L] <- pr; P }
  fit_path_C <- function(X, y, Cc, lambda = NULL) {
    ok <- y %in% (which(tabulate(y + 1L, Cc) >= 2) - 1L)
    glmnet(X[ok, , drop = FALSE], droplevels(factor(y[ok], levels = 0:(Cc - 1))), family = "multinomial", alpha = 1,
           type.multinomial = "grouped", nlambda = 40, lambda.min.ratio = 2e-4, lambda = lambda,
           standardize = FALSE, maxit = 1e5, thresh = 1e-4)
  }
  fit_predict_C <- function(Z, tr, y, Cc, pr, seed) {
    ok <- y %in% (which(tabulate(y + 1L, Cc) >= 5) - 1L)
    set.seed(seed); fid <- integer(sum(ok))
    for (cl in unique(y[ok])) { ii <- which(y[ok] == cl); fid[ii] <- sample(rep(1:5, length.out = length(ii))) }
    cvf <- cv.glmnet(Z[tr[ok], , drop = FALSE], droplevels(factor(y[ok], levels = 0:(Cc - 1))), family = "multinomial",
                     alpha = 1, type.multinomial = "grouped", nlambda = 40, lambda.min.ratio = 2e-4,
                     standardize = FALSE, maxit = 1e5, thresh = 1e-4, foldid = fid, type.measure = "deviance")
    i <- which.min(cvf$cvm); lam <- cvf$lambda[i]
    fit <- if (all(ok)) cvf$glmnet.fit else fit_path_C(Z[tr, , drop = FALSE], y, Cc, lambda = cvf$lambda[seq_len(i)])
    list(P = prob_C(fit, Z[pr, , drop = FALSE], lam, Cc), lam = lam, path = cvf$lambda)
  }
  refit_predict_C <- function(Z, tr, y, Cc, pr, path, lam) {
    fit <- fit_path_C(Z[tr, , drop = FALSE], y, Cc, lambda = c(path[path > lam], lam))
    prob_C(fit, Z[pr, , drop = FALSE], lam, Cc)
  }

  extra <- glm_extra()
  Pc_oof <- matrix(0, n_tr, C); Pc_te <- matrix(0, n_te, C)
  Ps_oof <- matrix(0, n_tr, K); Ps_te <- matrix(0, n_te, K)
  lam_c <- matrix(NA_real_, 5, 3, dimnames = list(NULL, glm_groups)); path_c <- list()
  lam_s <- matrix(NA_real_, 5, C, dimnames = list(NULL, coarse_classes)); path_s <- list()
  for (f in 1:5) {
    te <- folds[[f]]; tr <- setdiff(tr_rows, te)
    Z <- zs(cbind(readRDS(file.path(feat_dir, sprintf("F_fold%d.rds", f))), extra), tr)
    for (g in glm_groups) {
      trg <- tr[ei_cell[tr] == g]; teg <- te[ei_cell[te] == g]
      r <- fit_predict_C(Z, trg, yc_tr[trg], C, teg, 100 + f)
      Pc_oof[teg, ] <- r$P; lam_c[f, g] <- r$lam; path_c[[paste(f, g)]] <- r$path
    }
    for (ci in seq_len(C)) {
      cc <- coarse_classes[ci]; subs <- which(cls_coarse == cc)
      teg <- te[ei_cell[te] == ei_of_coarse[cc]]
      if (length(subs) == 1) { Ps_oof[teg, subs] <- 1; next }
      trc <- tr[yc_tr[tr] == ci - 1L]; ysub <- match(y_tr[trc] + 1L, subs) - 1L
      if (sum(tabulate(ysub + 1L, length(subs)) >= 2) < 2) {
        j <- which.max(tabulate(ysub + 1L, length(subs))); Ps_oof[teg, subs[j]] <- 1; next }
      r <- fit_predict_C(Z, trc, ysub, length(subs), teg, 200 + f)
      Ps_oof[teg, subs] <- r$P; lam_s[f, cc] <- r$lam; path_s[[paste(f, cc)]] <- r$path
    }
    say("hier2_a fold %d acc %.4f", f, acc(Ps_oof * Pc_oof[, g_of_class + 1L], te))
    rm(Z); gc(verbose = FALSE)
  }
  Z <- zs(cbind(readRDS(file.path(feat_dir, "F_fold0.rds")), extra), tr_rows)
  for (g in glm_groups) {
    trg <- tr_rows[ei_cell[tr_rows] == g]; teg <- te_rows[ei_cell[te_rows] == g]
    Pc_te[teg - n_tr, ] <- refit_predict_C(Z, trg, yc_tr[trg], C, teg, path_c[[paste(5, g)]], exp(mean(log(lam_c[, g]))))
  }
  for (ci in seq_len(C)) {
    cc <- coarse_classes[ci]; subs <- which(cls_coarse == cc)
    teg <- te_rows[ei_cell[te_rows] == ei_of_coarse[cc]]
    if (length(subs) == 1) { Ps_te[teg - n_tr, subs] <- 1; next }
    trc <- tr_rows[yc_tr == ci - 1L]; ysub <- match(y_tr[trc] + 1L, subs) - 1L
    if (all(is.na(lam_s[, cc]))) { j <- which.max(tabulate(ysub + 1L, length(subs))); Ps_te[teg - n_tr, subs[j]] <- 1; next }
    pth <- path_s[[paste(max(which(!is.na(lam_s[, cc]))), cc)]]
    Ps_te[teg - n_tr, subs] <- refit_predict_C(Z, trc, ysub, length(subs), teg, pth, exp(mean(log(lam_s[, cc]), na.rm = TRUE)))
  }
  save_result("hier2_a", Ps_oof * Pc_oof[, g_of_class + 1L], Ps_te * Pc_te[, g_of_class + 1L],
              notes = "P(coarse) x P(subtype | coarse), lambda by inner 5-fold cv")
  rm(Z, extra, Pc_oof, Pc_te, Ps_oof, Ps_te); gc(verbose = FALSE)
}

# bagged version of the spliced lasso: 32 bootstrap replicates per fold at the fold's selected lambda
if (!done("bag_glm")) {
  bag_dir <- file.path(scratch_dir, "bag"); dir.create(file.path(bag_dir, "parts"), showWarnings = FALSE, recursive = TRUE)
  extra <- glm_extra()
  lams_file <- file.path(bag_dir, "lams.rds")
  if (!file.exists(lams_file)) {
    # per-fold lambdas by the same inner rules as glm3_b (80/20 split, lmr 5e-4) and
    # glm3_f NA group (5-fold cv, lmr 2e-4)
    lam_single <- lam_na <- numeric(5)
    for (f in 1:5) {
      te <- folds[[f]]; tr <- setdiff(tr_rows, te)
      Z <- zs(cbind(readRDS(file.path(feat_dir, sprintf("F_fold%d.rds", f))), extra), tr)
      fp <- function(X, y, lmr, lambda = NULL) { ok <- y %in% (which(tabulate(y + 1L, K) >= 2) - 1L)
        glmnet(X[ok, , drop = FALSE], droplevels(factor(y[ok], levels = 0:(K - 1))), family = "multinomial", alpha = 1,
               type.multinomial = "grouped", nlambda = 40, lambda.min.ratio = lmr, lambda = lambda,
               standardize = FALSE, maxit = 1e5, thresh = 1e-4) }
      set.seed(100 + f); va <- sample(tr, round(0.2 * length(tr))); itr <- setdiff(tr, va)
      fit_in <- fp(Z[itr, ], y_tr[itr], 5e-4)
      ll_va <- sapply(fit_in$lambda, function(s) mlogloss(prob60(fit_in, Z[va, ], s), y_tr[va]))
      lam_single[f] <- fit_in$lambda[which.min(ll_va)]
      trg <- tr[ei_cell[tr] == "NA"]; y <- y_tr[trg]; ok <- y %in% (which(tabulate(y + 1L, K) >= 5) - 1L)
      set.seed(100 + f); fid <- integer(sum(ok))
      for (cl in unique(y[ok])) { ii <- which(y[ok] == cl); fid[ii] <- sample(rep(1:5, length.out = length(ii))) }
      cvf <- cv.glmnet(Z[trg[ok], , drop = FALSE], droplevels(factor(y[ok], levels = 0:(K - 1))), family = "multinomial",
                       alpha = 1, type.multinomial = "grouped", nlambda = 40, lambda.min.ratio = 2e-4,
                       standardize = FALSE, maxit = 1e5, thresh = 1e-4, foldid = fid, type.measure = "deviance")
      lam_na[f] <- cvf$lambda[which.min(cvf$cvm)]
      say("bag lambdas fold %d: single %.4g, NA %.4g", f, lam_single[f], lam_na[f])
      rm(Z); gc(verbose = FALSE)
    }
    saveRDS(list(single = lam_single, na = lam_na), lams_file)
  }
  lams <- readRDS(lams_file)
  for (fold in c(1:5, 0)) {           # fold 0 = test refit on all training rows
    part <- file.path(bag_dir, "parts", sprintf("plain_boot_fold%d.rds", fold))
    if (file.exists(part)) next
    if (fold > 0) { te <- folds[[fold]]; tr <- setdiff(tr_rows, te); lam_s <- lams$single[fold]; lam_n <- lams$na[fold]
    } else { te <- te_rows; tr <- tr_rows; lam_s <- exp(mean(log(lams$single))); lam_n <- exp(mean(log(lams$na))) }
    Z <- zs(cbind(readRDS(file.path(feat_dir, sprintf("F_fold%d.rds", fold))), extra), tr)
    tr_na <- tr[ei_cell[tr] == "NA"]; te_na <- te[ei_cell[te] == "NA"]
    fit_bag <- function(rows, lam, seed, pr_rows) {
      set.seed(seed); n <- length(rows); idx <- sample.int(n, n, replace = TRUE)
      yb <- y_tr[rows[idx]]; ok <- yb %in% (which(tabulate(yb + 1L, K) >= 2) - 1L)
      path <- exp(seq(log(0.5), log(lam), length.out = ceiling(log(0.5 / lam) / 0.2) + 1))
      fit <- glmnet(Z[rows[idx][ok], , drop = FALSE], droplevels(factor(yb[ok], levels = 0:(K - 1))), family = "multinomial",
                    alpha = 1, type.multinomial = "grouped", lambda = path, standardize = FALSE, maxit = 1e5, thresh = 1e-4)
      prob60(fit, Z[pr_rows, , drop = FALSE], lam)
    }
    P_s <- array(NA_real_, c(length(te), K, 32)); P_n <- array(NA_real_, c(length(te_na), K, 32))
    for (b in 1:32) {
      P_s[, , b] <- fit_bag(tr,    lam_s, 7000 + 100 * fold + b, te)
      P_n[, , b] <- fit_bag(tr_na, lam_n, 8000 + 100 * fold + b, te_na)
      if (b %% 8 == 0) say("bag fold %d: %d/32 bags", fold, b)
    }
    saveRDS(list(fold = fold, te = te, te_na = te_na, P_single = P_s, P_na = P_n, done = 32), part)
    rm(Z, P_s, P_n); gc(verbose = FALSE)
  }
  oof <- matrix(0, n_tr, K); p_te <- matrix(0, n_te, K)
  for (fold in 0:5) {
    p <- readRDS(file.path(bag_dir, "parts", sprintf("plain_boot_fold%d.rds", fold)))
    Ps <- apply(p$P_single, c(1, 2), mean)
    Ps[match(p$te_na, p$te), ] <- apply(p$P_na, c(1, 2), mean)
    if (fold > 0) oof[p$te, ] <- Ps else p_te <- Ps
  }
  save_result("bag_glm", oof, p_te, notes = "32 bootstrap bags per fold, spliced by E/I group")
  rm(extra); gc(verbose = FALSE)
}

# scikit-learn members (separate scripts; fixed random_state, n_jobs = nthread)
sk_members <- c(mlp = "diverse_mlp", rf = "diverse_rf", et = "diverse_et", hgb = "diverse_hgb",
                svc = "diverse_svc", lr = "diverse_lr", mlpbag = "diverse_mlp_b")
if (!all(sapply(sk_members, done))) {
  Sys.setenv(PIPE_WORK = normalizePath(work_dir), PIPE_NTHREAD = as.character(nthread))
  sk_out <- file.path(scratch_dir, "sk_out")
  if (!all(file.exists(file.path(sk_out, c("mlp_test.csv", "rf_test.csv", "et_test.csv", "hgb_test.csv", "svc_test.csv", "lr_test.csv"))))) {
    say("running sk_models.py")
    stopifnot(system2(python, c("sk_models.py", "mlp", "rf", "et", "hgb", "svc", "lr")) == 0)
  }
  if (!file.exists(file.path(sk_out, "mlpbag_test.csv"))) {
    say("running sk_mlp_bag.py")
    stopifnot(system2(python, "sk_mlp_bag.py") == 0)
  }
  for (nm in names(sk_members)) if (!done(sk_members[[nm]])) {
    oof <- as.matrix(fread(file.path(sk_out, paste0(nm, "_oof.csv")), header = FALSE))
    p_te <- as.matrix(fread(file.path(sk_out, paste0(nm, "_test.csv")), header = FALSE))
    save_result(sk_members[[nm]], oof, p_te, notes = paste("sklearn", nm))
  }
}

# single 60-class xgboost on the merged feature set (880 label-free cols + fold-wise nb10/30 label block)
if (!done("xgb3_a")) {
  fu_file <- file.path(feat_dir, "feat_unsup.rds")
  if (file.exists(fu_file)) feats <- readRDS(fu_file) else {
    edge  <- readRDS(edge_file); unsup <- readRDS(unsup_file); reps <- readRDS(reps_file)
    keep_u <- c("dx", "dy", "rad", "theta", "pc1", "pc2", "abs_pc2", "log_secn",
                "dist_k5", "dist_k25", "dist_k50", "cnt_r150", "cnt_r400",
                grep("^nbei|^nbreg", colnames(unsup), value = TRUE))
    U <- unsup[, keep_u]; colnames(U) <- paste0("sp_", colnames(U))
    E <- reps$pcs; colnames(E) <- paste0("epc_", seq_len(ncol(E)))
    B <- base_feat[, setdiff(colnames(base_feat), c("dataset", "mouse", "section"))]
    feats <- cbind(B, edge, reps$sm, E, U)
    saveRDS(feats, fu_file)
    rm(edge, unsup, reps, U, E, B)
  }
  stopifnot(nrow(feats) == N, ncol(feats) == 880)
  par <- modifyList(xgb_par, list(eta = 0.04, colsample_bytree = 0.35, min_child_weight = 3))
  oof <- matrix(0, n_tr, K); best <- integer(5)
  for (f in 1:5) {
    te <- folds[[f]]; tr <- setdiff(tr_rows, te)
    x <- cbind(feats, readRDS(lf_file(c(10, 30), f)))
    set.seed(1000 + f)
    fit <- xgb.train(par, xgb.DMatrix(x[tr, ], label = y_tr[tr]), nrounds = 3000,
                     watchlist = list(val = xgb.DMatrix(x[te, ], label = y_tr[te])),
                     early_stopping_rounds = 100, verbose = 0)
    best[f] <- fit$best_iteration
    oof[te, ] <- matrix(predict(fit, xgb.DMatrix(x[te, ]), iterationrange = c(1, best[f] + 1)), ncol = K, byrow = TRUE)
    say("xgb3_a fold %d best %d acc %.4f", f, best[f], acc(oof, te))
  }
  x <- cbind(feats, readRDS(lf_file(c(10, 30), 0)))
  set.seed(1000)
  fit <- xgb.train(par, xgb.DMatrix(x[tr_rows, ], label = y_tr), nrounds = ceiling(mean(best) * 1.1), verbose = 0)
  p_te <- matrix(predict(fit, xgb.DMatrix(x[te_rows, ])), ncol = K, byrow = TRUE)
  save_result("xgb3_a", oof, p_te, notes = sprintf("best iters %s", paste(best, collapse = ",")))
  rm(x, feats, fit); gc(verbose = FALSE)
}

# xgboost on base + smoothed expression + expression pcs (id codes dropped) + fold-wise label block
if (!done("expr_c")) {
  reps <- readRDS(reps_file)
  feats <- cbind(base_feat, reps$sm, reps$pcs)
  feats <- feats[, setdiff(colnames(feats), c("dataset", "mouse", "section"))]
  stopifnot(ncol(feats) == 846)
  par <- modifyList(xgb_par, list(eta = 0.05, colsample_bytree = 0.3, min_child_weight = 3))
  oof <- matrix(0, n_tr, K); best <- integer(5)
  for (f in 1:5) {
    te <- folds[[f]]; tr <- setdiff(tr_rows, te)
    x <- cbind(feats, readRDS(lf_file(c(10, 30), f)))
    set.seed(21000 + f)
    fit <- xgb.train(par, xgb.DMatrix(x[tr, ], label = y_tr[tr]), nrounds = 3000,
                     watchlist = list(val = xgb.DMatrix(x[te, ], label = y_tr[te])),
                     early_stopping_rounds = 100, verbose = 0)
    best[f] <- fit$best_iteration
    oof[te, ] <- matrix(predict(fit, xgb.DMatrix(x[te, ]), iterationrange = c(1, best[f] + 1)), ncol = K, byrow = TRUE)
    say("expr_c fold %d best %d acc %.4f", f, best[f], acc(oof, te))
  }
  x <- cbind(feats, readRDS(lf_file(c(10, 30), 0)))
  set.seed(21000)
  fit <- xgb.train(par, xgb.DMatrix(x[tr_rows, ], label = y_tr), nrounds = ceiling(mean(best) * 1.1), verbose = 0)
  p_te <- matrix(predict(fit, xgb.DMatrix(x[te_rows, ])), ncol = K, byrow = TRUE)
  save_result("expr_c", oof, p_te, notes = sprintf("best iters %s", paste(best, collapse = ",")))
  rm(x, feats, reps, fit); gc(verbose = FALSE)
}

# spatial xgboost, round 1: base + label-free spatial block (+ edge block for spatial_d) +
# label-derived block from the fold's training pool
spatial_dir <- file.path(scratch_dir, "spatial"); dir.create(spatial_dir, showWarnings = FALSE, recursive = TRUE)
run_round1 <- function(name, use_edge, seed_base, out_file) {
  U <- readRDS(unsup_file); if (use_edge) U <- cbind(U, readRDS(edge_file))
  oof <- matrix(0, n_tr, K); best <- integer(5); te_pred <- vector("list", 5)
  for (f in 1:5) {
    te <- folds[[f]]; tr <- setdiff(tr_rows, te)
    x <- cbind(base_feat, U, lab_feats(tr, onehot(y_tr[tr])))
    set.seed(seed_base + f)
    fit <- xgb.train(xgb_par, xgb.DMatrix(x[tr, ], label = y_tr[tr]), nrounds = 1500,
                     watchlist = list(val = xgb.DMatrix(x[te, ], label = y_tr[te])),
                     early_stopping_rounds = 60, verbose = 0)
    best[f] <- fit$best_iteration
    oof[te, ] <- matrix(predict(fit, xgb.DMatrix(x[te, ]), iterationrange = c(1, best[f] + 1)), ncol = K, byrow = TRUE)
    te_pred[[f]] <- matrix(predict(fit, xgb.DMatrix(x[te_rows, ]), iterationrange = c(1, best[f] + 1)), ncol = K, byrow = TRUE)
    say("%s fold %d best %d acc %.4f", name, f, best[f], acc(oof, te))
  }
  x <- cbind(base_feat, U, lab_feats(tr_rows, onehot(y_tr)))
  set.seed(seed_base)
  fit <- xgb.train(xgb_par, xgb.DMatrix(x[tr_rows, ], label = y_tr), nrounds = ceiling(mean(best) * 1.1), verbose = 0)
  P_te <- matrix(predict(fit, xgb.DMatrix(x[te_rows, ])), ncol = K, byrow = TRUE)
  r1 <- list(oof = oof, best = best, te_pred = te_pred, P_te = P_te)
  saveRDS(r1, out_file)
  save_result(name, oof, P_te, notes = sprintf("best iters %s", paste(best, collapse = ",")))
  rm(x, fit); gc(verbose = FALSE)
  r1
}
round1_file <- file.path(spatial_dir, "round1.rds")
if (!done("spatial_a") || !file.exists(round1_file)) invisible(run_round1("spatial_a", FALSE, 22000, round1_file))
if (!done("spatial_d")) invisible(run_round1("spatial_d", TRUE, 22100, file.path(spatial_dir, "round1_edge.rds")))

# round 2: the label-derived block is rebuilt with pool = true training labels + masked soft
# round-1 predictions of the test cells; training rows of fold g take the pseudo labels from the
# round-1 fold-g model (which never saw their labels), held-out rows from the fold-f model
if (!done("spatial_e")) {
  r1 <- readRDS(round1_file)
  U <- readRDS(unsup_file)
  lab_feats_fold <- function(f) {
    L <- NULL
    for (g in 1:5) {
      if (g == f) next
      tr_true <- if (f == 0) tr_rows else setdiff(tr_rows, folds[[f]])
      Lg <- lab_feats(c(tr_true, te_rows), rbind(onehot(y_tr[tr_true]), apply_mask(r1$te_pred[[g]], te_rows)))
      if (is.null(L)) L <- matrix(0, N, ncol(Lg), dimnames = list(NULL, colnames(Lg)))
      L[folds[[g]], ] <- Lg[folds[[g]], ]
    }
    if (f == 0) {
      Lt <- lab_feats(c(tr_rows, te_rows), rbind(onehot(y_tr), apply_mask(r1$P_te, te_rows)))
      L[te_rows, ] <- Lt[te_rows, ]
    } else {
      te <- folds[[f]]; tr <- setdiff(tr_rows, te)
      Lf <- lab_feats(c(tr, te, te_rows), rbind(onehot(y_tr[tr]), apply_mask(r1$oof[te, , drop = FALSE], te),
                                                apply_mask(r1$te_pred[[f]], te_rows)))
      L[te, ] <- Lf[te, ]
    }
    L
  }
  oof <- matrix(0, n_tr, K); best <- integer(5)
  for (f in 1:5) {
    te <- folds[[f]]; tr <- setdiff(tr_rows, te)
    x <- cbind(base_feat, U, lab_feats_fold(f))
    set.seed(23000 + f)
    fit <- xgb.train(xgb_par, xgb.DMatrix(x[tr, ], label = y_tr[tr]), nrounds = 1500,
                     watchlist = list(val = xgb.DMatrix(x[te, ], label = y_tr[te])),
                     early_stopping_rounds = 60, verbose = 0)
    best[f] <- fit$best_iteration
    oof[te, ] <- matrix(predict(fit, xgb.DMatrix(x[te, ]), iterationrange = c(1, best[f] + 1)), ncol = K, byrow = TRUE)
    say("spatial_e fold %d best %d acc %.4f", f, best[f], acc(oof, te))
  }
  x <- cbind(base_feat, U, lab_feats_fold(0))
  set.seed(23000)
  fit <- xgb.train(xgb_par, xgb.DMatrix(x[tr_rows, ], label = y_tr), nrounds = ceiling(mean(best) * 1.1), verbose = 0)
  p_te <- matrix(predict(fit, xgb.DMatrix(x[te_rows, ])), ncol = K, byrow = TRUE)
  save_result("spatial_e", oof, p_te, notes = sprintf("best iters %s", paste(best, collapse = ",")))
  rm(x, fit, r1, U); gc(verbose = FALSE)
}

# grouped xgboost (one model per E/I/NA group) with two label-block scales, averaged,
# then within-family expert models redistribute each family's total probability (NA group)
if (!done("hier_e")) {
  hier_dir <- file.path(scratch_dir, "hier"); dir.create(hier_dir, showWarnings = FALSE, recursive = TRUE)
  grp_names <- c("excitatory", "inhibitory", "NA")
  cls_idx <- lapply(grp_names, function(g) which(ei_of_class == g)); names(cls_idx) <- grp_names
  fit_grp <- function(Ftr, ytr, Fva, yva, Kg, par, nrounds = 2000, es = 60, fixed_rounds = NULL) {
    par$num_class <- Kg
    dtr <- xgb.DMatrix(Ftr, label = ytr)
    if (is.null(fixed_rounds)) {
      dva <- xgb.DMatrix(Fva, label = yva)
      fit <- xgb.train(par, dtr, nrounds = nrounds, watchlist = list(val = dva), early_stopping_rounds = es, verbose = 0)
      list(pred = matrix(predict(fit, dva, iterationrange = c(1, fit$best_iteration + 1)), ncol = Kg, byrow = TRUE),
           best = fit$best_iteration, fit = fit)
    } else {
      fit <- xgb.train(par, dtr, nrounds = fixed_rounds, verbose = 0)
      list(pred = matrix(predict(fit, xgb.DMatrix(Fva)), ncol = Kg, byrow = TRUE), best = fixed_rounds, fit = fit)
    }
  }
  run_grouped <- function(tag, ks, seed_base) {
    fn <- file.path(hier_dir, paste0("res_", tag, ".rds"))
    if (file.exists(fn)) return(readRDS(fn))
    par <- modifyList(xgb_par, list(eta = 0.04))
    oof <- matrix(0, n_tr, K); best <- matrix(NA_integer_, 5, 3, dimnames = list(NULL, grp_names))
    for (f in 1:5) {
      te <- folds[[f]]; tr <- setdiff(tr_rows, te)
      x <- cbind(base_feat, readRDS(lf_file(ks, f)))
      for (g in grp_names) {
        ci <- cls_idx[[g]]; Kg <- length(ci)
        trg <- tr[ei_cell[tr] == g]; teg <- te[ei_cell[te] == g]
        yl <- match(y_tr + 1L, ci) - 1L
        set.seed(seed_base + 100 * match(g, grp_names) + f)
        r <- fit_grp(x[trg, , drop = FALSE], yl[trg], x[teg, , drop = FALSE], yl[teg], Kg, par, nrounds = 3000, es = 80)
        oof[teg, ci] <- r$pred; best[f, g] <- r$best
      }
      say("%s fold %d best %s acc %.4f", tag, f, paste(best[f, ], collapse = "/"), acc(oof, te))
    }
    x <- cbind(base_feat, readRDS(lf_file(ks, 0)))
    test <- matrix(0, n_te, K)
    for (g in grp_names) {
      ci <- cls_idx[[g]]; Kg <- length(ci)
      trg <- tr_rows[ei_cell[tr_rows] == g]; teg <- te_rows[ei_cell[te_rows] == g]
      yl <- match(y_tr + 1L, ci) - 1L
      set.seed(seed_base + 100 * match(g, grp_names))
      r <- fit_grp(x[trg, , drop = FALSE], yl[trg], x[teg, , drop = FALSE], NULL, Kg, par,
                   fixed_rounds = ceiling(mean(best[, g]) * 1.1))
      test[teg - n_tr, ci] <- r$pred
    }
    res <- list(oof = oof, test = test, best = best)
    saveRDS(res, fn)
    save_result(tag, oof, test, notes = sprintf("grouped xgboost, label block k = %s", paste(ks, collapse = "/")))
    res
  }
  b <- run_grouped("hier_b", c(10, 30), 24000)
  d <- run_grouped("hier_d", c(5, 15, 50), 25000)
  hb <- list(oof = (b$oof + d$oof) / 2, test = (b$test + d$test) / 2)

  fams <- list(
    oligo  = c("oligodendrocyte_1", "oligodendrocyte_2", "oligodendrocyte_progenitor_1",
               "oligodendrocyte_progenitor_2", "oligodendrocyte_precursor_cell"),
    astro  = c("astrocyte_1", "astrocyte_2"),
    vasc   = c("endothelial", "pericyte"),
    mening = c("meninges_1", "meninges_2", "meninges_3"),
    moto   = c("alpha_motoneuron", "gamma_motoneuron", "beta_motoneuron"),
    pns    = c("Schwann_cell", "peripheral_glia"))
  stopifnot(all(unlist(fams) %in% classes))
  par_exp <- modifyList(xgb_par, list(eta = 0.04, max_depth = 5, min_child_weight = 2))
  na_tr <- tr_rows[ei_cell[tr_rows] == "NA"]; na_te <- te_rows[ei_cell[te_rows] == "NA"]
  na_te_loc <- na_te - n_tr
  exp_file <- file.path(hier_dir, "experts.rds")
  exp_oof <- list(); exp_test <- list()
  if (file.exists(exp_file)) { ex <- readRDS(exp_file); exp_oof <- ex$exp_oof; exp_test <- ex$exp_test }
  lf_full <- readRDS(lf_file(c(10, 30), 0))
  for (fn in setdiff(names(fams), names(exp_oof))) {
    ci <- match(fams[[fn]], classes); Kf <- length(ci); yl <- match(y_tr + 1L, ci) - 1L
    fi <- match(fn, names(fams))
    Po <- matrix(0, n_tr, Kf); bests <- c()
    for (f in 1:5) {
      te <- folds[[f]]; tr <- setdiff(tr_rows, te)
      x <- cbind(base_feat, readRDS(lf_file(c(10, 30), f)))
      tr_f <- tr[!is.na(yl[tr])]; te_f <- te[!is.na(yl[te])]; te_na <- te[ei_cell[te] == "NA"]
      set.seed(26000 + 10 * fi + f)
      r <- fit_grp(x[tr_f, ], yl[tr_f], x[te_f, ], yl[te_f], Kf, par_exp, nrounds = 3000, es = 80)
      bests <- c(bests, r$best)
      Po[te_na, ] <- matrix(predict(r$fit, xgb.DMatrix(x[te_na, ]), iterationrange = c(1, r$best + 1)), ncol = Kf, byrow = TRUE)
    }
    x <- cbind(base_feat, lf_full); tr_f <- tr_rows[!is.na(yl[tr_rows])]
    set.seed(26000 + 10 * fi)
    r <- fit_grp(x[tr_f, ], yl[tr_f], x[na_te, ], NULL, Kf, par_exp, fixed_rounds = ceiling(mean(bests) * 1.1))
    Pt <- matrix(0, n_te, Kf); Pt[na_te_loc, ] <- r$pred
    exp_oof[[fn]] <- Po; exp_test[[fn]] <- Pt
    saveRDS(list(exp_oof = exp_oof, exp_test = exp_test), exp_file)
    say("family expert %s done (best %s)", fn, paste(bests, collapse = "/"))
  }
  recombine <- function(P, Pe, ci, rows) { P[rows, ci] <- rowSums(P[rows, ci, drop = FALSE]) * Pe[rows, , drop = FALSE]; P }
  P2 <- hb$oof; Pt <- hb$test
  for (fn in c("oligo", "astro", "vasc", "moto")) { ci <- match(fams[[fn]], classes)
    P2 <- recombine(P2, exp_oof[[fn]], ci, na_tr); Pt <- recombine(Pt, exp_test[[fn]], ci, na_te_loc) }
  save_result("hier_e", P2, Pt, notes = "mean(hier_b, hier_d) + family experts oligo/astro/vasc/moto")
  rm(b, d, hb, exp_oof, exp_test, lf_full); gc(verbose = FALSE)
}

## stage 3: ensemble and submission file --------------------------------------

members <- c("glm3_h3", "hier_e", "xgb3_a", "expr_c", "spatial_d", "spatial_e",
             "diverse_hgb", "diverse_et", "diverse_mlp_b", "diverse_glmnet", "diverse_lr",
             "diverse_svc", "diverse_rf", "diverse_mlp", "nb_glm2", "bag_glm", "hier2_a")
stopifnot(sapply(members, done))
say("ensemble over %d members", length(members))

dirs <- list.dirs(model_dir, recursive = FALSE, full.names = TRUE)
dirs <- dirs[basename(dirs) %in% members]
mods <- lapply(dirs, function(d) list(oof = readRDS(file.path(d, "oof_prob.rds")),
                                      te  = readRDS(file.path(d, "test_prob.rds"))))
names(mods) <- basename(dirs)
for (m in names(mods)) {
  stopifnot(dim(mods[[m]]$oof) == c(n_tr, K), dim(mods[[m]]$te) == c(n_te, K),
            identical(colnames(mods[[m]]$oof), classes))
  mods[[m]]$oof <- apply_mask(mods[[m]]$oof, tr_rows); mods[[m]]$te <- apply_mask(mods[[m]]$te, te_rows)
  bad <- !is.finite(mods[[m]]$oof); if (any(bad)) mods[[m]]$oof[bad] <- 0
}
acc_rows <- function(P, rows) mean(max.col(P[rows, , drop = FALSE]) - 1 == y_tr[rows])
ind <- sort(sapply(mods, function(m) acc_rows(m$oof, tr_rows)), decreasing = TRUE)

# greedy forward selection with replacement (weights = selection counts), nested over the folds
wavg <- function(w, which = "oof") { P <- 0
  for (m in names(w)) if (w[m] > 0) P <- P + w[m] * mods[[m]][[which]]
  P / sum(w) }
greedy <- function(rows, iters = 60, patience = 12) {
  cnt <- setNames(rep(0, length(mods)), names(mods)); P <- matrix(0, n_tr, K)
  best <- -Inf; best_cnt <- cnt; since <- 0
  for (it in seq_len(iters)) {
    sc <- sapply(names(mods), function(m) acc_rows(P + mods[[m]]$oof, rows))
    b <- names(which.max(sc)); cnt[b] <- cnt[b] + 1; P <- P + mods[[b]]$oof
    if (max(sc) > best + 1e-12) { best <- max(sc); best_cnt <- cnt; since <- 0 } else since <- since + 1
    if (since >= patience) break
  }
  best_cnt / sum(best_cnt)
}
cv_pred <- matrix(0, n_tr, K)
for (f in 1:5) { te <- folds[[f]]; cv_pred[te, ] <- wavg(greedy(setdiff(tr_rows, te)))[te, ] }
cv_acc <- acc_rows(cv_pred, tr_rows)
w_full <- greedy(tr_rows)
final_w <- if (cv_acc >= ind[1] - 1e-9) w_full else setNames(as.numeric(names(mods) == names(ind)[1]), names(mods))
G <- list(oof = apply_mask(cv_pred, tr_rows), te = apply_mask(wavg(final_w, "te"), te_rows))
w_txt <- paste(sprintf("%s=%.3f", names(final_w)[final_w > 0], final_w[final_w > 0]), collapse = " ")
say("greedy nested acc %.4f | weights %s", cv_acc, w_txt)

# per-class ridge stacker on the members' oof probabilities, after correlation dedup (greedy by
# accuracy, keep a member only if its oof prob correlates < 0.97 with every kept one); the score
# of class c is a closed-form ridge regression of 1[y = c] on the members' P(c) columns, ridge
# penalty (as a fraction of n) picked per outer fold on an inner fold; nested like the greedy part
keep <- character(0)
for (m in names(ind)[ind >= 0.60]) {
  if (!length(keep)) { keep <- m; next }
  r <- sapply(keep, function(k) cor(as.vector(mods[[m]]$oof), as.vector(mods[[k]]$oof)))
  if (max(r) < 0.97) keep <- c(keep, m)
}
say("stacker members (%d): %s", length(keep), paste(keep, collapse = " "))
ridge_fit_pred <- function(Xtr, Ytr, Xte, fracs) {
  mu <- colMeans(Xtr); sdv <- apply(Xtr, 2, sd); sdv[!is.finite(sdv) | sdv < 1e-8] <- 1
  Xs <- sweep(sweep(Xtr, 2, mu), 2, sdv, "/"); Xt <- sweep(sweep(Xte, 2, mu), 2, sdv, "/")
  ym <- colMeans(Ytr); Yc <- sweep(Ytr, 2, ym)
  E <- eigen(crossprod(Xs), symmetric = TRUE); VtXtY <- crossprod(E$vectors, crossprod(Xs, Yc))
  lapply(fracs, function(fr) { B <- E$vectors %*% (VtXtY / (pmax(E$values, 0) + fr * nrow(Xtr)))
    sweep(Xt %*% B, 2, ym, "+") })
}
Y1 <- outer(y_tr, 0:(K - 1), "==") * 1
fracs <- c(3, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
Lo <- do.call(cbind, lapply(keep, function(m) mods[[m]]$oof))
Lt <- do.call(cbind, lapply(keep, function(m) mods[[m]]$te))
cols_of <- function(c) seq(c, ncol(Lo), by = K)
fit_pred <- function(tr, te_idx, Xte_all, fr) {
  outs <- lapply(fr, function(l) matrix(0, length(te_idx), K))
  for (c in seq_len(K)) {
    pc <- ridge_fit_pred(Lo[tr, cols_of(c), drop = FALSE], Y1[tr, c, drop = FALSE],
                         Xte_all[te_idx, cols_of(c), drop = FALSE], fr)
    for (j in seq_along(fr)) outs[[j]][, c] <- pc[[j]][, 1]
  }
  outs
}
post <- function(P, rows) { P <- pmax(P, 1e-6) * mask[rows, , drop = FALSE]; P / pmax(rowSums(P), 1e-12) }
inner_of <- function(f) { others <- setdiff(1:5, f); others[(f %% 4) + 1] }
stack_oof <- matrix(0, n_tr, K); lam_f <- numeric(5)
for (f in 1:5) {
  te <- folds[[f]]; tr <- setdiff(tr_rows, te); inn <- folds[[inner_of(f)]]; tr_in <- setdiff(tr, inn)
  pa <- fit_pred(tr_in, inn, Lo, fracs)
  acc_in <- sapply(pa, function(P) mean(max.col(post(P, inn)) - 1 == y_tr[inn]))
  j <- which.max(acc_in); lam_f[f] <- fracs[j]
  pb <- fit_pred(tr, te, Lo, fracs)
  stack_oof[te, ] <- post(pb[[j]], te)
}
stack_te <- post(fit_pred(tr_rows, seq_len(n_te), Lt, exp(mean(log(lam_f))))[[1]], te_rows)
S <- list(oof = apply_mask(stack_oof, tr_rows), te = apply_mask(stack_te, te_rows))
stack_acc <- acc_rows(stack_oof, tr_rows)
say("stacker nested acc %.4f (ridge fracs %s)", stack_acc, paste(lam_f, collapse = "/"))

# final = (greedy + stacker) / 2, E/I mask, argmax
set.seed(1)
fb_oof <- (G$oof + S$oof) / 2
fb_te  <- apply_mask((G$te + S$te) / 2, te_rows)
blend_acc <- acc_rows(apply_mask(fb_oof, tr_rows), tr_rows)
stopifnot(!anyNA(fb_te), all(is.finite(fb_te)))
pred_te <- classes[max.col(fb_te)]
fwrite(data.table(Cell_ID = ms$cell_id, MERFISH_cell_type_annotation.y = pred_te),
       file.path(out_dir, "prediction.csv"), quote = FALSE)
saveRDS(list(greedy = G, stack = S, blend = list(oof = fb_oof, te = fb_te), weights = final_w),
        file.path(out_dir, "final_prob.rds"))

lines <- c(sprintf("nested OOF accuracy: greedy %.4f | stacker %.4f | blend %.4f", cv_acc, stack_acc, blend_acc),
           sprintf("greedy weights: %s", w_txt),
           sprintf("prediction.csv: %d rows, %d distinct classes", length(pred_te), length(unique(pred_te))))
for (ref in c("candidate_seeded/prediction.csv", "reference/prediction_submitted.csv")) if (file.exists(ref) && nrow(fread(ref)) == n_te) {
  cur <- fread(ref, colClasses = "character")
  lines <- c(lines, sprintf("agreement with %s: %.4f (%d of %d differ)",
                            ref, mean(cur[[2]] == pred_te), sum(cur[[2]] != pred_te), n_te))
}
writeLines(lines, file.path(out_dir, "final_summary.txt"))
for (l in lines) say("%s", l)
say("finished")
