"""Exact variable-length top-k with row-private selection and ordered output."""
ROW_COHORT = 16
LEAF_ROWS = 4
UNROLL_CAP = 4
from golden import TensorSpec
import torch
import pypto.language as pl
RTOL = 1e-05
ATOL = 1e-05
ROWS_PER_TASK = 1
SCAN = 128
MAP = 512
K = 512
LEAF = 2048
PAIR_W = 2 * K
PAGE = 128
SORT_FLOOR = -3e+38
FIN_FLOOR = -1e+29

def _build_fp32(shape):
 M, N = (shape['M'], shape['N'])
 P = N // PAGE
 P8 = (P + 7) // 8 * 8
 NL = (N + LEAF - 1) // LEAF
 LEAF_COHORTS = (M + LEAF_ROWS - 1) // LEAF_ROWS
 WORKSPACE_WIDTH = (N + 1023) // 1024 * 1024
 SHORT_LIMIT = min(K, N - 1)
 if N <= 512:

  @pl.jit
  def dsa41_topk_guarded_root(scores: pl.Tensor[[M, N], pl.FP32], lengths: pl.Tensor[[M], pl.INT32], block_table: pl.Tensor[[1, P], pl.INT32], wpairs: pl.InOut[pl.Tensor[[M, WORKSPACE_WIDTH], pl.FP32]], cutoffs: pl.InOut[pl.Tensor[[M, 1], pl.FP32]], quotas: pl.InOut[pl.Tensor[[M, 1], pl.INT32]], indices: pl.Out[pl.Tensor[[M, K], pl.INT32]]):
   aux_discarded = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_cutoffs = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_quotas = pl.create_tensor([M, 16], dtype=pl.INT32)
   aux_front = pl.create_tensor([M, 16], dtype=pl.INT32)
   for cohort_base in pl.parallel(0, M, ROW_COHORT):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='decomposed_leaf'):
     for t in pl.range(cohort_base, pl.min(cohort_base + ROW_COHORT, M), 1):
      discarded = pl.cast(SORT_FLOOR, pl.FP32)
      frontier_position = pl.cast(2147483647, pl.INT32)
      zero_pairs = pl.tile.full([1, PAIR_W], dtype=pl.FP32, value=0.0)
      for clear_offset in pl.range(0, WORKSPACE_WIDTH, PAIR_W):
       pl.store(zero_pairs, [t, clear_offset], wpairs)
      visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
      wv = pl.max(visible, 1)
      for leaf in pl.range(NL):
       begin = leaf * LEAF
       if begin < wv:
        valid = pl.min(LEAF, wv - begin)
        narrow_raw = pl.load(scores, [t, begin], [1, 512], valid_shape=[1, valid])
        narrow_padded = pl.tile.fillpad(narrow_raw, pad_value=pl.PadValue.min)
        narrow_values = pl.maximum(narrow_padded, SORT_FLOOR)
        narrow_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
        narrow_pairs = pl.tile.sort32(narrow_values, pl.reinterpret_view(narrow_positions, pl.UINT32))
        narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=64)
        narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=256)
        pl.store(narrow_pairs, [t, leaf * PAIR_W], wpairs)
      pl.write(aux_discarded, [t, 0], discarded)
      pl.write(aux_front, [t, 0], frontier_position)
   for cohort_base in pl.parallel(0, M, ROW_COHORT):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='decomposed_merge'):
     for t in pl.range(cohort_base, pl.min(cohort_base + ROW_COHORT, M), 1):
      visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
      wv = pl.max(visible, 1)
      lc = (wv + LEAF - 1) // LEAF
      discarded = pl.read(aux_discarded, [t, 0])
      frontier_position = pl.read(aux_front, [t, 0])
      mtmp = pl.tile.create([1, 2 * PAIR_W], dtype=pl.FP32)
      for cb in pl.range(1, lc):
       mleft = pl.load(wpairs, [t, 0], [1, PAIR_W])
       mright = pl.load(wpairs, [t, cb * PAIR_W], [1, PAIR_W])
       mboth = pl.tile.mrgsort(mleft, mright, tmp=mtmp)
       merge_discarded = pl.tile.read(mboth, [0, PAIR_W])
       if merge_discarded > discarded:
        discarded = merge_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if merge_discarded == discarded:
        merge_front_boundary = pl.tile.read(mboth, [0, PAIR_W - 2])
        merge_front_bound = merge_front_boundary
        if merge_front_bound < FIN_FLOOR:
         merge_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if merge_discarded >= merge_front_bound:
         merge_front_tail = pl.tile.slice(mboth, [1, 1024], [0, PAIR_W])
         merge_front_values = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
         merge_front_positions = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         merge_front_candidates = pl.tile.select(pl.tile.cmps(merge_front_values, merge_discarded, cmp_type=0), merge_front_positions, 2147483647)
         merge_front_wide = pl.col_min(pl.reshape(merge_front_candidates, [8, 64]))
         merge_front_eight = pl.col_min(pl.reshape(merge_front_wide, [8, 8]))
         merge_front_result = pl.tile.read(merge_front_eight, [0, 0])
         merge_front_lane_1 = pl.tile.read(merge_front_eight, [0, 1])
         if merge_front_lane_1 < merge_front_result:
          merge_front_result = merge_front_lane_1
         merge_front_lane_2 = pl.tile.read(merge_front_eight, [0, 2])
         if merge_front_lane_2 < merge_front_result:
          merge_front_result = merge_front_lane_2
         merge_front_lane_3 = pl.tile.read(merge_front_eight, [0, 3])
         if merge_front_lane_3 < merge_front_result:
          merge_front_result = merge_front_lane_3
         merge_front_lane_4 = pl.tile.read(merge_front_eight, [0, 4])
         if merge_front_lane_4 < merge_front_result:
          merge_front_result = merge_front_lane_4
         merge_front_lane_5 = pl.tile.read(merge_front_eight, [0, 5])
         if merge_front_lane_5 < merge_front_result:
          merge_front_result = merge_front_lane_5
         merge_front_lane_6 = pl.tile.read(merge_front_eight, [0, 6])
         if merge_front_lane_6 < merge_front_result:
          merge_front_result = merge_front_lane_6
         merge_front_lane_7 = pl.tile.read(merge_front_eight, [0, 7])
         if merge_front_lane_7 < merge_front_result:
          merge_front_result = merge_front_lane_7
         if merge_front_result < frontier_position:
          frontier_position = merge_front_result
       pl.store(pl.tile.slice(mboth, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
      pl.write(aux_discarded, [t, 0], discarded)
      pl.write(aux_front, [t, 0], frontier_position)
   for cohort_base in pl.parallel(0, M, ROW_COHORT):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='decomposed_materialize'):
     for t in pl.range(cohort_base, pl.min(cohort_base + ROW_COHORT, M), 1):
      fin = pl.load(wpairs, [t, 0], [1, PAIR_W])
      values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
      finite = pl.maximum(values, FIN_FLOOR)
      cutoff = pl.tile.read(finite, [0, K - 1])
      ones_c = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
      flags = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=4), ones_c, 0.0)
      sums64 = pl.col_sum(pl.reshape(flags, [8, 64]))
      sums8 = pl.col_sum(pl.reshape(sums64, [8, 8]))
      above = pl.cast(0, pl.INDEX)
      for lane in pl.unroll(8):
       part = pl.cast(pl.tile.read(sums8, [0, lane]), pl.INT32)
       above = above + pl.cast(part, pl.INDEX)
      cutoff_stage_line = pl.tile.full([1, 16], dtype=pl.FP32, value=0)
      pl.tile.write(cutoff_stage_line, [0, 0], cutoff)
      pl.store(cutoff_stage_line, [t, 0], aux_cutoffs)
      quota_stage_line = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
      pl.tile.write(quota_stage_line, [0, 0], pl.cast(K - above, pl.INT32))
      pl.store(quota_stage_line, [t, 0], aux_quotas)
      discarded = pl.read(aux_discarded, [t, 0])
      frontier_position = pl.read(aux_front, [t, 0])
      replay_safe = discarded < cutoff
      if discarded == cutoff:
       certificate_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       certificate_candidates = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=0), certificate_positions, -1)
       certificate_wide = pl.col_max(pl.reshape(certificate_candidates, [8, 64]))
       certificate_eight = pl.col_max(pl.reshape(certificate_wide, [8, 8]))
       certificate_result = pl.tile.read(certificate_eight, [0, 0])
       certificate_lane_1 = pl.tile.read(certificate_eight, [0, 1])
       if certificate_lane_1 > certificate_result:
        certificate_result = certificate_lane_1
       certificate_lane_2 = pl.tile.read(certificate_eight, [0, 2])
       if certificate_lane_2 > certificate_result:
        certificate_result = certificate_lane_2
       certificate_lane_3 = pl.tile.read(certificate_eight, [0, 3])
       if certificate_lane_3 > certificate_result:
        certificate_result = certificate_lane_3
       certificate_lane_4 = pl.tile.read(certificate_eight, [0, 4])
       if certificate_lane_4 > certificate_result:
        certificate_result = certificate_lane_4
       certificate_lane_5 = pl.tile.read(certificate_eight, [0, 5])
       if certificate_lane_5 > certificate_result:
        certificate_result = certificate_lane_5
       certificate_lane_6 = pl.tile.read(certificate_eight, [0, 6])
       if certificate_lane_6 > certificate_result:
        certificate_result = certificate_lane_6
       certificate_lane_7 = pl.tile.read(certificate_eight, [0, 7])
       if certificate_lane_7 > certificate_result:
        certificate_result = certificate_lane_7
       if frontier_position > certificate_result:
        replay_safe = True
      if replay_safe:
       if cutoff > FIN_FLOOR:
        root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        selected = pl.cast(K, pl.INDEX)
        if selected > 0:
         root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(root_positions, pl.FP32))
         root_keys = root_priority
         root_order = pl.tile.sort32(root_keys, pl.reinterpret_view(root_positions, pl.UINT32))
         root_order = pl.tile.mrgsort(root_order, block_len=64)
         root_order = pl.tile.mrgsort(root_order, block_len=256)
         root_logical = pl.tile.gather_mask(root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         root_valid = pl.tile.set_validshape(root_logical, 1, selected)
         root_safe = pl.tile.fillpad(root_valid, pad_value=pl.PadValue.zero)
         if P == 1:
          single_page = pl.read(block_table, [0, 0])
          pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), single_page), pad_value=pl.PadValue.zero)
         else:
          pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
         root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for map_start in pl.range(0, selected, MAP):
          map_count = pl.min(MAP, selected - map_start)
          map_bits = pl.tile.slice(pl.reinterpret_view(root_safe, pl.FP32), [1, MAP], [0, map_start])
          map_logical = pl.reinterpret_view(map_bits, pl.INT32)
          root_page_idx = pl.tile.shrs(map_logical, 7)
          root_page_ids = pl.tile.gather(pages, root_page_idx, root_tmp)
          root_offsets = pl.sub(map_logical, pl.mul(root_page_idx, PAGE))
          root_physical = pl.add(pl.mul(root_page_ids, PAGE), root_offsets)
          pl.store(pl.tile.set_validshape(root_physical, 1, map_count), [t, map_start], indices)
       else:
        visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
        root = fin
        root_values = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P0101)
        root_positions = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
        root_strict = pl.tile.select(pl.tile.cmps(root_values, cutoff, cmp_type=4), ones_root, 0.0)
        root_equal = pl.tile.select(pl.tile.cmps(root_values, cutoff, cmp_type=0), ones_root, 0.0)
        root_eligible = pl.maximum(root_strict, root_equal)
        root_visible = pl.tile.cmps(pl.cast(root_positions, pl.FP32), pl.cast(pl.cast(visible, pl.INT32), pl.FP32), cmp_type=2)
        root_keep = pl.tile.select(root_visible, root_eligible, 0.0)
        root_sums64 = pl.col_sum(pl.reshape(root_keep, [8, 64]))
        root_sums8 = pl.col_sum(pl.reshape(root_sums64, [8, 8]))
        selected = pl.cast(0, pl.INDEX)
        for root_lane in pl.unroll(8):
         root_part = pl.cast(pl.tile.read(root_sums8, [0, root_lane]), pl.INT32)
         selected = selected + pl.cast(root_part, pl.INDEX)
        pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
        if selected > 0:
         root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(root_positions, pl.FP32))
         root_keys = pl.tile.select(pl.tile.cmps(root_keep, 0.0, cmp_type=4), root_priority, -1.0)
         root_order = pl.tile.sort32(root_keys, pl.reinterpret_view(root_positions, pl.UINT32))
         root_order = pl.tile.mrgsort(root_order, block_len=64)
         root_order = pl.tile.mrgsort(root_order, block_len=256)
         root_sorted_keys = pl.tile.gather_mask(root_order, mask_pattern=pl.tile.MaskPattern.P0101)
         root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), root_sorted_keys), pl.INT32)
         root_valid = pl.tile.set_validshape(root_logical, 1, selected)
         root_safe = pl.tile.fillpad(root_valid, pad_value=pl.PadValue.zero)
         if P == 1:
          single_page = pl.read(block_table, [0, 0])
          pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), single_page), pad_value=pl.PadValue.zero)
         else:
          pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
         root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for map_start in pl.range(0, selected, MAP):
          map_count = pl.min(MAP, selected - map_start)
          map_bits = pl.tile.slice(pl.reinterpret_view(root_safe, pl.FP32), [1, MAP], [0, map_start])
          map_logical = pl.reinterpret_view(map_bits, pl.INT32)
          root_page_idx = pl.tile.shrs(map_logical, 7)
          root_page_ids = pl.tile.gather(pages, root_page_idx, root_tmp)
          root_offsets = pl.sub(map_logical, pl.mul(root_page_idx, PAGE))
          root_physical = pl.add(pl.mul(root_page_ids, PAGE), root_offsets)
          pl.store(pl.tile.set_validshape(root_physical, 1, map_count), [t, map_start], indices)
      else:
       visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
       remaining = pl.cast(K - above, pl.INDEX)
       if P == 1:
        single_page = pl.read(block_table, [0, 0])
        pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), single_page), pad_value=pl.PadValue.zero)
       else:
        pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
       pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
       ones_m = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
       lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
       lanes_f = pl.cast(lanes, pl.FP32)
       lane_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
       priority = pl.cast(pl.sub(lane_origin, lanes), pl.FP32)
       gtmp = pl.tile.create([1, SCAN], dtype=pl.INT32)
       count = pl.cast(0, pl.INDEX)
       for start in pl.range(0, visible, SCAN):
        if count < K:
         valid = pl.min(SCAN, visible - start)
         scan_raw = pl.load(scores, [t, start], [1, SCAN], valid_shape=[1, valid])
         scan_vals = pl.tile.fillpad(scan_raw, pad_value=pl.PadValue.min)
         strict = pl.tile.select(pl.tile.cmps(scan_vals, cutoff, cmp_type=4), ones_m, 0.0)
         ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
         if remaining > 0:
          eq = pl.tile.select(pl.tile.cmps(scan_vals, cutoff, cmp_type=0), ones_m, 0.0)
          eq_valid = pl.tile.cmps(lanes_f, pl.cast(pl.cast(valid, pl.INT32), pl.FP32), cmp_type=2)
          eq = pl.tile.select(eq_valid, eq, 0.0)
          eq_sums64 = pl.col_sum(pl.reshape(eq, [SCAN // 64, 64]))
          eq_sums8 = pl.col_sum(pl.reshape(eq_sums64, [8, 8]))
          available = pl.cast(0, pl.INDEX)
          for eq_lane in pl.unroll(8):
           eq_part = pl.cast(pl.tile.read(eq_sums8, [0, eq_lane]), pl.INT32)
           available = available + pl.cast(eq_part, pl.INDEX)
          take = pl.min(available, remaining)
          if take > 0:
           if take == available:
            ties = eq
           else:
            tie_keys = pl.mul(eq, priority)
            tie_pairs = pl.tile.sort32(tie_keys, pl.reinterpret_view(lanes, pl.UINT32))
            tie_pairs = pl.tile.mrgsort(tie_pairs, block_len=64)
            tie_merged = tie_pairs
            equal_keys = pl.tile.gather_mask(tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
            last_key_f = pl.tile.read(equal_keys, [0, take - 1])
            boundary_f = 128.0 - last_key_f
            ties = pl.tile.select(pl.tile.cmps(lanes_f, boundary_f, cmp_type=3), eq, 0.0)
           remaining = remaining - take
         keep = pl.maximum(strict, ties)
         keep_valid = pl.tile.set_validshape(keep, 1, valid)
         keep_padded = pl.tile.fillpad(keep_valid, pad_value=pl.PadValue.zero)
         keep_sums64 = pl.col_sum(pl.reshape(keep_padded, [SCAN // 64, 64]))
         keep_sums8 = pl.col_sum(pl.reshape(keep_sums64, [8, 8]))
         keep_count = pl.cast(0, pl.INDEX)
         for keep_lane in pl.unroll(8):
          keep_part = pl.cast(pl.tile.read(keep_sums8, [0, keep_lane]), pl.INT32)
          keep_count = keep_count + pl.cast(keep_part, pl.INDEX)
         selected = pl.min(keep_count, K - count)
         if selected > 0:
          order_keys = pl.mul(keep_padded, priority)
          order_pairs = pl.tile.sort32(order_keys, pl.reinterpret_view(lanes, pl.UINT32))
          order_pairs = pl.tile.mrgsort(order_pairs, block_len=64)
          order_merged = order_pairs
          order_vals = pl.tile.gather_mask(order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
          offs = pl.sub(lane_origin, pl.cast(order_vals, pl.INT32))
          compact_valid = pl.tile.set_validshape(offs, 1, selected)
          compact_safe = pl.tile.fillpad(compact_valid, pad_value=pl.PadValue.zero)
          logical = pl.add(compact_safe, pl.cast(start, pl.INT32))
          page_idx = pl.tile.shrs(logical, 7)
          page_ids = pl.tile.gather(pages, page_idx, gtmp)
          offs_in_page = pl.sub(logical, pl.mul(page_idx, 128))
          physical = pl.add(pl.mul(page_ids, 128), offs_in_page)
          pl.store(pl.tile.set_validshape(physical, 1, selected), [t, count], indices)
          count = count + selected
   for pack_task in pl.spmd(1, name_hint='pack_auxiliary'):
    for output_row in pl.range(0, M, 16):
     cutoff_lines = pl.load(aux_cutoffs, [output_row, 0], [16, 16])
     quota_lines = pl.load(aux_quotas, [output_row, 0], [16, 16])
     cutoff_flat = pl.reshape(cutoff_lines, [1, 256])
     quota_flat = pl.reshape(quota_lines, [1, 256])
     pack_offsets = pl.mul(pl.tile.arange(0, [1, 16], dtype=pl.INT32), 16)
     cutoff_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     quota_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     cutoff_packed = pl.tile.gather(cutoff_flat, pack_offsets, cutoff_scratch)
     quota_packed = pl.tile.gather(quota_flat, pack_offsets, quota_scratch)
     pl.store(pl.reshape(cutoff_packed, [16, 1]), [output_row, 0], cutoffs)
     pl.store(pl.reshape(quota_packed, [16, 1]), [output_row, 0], quotas)
   return indices
 elif N <= 1024:

  @pl.jit
  def dsa41_topk_guarded_root(scores: pl.Tensor[[M, N], pl.FP32], lengths: pl.Tensor[[M], pl.INT32], block_table: pl.Tensor[[1, P], pl.INT32], wpairs: pl.InOut[pl.Tensor[[M, WORKSPACE_WIDTH], pl.FP32]], cutoffs: pl.InOut[pl.Tensor[[M, 1], pl.FP32]], quotas: pl.InOut[pl.Tensor[[M, 1], pl.INT32]], indices: pl.Out[pl.Tensor[[M, K], pl.INT32]]):
   aux_discarded = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_cutoffs = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_quotas = pl.create_tensor([M, 16], dtype=pl.INT32)
   aux_front = pl.create_tensor([M, 16], dtype=pl.INT32)
   leaf_discarded_bank = pl.create_tensor([M, NL * 16], dtype=pl.FP32)
   leaf_front_bank = pl.create_tensor([M, NL * 16], dtype=pl.INT32)
   for leaf_job in pl.spmd(LEAF_COHORTS * NL, name_hint='canonical_leaf_plane'):
    leaf = leaf_job // LEAF_COHORTS
    leaf_cohort_base = leaf_job % LEAF_COHORTS * LEAF_ROWS
    for t in pl.range(leaf_cohort_base, pl.min(leaf_cohort_base + LEAF_ROWS, M), 1):
     zero_pairs = pl.tile.full([1, PAIR_W], dtype=pl.FP32, value=0.0)
     pl.store(zero_pairs, [t, leaf * PAIR_W], wpairs)
     if leaf == 0:
      for clear_offset in pl.range(NL * PAIR_W, WORKSPACE_WIDTH, PAIR_W):
       pl.store(zero_pairs, [t, clear_offset], wpairs)
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     begin = leaf * LEAF
     if begin < wv:
      valid = pl.min(LEAF, wv - begin)
      if valid <= 512:
       narrow_raw = pl.load(scores, [t, begin], [1, 512], valid_shape=[1, valid])
       narrow_padded = pl.tile.fillpad(narrow_raw, pad_value=pl.PadValue.min)
       narrow_values = pl.maximum(narrow_padded, SORT_FLOOR)
       narrow_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       narrow_pairs = pl.tile.sort32(narrow_values, pl.reinterpret_view(narrow_positions, pl.UINT32))
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=64)
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=256)
       pl.store(narrow_pairs, [t, leaf * PAIR_W], wpairs)
      else:
       left_valid = pl.min(valid, 512)
       right_valid = pl.max(valid - 512, 0)
       left_raw = pl.load(scores, [t, begin], [1, 512], valid_shape=[1, left_valid])
       right_raw = pl.load(scores, [t, begin + 512], [1, 512], valid_shape=[1, right_valid])
       left_values = pl.maximum(pl.tile.fillpad(left_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       right_values = pl.maximum(pl.tile.fillpad(right_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       left_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       right_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin + 512, pl.INT32))
       left_pairs = pl.tile.sort32(left_values, pl.reinterpret_view(left_positions, pl.UINT32))
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=64)
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=256)
       right_pairs = pl.tile.sort32(right_values, pl.reinterpret_view(right_positions, pl.UINT32))
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=64)
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=256)
       native_scratch = pl.tile.create([1, 2048], dtype=pl.FP32)
       native_pairs = pl.tile.mrgsort(left_pairs, right_pairs, tmp=native_scratch)
       native_discarded = pl.tile.read(native_pairs, [0, PAIR_W])
       if native_discarded > discarded:
        discarded = native_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if native_discarded == discarded:
        native_front_boundary = pl.tile.read(native_pairs, [0, PAIR_W - 2])
        native_front_bound = native_front_boundary
        if native_front_bound < FIN_FLOOR:
         native_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if native_discarded >= native_front_bound:
         native_arity_third = pl.tile.read(native_pairs, [0, PAIR_W + 4])
         if native_arity_third < native_discarded:
          native_arity_second = pl.tile.read(native_pairs, [0, PAIR_W + 2])
          native_arity_bits = pl.reinterpret_view(native_pairs, pl.INT32)
          native_front_result = pl.tile.read(native_arity_bits, [0, PAIR_W + 1])
          if native_arity_second == native_discarded:
           native_arity_second_position = pl.tile.read(native_arity_bits, [0, PAIR_W + 3])
           native_front_result = pl.min(native_front_result, native_arity_second_position)
         else:
          native_front_tail = pl.tile.slice(native_pairs, [1, 1024], [0, PAIR_W])
          native_front_values = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
          native_front_positions = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          native_front_candidates = pl.tile.select(pl.tile.cmps(native_front_values, native_discarded, cmp_type=0), native_front_positions, 2147483647)
          native_front_wide = pl.col_min(pl.reshape(native_front_candidates, [8, 64]))
          native_front_eight = pl.col_min(pl.reshape(native_front_wide, [8, 8]))
          native_front_result = pl.tile.read(native_front_eight, [0, 0])
          native_front_lane_1 = pl.tile.read(native_front_eight, [0, 1])
          if native_front_lane_1 < native_front_result:
           native_front_result = native_front_lane_1
          native_front_lane_2 = pl.tile.read(native_front_eight, [0, 2])
          if native_front_lane_2 < native_front_result:
           native_front_result = native_front_lane_2
          native_front_lane_3 = pl.tile.read(native_front_eight, [0, 3])
          if native_front_lane_3 < native_front_result:
           native_front_result = native_front_lane_3
          native_front_lane_4 = pl.tile.read(native_front_eight, [0, 4])
          if native_front_lane_4 < native_front_result:
           native_front_result = native_front_lane_4
          native_front_lane_5 = pl.tile.read(native_front_eight, [0, 5])
          if native_front_lane_5 < native_front_result:
           native_front_result = native_front_lane_5
          native_front_lane_6 = pl.tile.read(native_front_eight, [0, 6])
          if native_front_lane_6 < native_front_result:
           native_front_result = native_front_lane_6
          native_front_lane_7 = pl.tile.read(native_front_eight, [0, 7])
          if native_front_lane_7 < native_front_result:
           native_front_result = native_front_lane_7
         if native_front_result < frontier_position:
          frontier_position = native_front_result
       pl.store(pl.tile.slice(native_pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
     leaf_score_line = pl.tile.full([1, 16], dtype=pl.FP32, value=SORT_FLOOR)
     leaf_position_line = pl.tile.full([1, 16], dtype=pl.INT32, value=2147483647)
     pl.tile.write(leaf_score_line, [0, 0], discarded)
     pl.tile.write(leaf_position_line, [0, 0], frontier_position)
     pl.store(leaf_score_line, [t, leaf * 16], leaf_discarded_bank)
     pl.store(leaf_position_line, [t, leaf * 16], leaf_front_bank)
   for cohort_base in pl.parallel(0, M, ROW_COHORT):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='decomposed_merge'):
     for t in pl.range(cohort_base, pl.min(cohort_base + ROW_COHORT, M), 1):
      visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
      wv = pl.max(visible, 1)
      lc = (wv + LEAF - 1) // LEAF
      discarded = pl.cast(SORT_FLOOR, pl.FP32)
      frontier_position = pl.cast(2147483647, pl.INT32)
      for proof_leaf in pl.range(NL):
       proof_score = pl.read(leaf_discarded_bank, [t, proof_leaf * 16])
       proof_position = pl.read(leaf_front_bank, [t, proof_leaf * 16])
       if proof_score > discarded:
        discarded = proof_score
        frontier_position = pl.cast(2147483647, pl.INT32)
       if proof_score == discarded:
        if proof_position < frontier_position:
         frontier_position = proof_position
      mtmp = pl.tile.create([1, 2 * PAIR_W], dtype=pl.FP32)
      for cb in pl.range(1, lc):
       mleft = pl.load(wpairs, [t, 0], [1, PAIR_W])
       mright = pl.load(wpairs, [t, cb * PAIR_W], [1, PAIR_W])
       mboth = pl.tile.mrgsort(mleft, mright, tmp=mtmp)
       merge_discarded = pl.tile.read(mboth, [0, PAIR_W])
       if merge_discarded > discarded:
        discarded = merge_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if merge_discarded == discarded:
        merge_front_boundary = pl.tile.read(mboth, [0, PAIR_W - 2])
        merge_front_bound = merge_front_boundary
        if merge_front_bound < FIN_FLOOR:
         merge_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if merge_discarded >= merge_front_bound:
         merge_front_tail = pl.tile.slice(mboth, [1, 1024], [0, PAIR_W])
         merge_front_values = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
         merge_front_positions = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         merge_front_candidates = pl.tile.select(pl.tile.cmps(merge_front_values, merge_discarded, cmp_type=0), merge_front_positions, 2147483647)
         merge_front_wide = pl.col_min(pl.reshape(merge_front_candidates, [8, 64]))
         merge_front_eight = pl.col_min(pl.reshape(merge_front_wide, [8, 8]))
         merge_front_result = pl.tile.read(merge_front_eight, [0, 0])
         merge_front_lane_1 = pl.tile.read(merge_front_eight, [0, 1])
         if merge_front_lane_1 < merge_front_result:
          merge_front_result = merge_front_lane_1
         merge_front_lane_2 = pl.tile.read(merge_front_eight, [0, 2])
         if merge_front_lane_2 < merge_front_result:
          merge_front_result = merge_front_lane_2
         merge_front_lane_3 = pl.tile.read(merge_front_eight, [0, 3])
         if merge_front_lane_3 < merge_front_result:
          merge_front_result = merge_front_lane_3
         merge_front_lane_4 = pl.tile.read(merge_front_eight, [0, 4])
         if merge_front_lane_4 < merge_front_result:
          merge_front_result = merge_front_lane_4
         merge_front_lane_5 = pl.tile.read(merge_front_eight, [0, 5])
         if merge_front_lane_5 < merge_front_result:
          merge_front_result = merge_front_lane_5
         merge_front_lane_6 = pl.tile.read(merge_front_eight, [0, 6])
         if merge_front_lane_6 < merge_front_result:
          merge_front_result = merge_front_lane_6
         merge_front_lane_7 = pl.tile.read(merge_front_eight, [0, 7])
         if merge_front_lane_7 < merge_front_result:
          merge_front_result = merge_front_lane_7
         if merge_front_result < frontier_position:
          frontier_position = merge_front_result
       pl.store(pl.tile.slice(mboth, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
      pl.write(aux_discarded, [t, 0], discarded)
      pl.write(aux_front, [t, 0], frontier_position)
   for cohort_base in pl.parallel(0, M, ROW_COHORT):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='decomposed_materialize'):
     for t in pl.range(cohort_base, pl.min(cohort_base + ROW_COHORT, M), 1):
      fin = pl.load(wpairs, [t, 0], [1, PAIR_W])
      values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
      finite = pl.maximum(values, FIN_FLOOR)
      cutoff = pl.tile.read(finite, [0, K - 1])
      ones_c = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
      flags = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=4), ones_c, 0.0)
      sums64 = pl.col_sum(pl.reshape(flags, [8, 64]))
      sums8 = pl.col_sum(pl.reshape(sums64, [8, 8]))
      above = pl.cast(0, pl.INDEX)
      for lane in pl.unroll(8):
       part = pl.cast(pl.tile.read(sums8, [0, lane]), pl.INT32)
       above = above + pl.cast(part, pl.INDEX)
      cutoff_stage_line = pl.tile.full([1, 16], dtype=pl.FP32, value=0)
      pl.tile.write(cutoff_stage_line, [0, 0], cutoff)
      pl.store(cutoff_stage_line, [t, 0], aux_cutoffs)
      quota_stage_line = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
      pl.tile.write(quota_stage_line, [0, 0], pl.cast(K - above, pl.INT32))
      pl.store(quota_stage_line, [t, 0], aux_quotas)
      discarded = pl.read(aux_discarded, [t, 0])
      frontier_position = pl.read(aux_front, [t, 0])
      replay_safe = discarded < cutoff
      if discarded == cutoff:
       certificate_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       certificate_candidates = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=0), certificate_positions, -1)
       certificate_wide = pl.col_max(pl.reshape(certificate_candidates, [8, 64]))
       certificate_eight = pl.col_max(pl.reshape(certificate_wide, [8, 8]))
       certificate_result = pl.tile.read(certificate_eight, [0, 0])
       certificate_lane_1 = pl.tile.read(certificate_eight, [0, 1])
       if certificate_lane_1 > certificate_result:
        certificate_result = certificate_lane_1
       certificate_lane_2 = pl.tile.read(certificate_eight, [0, 2])
       if certificate_lane_2 > certificate_result:
        certificate_result = certificate_lane_2
       certificate_lane_3 = pl.tile.read(certificate_eight, [0, 3])
       if certificate_lane_3 > certificate_result:
        certificate_result = certificate_lane_3
       certificate_lane_4 = pl.tile.read(certificate_eight, [0, 4])
       if certificate_lane_4 > certificate_result:
        certificate_result = certificate_lane_4
       certificate_lane_5 = pl.tile.read(certificate_eight, [0, 5])
       if certificate_lane_5 > certificate_result:
        certificate_result = certificate_lane_5
       certificate_lane_6 = pl.tile.read(certificate_eight, [0, 6])
       if certificate_lane_6 > certificate_result:
        certificate_result = certificate_lane_6
       certificate_lane_7 = pl.tile.read(certificate_eight, [0, 7])
       if certificate_lane_7 > certificate_result:
        certificate_result = certificate_lane_7
       if frontier_position > certificate_result:
        replay_safe = True
      if replay_safe:
       if cutoff > FIN_FLOOR:
        root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        selected = pl.cast(K, pl.INDEX)
        if selected > 0:
         root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(root_positions, pl.FP32))
         root_keys = root_priority
         root_order = pl.tile.sort32(root_keys, pl.reinterpret_view(root_positions, pl.UINT32))
         root_order = pl.tile.mrgsort(root_order, block_len=64)
         root_order = pl.tile.mrgsort(root_order, block_len=256)
         root_logical = pl.tile.gather_mask(root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         root_valid = pl.tile.set_validshape(root_logical, 1, selected)
         root_safe = pl.tile.fillpad(root_valid, pad_value=pl.PadValue.zero)
         if P == 1:
          single_page = pl.read(block_table, [0, 0])
          pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), single_page), pad_value=pl.PadValue.zero)
         else:
          pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
         root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for map_start in pl.range(0, selected, MAP):
          map_count = pl.min(MAP, selected - map_start)
          map_bits = pl.tile.slice(pl.reinterpret_view(root_safe, pl.FP32), [1, MAP], [0, map_start])
          map_logical = pl.reinterpret_view(map_bits, pl.INT32)
          root_page_idx = pl.tile.shrs(map_logical, 7)
          root_page_ids = pl.tile.gather(pages, root_page_idx, root_tmp)
          root_offsets = pl.sub(map_logical, pl.mul(root_page_idx, PAGE))
          root_physical = pl.add(pl.mul(root_page_ids, PAGE), root_offsets)
          pl.store(pl.tile.set_validshape(root_physical, 1, map_count), [t, map_start], indices)
       else:
        visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
        root = fin
        root_values = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P0101)
        root_positions = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
        root_strict = pl.tile.select(pl.tile.cmps(root_values, cutoff, cmp_type=4), ones_root, 0.0)
        root_equal = pl.tile.select(pl.tile.cmps(root_values, cutoff, cmp_type=0), ones_root, 0.0)
        root_eligible = pl.maximum(root_strict, root_equal)
        root_visible = pl.tile.cmps(pl.cast(root_positions, pl.FP32), pl.cast(pl.cast(visible, pl.INT32), pl.FP32), cmp_type=2)
        root_keep = pl.tile.select(root_visible, root_eligible, 0.0)
        root_sums64 = pl.col_sum(pl.reshape(root_keep, [8, 64]))
        root_sums8 = pl.col_sum(pl.reshape(root_sums64, [8, 8]))
        selected = pl.cast(0, pl.INDEX)
        for root_lane in pl.unroll(8):
         root_part = pl.cast(pl.tile.read(root_sums8, [0, root_lane]), pl.INT32)
         selected = selected + pl.cast(root_part, pl.INDEX)
        pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
        if selected > 0:
         root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(root_positions, pl.FP32))
         root_keys = pl.tile.select(pl.tile.cmps(root_keep, 0.0, cmp_type=4), root_priority, -1.0)
         root_order = pl.tile.sort32(root_keys, pl.reinterpret_view(root_positions, pl.UINT32))
         root_order = pl.tile.mrgsort(root_order, block_len=64)
         root_order = pl.tile.mrgsort(root_order, block_len=256)
         root_sorted_keys = pl.tile.gather_mask(root_order, mask_pattern=pl.tile.MaskPattern.P0101)
         root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), root_sorted_keys), pl.INT32)
         root_valid = pl.tile.set_validshape(root_logical, 1, selected)
         root_safe = pl.tile.fillpad(root_valid, pad_value=pl.PadValue.zero)
         if P == 1:
          single_page = pl.read(block_table, [0, 0])
          pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), single_page), pad_value=pl.PadValue.zero)
         else:
          pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
         root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for map_start in pl.range(0, selected, MAP):
          map_count = pl.min(MAP, selected - map_start)
          map_bits = pl.tile.slice(pl.reinterpret_view(root_safe, pl.FP32), [1, MAP], [0, map_start])
          map_logical = pl.reinterpret_view(map_bits, pl.INT32)
          root_page_idx = pl.tile.shrs(map_logical, 7)
          root_page_ids = pl.tile.gather(pages, root_page_idx, root_tmp)
          root_offsets = pl.sub(map_logical, pl.mul(root_page_idx, PAGE))
          root_physical = pl.add(pl.mul(root_page_ids, PAGE), root_offsets)
          pl.store(pl.tile.set_validshape(root_physical, 1, map_count), [t, map_start], indices)
      else:
       visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
       remaining = pl.cast(K - above, pl.INDEX)
       if P == 1:
        single_page = pl.read(block_table, [0, 0])
        pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), single_page), pad_value=pl.PadValue.zero)
       else:
        pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
       pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
       ones_m = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
       lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
       lanes_f = pl.cast(lanes, pl.FP32)
       lane_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
       priority = pl.cast(pl.sub(lane_origin, lanes), pl.FP32)
       gtmp = pl.tile.create([1, SCAN], dtype=pl.INT32)
       count = pl.cast(0, pl.INDEX)
       for start in pl.range(0, visible, SCAN):
        if count < K:
         valid = pl.min(SCAN, visible - start)
         scan_raw = pl.load(scores, [t, start], [1, SCAN], valid_shape=[1, valid])
         scan_vals = pl.tile.fillpad(scan_raw, pad_value=pl.PadValue.min)
         strict = pl.tile.select(pl.tile.cmps(scan_vals, cutoff, cmp_type=4), ones_m, 0.0)
         ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
         if remaining > 0:
          eq = pl.tile.select(pl.tile.cmps(scan_vals, cutoff, cmp_type=0), ones_m, 0.0)
          eq_valid = pl.tile.cmps(lanes_f, pl.cast(pl.cast(valid, pl.INT32), pl.FP32), cmp_type=2)
          eq = pl.tile.select(eq_valid, eq, 0.0)
          eq_sums64 = pl.col_sum(pl.reshape(eq, [SCAN // 64, 64]))
          eq_sums8 = pl.col_sum(pl.reshape(eq_sums64, [8, 8]))
          available = pl.cast(0, pl.INDEX)
          for eq_lane in pl.unroll(8):
           eq_part = pl.cast(pl.tile.read(eq_sums8, [0, eq_lane]), pl.INT32)
           available = available + pl.cast(eq_part, pl.INDEX)
          take = pl.min(available, remaining)
          if take > 0:
           if take == available:
            ties = eq
           else:
            tie_keys = pl.mul(eq, priority)
            tie_pairs = pl.tile.sort32(tie_keys, pl.reinterpret_view(lanes, pl.UINT32))
            tie_pairs = pl.tile.mrgsort(tie_pairs, block_len=64)
            tie_merged = tie_pairs
            equal_keys = pl.tile.gather_mask(tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
            last_key_f = pl.tile.read(equal_keys, [0, take - 1])
            boundary_f = 128.0 - last_key_f
            ties = pl.tile.select(pl.tile.cmps(lanes_f, boundary_f, cmp_type=3), eq, 0.0)
           remaining = remaining - take
         keep = pl.maximum(strict, ties)
         keep_valid = pl.tile.set_validshape(keep, 1, valid)
         keep_padded = pl.tile.fillpad(keep_valid, pad_value=pl.PadValue.zero)
         keep_sums64 = pl.col_sum(pl.reshape(keep_padded, [SCAN // 64, 64]))
         keep_sums8 = pl.col_sum(pl.reshape(keep_sums64, [8, 8]))
         keep_count = pl.cast(0, pl.INDEX)
         for keep_lane in pl.unroll(8):
          keep_part = pl.cast(pl.tile.read(keep_sums8, [0, keep_lane]), pl.INT32)
          keep_count = keep_count + pl.cast(keep_part, pl.INDEX)
         selected = pl.min(keep_count, K - count)
         if selected > 0:
          order_keys = pl.mul(keep_padded, priority)
          order_pairs = pl.tile.sort32(order_keys, pl.reinterpret_view(lanes, pl.UINT32))
          order_pairs = pl.tile.mrgsort(order_pairs, block_len=64)
          order_merged = order_pairs
          order_vals = pl.tile.gather_mask(order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
          offs = pl.sub(lane_origin, pl.cast(order_vals, pl.INT32))
          compact_valid = pl.tile.set_validshape(offs, 1, selected)
          compact_safe = pl.tile.fillpad(compact_valid, pad_value=pl.PadValue.zero)
          logical = pl.add(compact_safe, pl.cast(start, pl.INT32))
          page_idx = pl.tile.shrs(logical, 7)
          page_ids = pl.tile.gather(pages, page_idx, gtmp)
          offs_in_page = pl.sub(logical, pl.mul(page_idx, 128))
          physical = pl.add(pl.mul(page_ids, 128), offs_in_page)
          pl.store(pl.tile.set_validshape(physical, 1, selected), [t, count], indices)
          count = count + selected
   for pack_task in pl.spmd(1, name_hint='pack_auxiliary'):
    for output_row in pl.range(0, M, 16):
     cutoff_lines = pl.load(aux_cutoffs, [output_row, 0], [16, 16])
     quota_lines = pl.load(aux_quotas, [output_row, 0], [16, 16])
     cutoff_flat = pl.reshape(cutoff_lines, [1, 256])
     quota_flat = pl.reshape(quota_lines, [1, 256])
     pack_offsets = pl.mul(pl.tile.arange(0, [1, 16], dtype=pl.INT32), 16)
     cutoff_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     quota_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     cutoff_packed = pl.tile.gather(cutoff_flat, pack_offsets, cutoff_scratch)
     quota_packed = pl.tile.gather(quota_flat, pack_offsets, quota_scratch)
     pl.store(pl.reshape(cutoff_packed, [16, 1]), [output_row, 0], cutoffs)
     pl.store(pl.reshape(quota_packed, [16, 1]), [output_row, 0], quotas)
   return indices
 elif 8192 <= N <= 32768:

  @pl.jit
  def dsa41_topk_guarded_root(scores: pl.Tensor[[M, N], pl.FP32], lengths: pl.Tensor[[M], pl.INT32], block_table: pl.Tensor[[1, P], pl.INT32], wpairs: pl.InOut[pl.Tensor[[M, WORKSPACE_WIDTH], pl.FP32]], cutoffs: pl.InOut[pl.Tensor[[M, 1], pl.FP32]], quotas: pl.InOut[pl.Tensor[[M, 1], pl.INT32]], indices: pl.Out[pl.Tensor[[M, K], pl.INT32]]):
   aux_discarded = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_cutoffs = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_quotas = pl.create_tensor([M, 16], dtype=pl.INT32)
   aux_front = pl.create_tensor([M, 16], dtype=pl.INT32)
   leaf_discarded_bank = pl.create_tensor([M, NL * 16], dtype=pl.FP32)
   leaf_front_bank = pl.create_tensor([M, NL * 16], dtype=pl.INT32)
   for leaf_job in pl.spmd(LEAF_COHORTS * NL, name_hint='canonical_leaf_plane'):
    leaf = leaf_job // LEAF_COHORTS
    leaf_cohort_base = leaf_job % LEAF_COHORTS * LEAF_ROWS
    for t in pl.range(leaf_cohort_base, pl.min(leaf_cohort_base + LEAF_ROWS, M), 1):
     zero_pairs = pl.tile.full([1, PAIR_W], dtype=pl.FP32, value=0.0)
     if leaf == 0:
      for clear_offset in pl.range(NL * PAIR_W, WORKSPACE_WIDTH, PAIR_W):
       pl.store(zero_pairs, [t, clear_offset], wpairs)
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     begin = leaf * LEAF
     if begin < wv:
      valid = pl.min(LEAF, wv - begin)
      if valid <= 512:
       narrow_raw = pl.load(scores, [t, begin], [1, 512], valid_shape=[1, valid])
       narrow_padded = pl.tile.fillpad(narrow_raw, pad_value=pl.PadValue.min)
       narrow_values = pl.maximum(narrow_padded, SORT_FLOOR)
       narrow_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       narrow_pairs = pl.tile.sort32(narrow_values, pl.reinterpret_view(narrow_positions, pl.UINT32))
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=64)
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=256)
       pl.store(narrow_pairs, [t, leaf * PAIR_W], wpairs)
      elif valid <= 1024:
       left_valid = pl.min(valid, 512)
       right_valid = pl.max(valid - 512, 0)
       left_raw = pl.load(scores, [t, begin], [1, 512], valid_shape=[1, left_valid])
       right_raw = pl.load(scores, [t, begin + 512], [1, 512], valid_shape=[1, right_valid])
       left_values = pl.maximum(pl.tile.fillpad(left_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       right_values = pl.maximum(pl.tile.fillpad(right_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       left_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       right_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin + 512, pl.INT32))
       left_pairs = pl.tile.sort32(left_values, pl.reinterpret_view(left_positions, pl.UINT32))
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=64)
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=256)
       right_pairs = pl.tile.sort32(right_values, pl.reinterpret_view(right_positions, pl.UINT32))
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=64)
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=256)
       native_scratch = pl.tile.create([1, 2048], dtype=pl.FP32)
       native_pairs = pl.tile.mrgsort(left_pairs, right_pairs, tmp=native_scratch)
       native_discarded = pl.tile.read(native_pairs, [0, PAIR_W])
       if native_discarded > discarded:
        discarded = native_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if native_discarded == discarded:
        native_front_boundary = pl.tile.read(native_pairs, [0, PAIR_W - 2])
        native_front_bound = native_front_boundary
        if native_front_bound < FIN_FLOOR:
         native_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if native_discarded >= native_front_bound:
         native_arity_third = pl.tile.read(native_pairs, [0, PAIR_W + 4])
         if native_arity_third < native_discarded:
          native_arity_second = pl.tile.read(native_pairs, [0, PAIR_W + 2])
          native_arity_bits = pl.reinterpret_view(native_pairs, pl.INT32)
          native_front_result = pl.tile.read(native_arity_bits, [0, PAIR_W + 1])
          if native_arity_second == native_discarded:
           native_arity_second_position = pl.tile.read(native_arity_bits, [0, PAIR_W + 3])
           native_front_result = pl.min(native_front_result, native_arity_second_position)
         else:
          native_front_tail = pl.tile.slice(native_pairs, [1, 1024], [0, PAIR_W])
          native_front_values = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
          native_front_positions = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          native_front_candidates = pl.tile.select(pl.tile.cmps(native_front_values, native_discarded, cmp_type=0), native_front_positions, 2147483647)
          native_front_wide = pl.col_min(pl.reshape(native_front_candidates, [8, 64]))
          native_front_eight = pl.col_min(pl.reshape(native_front_wide, [8, 8]))
          native_front_result = pl.tile.read(native_front_eight, [0, 0])
          native_front_lane_1 = pl.tile.read(native_front_eight, [0, 1])
          if native_front_lane_1 < native_front_result:
           native_front_result = native_front_lane_1
          native_front_lane_2 = pl.tile.read(native_front_eight, [0, 2])
          if native_front_lane_2 < native_front_result:
           native_front_result = native_front_lane_2
          native_front_lane_3 = pl.tile.read(native_front_eight, [0, 3])
          if native_front_lane_3 < native_front_result:
           native_front_result = native_front_lane_3
          native_front_lane_4 = pl.tile.read(native_front_eight, [0, 4])
          if native_front_lane_4 < native_front_result:
           native_front_result = native_front_lane_4
          native_front_lane_5 = pl.tile.read(native_front_eight, [0, 5])
          if native_front_lane_5 < native_front_result:
           native_front_result = native_front_lane_5
          native_front_lane_6 = pl.tile.read(native_front_eight, [0, 6])
          if native_front_lane_6 < native_front_result:
           native_front_result = native_front_lane_6
          native_front_lane_7 = pl.tile.read(native_front_eight, [0, 7])
          if native_front_lane_7 < native_front_result:
           native_front_result = native_front_lane_7
         if native_front_result < frontier_position:
          frontier_position = native_front_result
       pl.store(pl.tile.slice(native_pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
      else:
       raw = pl.load(scores, [t, begin], [1, LEAF], valid_shape=[1, valid])
       padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
       flr = pl.tile.full([1, LEAF], dtype=pl.FP32, value=SORT_FLOOR)
       vals = pl.maximum(padded, flr)
       idx_i32 = pl.add(pl.tile.arange(0, [1, LEAF], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       pairs = pl.tile.sort32(vals, pl.reinterpret_view(idx_i32, pl.UINT32))
       pairs = pl.tile.mrgsort(pairs, block_len=64)
       pairs = pl.tile.mrgsort(pairs, block_len=256)
       pairs = pl.tile.mrgsort(pairs, block_len=1024)
       leaf_discarded = pl.tile.read(pairs, [0, PAIR_W])
       if leaf_discarded > discarded:
        discarded = leaf_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if leaf_discarded == discarded:
        leaf_front_boundary = pl.tile.read(pairs, [0, PAIR_W - 2])
        leaf_front_bound = leaf_front_boundary
        if leaf_front_bound < FIN_FLOOR:
         leaf_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if leaf_discarded >= leaf_front_bound:
         leaf_arity_third = pl.tile.read(pairs, [0, PAIR_W + 4])
         if leaf_arity_third < leaf_discarded:
          leaf_arity_second = pl.tile.read(pairs, [0, PAIR_W + 2])
          leaf_arity_bits = pl.reinterpret_view(pairs, pl.INT32)
          leaf_front_result = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 1])
          if leaf_arity_second == leaf_discarded:
           leaf_arity_second_position = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 3])
           leaf_front_result = pl.min(leaf_front_result, leaf_arity_second_position)
         else:
          leaf_front_tail = pl.tile.slice(pairs, [1, 3072], [0, PAIR_W])
          leaf_front_values = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
          leaf_front_positions = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          leaf_front_candidates = pl.tile.select(pl.tile.cmps(leaf_front_values, leaf_discarded, cmp_type=0), leaf_front_positions, 2147483647)
          leaf_front_wide = pl.col_min(pl.reshape(leaf_front_candidates, [24, 64]))
          leaf_front_eight = pl.col_min(pl.reshape(leaf_front_wide, [8, 8]))
          leaf_front_result = pl.tile.read(leaf_front_eight, [0, 0])
          leaf_front_lane_1 = pl.tile.read(leaf_front_eight, [0, 1])
          if leaf_front_lane_1 < leaf_front_result:
           leaf_front_result = leaf_front_lane_1
          leaf_front_lane_2 = pl.tile.read(leaf_front_eight, [0, 2])
          if leaf_front_lane_2 < leaf_front_result:
           leaf_front_result = leaf_front_lane_2
          leaf_front_lane_3 = pl.tile.read(leaf_front_eight, [0, 3])
          if leaf_front_lane_3 < leaf_front_result:
           leaf_front_result = leaf_front_lane_3
          leaf_front_lane_4 = pl.tile.read(leaf_front_eight, [0, 4])
          if leaf_front_lane_4 < leaf_front_result:
           leaf_front_result = leaf_front_lane_4
          leaf_front_lane_5 = pl.tile.read(leaf_front_eight, [0, 5])
          if leaf_front_lane_5 < leaf_front_result:
           leaf_front_result = leaf_front_lane_5
          leaf_front_lane_6 = pl.tile.read(leaf_front_eight, [0, 6])
          if leaf_front_lane_6 < leaf_front_result:
           leaf_front_result = leaf_front_lane_6
          leaf_front_lane_7 = pl.tile.read(leaf_front_eight, [0, 7])
          if leaf_front_lane_7 < leaf_front_result:
           leaf_front_result = leaf_front_lane_7
         if leaf_front_result < frontier_position:
          frontier_position = leaf_front_result
       pl.store(pl.tile.slice(pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
     else:
      pl.store(zero_pairs, [t, leaf * PAIR_W], wpairs)
     leaf_score_line = pl.tile.full([1, 16], dtype=pl.FP32, value=SORT_FLOOR)
     leaf_position_line = pl.tile.full([1, 16], dtype=pl.INT32, value=2147483647)
     pl.tile.write(leaf_score_line, [0, 0], discarded)
     pl.tile.write(leaf_position_line, [0, 0], frontier_position)
     pl.store(leaf_score_line, [t, leaf * 16], leaf_discarded_bank)
     pl.store(leaf_position_line, [t, leaf * 16], leaf_front_bank)
   for merge_cohort_job in pl.spmd((M + ROW_COHORT - 1) // ROW_COHORT, name_hint='decomposed_merge'):
    cohort_base = merge_cohort_job * ROW_COHORT
    for t in pl.range(cohort_base, pl.min(cohort_base + ROW_COHORT, M), 1):
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     lc = (wv + LEAF - 1) // LEAF
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     for proof_leaf in pl.range(NL):
      proof_score = pl.read(leaf_discarded_bank, [t, proof_leaf * 16])
      proof_position = pl.read(leaf_front_bank, [t, proof_leaf * 16])
      if proof_score > discarded:
       discarded = proof_score
       frontier_position = pl.cast(2147483647, pl.INT32)
      if proof_score == discarded:
       if proof_position < frontier_position:
        frontier_position = proof_position
     pool_accepted = False
     if lc >= 14:
      pool64_outside = discarded
      pool64_pairs = pl.tile.full([1, 2048], dtype=pl.FP32, value=SORT_FLOOR)
      for pool64_leaf in pl.unroll(16):
       pool64_piece = pl.tile.full([1, 128], dtype=pl.FP32, value=SORT_FLOOR)
       if pool64_leaf < lc:
        pool64_safe_leaf = pl.min(pool64_leaf, NL - 1)
        pool64_piece = pl.load(wpairs, [t, pool64_safe_leaf * PAIR_W], [1, 128])
        pool64_excluded = pl.read(wpairs, [t, pool64_safe_leaf * PAIR_W + 128])
        if pool64_excluded > pool64_outside:
         pool64_outside = pool64_excluded
       pool64_pairs = pl.tile.assemble(pool64_pairs, pool64_piece, [0, pool64_leaf * 128])
      pool64_pairs = pl.tile.mrgsort(pool64_pairs, block_len=128)
      pool64_pairs = pl.tile.mrgsort(pool64_pairs, block_len=512)
      pool64_raw_cutoff = pl.tile.read(pool64_pairs, [0, PAIR_W - 2])
      pool64_rank513 = pl.tile.read(pool64_pairs, [0, PAIR_W])
      if pool64_outside < pool64_raw_cutoff and pool64_raw_cutoff > FIN_FLOOR:
       pool_accepted = True
       discarded = pool64_outside
       frontier_position = pl.cast(2147483647, pl.INT32)
       if pool64_rank513 > discarded:
        discarded = pool64_rank513
       if pool64_rank513 == pool64_raw_cutoff:
        for pool64_tail_chunk in pl.unroll(1):
         pool64_front_tail = pl.tile.slice(pool64_pairs, [1, 1024], [0, (pool64_tail_chunk + 1) * PAIR_W])
         pool64_front_values = pl.tile.gather_mask(pool64_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
         pool64_front_positions = pl.tile.gather_mask(pool64_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         pool64_front_candidates = pl.tile.select(pl.tile.cmps(pool64_front_values, pool64_rank513, cmp_type=0), pool64_front_positions, 2147483647)
         pool64_front_wide = pl.col_min(pl.reshape(pool64_front_candidates, [8, 64]))
         pool64_front_eight = pl.col_min(pl.reshape(pool64_front_wide, [8, 8]))
         pool64_front_result = pl.tile.read(pool64_front_eight, [0, 0])
         pool64_front_lane_1 = pl.tile.read(pool64_front_eight, [0, 1])
         if pool64_front_lane_1 < pool64_front_result:
          pool64_front_result = pool64_front_lane_1
         pool64_front_lane_2 = pl.tile.read(pool64_front_eight, [0, 2])
         if pool64_front_lane_2 < pool64_front_result:
          pool64_front_result = pool64_front_lane_2
         pool64_front_lane_3 = pl.tile.read(pool64_front_eight, [0, 3])
         if pool64_front_lane_3 < pool64_front_result:
          pool64_front_result = pool64_front_lane_3
         pool64_front_lane_4 = pl.tile.read(pool64_front_eight, [0, 4])
         if pool64_front_lane_4 < pool64_front_result:
          pool64_front_result = pool64_front_lane_4
         pool64_front_lane_5 = pl.tile.read(pool64_front_eight, [0, 5])
         if pool64_front_lane_5 < pool64_front_result:
          pool64_front_result = pool64_front_lane_5
         pool64_front_lane_6 = pl.tile.read(pool64_front_eight, [0, 6])
         if pool64_front_lane_6 < pool64_front_result:
          pool64_front_result = pool64_front_lane_6
         pool64_front_lane_7 = pl.tile.read(pool64_front_eight, [0, 7])
         if pool64_front_lane_7 < pool64_front_result:
          pool64_front_result = pool64_front_lane_7
         if pool64_front_result < frontier_position:
          frontier_position = pool64_front_result
       pl.store(pl.tile.slice(pool64_pairs, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
     elif lc >= 8:
      pool128_outside = discarded
      pool128_pairs = pl.tile.full([1, 4096], dtype=pl.FP32, value=SORT_FLOOR)
      for pool128_leaf in pl.unroll(16):
       pool128_piece = pl.tile.full([1, 256], dtype=pl.FP32, value=SORT_FLOOR)
       if pool128_leaf < lc:
        pool128_safe_leaf = pl.min(pool128_leaf, NL - 1)
        pool128_piece = pl.load(wpairs, [t, pool128_safe_leaf * PAIR_W], [1, 256])
        pool128_excluded = pl.read(wpairs, [t, pool128_safe_leaf * PAIR_W + 256])
        if pool128_excluded > pool128_outside:
         pool128_outside = pool128_excluded
       pool128_pairs = pl.tile.assemble(pool128_pairs, pool128_piece, [0, pool128_leaf * 256])
      pool128_pairs = pl.tile.mrgsort(pool128_pairs, block_len=256)
      pool128_pairs = pl.tile.mrgsort(pool128_pairs, block_len=1024)
      pool128_raw_cutoff = pl.tile.read(pool128_pairs, [0, PAIR_W - 2])
      pool128_rank513 = pl.tile.read(pool128_pairs, [0, PAIR_W])
      if pool128_outside < pool128_raw_cutoff and pool128_raw_cutoff > FIN_FLOOR:
       pool_accepted = True
       discarded = pool128_outside
       frontier_position = pl.cast(2147483647, pl.INT32)
       if pool128_rank513 > discarded:
        discarded = pool128_rank513
       if pool128_rank513 == pool128_raw_cutoff:
        for pool128_tail_chunk in pl.unroll(3):
         pool128_front_tail = pl.tile.slice(pool128_pairs, [1, 1024], [0, (pool128_tail_chunk + 1) * PAIR_W])
         pool128_front_values = pl.tile.gather_mask(pool128_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
         pool128_front_positions = pl.tile.gather_mask(pool128_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         pool128_front_candidates = pl.tile.select(pl.tile.cmps(pool128_front_values, pool128_rank513, cmp_type=0), pool128_front_positions, 2147483647)
         pool128_front_wide = pl.col_min(pl.reshape(pool128_front_candidates, [8, 64]))
         pool128_front_eight = pl.col_min(pl.reshape(pool128_front_wide, [8, 8]))
         pool128_front_result = pl.tile.read(pool128_front_eight, [0, 0])
         pool128_front_lane_1 = pl.tile.read(pool128_front_eight, [0, 1])
         if pool128_front_lane_1 < pool128_front_result:
          pool128_front_result = pool128_front_lane_1
         pool128_front_lane_2 = pl.tile.read(pool128_front_eight, [0, 2])
         if pool128_front_lane_2 < pool128_front_result:
          pool128_front_result = pool128_front_lane_2
         pool128_front_lane_3 = pl.tile.read(pool128_front_eight, [0, 3])
         if pool128_front_lane_3 < pool128_front_result:
          pool128_front_result = pool128_front_lane_3
         pool128_front_lane_4 = pl.tile.read(pool128_front_eight, [0, 4])
         if pool128_front_lane_4 < pool128_front_result:
          pool128_front_result = pool128_front_lane_4
         pool128_front_lane_5 = pl.tile.read(pool128_front_eight, [0, 5])
         if pool128_front_lane_5 < pool128_front_result:
          pool128_front_result = pool128_front_lane_5
         pool128_front_lane_6 = pl.tile.read(pool128_front_eight, [0, 6])
         if pool128_front_lane_6 < pool128_front_result:
          pool128_front_result = pool128_front_lane_6
         pool128_front_lane_7 = pl.tile.read(pool128_front_eight, [0, 7])
         if pool128_front_lane_7 < pool128_front_result:
          pool128_front_result = pool128_front_lane_7
         if pool128_front_result < frontier_position:
          frontier_position = pool128_front_result
       pl.store(pl.tile.slice(pool128_pairs, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
     if not pool_accepted:
      mtmp = pl.tile.create([1, 2 * PAIR_W], dtype=pl.FP32)
      for cb in pl.range(1, lc):
       mleft = pl.load(wpairs, [t, 0], [1, PAIR_W])
       mright = pl.load(wpairs, [t, cb * PAIR_W], [1, PAIR_W])
       mboth = pl.tile.mrgsort(mleft, mright, tmp=mtmp)
       merge_discarded = pl.tile.read(mboth, [0, PAIR_W])
       if merge_discarded > discarded:
        discarded = merge_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if merge_discarded == discarded:
        merge_front_boundary = pl.tile.read(mboth, [0, PAIR_W - 2])
        merge_front_bound = merge_front_boundary
        if merge_front_bound < FIN_FLOOR:
         merge_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if merge_discarded >= merge_front_bound:
         merge_front_tail = pl.tile.slice(mboth, [1, 1024], [0, PAIR_W])
         merge_front_values = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
         merge_front_positions = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         merge_front_candidates = pl.tile.select(pl.tile.cmps(merge_front_values, merge_discarded, cmp_type=0), merge_front_positions, 2147483647)
         merge_front_wide = pl.col_min(pl.reshape(merge_front_candidates, [8, 64]))
         merge_front_eight = pl.col_min(pl.reshape(merge_front_wide, [8, 8]))
         merge_front_result = pl.tile.read(merge_front_eight, [0, 0])
         merge_front_lane_1 = pl.tile.read(merge_front_eight, [0, 1])
         if merge_front_lane_1 < merge_front_result:
          merge_front_result = merge_front_lane_1
         merge_front_lane_2 = pl.tile.read(merge_front_eight, [0, 2])
         if merge_front_lane_2 < merge_front_result:
          merge_front_result = merge_front_lane_2
         merge_front_lane_3 = pl.tile.read(merge_front_eight, [0, 3])
         if merge_front_lane_3 < merge_front_result:
          merge_front_result = merge_front_lane_3
         merge_front_lane_4 = pl.tile.read(merge_front_eight, [0, 4])
         if merge_front_lane_4 < merge_front_result:
          merge_front_result = merge_front_lane_4
         merge_front_lane_5 = pl.tile.read(merge_front_eight, [0, 5])
         if merge_front_lane_5 < merge_front_result:
          merge_front_result = merge_front_lane_5
         merge_front_lane_6 = pl.tile.read(merge_front_eight, [0, 6])
         if merge_front_lane_6 < merge_front_result:
          merge_front_result = merge_front_lane_6
         merge_front_lane_7 = pl.tile.read(merge_front_eight, [0, 7])
         if merge_front_lane_7 < merge_front_result:
          merge_front_result = merge_front_lane_7
         if merge_front_result < frontier_position:
          frontier_position = merge_front_result
       pl.store(pl.tile.slice(mboth, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
     pl.write(aux_discarded, [t, 0], discarded)
     pl.write(aux_front, [t, 0], frontier_position)
   for materialize_row_job in pl.spmd((M + 4 - 1) // 4, name_hint='decomposed_materialize'):
    root_row_base = materialize_row_job * 4
    if P == 1:
     materialize_single_page = pl.read(block_table, [0, 0])
     materialize_pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), materialize_single_page), pad_value=pl.PadValue.zero)
    else:
     materialize_pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
    materialize_page_lanes = pl.tile.arange(0, [1, P8], dtype=pl.INT32)
    materialize_displacements = pl.mul(pl.sub(materialize_pages, materialize_page_lanes), PAGE)
    for t in pl.range(root_row_base, pl.min(root_row_base + 4, M), 1):
     fin = pl.load(wpairs, [t, 0], [1, PAIR_W])
     values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
     finite = pl.maximum(values, FIN_FLOOR)
     cutoff = pl.tile.read(finite, [0, K - 1])
     ones_c = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
     flags = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=4), ones_c, 0.0)
     sums64 = pl.col_sum(pl.reshape(flags, [8, 64]))
     sums8 = pl.col_sum(pl.reshape(sums64, [8, 8]))
     above = pl.cast(0, pl.INDEX)
     for lane in pl.unroll(8):
      part = pl.cast(pl.tile.read(sums8, [0, lane]), pl.INT32)
      above = above + pl.cast(part, pl.INDEX)
     cutoff_stage_line = pl.tile.full([1, 16], dtype=pl.FP32, value=0)
     pl.tile.write(cutoff_stage_line, [0, 0], cutoff)
     pl.store(cutoff_stage_line, [t, 0], aux_cutoffs)
     quota_stage_line = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
     pl.tile.write(quota_stage_line, [0, 0], pl.cast(K - above, pl.INT32))
     pl.store(quota_stage_line, [t, 0], aux_quotas)
     discarded = pl.read(aux_discarded, [t, 0])
     frontier_position = pl.read(aux_front, [t, 0])
     replay_safe = discarded < cutoff
     if discarded == cutoff:
      certificate_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
      certificate_candidates = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=0), certificate_positions, -1)
      certificate_wide = pl.col_max(pl.reshape(certificate_candidates, [8, 64]))
      certificate_eight = pl.col_max(pl.reshape(certificate_wide, [8, 8]))
      certificate_result = pl.tile.read(certificate_eight, [0, 0])
      certificate_lane_1 = pl.tile.read(certificate_eight, [0, 1])
      if certificate_lane_1 > certificate_result:
       certificate_result = certificate_lane_1
      certificate_lane_2 = pl.tile.read(certificate_eight, [0, 2])
      if certificate_lane_2 > certificate_result:
       certificate_result = certificate_lane_2
      certificate_lane_3 = pl.tile.read(certificate_eight, [0, 3])
      if certificate_lane_3 > certificate_result:
       certificate_result = certificate_lane_3
      certificate_lane_4 = pl.tile.read(certificate_eight, [0, 4])
      if certificate_lane_4 > certificate_result:
       certificate_result = certificate_lane_4
      certificate_lane_5 = pl.tile.read(certificate_eight, [0, 5])
      if certificate_lane_5 > certificate_result:
       certificate_result = certificate_lane_5
      certificate_lane_6 = pl.tile.read(certificate_eight, [0, 6])
      if certificate_lane_6 > certificate_result:
       certificate_result = certificate_lane_6
      certificate_lane_7 = pl.tile.read(certificate_eight, [0, 7])
      if certificate_lane_7 > certificate_result:
       certificate_result = certificate_lane_7
      if frontier_position > certificate_result:
       replay_safe = True
     if replay_safe:
      if cutoff > FIN_FLOOR:
       root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       selected = pl.cast(K, pl.INDEX)
       if selected > 0:
        root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(root_positions, pl.FP32))
        root_keys = root_priority
        root_order = pl.tile.sort32(root_keys, pl.reinterpret_view(root_positions, pl.UINT32))
        root_order = pl.tile.mrgsort(root_order, block_len=64)
        root_order = pl.tile.mrgsort(root_order, block_len=256)
        root_logical = pl.tile.gather_mask(root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        root_valid = pl.tile.set_validshape(root_logical, 1, selected)
        root_safe = pl.tile.fillpad(root_valid, pad_value=pl.PadValue.zero)
        root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
        for map_start in pl.range(0, selected, MAP):
         map_count = pl.min(MAP, selected - map_start)
         map_bits = pl.tile.slice(pl.reinterpret_view(root_safe, pl.FP32), [1, MAP], [0, map_start])
         map_logical = pl.reinterpret_view(map_bits, pl.INT32)
         root_page_idx = pl.tile.shrs(map_logical, 7)
         root_page_ids = pl.tile.gather(materialize_displacements, root_page_idx, root_tmp)
         root_physical = pl.add(map_logical, root_page_ids)
         pl.store(pl.tile.set_validshape(root_physical, 1, map_count), [t, map_start], indices)
      else:
       visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
       root = fin
       root_values = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P0101)
       root_positions = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
       root_strict = pl.tile.select(pl.tile.cmps(root_values, cutoff, cmp_type=4), ones_root, 0.0)
       root_equal = pl.tile.select(pl.tile.cmps(root_values, cutoff, cmp_type=0), ones_root, 0.0)
       root_eligible = pl.maximum(root_strict, root_equal)
       root_visible = pl.tile.cmps(pl.cast(root_positions, pl.FP32), pl.cast(pl.cast(visible, pl.INT32), pl.FP32), cmp_type=2)
       root_keep = pl.tile.select(root_visible, root_eligible, 0.0)
       root_sums64 = pl.col_sum(pl.reshape(root_keep, [8, 64]))
       root_sums8 = pl.col_sum(pl.reshape(root_sums64, [8, 8]))
       selected = pl.cast(0, pl.INDEX)
       for root_lane in pl.unroll(8):
        root_part = pl.cast(pl.tile.read(root_sums8, [0, root_lane]), pl.INT32)
        selected = selected + pl.cast(root_part, pl.INDEX)
       pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
       if selected > 0:
        root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(root_positions, pl.FP32))
        root_keys = pl.tile.select(pl.tile.cmps(root_keep, 0.0, cmp_type=4), root_priority, -1.0)
        root_order = pl.tile.sort32(root_keys, pl.reinterpret_view(root_positions, pl.UINT32))
        root_order = pl.tile.mrgsort(root_order, block_len=64)
        root_order = pl.tile.mrgsort(root_order, block_len=256)
        root_sorted_keys = pl.tile.gather_mask(root_order, mask_pattern=pl.tile.MaskPattern.P0101)
        root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), root_sorted_keys), pl.INT32)
        root_valid = pl.tile.set_validshape(root_logical, 1, selected)
        root_safe = pl.tile.fillpad(root_valid, pad_value=pl.PadValue.zero)
        root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
        for map_start in pl.range(0, selected, MAP):
         map_count = pl.min(MAP, selected - map_start)
         map_bits = pl.tile.slice(pl.reinterpret_view(root_safe, pl.FP32), [1, MAP], [0, map_start])
         map_logical = pl.reinterpret_view(map_bits, pl.INT32)
         root_page_idx = pl.tile.shrs(map_logical, 7)
         root_page_ids = pl.tile.gather(materialize_displacements, root_page_idx, root_tmp)
         root_physical = pl.add(map_logical, root_page_ids)
         pl.store(pl.tile.set_validshape(root_physical, 1, map_count), [t, map_start], indices)
     else:
      visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
      remaining = pl.cast(K - above, pl.INDEX)
      pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
      ones_m = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
      lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
      lanes_f = pl.cast(lanes, pl.FP32)
      lane_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
      priority = pl.cast(pl.sub(lane_origin, lanes), pl.FP32)
      gtmp = pl.tile.create([1, SCAN], dtype=pl.INT32)
      count = pl.cast(0, pl.INDEX)
      for start in pl.range(0, visible, SCAN):
       if count < K:
        valid = pl.min(SCAN, visible - start)
        scan_raw = pl.load(scores, [t, start], [1, SCAN], valid_shape=[1, valid])
        scan_vals = pl.tile.fillpad(scan_raw, pad_value=pl.PadValue.min)
        strict = pl.tile.select(pl.tile.cmps(scan_vals, cutoff, cmp_type=4), ones_m, 0.0)
        ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
        if remaining > 0:
         eq = pl.tile.select(pl.tile.cmps(scan_vals, cutoff, cmp_type=0), ones_m, 0.0)
         eq_valid = pl.tile.cmps(lanes_f, pl.cast(pl.cast(valid, pl.INT32), pl.FP32), cmp_type=2)
         eq = pl.tile.select(eq_valid, eq, 0.0)
         eq_sums64 = pl.col_sum(pl.reshape(eq, [SCAN // 64, 64]))
         eq_sums8 = pl.col_sum(pl.reshape(eq_sums64, [8, 8]))
         available = pl.cast(0, pl.INDEX)
         for eq_lane in pl.unroll(8):
          eq_part = pl.cast(pl.tile.read(eq_sums8, [0, eq_lane]), pl.INT32)
          available = available + pl.cast(eq_part, pl.INDEX)
         take = pl.min(available, remaining)
         if take > 0:
          if take == available:
           ties = eq
          else:
           tie_keys = pl.mul(eq, priority)
           tie_pairs = pl.tile.sort32(tie_keys, pl.reinterpret_view(lanes, pl.UINT32))
           tie_pairs = pl.tile.mrgsort(tie_pairs, block_len=64)
           tie_merged = tie_pairs
           equal_keys = pl.tile.gather_mask(tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
           last_key_f = pl.tile.read(equal_keys, [0, take - 1])
           boundary_f = 128.0 - last_key_f
           ties = pl.tile.select(pl.tile.cmps(lanes_f, boundary_f, cmp_type=3), eq, 0.0)
          remaining = remaining - take
        keep = pl.maximum(strict, ties)
        keep_valid = pl.tile.set_validshape(keep, 1, valid)
        keep_padded = pl.tile.fillpad(keep_valid, pad_value=pl.PadValue.zero)
        keep_sums64 = pl.col_sum(pl.reshape(keep_padded, [SCAN // 64, 64]))
        keep_sums8 = pl.col_sum(pl.reshape(keep_sums64, [8, 8]))
        keep_count = pl.cast(0, pl.INDEX)
        for keep_lane in pl.unroll(8):
         keep_part = pl.cast(pl.tile.read(keep_sums8, [0, keep_lane]), pl.INT32)
         keep_count = keep_count + pl.cast(keep_part, pl.INDEX)
        selected = pl.min(keep_count, K - count)
        if selected > 0:
         order_keys = pl.mul(keep_padded, priority)
         order_pairs = pl.tile.sort32(order_keys, pl.reinterpret_view(lanes, pl.UINT32))
         order_pairs = pl.tile.mrgsort(order_pairs, block_len=64)
         order_merged = order_pairs
         order_vals = pl.tile.gather_mask(order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
         offs = pl.sub(lane_origin, pl.cast(order_vals, pl.INT32))
         compact_valid = pl.tile.set_validshape(offs, 1, selected)
         compact_safe = pl.tile.fillpad(compact_valid, pad_value=pl.PadValue.zero)
         logical = pl.add(compact_safe, pl.cast(start, pl.INT32))
         page_idx = pl.tile.shrs(logical, 7)
         page_ids = pl.tile.gather(materialize_displacements, page_idx, gtmp)
         physical = pl.add(logical, page_ids)
         pl.store(pl.tile.set_validshape(physical, 1, selected), [t, count], indices)
         count = count + selected
   for pack_task in pl.spmd(1, name_hint='pack_auxiliary'):
    for output_row in pl.range(0, M, 16):
     cutoff_lines = pl.load(aux_cutoffs, [output_row, 0], [16, 16])
     quota_lines = pl.load(aux_quotas, [output_row, 0], [16, 16])
     cutoff_flat = pl.reshape(cutoff_lines, [1, 256])
     quota_flat = pl.reshape(quota_lines, [1, 256])
     pack_offsets = pl.mul(pl.tile.arange(0, [1, 16], dtype=pl.INT32), 16)
     cutoff_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     quota_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     cutoff_packed = pl.tile.gather(cutoff_flat, pack_offsets, cutoff_scratch)
     quota_packed = pl.tile.gather(quota_flat, pack_offsets, quota_scratch)
     pl.store(pl.reshape(cutoff_packed, [16, 1]), [output_row, 0], cutoffs)
     pl.store(pl.reshape(quota_packed, [16, 1]), [output_row, 0], quotas)
   return indices
 elif NL == 1:

  @pl.jit
  def dsa41_topk_guarded_root(scores: pl.Tensor[[M, N], pl.FP32], lengths: pl.Tensor[[M], pl.INT32], block_table: pl.Tensor[[1, P], pl.INT32], wpairs: pl.InOut[pl.Tensor[[M, WORKSPACE_WIDTH], pl.FP32]], cutoffs: pl.InOut[pl.Tensor[[M, 1], pl.FP32]], quotas: pl.InOut[pl.Tensor[[M, 1], pl.INT32]], indices: pl.Out[pl.Tensor[[M, K], pl.INT32]]):
   aux_discarded = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_cutoffs = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_quotas = pl.create_tensor([M, 16], dtype=pl.INT32)
   aux_front = pl.create_tensor([M, 16], dtype=pl.INT32)
   leaf_discarded_bank = pl.create_tensor([M, NL * 16], dtype=pl.FP32)
   leaf_front_bank = pl.create_tensor([M, NL * 16], dtype=pl.INT32)
   for leaf_job in pl.spmd(LEAF_COHORTS * NL, name_hint='canonical_leaf_plane'):
    leaf = leaf_job // LEAF_COHORTS
    leaf_cohort_base = leaf_job % LEAF_COHORTS * LEAF_ROWS
    for t in pl.range(leaf_cohort_base, pl.min(leaf_cohort_base + LEAF_ROWS, M), 1):
     zero_pairs = pl.tile.full([1, PAIR_W], dtype=pl.FP32, value=0.0)
     if leaf == 0:
      for clear_offset in pl.range(NL * PAIR_W, WORKSPACE_WIDTH, PAIR_W):
       pl.store(zero_pairs, [t, clear_offset], wpairs)
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     begin = leaf * LEAF
     if begin < wv:
      valid = pl.min(LEAF, wv - begin)
      if valid <= 512:
       narrow_raw = pl.load(scores, [t, begin], [1, 512], valid_shape=[1, valid])
       narrow_padded = pl.tile.fillpad(narrow_raw, pad_value=pl.PadValue.min)
       narrow_values = pl.maximum(narrow_padded, SORT_FLOOR)
       narrow_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       narrow_pairs = pl.tile.sort32(narrow_values, pl.reinterpret_view(narrow_positions, pl.UINT32))
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=64)
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=256)
       pl.store(narrow_pairs, [t, leaf * PAIR_W], wpairs)
      elif valid <= 1024:
       left_valid = pl.min(valid, 512)
       right_valid = pl.max(valid - 512, 0)
       left_raw = pl.load(scores, [t, begin], [1, 512], valid_shape=[1, left_valid])
       right_raw = pl.load(scores, [t, begin + 512], [1, 512], valid_shape=[1, right_valid])
       left_values = pl.maximum(pl.tile.fillpad(left_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       right_values = pl.maximum(pl.tile.fillpad(right_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       left_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       right_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin + 512, pl.INT32))
       left_pairs = pl.tile.sort32(left_values, pl.reinterpret_view(left_positions, pl.UINT32))
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=64)
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=256)
       right_pairs = pl.tile.sort32(right_values, pl.reinterpret_view(right_positions, pl.UINT32))
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=64)
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=256)
       native_scratch = pl.tile.create([1, 2048], dtype=pl.FP32)
       native_pairs = pl.tile.mrgsort(left_pairs, right_pairs, tmp=native_scratch)
       native_discarded = pl.tile.read(native_pairs, [0, PAIR_W])
       if native_discarded > discarded:
        discarded = native_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if native_discarded == discarded:
        native_front_boundary = pl.tile.read(native_pairs, [0, PAIR_W - 2])
        native_front_bound = native_front_boundary
        if native_front_bound < FIN_FLOOR:
         native_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if native_discarded >= native_front_bound:
         native_arity_third = pl.tile.read(native_pairs, [0, PAIR_W + 4])
         if native_arity_third < native_discarded:
          native_arity_second = pl.tile.read(native_pairs, [0, PAIR_W + 2])
          native_arity_bits = pl.reinterpret_view(native_pairs, pl.INT32)
          native_front_result = pl.tile.read(native_arity_bits, [0, PAIR_W + 1])
          if native_arity_second == native_discarded:
           native_arity_second_position = pl.tile.read(native_arity_bits, [0, PAIR_W + 3])
           native_front_result = pl.min(native_front_result, native_arity_second_position)
         else:
          native_front_tail = pl.tile.slice(native_pairs, [1, 1024], [0, PAIR_W])
          native_front_values = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
          native_front_positions = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          native_front_candidates = pl.tile.select(pl.tile.cmps(native_front_values, native_discarded, cmp_type=0), native_front_positions, 2147483647)
          native_front_wide = pl.col_min(pl.reshape(native_front_candidates, [8, 64]))
          native_front_eight = pl.col_min(pl.reshape(native_front_wide, [8, 8]))
          native_front_result = pl.tile.read(native_front_eight, [0, 0])
          native_front_lane_1 = pl.tile.read(native_front_eight, [0, 1])
          if native_front_lane_1 < native_front_result:
           native_front_result = native_front_lane_1
          native_front_lane_2 = pl.tile.read(native_front_eight, [0, 2])
          if native_front_lane_2 < native_front_result:
           native_front_result = native_front_lane_2
          native_front_lane_3 = pl.tile.read(native_front_eight, [0, 3])
          if native_front_lane_3 < native_front_result:
           native_front_result = native_front_lane_3
          native_front_lane_4 = pl.tile.read(native_front_eight, [0, 4])
          if native_front_lane_4 < native_front_result:
           native_front_result = native_front_lane_4
          native_front_lane_5 = pl.tile.read(native_front_eight, [0, 5])
          if native_front_lane_5 < native_front_result:
           native_front_result = native_front_lane_5
          native_front_lane_6 = pl.tile.read(native_front_eight, [0, 6])
          if native_front_lane_6 < native_front_result:
           native_front_result = native_front_lane_6
          native_front_lane_7 = pl.tile.read(native_front_eight, [0, 7])
          if native_front_lane_7 < native_front_result:
           native_front_result = native_front_lane_7
         if native_front_result < frontier_position:
          frontier_position = native_front_result
       pl.store(pl.tile.slice(native_pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
      else:
       raw = pl.load(scores, [t, begin], [1, LEAF], valid_shape=[1, valid])
       padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
       flr = pl.tile.full([1, LEAF], dtype=pl.FP32, value=SORT_FLOOR)
       vals = pl.maximum(padded, flr)
       idx_i32 = pl.add(pl.tile.arange(0, [1, LEAF], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       pairs = pl.tile.sort32(vals, pl.reinterpret_view(idx_i32, pl.UINT32))
       pairs = pl.tile.mrgsort(pairs, block_len=64)
       pairs = pl.tile.mrgsort(pairs, block_len=256)
       pairs = pl.tile.mrgsort(pairs, block_len=1024)
       leaf_discarded = pl.tile.read(pairs, [0, PAIR_W])
       if leaf_discarded > discarded:
        discarded = leaf_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if leaf_discarded == discarded:
        leaf_front_boundary = pl.tile.read(pairs, [0, PAIR_W - 2])
        leaf_front_bound = leaf_front_boundary
        if leaf_front_bound < FIN_FLOOR:
         leaf_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if leaf_discarded >= leaf_front_bound:
         leaf_arity_third = pl.tile.read(pairs, [0, PAIR_W + 4])
         if leaf_arity_third < leaf_discarded:
          leaf_arity_second = pl.tile.read(pairs, [0, PAIR_W + 2])
          leaf_arity_bits = pl.reinterpret_view(pairs, pl.INT32)
          leaf_front_result = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 1])
          if leaf_arity_second == leaf_discarded:
           leaf_arity_second_position = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 3])
           leaf_front_result = pl.min(leaf_front_result, leaf_arity_second_position)
         else:
          leaf_front_tail = pl.tile.slice(pairs, [1, 3072], [0, PAIR_W])
          leaf_front_values = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
          leaf_front_positions = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          leaf_front_candidates = pl.tile.select(pl.tile.cmps(leaf_front_values, leaf_discarded, cmp_type=0), leaf_front_positions, 2147483647)
          leaf_front_wide = pl.col_min(pl.reshape(leaf_front_candidates, [24, 64]))
          leaf_front_eight = pl.col_min(pl.reshape(leaf_front_wide, [8, 8]))
          leaf_front_result = pl.tile.read(leaf_front_eight, [0, 0])
          leaf_front_lane_1 = pl.tile.read(leaf_front_eight, [0, 1])
          if leaf_front_lane_1 < leaf_front_result:
           leaf_front_result = leaf_front_lane_1
          leaf_front_lane_2 = pl.tile.read(leaf_front_eight, [0, 2])
          if leaf_front_lane_2 < leaf_front_result:
           leaf_front_result = leaf_front_lane_2
          leaf_front_lane_3 = pl.tile.read(leaf_front_eight, [0, 3])
          if leaf_front_lane_3 < leaf_front_result:
           leaf_front_result = leaf_front_lane_3
          leaf_front_lane_4 = pl.tile.read(leaf_front_eight, [0, 4])
          if leaf_front_lane_4 < leaf_front_result:
           leaf_front_result = leaf_front_lane_4
          leaf_front_lane_5 = pl.tile.read(leaf_front_eight, [0, 5])
          if leaf_front_lane_5 < leaf_front_result:
           leaf_front_result = leaf_front_lane_5
          leaf_front_lane_6 = pl.tile.read(leaf_front_eight, [0, 6])
          if leaf_front_lane_6 < leaf_front_result:
           leaf_front_result = leaf_front_lane_6
          leaf_front_lane_7 = pl.tile.read(leaf_front_eight, [0, 7])
          if leaf_front_lane_7 < leaf_front_result:
           leaf_front_result = leaf_front_lane_7
         if leaf_front_result < frontier_position:
          frontier_position = leaf_front_result
       pl.store(pl.tile.slice(pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
     else:
      pl.store(zero_pairs, [t, leaf * PAIR_W], wpairs)
     leaf_score_line = pl.tile.full([1, 16], dtype=pl.FP32, value=SORT_FLOOR)
     leaf_position_line = pl.tile.full([1, 16], dtype=pl.INT32, value=2147483647)
     pl.tile.write(leaf_score_line, [0, 0], discarded)
     pl.tile.write(leaf_position_line, [0, 0], frontier_position)
     pl.store(leaf_score_line, [t, leaf * 16], leaf_discarded_bank)
     pl.store(leaf_position_line, [t, leaf * 16], leaf_front_bank)
   for merge_cohort_job in pl.spmd((M + 4 - 1) // 4, name_hint='bounded_root_handoff'):
    cohort_base = merge_cohort_job * 4
    if P == 1:
     materialize_single_page = pl.read(block_table, [0, 0])
     materialize_pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), materialize_single_page), pad_value=pl.PadValue.zero)
    else:
     materialize_pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
    materialize_page_lanes = pl.tile.arange(0, [1, P8], dtype=pl.INT32)
    materialize_displacements = pl.mul(pl.sub(materialize_pages, materialize_page_lanes), PAGE)
    for t in pl.range(cohort_base, pl.min(cohort_base + 4, M), 1):
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     lc = (wv + LEAF - 1) // LEAF
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     for proof_leaf in pl.range(NL):
      proof_score = pl.read(leaf_discarded_bank, [t, proof_leaf * 16])
      proof_position = pl.read(leaf_front_bank, [t, proof_leaf * 16])
      if proof_score > discarded:
       discarded = proof_score
       frontier_position = pl.cast(2147483647, pl.INT32)
      if proof_score == discarded:
       if proof_position < frontier_position:
        frontier_position = proof_position
     fin = pl.load(wpairs, [t, 0], [1, PAIR_W])
     one_values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
     one_finite = pl.maximum(one_values, FIN_FLOOR)
     one_cutoff = pl.tile.read(one_finite, [0, K - 1])
     one_ones_c = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
     one_flags = pl.tile.select(pl.tile.cmps(one_values, one_cutoff, cmp_type=4), one_ones_c, 0.0)
     one_sums64 = pl.col_sum(pl.reshape(one_flags, [8, 64]))
     one_sums8 = pl.col_sum(pl.reshape(one_sums64, [8, 8]))
     one_above = pl.cast(0, pl.INDEX)
     for one_lane in pl.unroll(8):
      one_part = pl.cast(pl.tile.read(one_sums8, [0, one_lane]), pl.INT32)
      one_above = one_above + pl.cast(one_part, pl.INDEX)
     one_cutoff_stage_line = pl.tile.full([1, 16], dtype=pl.FP32, value=0)
     pl.tile.write(one_cutoff_stage_line, [0, 0], one_cutoff)
     pl.store(one_cutoff_stage_line, [t, 0], aux_cutoffs)
     one_quota_stage_line = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
     pl.tile.write(one_quota_stage_line, [0, 0], pl.cast(K - one_above, pl.INT32))
     pl.store(one_quota_stage_line, [t, 0], aux_quotas)
     one_replay_safe = discarded < one_cutoff
     if discarded == one_cutoff:
      one_cert_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
      one_cert_candidates = pl.tile.select(pl.tile.cmps(one_values, one_cutoff, cmp_type=0), one_cert_positions, -1)
      one_cert_wide = pl.col_max(pl.reshape(one_cert_candidates, [8, 64]))
      one_cert_eight = pl.col_max(pl.reshape(one_cert_wide, [8, 8]))
      one_cert_result = pl.tile.read(one_cert_eight, [0, 0])
      one_cert_lane_1 = pl.tile.read(one_cert_eight, [0, 1])
      if one_cert_lane_1 > one_cert_result:
       one_cert_result = one_cert_lane_1
      one_cert_lane_2 = pl.tile.read(one_cert_eight, [0, 2])
      if one_cert_lane_2 > one_cert_result:
       one_cert_result = one_cert_lane_2
      one_cert_lane_3 = pl.tile.read(one_cert_eight, [0, 3])
      if one_cert_lane_3 > one_cert_result:
       one_cert_result = one_cert_lane_3
      one_cert_lane_4 = pl.tile.read(one_cert_eight, [0, 4])
      if one_cert_lane_4 > one_cert_result:
       one_cert_result = one_cert_lane_4
      one_cert_lane_5 = pl.tile.read(one_cert_eight, [0, 5])
      if one_cert_lane_5 > one_cert_result:
       one_cert_result = one_cert_lane_5
      one_cert_lane_6 = pl.tile.read(one_cert_eight, [0, 6])
      if one_cert_lane_6 > one_cert_result:
       one_cert_result = one_cert_lane_6
      one_cert_lane_7 = pl.tile.read(one_cert_eight, [0, 7])
      if one_cert_lane_7 > one_cert_result:
       one_cert_result = one_cert_lane_7
      if frontier_position > one_cert_result:
       one_replay_safe = True
     if one_replay_safe:
      if one_cutoff > FIN_FLOOR:
       one_root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       one_selected = pl.cast(K, pl.INDEX)
       if one_selected > 0:
        one_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(one_root_positions, pl.FP32))
        one_root_keys = one_root_priority
        one_root_order = pl.tile.sort32(one_root_keys, pl.reinterpret_view(one_root_positions, pl.UINT32))
        one_root_order = pl.tile.mrgsort(one_root_order, block_len=64)
        one_root_order = pl.tile.mrgsort(one_root_order, block_len=256)
        one_root_logical = pl.tile.gather_mask(one_root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        one_root_valid = pl.tile.set_validshape(one_root_logical, 1, one_selected)
        one_root_safe = pl.tile.fillpad(one_root_valid, pad_value=pl.PadValue.zero)
        one_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
        for one_map_start in pl.range(0, one_selected, MAP):
         one_map_count = pl.min(MAP, one_selected - one_map_start)
         one_map_bits = pl.tile.slice(pl.reinterpret_view(one_root_safe, pl.FP32), [1, MAP], [0, one_map_start])
         one_map_logical = pl.reinterpret_view(one_map_bits, pl.INT32)
         one_root_page_idx = pl.tile.shrs(one_map_logical, 7)
         one_root_page_ids = pl.tile.gather(materialize_displacements, one_root_page_idx, one_root_tmp)
         one_root_physical = pl.add(one_map_logical, one_root_page_ids)
         pl.store(pl.tile.set_validshape(one_root_physical, 1, one_map_count), [t, one_map_start], indices)
      else:
       one_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
       one_root = fin
       one_root_values = pl.tile.gather_mask(one_root, mask_pattern=pl.tile.MaskPattern.P0101)
       one_root_positions = pl.tile.gather_mask(one_root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       one_ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
       one_root_strict = pl.tile.select(pl.tile.cmps(one_root_values, one_cutoff, cmp_type=4), one_ones_root, 0.0)
       one_root_equal = pl.tile.select(pl.tile.cmps(one_root_values, one_cutoff, cmp_type=0), one_ones_root, 0.0)
       one_root_eligible = pl.maximum(one_root_strict, one_root_equal)
       one_root_visible = pl.tile.cmps(pl.cast(one_root_positions, pl.FP32), pl.cast(pl.cast(one_visible, pl.INT32), pl.FP32), cmp_type=2)
       one_root_keep = pl.tile.select(one_root_visible, one_root_eligible, 0.0)
       one_root_sums64 = pl.col_sum(pl.reshape(one_root_keep, [8, 64]))
       one_root_sums8 = pl.col_sum(pl.reshape(one_root_sums64, [8, 8]))
       one_selected = pl.cast(0, pl.INDEX)
       for one_root_lane in pl.unroll(8):
        one_root_part = pl.cast(pl.tile.read(one_root_sums8, [0, one_root_lane]), pl.INT32)
        one_selected = one_selected + pl.cast(one_root_part, pl.INDEX)
       pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
       if one_selected > 0:
        one_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(one_root_positions, pl.FP32))
        one_root_keys = pl.tile.select(pl.tile.cmps(one_root_keep, 0.0, cmp_type=4), one_root_priority, -1.0)
        one_root_order = pl.tile.sort32(one_root_keys, pl.reinterpret_view(one_root_positions, pl.UINT32))
        one_root_order = pl.tile.mrgsort(one_root_order, block_len=64)
        one_root_order = pl.tile.mrgsort(one_root_order, block_len=256)
        one_root_sorted_keys = pl.tile.gather_mask(one_root_order, mask_pattern=pl.tile.MaskPattern.P0101)
        one_root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), one_root_sorted_keys), pl.INT32)
        one_root_valid = pl.tile.set_validshape(one_root_logical, 1, one_selected)
        one_root_safe = pl.tile.fillpad(one_root_valid, pad_value=pl.PadValue.zero)
        one_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
        for one_map_start in pl.range(0, one_selected, MAP):
         one_map_count = pl.min(MAP, one_selected - one_map_start)
         one_map_bits = pl.tile.slice(pl.reinterpret_view(one_root_safe, pl.FP32), [1, MAP], [0, one_map_start])
         one_map_logical = pl.reinterpret_view(one_map_bits, pl.INT32)
         one_root_page_idx = pl.tile.shrs(one_map_logical, 7)
         one_root_page_ids = pl.tile.gather(materialize_displacements, one_root_page_idx, one_root_tmp)
         one_root_physical = pl.add(one_map_logical, one_root_page_ids)
         pl.store(pl.tile.set_validshape(one_root_physical, 1, one_map_count), [t, one_map_start], indices)
     else:
      one_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
      one_remaining = pl.cast(K - one_above, pl.INDEX)
      pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
      one_ones_m = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
      one_lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
      one_lanes_f = pl.cast(one_lanes, pl.FP32)
      one_lane_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
      one_priority = pl.cast(pl.sub(one_lane_origin, one_lanes), pl.FP32)
      one_gtmp = pl.tile.create([1, SCAN], dtype=pl.INT32)
      one_count = pl.cast(0, pl.INDEX)
      for one_start in pl.range(0, one_visible, SCAN):
       if one_count < K:
        one_valid = pl.min(SCAN, one_visible - one_start)
        one_scan_raw = pl.load(scores, [t, one_start], [1, SCAN], valid_shape=[1, one_valid])
        one_scan_vals = pl.tile.fillpad(one_scan_raw, pad_value=pl.PadValue.min)
        one_strict = pl.tile.select(pl.tile.cmps(one_scan_vals, one_cutoff, cmp_type=4), one_ones_m, 0.0)
        one_ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
        if one_remaining > 0:
         one_eq = pl.tile.select(pl.tile.cmps(one_scan_vals, one_cutoff, cmp_type=0), one_ones_m, 0.0)
         one_eq_valid = pl.tile.cmps(one_lanes_f, pl.cast(pl.cast(one_valid, pl.INT32), pl.FP32), cmp_type=2)
         one_eq = pl.tile.select(one_eq_valid, one_eq, 0.0)
         one_eq_sums64 = pl.col_sum(pl.reshape(one_eq, [SCAN // 64, 64]))
         one_eq_sums8 = pl.col_sum(pl.reshape(one_eq_sums64, [8, 8]))
         one_available = pl.cast(0, pl.INDEX)
         for one_eq_lane in pl.unroll(8):
          one_eq_part = pl.cast(pl.tile.read(one_eq_sums8, [0, one_eq_lane]), pl.INT32)
          one_available = one_available + pl.cast(one_eq_part, pl.INDEX)
         one_take = pl.min(one_available, one_remaining)
         if one_take > 0:
          if one_take == one_available:
           one_ties = one_eq
          else:
           one_tie_keys = pl.mul(one_eq, one_priority)
           one_tie_pairs = pl.tile.sort32(one_tie_keys, pl.reinterpret_view(one_lanes, pl.UINT32))
           one_tie_pairs = pl.tile.mrgsort(one_tie_pairs, block_len=64)
           one_tie_merged = one_tie_pairs
           one_equal_keys = pl.tile.gather_mask(one_tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
           one_last_key_f = pl.tile.read(one_equal_keys, [0, one_take - 1])
           one_boundary_f = 128.0 - one_last_key_f
           one_ties = pl.tile.select(pl.tile.cmps(one_lanes_f, one_boundary_f, cmp_type=3), one_eq, 0.0)
          one_remaining = one_remaining - one_take
        one_keep = pl.maximum(one_strict, one_ties)
        one_keep_valid = pl.tile.set_validshape(one_keep, 1, one_valid)
        one_keep_padded = pl.tile.fillpad(one_keep_valid, pad_value=pl.PadValue.zero)
        one_keep_sums64 = pl.col_sum(pl.reshape(one_keep_padded, [SCAN // 64, 64]))
        one_keep_sums8 = pl.col_sum(pl.reshape(one_keep_sums64, [8, 8]))
        one_keep_count = pl.cast(0, pl.INDEX)
        for one_keep_lane in pl.unroll(8):
         one_keep_part = pl.cast(pl.tile.read(one_keep_sums8, [0, one_keep_lane]), pl.INT32)
         one_keep_count = one_keep_count + pl.cast(one_keep_part, pl.INDEX)
        one_selected = pl.min(one_keep_count, K - one_count)
        if one_selected > 0:
         one_order_keys = pl.mul(one_keep_padded, one_priority)
         one_order_pairs = pl.tile.sort32(one_order_keys, pl.reinterpret_view(one_lanes, pl.UINT32))
         one_order_pairs = pl.tile.mrgsort(one_order_pairs, block_len=64)
         one_order_merged = one_order_pairs
         one_order_vals = pl.tile.gather_mask(one_order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
         one_offs = pl.sub(one_lane_origin, pl.cast(one_order_vals, pl.INT32))
         one_compact_valid = pl.tile.set_validshape(one_offs, 1, one_selected)
         one_compact_safe = pl.tile.fillpad(one_compact_valid, pad_value=pl.PadValue.zero)
         one_logical = pl.add(one_compact_safe, pl.cast(one_start, pl.INT32))
         one_page_idx = pl.tile.shrs(one_logical, 7)
         one_page_ids = pl.tile.gather(materialize_displacements, one_page_idx, one_gtmp)
         one_physical = pl.add(one_logical, one_page_ids)
         pl.store(pl.tile.set_validshape(one_physical, 1, one_selected), [t, one_count], indices)
         one_count = one_count + one_selected
   for pack_task in pl.spmd(1, name_hint='pack_auxiliary'):
    for output_row in pl.range(0, M, 16):
     cutoff_lines = pl.load(aux_cutoffs, [output_row, 0], [16, 16])
     quota_lines = pl.load(aux_quotas, [output_row, 0], [16, 16])
     cutoff_flat = pl.reshape(cutoff_lines, [1, 256])
     quota_flat = pl.reshape(quota_lines, [1, 256])
     pack_offsets = pl.mul(pl.tile.arange(0, [1, 16], dtype=pl.INT32), 16)
     cutoff_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     quota_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     cutoff_packed = pl.tile.gather(cutoff_flat, pack_offsets, cutoff_scratch)
     quota_packed = pl.tile.gather(quota_flat, pack_offsets, quota_scratch)
     pl.store(pl.reshape(cutoff_packed, [16, 1]), [output_row, 0], cutoffs)
     pl.store(pl.reshape(quota_packed, [16, 1]), [output_row, 0], quotas)
   return indices
 elif NL == 2:

  @pl.jit
  def dsa41_topk_guarded_root(scores: pl.Tensor[[M, N], pl.FP32], lengths: pl.Tensor[[M], pl.INT32], block_table: pl.Tensor[[1, P], pl.INT32], wpairs: pl.InOut[pl.Tensor[[M, WORKSPACE_WIDTH], pl.FP32]], cutoffs: pl.InOut[pl.Tensor[[M, 1], pl.FP32]], quotas: pl.InOut[pl.Tensor[[M, 1], pl.INT32]], indices: pl.Out[pl.Tensor[[M, K], pl.INT32]]):
   aux_discarded = pl.create_tensor([M, 16], dtype=pl.FP32)
   publication_bank = pl.create_tensor([M // 4, 128], dtype=pl.FP32)
   aux_front = pl.create_tensor([M, 16], dtype=pl.INT32)
   leaf_discarded_bank = pl.create_tensor([M, NL * 16], dtype=pl.FP32)
   leaf_front_bank = pl.create_tensor([M, NL * 16], dtype=pl.INT32)
   for leaf_job in pl.spmd(LEAF_COHORTS * NL, name_hint='canonical_leaf_plane', allow_early_resolve=True):
    leaf = leaf_job // LEAF_COHORTS
    leaf_cohort_base = leaf_job % LEAF_COHORTS * LEAF_ROWS
    for t in pl.range(leaf_cohort_base, pl.min(leaf_cohort_base + LEAF_ROWS, M), 1):
     zero_pairs = pl.tile.full([1, PAIR_W], dtype=pl.FP32, value=0.0)
     if leaf == 0:
      for clear_offset in pl.range(NL * PAIR_W, WORKSPACE_WIDTH, PAIR_W):
       pl.store(zero_pairs, [t, clear_offset], wpairs)
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     begin = leaf * LEAF
     if begin < wv:
      valid = pl.min(LEAF, wv - begin)
      if valid <= 512:
       narrow_raw = pl.load(scores, [t, begin], [1, 512], valid_shape=[1, valid])
       narrow_padded = pl.tile.fillpad(narrow_raw, pad_value=pl.PadValue.min)
       narrow_values = pl.maximum(narrow_padded, SORT_FLOOR)
       narrow_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       narrow_pairs = pl.tile.sort32(narrow_values, pl.reinterpret_view(narrow_positions, pl.UINT32))
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=64)
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=256)
       pl.store(narrow_pairs, [t, leaf * PAIR_W], wpairs)
      elif valid <= 1024:
       left_valid = pl.min(valid, 512)
       right_valid = pl.max(valid - 512, 0)
       left_raw = pl.load(scores, [t, begin], [1, 512], valid_shape=[1, left_valid])
       right_raw = pl.load(scores, [t, begin + 512], [1, 512], valid_shape=[1, right_valid])
       left_values = pl.maximum(pl.tile.fillpad(left_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       right_values = pl.maximum(pl.tile.fillpad(right_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       left_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       right_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin + 512, pl.INT32))
       left_pairs = pl.tile.sort32(left_values, pl.reinterpret_view(left_positions, pl.UINT32))
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=64)
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=256)
       right_pairs = pl.tile.sort32(right_values, pl.reinterpret_view(right_positions, pl.UINT32))
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=64)
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=256)
       native_scratch = pl.tile.create([1, 2048], dtype=pl.FP32)
       native_pairs = pl.tile.mrgsort(left_pairs, right_pairs, tmp=native_scratch)
       native_discarded = pl.tile.read(native_pairs, [0, PAIR_W])
       if native_discarded > discarded:
        discarded = native_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if native_discarded == discarded:
        native_front_boundary = pl.tile.read(native_pairs, [0, PAIR_W - 2])
        native_front_bound = native_front_boundary
        if native_front_bound < FIN_FLOOR:
         native_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if native_discarded >= native_front_bound:
         native_arity_third = pl.tile.read(native_pairs, [0, PAIR_W + 4])
         if native_arity_third < native_discarded:
          native_arity_second = pl.tile.read(native_pairs, [0, PAIR_W + 2])
          native_arity_bits = pl.reinterpret_view(native_pairs, pl.INT32)
          native_front_result = pl.tile.read(native_arity_bits, [0, PAIR_W + 1])
          if native_arity_second == native_discarded:
           native_arity_second_position = pl.tile.read(native_arity_bits, [0, PAIR_W + 3])
           native_front_result = pl.min(native_front_result, native_arity_second_position)
         else:
          native_front_lookahead = pl.tile.read(native_pairs, [0, PAIR_W + 64])
          if native_front_lookahead < native_discarded:
           native_front_compact_lanes = pl.add(pl.tile.arange(0, [1, 64], dtype=pl.INT32), PAIR_W)
           native_front_compact_scratch = pl.tile.create([1, 2048], dtype=pl.INT32)
           native_front_compact = pl.tile.gather(native_pairs, native_front_compact_lanes, native_front_compact_scratch)
           native_front_compact_values = pl.tile.gather_mask(native_front_compact, mask_pattern=pl.tile.MaskPattern.P0101)
           native_front_compact_positions = pl.tile.gather_mask(native_front_compact, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
           native_front_compact_candidates = pl.tile.select(pl.tile.cmps(native_front_compact_values, native_discarded, cmp_type=0), native_front_compact_positions, 2147483647)
           native_front_compact_carrier = pl.add(pl.tile.full([8, 32], dtype=pl.INT32, value=0), native_front_compact_candidates)
           native_front_compact_reduce_scratch = pl.tile.create([8, 32], dtype=pl.INT32)
           native_front_compact_min = pl.row_min(native_front_compact_carrier, native_front_compact_reduce_scratch)
           native_front_result = pl.tile.read(native_front_compact_min, [0, 0])
          else:
           native_front_tail = pl.tile.slice(native_pairs, [1, 1024], [0, PAIR_W])
           native_front_values = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
           native_front_positions = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
           native_front_candidates = pl.tile.select(pl.tile.cmps(native_front_values, native_discarded, cmp_type=0), native_front_positions, 2147483647)
           native_front_wide = pl.col_min(pl.reshape(native_front_candidates, [8, 64]))
           native_front_eight = pl.col_min(pl.reshape(native_front_wide, [8, 8]))
           native_front_result = pl.tile.read(native_front_eight, [0, 0])
           native_front_lane_1 = pl.tile.read(native_front_eight, [0, 1])
           if native_front_lane_1 < native_front_result:
            native_front_result = native_front_lane_1
           native_front_lane_2 = pl.tile.read(native_front_eight, [0, 2])
           if native_front_lane_2 < native_front_result:
            native_front_result = native_front_lane_2
           native_front_lane_3 = pl.tile.read(native_front_eight, [0, 3])
           if native_front_lane_3 < native_front_result:
            native_front_result = native_front_lane_3
           native_front_lane_4 = pl.tile.read(native_front_eight, [0, 4])
           if native_front_lane_4 < native_front_result:
            native_front_result = native_front_lane_4
           native_front_lane_5 = pl.tile.read(native_front_eight, [0, 5])
           if native_front_lane_5 < native_front_result:
            native_front_result = native_front_lane_5
           native_front_lane_6 = pl.tile.read(native_front_eight, [0, 6])
           if native_front_lane_6 < native_front_result:
            native_front_result = native_front_lane_6
           native_front_lane_7 = pl.tile.read(native_front_eight, [0, 7])
           if native_front_lane_7 < native_front_result:
            native_front_result = native_front_lane_7
         if native_front_result < frontier_position:
          frontier_position = native_front_result
       pl.store(pl.tile.slice(native_pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
      else:
       raw = pl.load(scores, [t, begin], [1, LEAF], valid_shape=[1, valid])
       padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
       flr = pl.tile.full([1, LEAF], dtype=pl.FP32, value=SORT_FLOOR)
       vals = pl.maximum(padded, flr)
       idx_i32 = pl.add(pl.tile.arange(0, [1, LEAF], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       pairs = pl.tile.sort32(vals, pl.reinterpret_view(idx_i32, pl.UINT32))
       pairs = pl.tile.mrgsort(pairs, block_len=64)
       pairs = pl.tile.mrgsort(pairs, block_len=256)
       pairs = pl.tile.mrgsort(pairs, block_len=1024)
       leaf_discarded = pl.tile.read(pairs, [0, PAIR_W])
       if leaf_discarded > discarded:
        discarded = leaf_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if leaf_discarded == discarded:
        leaf_front_boundary = pl.tile.read(pairs, [0, PAIR_W - 2])
        leaf_front_bound = leaf_front_boundary
        if leaf_front_bound < FIN_FLOOR:
         leaf_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if leaf_discarded >= leaf_front_bound:
         leaf_arity_third = pl.tile.read(pairs, [0, PAIR_W + 4])
         if leaf_arity_third < leaf_discarded:
          leaf_arity_second = pl.tile.read(pairs, [0, PAIR_W + 2])
          leaf_arity_bits = pl.reinterpret_view(pairs, pl.INT32)
          leaf_front_result = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 1])
          if leaf_arity_second == leaf_discarded:
           leaf_arity_second_position = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 3])
           leaf_front_result = pl.min(leaf_front_result, leaf_arity_second_position)
         else:
          leaf_front_lookahead = pl.tile.read(pairs, [0, PAIR_W + 64])
          if leaf_front_lookahead < leaf_discarded:
           leaf_front_compact_lanes = pl.add(pl.tile.arange(0, [1, 64], dtype=pl.INT32), PAIR_W)
           leaf_front_compact_scratch = pl.tile.create([1, 4096], dtype=pl.INT32)
           leaf_front_compact = pl.tile.gather(pairs, leaf_front_compact_lanes, leaf_front_compact_scratch)
           leaf_front_compact_values = pl.tile.gather_mask(leaf_front_compact, mask_pattern=pl.tile.MaskPattern.P0101)
           leaf_front_compact_positions = pl.tile.gather_mask(leaf_front_compact, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
           leaf_front_compact_candidates = pl.tile.select(pl.tile.cmps(leaf_front_compact_values, leaf_discarded, cmp_type=0), leaf_front_compact_positions, 2147483647)
           leaf_front_compact_carrier = pl.add(pl.tile.full([8, 32], dtype=pl.INT32, value=0), leaf_front_compact_candidates)
           leaf_front_compact_reduce_scratch = pl.tile.create([8, 32], dtype=pl.INT32)
           leaf_front_compact_min = pl.row_min(leaf_front_compact_carrier, leaf_front_compact_reduce_scratch)
           leaf_front_result = pl.tile.read(leaf_front_compact_min, [0, 0])
          else:
           leaf_front_tail = pl.tile.slice(pairs, [1, 3072], [0, PAIR_W])
           leaf_front_values = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
           leaf_front_positions = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
           leaf_front_candidates = pl.tile.select(pl.tile.cmps(leaf_front_values, leaf_discarded, cmp_type=0), leaf_front_positions, 2147483647)
           leaf_front_wide = pl.col_min(pl.reshape(leaf_front_candidates, [24, 64]))
           leaf_front_eight = pl.col_min(pl.reshape(leaf_front_wide, [8, 8]))
           leaf_front_result = pl.tile.read(leaf_front_eight, [0, 0])
           leaf_front_lane_1 = pl.tile.read(leaf_front_eight, [0, 1])
           if leaf_front_lane_1 < leaf_front_result:
            leaf_front_result = leaf_front_lane_1
           leaf_front_lane_2 = pl.tile.read(leaf_front_eight, [0, 2])
           if leaf_front_lane_2 < leaf_front_result:
            leaf_front_result = leaf_front_lane_2
           leaf_front_lane_3 = pl.tile.read(leaf_front_eight, [0, 3])
           if leaf_front_lane_3 < leaf_front_result:
            leaf_front_result = leaf_front_lane_3
           leaf_front_lane_4 = pl.tile.read(leaf_front_eight, [0, 4])
           if leaf_front_lane_4 < leaf_front_result:
            leaf_front_result = leaf_front_lane_4
           leaf_front_lane_5 = pl.tile.read(leaf_front_eight, [0, 5])
           if leaf_front_lane_5 < leaf_front_result:
            leaf_front_result = leaf_front_lane_5
           leaf_front_lane_6 = pl.tile.read(leaf_front_eight, [0, 6])
           if leaf_front_lane_6 < leaf_front_result:
            leaf_front_result = leaf_front_lane_6
           leaf_front_lane_7 = pl.tile.read(leaf_front_eight, [0, 7])
           if leaf_front_lane_7 < leaf_front_result:
            leaf_front_result = leaf_front_lane_7
         if leaf_front_result < frontier_position:
          frontier_position = leaf_front_result
       pl.store(pl.tile.slice(pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
     else:
      pl.store(zero_pairs, [t, leaf * PAIR_W], wpairs)
     leaf_score_line = pl.tile.full([1, 16], dtype=pl.FP32, value=SORT_FLOOR)
     leaf_position_line = pl.tile.full([1, 16], dtype=pl.INT32, value=2147483647)
     pl.tile.write(leaf_score_line, [0, 0], discarded)
     pl.tile.write(leaf_position_line, [0, 0], frontier_position)
     pl.store(leaf_score_line, [t, leaf * 16], leaf_discarded_bank)
     pl.store(leaf_position_line, [t, leaf * 16], leaf_front_bank)
   for merge_cohort_job in pl.spmd((M + 4 - 1) // 4, name_hint='bounded_root_handoff', allow_early_resolve=True):
    cohort_base = merge_cohort_job * 4
    publication_record = pl.tile.full([1, 128], dtype=pl.FP32, value=0.0)
    if P == 1:
     materialize_single_page = pl.read(block_table, [0, 0])
     materialize_pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), materialize_single_page), pad_value=pl.PadValue.zero)
    else:
     materialize_pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
    materialize_page_lanes = pl.tile.arange(0, [1, P8], dtype=pl.INT32)
    materialize_displacements = pl.mul(pl.sub(materialize_pages, materialize_page_lanes), PAGE)
    for t in pl.range(cohort_base, pl.min(cohort_base + 4, M), 1):
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     lc = (wv + LEAF - 1) // LEAF
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     for proof_leaf in pl.range(NL):
      proof_score = pl.read(leaf_discarded_bank, [t, proof_leaf * 16])
      proof_position = pl.read(leaf_front_bank, [t, proof_leaf * 16])
      if proof_score > discarded:
       discarded = proof_score
       frontier_position = pl.cast(2147483647, pl.INT32)
      if proof_score == discarded:
       if proof_position < frontier_position:
        frontier_position = proof_position
     if lc == 1:
      fin = pl.load(wpairs, [t, 0], [1, PAIR_W])
      short_complete = False
      if visible > 0:
       if visible < K:
        short_minimum = pl.tile.read(fin, [0, 2 * (visible - 1)])
        if short_minimum > FIN_FLOOR:
         short_complete = True
      if short_complete:
       short_cutoff = pl.cast(FIN_FLOOR, pl.FP32)
       pl.tile.write(publication_record, [0, t - cohort_base], short_cutoff)
       pl.tile.write(publication_record, [0, 4 + t - cohort_base], pl.cast(pl.cast(K - visible, pl.INT32), pl.FP32))
       short_lanes = pl.tile.arange(0, [1, K], dtype=pl.INT32)
       short_keep = pl.tile.cmps(pl.cast(short_lanes, pl.FP32), pl.cast(pl.cast(visible, pl.INT32), pl.FP32), cmp_type=2)
       short_safe = pl.tile.select(short_keep, short_lanes, 0)
       short_pages = pl.tile.shrs(short_safe, 7)
       short_scratch = pl.tile.create([1, MAP], dtype=pl.INT32)
       short_displacements = pl.tile.gather(materialize_displacements, short_pages, short_scratch)
       short_physical = pl.add(short_safe, short_displacements)
       short_output = pl.tile.select(short_keep, short_physical, -1)
       pl.store(short_output, [t, 0], indices)
      else:
       one_cutoff = pl.tile.read(fin, [0, 2 * (K - 1)])
       if one_cutoff < FIN_FLOOR:
        one_cutoff = pl.cast(FIN_FLOOR, pl.FP32)
       one_above = pl.cast(0, pl.INDEX)
       one_neighbor = pl.tile.read(fin, [0, 2 * (K - 2)])
       one_witness = False
       if visible >= K:
        if one_cutoff > FIN_FLOOR:
         if one_neighbor > one_cutoff:
          one_witness = True
       if one_witness:
        one_above = pl.cast(K - 1, pl.INDEX)
       else:
        one_rank_probe_256 = pl.tile.read(fin, [0, 2 * (one_above + 256 - 1)])
        one_rank_bit_256 = pl.cast(pl.cast(one_rank_probe_256 > one_cutoff, pl.INT32), pl.INDEX)
        one_above = one_above + 256 * one_rank_bit_256
        one_rank_probe_128 = pl.tile.read(fin, [0, 2 * (one_above + 128 - 1)])
        one_rank_bit_128 = pl.cast(pl.cast(one_rank_probe_128 > one_cutoff, pl.INT32), pl.INDEX)
        one_above = one_above + 128 * one_rank_bit_128
        one_rank_probe_64 = pl.tile.read(fin, [0, 2 * (one_above + 64 - 1)])
        one_rank_bit_64 = pl.cast(pl.cast(one_rank_probe_64 > one_cutoff, pl.INT32), pl.INDEX)
        one_above = one_above + 64 * one_rank_bit_64
        one_rank_probe_32 = pl.tile.read(fin, [0, 2 * (one_above + 32 - 1)])
        one_rank_bit_32 = pl.cast(pl.cast(one_rank_probe_32 > one_cutoff, pl.INT32), pl.INDEX)
        one_above = one_above + 32 * one_rank_bit_32
        one_rank_probe_16 = pl.tile.read(fin, [0, 2 * (one_above + 16 - 1)])
        one_rank_bit_16 = pl.cast(pl.cast(one_rank_probe_16 > one_cutoff, pl.INT32), pl.INDEX)
        one_above = one_above + 16 * one_rank_bit_16
        one_rank_probe_8 = pl.tile.read(fin, [0, 2 * (one_above + 8 - 1)])
        one_rank_bit_8 = pl.cast(pl.cast(one_rank_probe_8 > one_cutoff, pl.INT32), pl.INDEX)
        one_above = one_above + 8 * one_rank_bit_8
        one_rank_probe_4 = pl.tile.read(fin, [0, 2 * (one_above + 4 - 1)])
        one_rank_bit_4 = pl.cast(pl.cast(one_rank_probe_4 > one_cutoff, pl.INT32), pl.INDEX)
        one_above = one_above + 4 * one_rank_bit_4
        one_rank_probe_2 = pl.tile.read(fin, [0, 2 * (one_above + 2 - 1)])
        one_rank_bit_2 = pl.cast(pl.cast(one_rank_probe_2 > one_cutoff, pl.INT32), pl.INDEX)
        one_above = one_above + 2 * one_rank_bit_2
        one_rank_probe_1 = pl.tile.read(fin, [0, 2 * (one_above + 1 - 1)])
        one_rank_bit_1 = pl.cast(pl.cast(one_rank_probe_1 > one_cutoff, pl.INT32), pl.INDEX)
        one_above = one_above + 1 * one_rank_bit_1
       pl.tile.write(publication_record, [0, t - cohort_base], one_cutoff)
       pl.tile.write(publication_record, [0, 4 + t - cohort_base], pl.cast(pl.cast(K - one_above, pl.INT32), pl.FP32))
       one_replay_safe = discarded < one_cutoff
       one_single = one_witness
       if discarded > one_cutoff:
        one_single = False
       if one_single:
        one_replay_safe = True
       else:
        if discarded == one_cutoff:
         if K - one_above <= 8:
          one_cert_result = pl.cast(-1, pl.INT32)
          one_tbits = pl.reinterpret_view(fin, pl.INT32)
          for one_tidx in pl.range(one_above, K):
           one_tpos = pl.tile.read(one_tbits, [0, 2 * one_tidx + 1])
           if pl.tile.read(fin, [0, 2 * one_tidx]) == one_cutoff:
            one_cert_result = pl.max(one_cert_result, one_tpos)
         else:
          one_cert_values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
          one_cert_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          one_cert_candidates = pl.tile.select(pl.tile.cmps(one_cert_values, one_cutoff, cmp_type=0), one_cert_positions, -1)
          one_cert_wide = pl.col_max(pl.reshape(one_cert_candidates, [8, 64]))
          one_cert_eight = pl.col_max(pl.reshape(one_cert_wide, [8, 8]))
          one_cert_result = pl.tile.read(one_cert_eight, [0, 0])
          one_cert_lane_1 = pl.tile.read(one_cert_eight, [0, 1])
          if one_cert_lane_1 > one_cert_result:
           one_cert_result = one_cert_lane_1
          one_cert_lane_2 = pl.tile.read(one_cert_eight, [0, 2])
          if one_cert_lane_2 > one_cert_result:
           one_cert_result = one_cert_lane_2
          one_cert_lane_3 = pl.tile.read(one_cert_eight, [0, 3])
          if one_cert_lane_3 > one_cert_result:
           one_cert_result = one_cert_lane_3
          one_cert_lane_4 = pl.tile.read(one_cert_eight, [0, 4])
          if one_cert_lane_4 > one_cert_result:
           one_cert_result = one_cert_lane_4
          one_cert_lane_5 = pl.tile.read(one_cert_eight, [0, 5])
          if one_cert_lane_5 > one_cert_result:
           one_cert_result = one_cert_lane_5
          one_cert_lane_6 = pl.tile.read(one_cert_eight, [0, 6])
          if one_cert_lane_6 > one_cert_result:
           one_cert_result = one_cert_lane_6
          one_cert_lane_7 = pl.tile.read(one_cert_eight, [0, 7])
          if one_cert_lane_7 > one_cert_result:
           one_cert_result = one_cert_lane_7
         if frontier_position > one_cert_result:
          one_replay_safe = True
       if one_replay_safe:
        if one_cutoff > FIN_FLOOR:
         one_root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         if one_single:
          one_endpoint = pl.tile.read(one_root_positions, [0, K - 1])
          if discarded == one_cutoff:
           one_endpoint = pl.min(one_endpoint, frontier_position)
          pl.tile.write(one_root_positions, [0, K - 1], one_endpoint)
         one_selected = pl.cast(K, pl.INDEX)
         if one_selected > 0:
          one_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(one_root_positions, pl.FP32))
          one_root_keys = one_root_priority
          one_root_order = pl.tile.sort32(one_root_keys, pl.reinterpret_view(one_root_positions, pl.UINT32))
          one_root_order = pl.tile.mrgsort(one_root_order, block_len=64)
          one_root_order = pl.tile.mrgsort(one_root_order, block_len=256)
          one_root_logical = pl.tile.gather_mask(one_root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          one_root_valid = pl.tile.set_validshape(one_root_logical, 1, one_selected)
          one_root_safe = pl.tile.fillpad(one_root_valid, pad_value=pl.PadValue.zero)
          one_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
          for one_map_start in pl.range(0, one_selected, MAP):
           one_map_count = pl.min(MAP, one_selected - one_map_start)
           one_map_bits = pl.tile.slice(pl.reinterpret_view(one_root_safe, pl.FP32), [1, MAP], [0, one_map_start])
           one_map_logical = pl.reinterpret_view(one_map_bits, pl.INT32)
           one_root_page_idx = pl.tile.shrs(one_map_logical, 7)
           one_root_page_ids = pl.tile.gather(materialize_displacements, one_root_page_idx, one_root_tmp)
           one_root_physical = pl.add(one_map_logical, one_root_page_ids)
           pl.store(pl.tile.set_validshape(one_root_physical, 1, one_map_count), [t, one_map_start], indices)
        else:
         one_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
         one_root = fin
         one_root_values = pl.tile.gather_mask(one_root, mask_pattern=pl.tile.MaskPattern.P0101)
         one_root_positions = pl.tile.gather_mask(one_root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         one_ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
         one_root_strict = pl.tile.select(pl.tile.cmps(one_root_values, one_cutoff, cmp_type=4), one_ones_root, 0.0)
         one_root_equal = pl.tile.select(pl.tile.cmps(one_root_values, one_cutoff, cmp_type=0), one_ones_root, 0.0)
         one_root_eligible = pl.maximum(one_root_strict, one_root_equal)
         one_root_visible = pl.tile.cmps(pl.cast(one_root_positions, pl.FP32), pl.cast(pl.cast(one_visible, pl.INT32), pl.FP32), cmp_type=2)
         one_root_keep = pl.tile.select(one_root_visible, one_root_eligible, 0.0)
         one_root_sums64 = pl.col_sum(pl.reshape(one_root_keep, [8, 64]))
         one_root_sums8 = pl.col_sum(pl.reshape(one_root_sums64, [8, 8]))
         one_selected = pl.cast(0, pl.INDEX)
         for one_root_lane in pl.unroll(8):
          one_root_part = pl.cast(pl.tile.read(one_root_sums8, [0, one_root_lane]), pl.INT32)
          one_selected = one_selected + pl.cast(one_root_part, pl.INDEX)
         pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
         if one_selected > 0:
          one_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(one_root_positions, pl.FP32))
          one_root_keys = pl.tile.select(pl.tile.cmps(one_root_keep, 0.0, cmp_type=4), one_root_priority, -1.0)
          one_root_order = pl.tile.sort32(one_root_keys, pl.reinterpret_view(one_root_positions, pl.UINT32))
          one_root_order = pl.tile.mrgsort(one_root_order, block_len=64)
          one_root_order = pl.tile.mrgsort(one_root_order, block_len=256)
          one_root_sorted_keys = pl.tile.gather_mask(one_root_order, mask_pattern=pl.tile.MaskPattern.P0101)
          one_root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), one_root_sorted_keys), pl.INT32)
          one_root_valid = pl.tile.set_validshape(one_root_logical, 1, one_selected)
          one_root_safe = pl.tile.fillpad(one_root_valid, pad_value=pl.PadValue.zero)
          one_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
          for one_map_start in pl.range(0, one_selected, MAP):
           one_map_count = pl.min(MAP, one_selected - one_map_start)
           one_map_bits = pl.tile.slice(pl.reinterpret_view(one_root_safe, pl.FP32), [1, MAP], [0, one_map_start])
           one_map_logical = pl.reinterpret_view(one_map_bits, pl.INT32)
           one_root_page_idx = pl.tile.shrs(one_map_logical, 7)
           one_root_page_ids = pl.tile.gather(materialize_displacements, one_root_page_idx, one_root_tmp)
           one_root_physical = pl.add(one_map_logical, one_root_page_ids)
           pl.store(pl.tile.set_validshape(one_root_physical, 1, one_map_count), [t, one_map_start], indices)
       else:
        one_bulk_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
        one_bulk_quota = pl.cast(K - one_above, pl.INT32)
        pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
        one_bulk_raw = pl.load(scores, [t, 0], [1, 4096], valid_shape=[1, N])
        one_bulk_values = pl.reshape(pl.tile.fillpad(one_bulk_raw, pad_value=pl.PadValue.min), [32, SCAN])
        one_bulk_positions = pl.reshape(pl.tile.arange(0, [1, 4096], dtype=pl.INT32), [32, SCAN])
        one_bulk_valid = pl.tile.cmps(pl.cast(one_bulk_positions, pl.FP32), pl.cast(pl.cast(one_bulk_visible, pl.INT32), pl.FP32), cmp_type=2)
        one_bulk_strict = pl.tile.select(one_bulk_valid, pl.tile.select(pl.tile.cmps(one_bulk_values, one_cutoff, cmp_type=4), pl.tile.full([32, SCAN], dtype=pl.FP32, value=1.0), 0.0), 0.0)
        one_bulk_strict_scratch = pl.tile.create([32, SCAN], dtype=pl.FP32)
        one_bulk_count_offsets = pl.mul(pl.tile.arange(0, [1, 32], dtype=pl.INT32), 8)
        one_bulk_count_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
        one_bulk_equal = pl.tile.select(one_bulk_valid, pl.tile.select(pl.tile.cmps(one_bulk_values, one_cutoff, cmp_type=0), pl.tile.full([32, SCAN], dtype=pl.FP32, value=1.0), 0.0), 0.0)
        one_bulk_packed_flags = pl.add(pl.mul(one_bulk_strict, 8192.0), one_bulk_equal)
        one_bulk_packed_reduced = pl.row_sum(one_bulk_packed_flags, one_bulk_strict_scratch)
        one_bulk_packed_carrier = pl.row_expand_add(pl.tile.full([32, 8], dtype=pl.FP32, value=0.0), one_bulk_packed_reduced)
        one_bulk_packed_counts = pl.tile.gather(pl.reshape(one_bulk_packed_carrier, [1, 256]), one_bulk_count_offsets, one_bulk_count_scratch)
        one_bulk_blocks = pl.tile.arange(0, [1, 32], dtype=pl.INT32)
        one_bulk_prefix_scratch = pl.tile.create([1, 32], dtype=pl.INT32)
        one_bulk_packed_prefix = one_bulk_packed_counts
        one_bulk_shift_index = pl.maximum(pl.sub(one_bulk_blocks, 1), 0)
        one_bulk_shift_valid = pl.tile.cmps(pl.cast(one_bulk_blocks, pl.FP32), 1.0, cmp_type=5)
        one_bulk_packed_shift = pl.tile.gather(one_bulk_packed_prefix, one_bulk_shift_index, one_bulk_prefix_scratch)
        one_bulk_packed_prefix = pl.add(one_bulk_packed_prefix, pl.tile.select(one_bulk_shift_valid, one_bulk_packed_shift, 0.0))
        one_bulk_shift_index = pl.maximum(pl.sub(one_bulk_blocks, 2), 0)
        one_bulk_shift_valid = pl.tile.cmps(pl.cast(one_bulk_blocks, pl.FP32), 2.0, cmp_type=5)
        one_bulk_packed_shift = pl.tile.gather(one_bulk_packed_prefix, one_bulk_shift_index, one_bulk_prefix_scratch)
        one_bulk_packed_prefix = pl.add(one_bulk_packed_prefix, pl.tile.select(one_bulk_shift_valid, one_bulk_packed_shift, 0.0))
        one_bulk_shift_index = pl.maximum(pl.sub(one_bulk_blocks, 4), 0)
        one_bulk_shift_valid = pl.tile.cmps(pl.cast(one_bulk_blocks, pl.FP32), 4.0, cmp_type=5)
        one_bulk_packed_shift = pl.tile.gather(one_bulk_packed_prefix, one_bulk_shift_index, one_bulk_prefix_scratch)
        one_bulk_packed_prefix = pl.add(one_bulk_packed_prefix, pl.tile.select(one_bulk_shift_valid, one_bulk_packed_shift, 0.0))
        one_bulk_shift_index = pl.maximum(pl.sub(one_bulk_blocks, 8), 0)
        one_bulk_shift_valid = pl.tile.cmps(pl.cast(one_bulk_blocks, pl.FP32), 8.0, cmp_type=5)
        one_bulk_packed_shift = pl.tile.gather(one_bulk_packed_prefix, one_bulk_shift_index, one_bulk_prefix_scratch)
        one_bulk_packed_prefix = pl.add(one_bulk_packed_prefix, pl.tile.select(one_bulk_shift_valid, one_bulk_packed_shift, 0.0))
        one_bulk_shift_index = pl.maximum(pl.sub(one_bulk_blocks, 16), 0)
        one_bulk_shift_valid = pl.tile.cmps(pl.cast(one_bulk_blocks, pl.FP32), 16.0, cmp_type=5)
        one_bulk_packed_shift = pl.tile.gather(one_bulk_packed_prefix, one_bulk_shift_index, one_bulk_prefix_scratch)
        one_bulk_packed_prefix = pl.add(one_bulk_packed_prefix, pl.tile.select(one_bulk_shift_valid, one_bulk_packed_shift, 0.0))
        one_bulk_strict_counts = pl.cast(pl.cast(pl.mul(one_bulk_packed_counts, 0.0001220703125), pl.INT32, mode='floor'), pl.FP32)
        one_bulk_equal_counts = pl.sub(one_bulk_packed_counts, pl.mul(one_bulk_strict_counts, 8192.0))
        one_bulk_strict_prefix = pl.cast(pl.cast(pl.mul(one_bulk_packed_prefix, 0.0001220703125), pl.INT32, mode='floor'), pl.FP32)
        one_bulk_equal_prefix = pl.sub(one_bulk_packed_prefix, pl.mul(one_bulk_strict_prefix, 8192.0))
        one_bulk_strict_before = pl.sub(one_bulk_strict_prefix, one_bulk_strict_counts)
        one_bulk_equal_before = pl.sub(one_bulk_equal_prefix, one_bulk_equal_counts)
        one_bulk_tie_take = pl.minimum(one_bulk_equal_counts, pl.maximum(pl.sub(pl.add(pl.tile.full([1, 32], dtype=pl.FP32, value=0.0), pl.cast(one_bulk_quota, pl.FP32)), one_bulk_equal_before), 0.0))
        one_bulk_output_begin = pl.add(one_bulk_strict_before, pl.minimum(one_bulk_equal_before, pl.cast(one_bulk_quota, pl.FP32)))
        one_bulk_output_count = pl.add(one_bulk_strict_counts, one_bulk_tie_take)
        one_bulk_lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
        one_bulk_lanes_f = pl.cast(one_bulk_lanes, pl.FP32)
        one_bulk_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
        one_bulk_priority = pl.cast(pl.sub(one_bulk_origin, one_bulk_lanes), pl.FP32)
        one_bulk_map_scratch = pl.tile.create([1, SCAN], dtype=pl.INT32)
        for one_bulk_block in pl.range(0, (one_bulk_visible + SCAN - 1) // SCAN):
          one_bulk_selected = pl.cast(pl.cast(pl.tile.read(one_bulk_output_count, [0, one_bulk_block]), pl.INT32), pl.INDEX)
          if one_bulk_selected > 0:
            one_bulk_start = one_bulk_block * SCAN
            one_bulk_count = pl.cast(pl.cast(pl.tile.read(one_bulk_output_begin, [0, one_bulk_block]), pl.INT32), pl.INDEX)
            one_bulk_take = pl.cast(pl.cast(pl.tile.read(one_bulk_tie_take, [0, one_bulk_block]), pl.INT32), pl.INDEX)
            one_bulk_available = pl.cast(pl.cast(pl.tile.read(one_bulk_equal_counts, [0, one_bulk_block]), pl.INT32), pl.INDEX)
            one_bulk_block_flags = pl.tile.slice(pl.reshape(one_bulk_packed_flags, [1, 4096]), [1, SCAN], [0, one_bulk_start])
            one_bulk_block_ones = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
            one_bulk_block_strict = pl.tile.select(pl.tile.cmps(one_bulk_block_flags, 8192.0, cmp_type=0), one_bulk_block_ones, 0.0)
            one_bulk_block_equal = pl.tile.select(pl.tile.cmps(one_bulk_block_flags, 1.0, cmp_type=0), one_bulk_block_ones, 0.0)
            one_bulk_ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
            if one_bulk_take > 0:
              if one_bulk_take == one_bulk_available:
                one_bulk_ties = one_bulk_block_equal
              else:
                one_bulk_tie_keys = pl.mul(one_bulk_block_equal, one_bulk_priority)
                one_bulk_tie_pairs = pl.tile.sort32(one_bulk_tie_keys, pl.reinterpret_view(one_bulk_lanes, pl.UINT32))
                one_bulk_tie_pairs = pl.tile.mrgsort(one_bulk_tie_pairs, block_len=64)
                one_bulk_equal_keys = pl.tile.gather_mask(one_bulk_tie_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
                one_bulk_last_key = pl.tile.read(one_bulk_equal_keys, [0, one_bulk_take - 1])
                one_bulk_boundary = 128.0 - one_bulk_last_key
                one_bulk_ties = pl.tile.select(pl.tile.cmps(one_bulk_lanes_f, one_bulk_boundary, cmp_type=3), one_bulk_block_equal, 0.0)
            one_bulk_keep = pl.maximum(one_bulk_block_strict, one_bulk_ties)
            one_bulk_order_keys = pl.mul(one_bulk_keep, one_bulk_priority)
            one_bulk_order_pairs = pl.tile.sort32(one_bulk_order_keys, pl.reinterpret_view(one_bulk_lanes, pl.UINT32))
            one_bulk_order_pairs = pl.tile.mrgsort(one_bulk_order_pairs, block_len=64)
            one_bulk_order_values = pl.tile.gather_mask(one_bulk_order_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
            one_bulk_offsets = pl.sub(one_bulk_origin, pl.cast(one_bulk_order_values, pl.INT32))
            one_bulk_compact_valid = pl.tile.set_validshape(one_bulk_offsets, 1, one_bulk_selected)
            one_bulk_compact_safe = pl.tile.fillpad(one_bulk_compact_valid, pad_value=pl.PadValue.zero)
            one_bulk_logical = pl.add(one_bulk_compact_safe, pl.cast(one_bulk_start, pl.INT32))
            one_bulk_page_index = pl.tile.shrs(one_bulk_logical, 7)
            one_bulk_page_ids = pl.tile.gather(materialize_displacements, one_bulk_page_index, one_bulk_map_scratch)
            one_bulk_physical = pl.add(one_bulk_logical, one_bulk_page_ids)
            pl.store(pl.tile.set_validshape(one_bulk_physical, 1, one_bulk_selected), [t, one_bulk_count], indices)
     else:
      handoff_scratch = pl.tile.create([1, 2 * PAIR_W], dtype=pl.FP32)
      handoff_left = pl.load(wpairs, [t, 0], [1, PAIR_W])
      handoff_right = pl.load(wpairs, [t, PAIR_W], [1, PAIR_W])
      mboth = pl.tile.mrgsort(handoff_left, handoff_right, tmp=handoff_scratch)
      merge_discarded = pl.tile.read(mboth, [0, PAIR_W])
      if merge_discarded > discarded:
       discarded = merge_discarded
       frontier_position = pl.cast(2147483647, pl.INT32)
      if merge_discarded == discarded:
       merge_front_boundary = pl.tile.read(mboth, [0, PAIR_W - 2])
       merge_front_bound = merge_front_boundary
       if merge_front_bound < FIN_FLOOR:
        merge_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
       if merge_discarded >= merge_front_bound:
        merge_arity_third = pl.tile.read(mboth, [0, PAIR_W + 4])
        if merge_arity_third < merge_discarded:
         merge_arity_second = pl.tile.read(mboth, [0, PAIR_W + 2])
         merge_arity_bits = pl.reinterpret_view(mboth, pl.INT32)
         merge_front_result = pl.tile.read(merge_arity_bits, [0, PAIR_W + 1])
         if merge_arity_second == merge_discarded:
          merge_arity_second_position = pl.tile.read(merge_arity_bits, [0, PAIR_W + 3])
          merge_front_result = pl.min(merge_front_result, merge_arity_second_position)
        else:
         tie_edge = pl.tile.read(mboth, [0, PAIR_W + 16])
         if tie_edge != merge_discarded:
          merge_front_result = pl.cast(2147483647, pl.INT32)
          merge_tbits = pl.reinterpret_view(mboth, pl.INT32)
          for merge_tidx in pl.range(8):
           merge_tpos = pl.tile.read(merge_tbits, [0, PAIR_W + 2 * merge_tidx + 1])
           if pl.tile.read(mboth, [0, PAIR_W + 2 * merge_tidx]) == merge_discarded:
            merge_front_result = pl.min(merge_front_result, merge_tpos)
         else:
          merge_front_tail = pl.tile.slice(mboth, [1, 1024], [0, PAIR_W])
          merge_front_values = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
          merge_front_positions = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          merge_front_candidates = pl.tile.select(pl.tile.cmps(merge_front_values, merge_discarded, cmp_type=0), merge_front_positions, 2147483647)
          merge_front_wide = pl.col_min(pl.reshape(merge_front_candidates, [8, 64]))
          merge_front_eight = pl.col_min(pl.reshape(merge_front_wide, [8, 8]))
          merge_front_result = pl.tile.read(merge_front_eight, [0, 0])
          merge_front_lane_1 = pl.tile.read(merge_front_eight, [0, 1])
          if merge_front_lane_1 < merge_front_result:
           merge_front_result = merge_front_lane_1
          merge_front_lane_2 = pl.tile.read(merge_front_eight, [0, 2])
          if merge_front_lane_2 < merge_front_result:
           merge_front_result = merge_front_lane_2
          merge_front_lane_3 = pl.tile.read(merge_front_eight, [0, 3])
          if merge_front_lane_3 < merge_front_result:
           merge_front_result = merge_front_lane_3
          merge_front_lane_4 = pl.tile.read(merge_front_eight, [0, 4])
          if merge_front_lane_4 < merge_front_result:
           merge_front_result = merge_front_lane_4
          merge_front_lane_5 = pl.tile.read(merge_front_eight, [0, 5])
          if merge_front_lane_5 < merge_front_result:
           merge_front_result = merge_front_lane_5
          merge_front_lane_6 = pl.tile.read(merge_front_eight, [0, 6])
          if merge_front_lane_6 < merge_front_result:
           merge_front_result = merge_front_lane_6
          merge_front_lane_7 = pl.tile.read(merge_front_eight, [0, 7])
          if merge_front_lane_7 < merge_front_result:
           merge_front_result = merge_front_lane_7
        if merge_front_result < frontier_position:
         frontier_position = merge_front_result
      pl.store(pl.tile.slice(mboth, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
      fin = pl.tile.slice(mboth, [1, PAIR_W], [0, 0])
      two_cutoff = pl.tile.read(fin, [0, 2 * (K - 1)])
      if two_cutoff < FIN_FLOOR:
       two_cutoff = pl.cast(FIN_FLOOR, pl.FP32)
      two_above = pl.cast(0, pl.INDEX)
      two_neighbor = pl.tile.read(fin, [0, 2 * (K - 2)])
      two_witness = False
      if visible >= K:
       if two_cutoff > FIN_FLOOR:
        if two_neighbor > two_cutoff:
         two_witness = True
      if two_witness:
       two_above = pl.cast(K - 1, pl.INDEX)
      else:
       two_rank_probe_256 = pl.tile.read(fin, [0, 2 * (two_above + 256 - 1)])
       two_rank_bit_256 = pl.cast(pl.cast(two_rank_probe_256 > two_cutoff, pl.INT32), pl.INDEX)
       two_above = two_above + 256 * two_rank_bit_256
       two_rank_probe_128 = pl.tile.read(fin, [0, 2 * (two_above + 128 - 1)])
       two_rank_bit_128 = pl.cast(pl.cast(two_rank_probe_128 > two_cutoff, pl.INT32), pl.INDEX)
       two_above = two_above + 128 * two_rank_bit_128
       two_rank_probe_64 = pl.tile.read(fin, [0, 2 * (two_above + 64 - 1)])
       two_rank_bit_64 = pl.cast(pl.cast(two_rank_probe_64 > two_cutoff, pl.INT32), pl.INDEX)
       two_above = two_above + 64 * two_rank_bit_64
       two_rank_probe_32 = pl.tile.read(fin, [0, 2 * (two_above + 32 - 1)])
       two_rank_bit_32 = pl.cast(pl.cast(two_rank_probe_32 > two_cutoff, pl.INT32), pl.INDEX)
       two_above = two_above + 32 * two_rank_bit_32
       two_rank_probe_16 = pl.tile.read(fin, [0, 2 * (two_above + 16 - 1)])
       two_rank_bit_16 = pl.cast(pl.cast(two_rank_probe_16 > two_cutoff, pl.INT32), pl.INDEX)
       two_above = two_above + 16 * two_rank_bit_16
       two_rank_probe_8 = pl.tile.read(fin, [0, 2 * (two_above + 8 - 1)])
       two_rank_bit_8 = pl.cast(pl.cast(two_rank_probe_8 > two_cutoff, pl.INT32), pl.INDEX)
       two_above = two_above + 8 * two_rank_bit_8
       two_rank_probe_4 = pl.tile.read(fin, [0, 2 * (two_above + 4 - 1)])
       two_rank_bit_4 = pl.cast(pl.cast(two_rank_probe_4 > two_cutoff, pl.INT32), pl.INDEX)
       two_above = two_above + 4 * two_rank_bit_4
       two_rank_probe_2 = pl.tile.read(fin, [0, 2 * (two_above + 2 - 1)])
       two_rank_bit_2 = pl.cast(pl.cast(two_rank_probe_2 > two_cutoff, pl.INT32), pl.INDEX)
       two_above = two_above + 2 * two_rank_bit_2
       two_rank_probe_1 = pl.tile.read(fin, [0, 2 * (two_above + 1 - 1)])
       two_rank_bit_1 = pl.cast(pl.cast(two_rank_probe_1 > two_cutoff, pl.INT32), pl.INDEX)
       two_above = two_above + 1 * two_rank_bit_1
      pl.tile.write(publication_record, [0, t - cohort_base], two_cutoff)
      pl.tile.write(publication_record, [0, 4 + t - cohort_base], pl.cast(pl.cast(K - two_above, pl.INT32), pl.FP32))
      two_replay_safe = discarded < two_cutoff
      two_single = two_witness
      if discarded > two_cutoff:
       two_single = False
      if two_single:
       two_replay_safe = True
      else:
       if discarded == two_cutoff:
        if K - two_above <= 8:
         two_cert_result = pl.cast(-1, pl.INT32)
         two_tbits = pl.reinterpret_view(fin, pl.INT32)
         for two_tidx in pl.range(two_above, K):
          two_tpos = pl.tile.read(two_tbits, [0, 2 * two_tidx + 1])
          if pl.tile.read(fin, [0, 2 * two_tidx]) == two_cutoff:
           two_cert_result = pl.max(two_cert_result, two_tpos)
        else:
         two_cert_values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
         two_cert_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         two_cert_candidates = pl.tile.select(pl.tile.cmps(two_cert_values, two_cutoff, cmp_type=0), two_cert_positions, -1)
         two_cert_wide = pl.col_max(pl.reshape(two_cert_candidates, [8, 64]))
         two_cert_eight = pl.col_max(pl.reshape(two_cert_wide, [8, 8]))
         two_cert_result = pl.tile.read(two_cert_eight, [0, 0])
         two_cert_lane_1 = pl.tile.read(two_cert_eight, [0, 1])
         if two_cert_lane_1 > two_cert_result:
          two_cert_result = two_cert_lane_1
         two_cert_lane_2 = pl.tile.read(two_cert_eight, [0, 2])
         if two_cert_lane_2 > two_cert_result:
          two_cert_result = two_cert_lane_2
         two_cert_lane_3 = pl.tile.read(two_cert_eight, [0, 3])
         if two_cert_lane_3 > two_cert_result:
          two_cert_result = two_cert_lane_3
         two_cert_lane_4 = pl.tile.read(two_cert_eight, [0, 4])
         if two_cert_lane_4 > two_cert_result:
          two_cert_result = two_cert_lane_4
         two_cert_lane_5 = pl.tile.read(two_cert_eight, [0, 5])
         if two_cert_lane_5 > two_cert_result:
          two_cert_result = two_cert_lane_5
         two_cert_lane_6 = pl.tile.read(two_cert_eight, [0, 6])
         if two_cert_lane_6 > two_cert_result:
          two_cert_result = two_cert_lane_6
         two_cert_lane_7 = pl.tile.read(two_cert_eight, [0, 7])
         if two_cert_lane_7 > two_cert_result:
          two_cert_result = two_cert_lane_7
        if frontier_position > two_cert_result:
         two_replay_safe = True
      if two_replay_safe:
       if two_cutoff > FIN_FLOOR:
        two_root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        if two_single:
         two_endpoint = pl.tile.read(two_root_positions, [0, K - 1])
         if discarded == two_cutoff:
          two_endpoint = pl.min(two_endpoint, frontier_position)
         pl.tile.write(two_root_positions, [0, K - 1], two_endpoint)
        two_selected = pl.cast(K, pl.INDEX)
        if two_selected > 0:
         two_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(two_root_positions, pl.FP32))
         two_root_keys = two_root_priority
         two_root_order = pl.tile.sort32(two_root_keys, pl.reinterpret_view(two_root_positions, pl.UINT32))
         two_root_order = pl.tile.mrgsort(two_root_order, block_len=64)
         two_root_order = pl.tile.mrgsort(two_root_order, block_len=256)
         two_root_logical = pl.tile.gather_mask(two_root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         two_root_valid = pl.tile.set_validshape(two_root_logical, 1, two_selected)
         two_root_safe = pl.tile.fillpad(two_root_valid, pad_value=pl.PadValue.zero)
         two_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for two_map_start in pl.range(0, two_selected, MAP):
          two_map_count = pl.min(MAP, two_selected - two_map_start)
          two_map_bits = pl.tile.slice(pl.reinterpret_view(two_root_safe, pl.FP32), [1, MAP], [0, two_map_start])
          two_map_logical = pl.reinterpret_view(two_map_bits, pl.INT32)
          two_root_page_idx = pl.tile.shrs(two_map_logical, 7)
          two_root_page_ids = pl.tile.gather(materialize_displacements, two_root_page_idx, two_root_tmp)
          two_root_physical = pl.add(two_map_logical, two_root_page_ids)
          pl.store(pl.tile.set_validshape(two_root_physical, 1, two_map_count), [t, two_map_start], indices)
       else:
        two_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
        two_root = fin
        two_root_values = pl.tile.gather_mask(two_root, mask_pattern=pl.tile.MaskPattern.P0101)
        two_root_positions = pl.tile.gather_mask(two_root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        two_ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
        two_root_strict = pl.tile.select(pl.tile.cmps(two_root_values, two_cutoff, cmp_type=4), two_ones_root, 0.0)
        two_root_equal = pl.tile.select(pl.tile.cmps(two_root_values, two_cutoff, cmp_type=0), two_ones_root, 0.0)
        two_root_eligible = pl.maximum(two_root_strict, two_root_equal)
        two_root_visible = pl.tile.cmps(pl.cast(two_root_positions, pl.FP32), pl.cast(pl.cast(two_visible, pl.INT32), pl.FP32), cmp_type=2)
        two_root_keep = pl.tile.select(two_root_visible, two_root_eligible, 0.0)
        two_root_sums64 = pl.col_sum(pl.reshape(two_root_keep, [8, 64]))
        two_root_sums8 = pl.col_sum(pl.reshape(two_root_sums64, [8, 8]))
        two_selected = pl.cast(0, pl.INDEX)
        for two_root_lane in pl.unroll(8):
         two_root_part = pl.cast(pl.tile.read(two_root_sums8, [0, two_root_lane]), pl.INT32)
         two_selected = two_selected + pl.cast(two_root_part, pl.INDEX)
        pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
        if two_selected > 0:
         two_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(two_root_positions, pl.FP32))
         two_root_keys = pl.tile.select(pl.tile.cmps(two_root_keep, 0.0, cmp_type=4), two_root_priority, -1.0)
         two_root_order = pl.tile.sort32(two_root_keys, pl.reinterpret_view(two_root_positions, pl.UINT32))
         two_root_order = pl.tile.mrgsort(two_root_order, block_len=64)
         two_root_order = pl.tile.mrgsort(two_root_order, block_len=256)
         two_root_sorted_keys = pl.tile.gather_mask(two_root_order, mask_pattern=pl.tile.MaskPattern.P0101)
         two_root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), two_root_sorted_keys), pl.INT32)
         two_root_valid = pl.tile.set_validshape(two_root_logical, 1, two_selected)
         two_root_safe = pl.tile.fillpad(two_root_valid, pad_value=pl.PadValue.zero)
         two_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for two_map_start in pl.range(0, two_selected, MAP):
          two_map_count = pl.min(MAP, two_selected - two_map_start)
          two_map_bits = pl.tile.slice(pl.reinterpret_view(two_root_safe, pl.FP32), [1, MAP], [0, two_map_start])
          two_map_logical = pl.reinterpret_view(two_map_bits, pl.INT32)
          two_root_page_idx = pl.tile.shrs(two_map_logical, 7)
          two_root_page_ids = pl.tile.gather(materialize_displacements, two_root_page_idx, two_root_tmp)
          two_root_physical = pl.add(two_map_logical, two_root_page_ids)
          pl.store(pl.tile.set_validshape(two_root_physical, 1, two_map_count), [t, two_map_start], indices)
      else:
       two_bulk_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
       two_bulk_quota = pl.cast(K - two_above, pl.INT32)
       pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
       two_bulk_raw = pl.load(scores, [t, 0], [1, 4096], valid_shape=[1, N])
       two_bulk_values = pl.reshape(pl.tile.fillpad(two_bulk_raw, pad_value=pl.PadValue.min), [32, SCAN])
       two_bulk_positions = pl.reshape(pl.tile.arange(0, [1, 4096], dtype=pl.INT32), [32, SCAN])
       two_bulk_valid = pl.tile.cmps(pl.cast(two_bulk_positions, pl.FP32), pl.cast(pl.cast(two_bulk_visible, pl.INT32), pl.FP32), cmp_type=2)
       two_bulk_strict = pl.tile.select(two_bulk_valid, pl.tile.select(pl.tile.cmps(two_bulk_values, two_cutoff, cmp_type=4), pl.tile.full([32, SCAN], dtype=pl.FP32, value=1.0), 0.0), 0.0)
       two_bulk_strict_scratch = pl.tile.create([32, SCAN], dtype=pl.FP32)
       two_bulk_count_offsets = pl.mul(pl.tile.arange(0, [1, 32], dtype=pl.INT32), 8)
       two_bulk_count_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
       two_bulk_equal = pl.tile.select(two_bulk_valid, pl.tile.select(pl.tile.cmps(two_bulk_values, two_cutoff, cmp_type=0), pl.tile.full([32, SCAN], dtype=pl.FP32, value=1.0), 0.0), 0.0)
       two_bulk_packed_flags = pl.add(pl.mul(two_bulk_strict, 8192.0), two_bulk_equal)
       two_bulk_packed_reduced = pl.row_sum(two_bulk_packed_flags, two_bulk_strict_scratch)
       two_bulk_packed_carrier = pl.row_expand_add(pl.tile.full([32, 8], dtype=pl.FP32, value=0.0), two_bulk_packed_reduced)
       two_bulk_packed_counts = pl.tile.gather(pl.reshape(two_bulk_packed_carrier, [1, 256]), two_bulk_count_offsets, two_bulk_count_scratch)
       two_bulk_blocks = pl.tile.arange(0, [1, 32], dtype=pl.INT32)
       two_bulk_prefix_scratch = pl.tile.create([1, 32], dtype=pl.INT32)
       two_bulk_packed_prefix = two_bulk_packed_counts
       two_bulk_shift_index = pl.maximum(pl.sub(two_bulk_blocks, 1), 0)
       two_bulk_shift_valid = pl.tile.cmps(pl.cast(two_bulk_blocks, pl.FP32), 1.0, cmp_type=5)
       two_bulk_packed_shift = pl.tile.gather(two_bulk_packed_prefix, two_bulk_shift_index, two_bulk_prefix_scratch)
       two_bulk_packed_prefix = pl.add(two_bulk_packed_prefix, pl.tile.select(two_bulk_shift_valid, two_bulk_packed_shift, 0.0))
       two_bulk_shift_index = pl.maximum(pl.sub(two_bulk_blocks, 2), 0)
       two_bulk_shift_valid = pl.tile.cmps(pl.cast(two_bulk_blocks, pl.FP32), 2.0, cmp_type=5)
       two_bulk_packed_shift = pl.tile.gather(two_bulk_packed_prefix, two_bulk_shift_index, two_bulk_prefix_scratch)
       two_bulk_packed_prefix = pl.add(two_bulk_packed_prefix, pl.tile.select(two_bulk_shift_valid, two_bulk_packed_shift, 0.0))
       two_bulk_shift_index = pl.maximum(pl.sub(two_bulk_blocks, 4), 0)
       two_bulk_shift_valid = pl.tile.cmps(pl.cast(two_bulk_blocks, pl.FP32), 4.0, cmp_type=5)
       two_bulk_packed_shift = pl.tile.gather(two_bulk_packed_prefix, two_bulk_shift_index, two_bulk_prefix_scratch)
       two_bulk_packed_prefix = pl.add(two_bulk_packed_prefix, pl.tile.select(two_bulk_shift_valid, two_bulk_packed_shift, 0.0))
       two_bulk_shift_index = pl.maximum(pl.sub(two_bulk_blocks, 8), 0)
       two_bulk_shift_valid = pl.tile.cmps(pl.cast(two_bulk_blocks, pl.FP32), 8.0, cmp_type=5)
       two_bulk_packed_shift = pl.tile.gather(two_bulk_packed_prefix, two_bulk_shift_index, two_bulk_prefix_scratch)
       two_bulk_packed_prefix = pl.add(two_bulk_packed_prefix, pl.tile.select(two_bulk_shift_valid, two_bulk_packed_shift, 0.0))
       two_bulk_shift_index = pl.maximum(pl.sub(two_bulk_blocks, 16), 0)
       two_bulk_shift_valid = pl.tile.cmps(pl.cast(two_bulk_blocks, pl.FP32), 16.0, cmp_type=5)
       two_bulk_packed_shift = pl.tile.gather(two_bulk_packed_prefix, two_bulk_shift_index, two_bulk_prefix_scratch)
       two_bulk_packed_prefix = pl.add(two_bulk_packed_prefix, pl.tile.select(two_bulk_shift_valid, two_bulk_packed_shift, 0.0))
       two_bulk_strict_counts = pl.cast(pl.cast(pl.mul(two_bulk_packed_counts, 0.0001220703125), pl.INT32, mode='floor'), pl.FP32)
       two_bulk_equal_counts = pl.sub(two_bulk_packed_counts, pl.mul(two_bulk_strict_counts, 8192.0))
       two_bulk_strict_prefix = pl.cast(pl.cast(pl.mul(two_bulk_packed_prefix, 0.0001220703125), pl.INT32, mode='floor'), pl.FP32)
       two_bulk_equal_prefix = pl.sub(two_bulk_packed_prefix, pl.mul(two_bulk_strict_prefix, 8192.0))
       two_bulk_strict_before = pl.sub(two_bulk_strict_prefix, two_bulk_strict_counts)
       two_bulk_equal_before = pl.sub(two_bulk_equal_prefix, two_bulk_equal_counts)
       two_bulk_tie_take = pl.minimum(two_bulk_equal_counts, pl.maximum(pl.sub(pl.add(pl.tile.full([1, 32], dtype=pl.FP32, value=0.0), pl.cast(two_bulk_quota, pl.FP32)), two_bulk_equal_before), 0.0))
       two_bulk_output_begin = pl.add(two_bulk_strict_before, pl.minimum(two_bulk_equal_before, pl.cast(two_bulk_quota, pl.FP32)))
       two_bulk_output_count = pl.add(two_bulk_strict_counts, two_bulk_tie_take)
       two_bulk_lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
       two_bulk_lanes_f = pl.cast(two_bulk_lanes, pl.FP32)
       two_bulk_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
       two_bulk_priority = pl.cast(pl.sub(two_bulk_origin, two_bulk_lanes), pl.FP32)
       two_bulk_map_scratch = pl.tile.create([1, SCAN], dtype=pl.INT32)
       for two_bulk_block in pl.range(0, (two_bulk_visible + SCAN - 1) // SCAN):
         two_bulk_selected = pl.cast(pl.cast(pl.tile.read(two_bulk_output_count, [0, two_bulk_block]), pl.INT32), pl.INDEX)
         if two_bulk_selected > 0:
           two_bulk_start = two_bulk_block * SCAN
           two_bulk_count = pl.cast(pl.cast(pl.tile.read(two_bulk_output_begin, [0, two_bulk_block]), pl.INT32), pl.INDEX)
           two_bulk_take = pl.cast(pl.cast(pl.tile.read(two_bulk_tie_take, [0, two_bulk_block]), pl.INT32), pl.INDEX)
           two_bulk_available = pl.cast(pl.cast(pl.tile.read(two_bulk_equal_counts, [0, two_bulk_block]), pl.INT32), pl.INDEX)
           two_bulk_block_flags = pl.tile.slice(pl.reshape(two_bulk_packed_flags, [1, 4096]), [1, SCAN], [0, two_bulk_start])
           two_bulk_block_ones = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
           two_bulk_block_strict = pl.tile.select(pl.tile.cmps(two_bulk_block_flags, 8192.0, cmp_type=0), two_bulk_block_ones, 0.0)
           two_bulk_block_equal = pl.tile.select(pl.tile.cmps(two_bulk_block_flags, 1.0, cmp_type=0), two_bulk_block_ones, 0.0)
           two_bulk_ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
           if two_bulk_take > 0:
             if two_bulk_take == two_bulk_available:
               two_bulk_ties = two_bulk_block_equal
             else:
               two_bulk_tie_keys = pl.mul(two_bulk_block_equal, two_bulk_priority)
               two_bulk_tie_pairs = pl.tile.sort32(two_bulk_tie_keys, pl.reinterpret_view(two_bulk_lanes, pl.UINT32))
               two_bulk_tie_pairs = pl.tile.mrgsort(two_bulk_tie_pairs, block_len=64)
               two_bulk_equal_keys = pl.tile.gather_mask(two_bulk_tie_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
               two_bulk_last_key = pl.tile.read(two_bulk_equal_keys, [0, two_bulk_take - 1])
               two_bulk_boundary = 128.0 - two_bulk_last_key
               two_bulk_ties = pl.tile.select(pl.tile.cmps(two_bulk_lanes_f, two_bulk_boundary, cmp_type=3), two_bulk_block_equal, 0.0)
           two_bulk_keep = pl.maximum(two_bulk_block_strict, two_bulk_ties)
           two_bulk_order_keys = pl.mul(two_bulk_keep, two_bulk_priority)
           two_bulk_order_pairs = pl.tile.sort32(two_bulk_order_keys, pl.reinterpret_view(two_bulk_lanes, pl.UINT32))
           two_bulk_order_pairs = pl.tile.mrgsort(two_bulk_order_pairs, block_len=64)
           two_bulk_order_values = pl.tile.gather_mask(two_bulk_order_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
           two_bulk_offsets = pl.sub(two_bulk_origin, pl.cast(two_bulk_order_values, pl.INT32))
           two_bulk_compact_valid = pl.tile.set_validshape(two_bulk_offsets, 1, two_bulk_selected)
           two_bulk_compact_safe = pl.tile.fillpad(two_bulk_compact_valid, pad_value=pl.PadValue.zero)
           two_bulk_logical = pl.add(two_bulk_compact_safe, pl.cast(two_bulk_start, pl.INT32))
           two_bulk_page_index = pl.tile.shrs(two_bulk_logical, 7)
           two_bulk_page_ids = pl.tile.gather(materialize_displacements, two_bulk_page_index, two_bulk_map_scratch)
           two_bulk_physical = pl.add(two_bulk_logical, two_bulk_page_ids)
           pl.store(pl.tile.set_validshape(two_bulk_physical, 1, two_bulk_selected), [t, two_bulk_count], indices)
    pl.store(publication_record, [merge_cohort_job, 0], publication_bank)
   cutoff_output = pl.reshape(cutoffs, [1, M])
   quota_output = pl.reshape(quotas, [1, M])
   for pack_task in pl.spmd(1, name_hint='pack_auxiliary'):
    for output_row in pl.range(0, M, 64):
     record_lines = pl.load(publication_bank, [output_row // 4, 0], [16, 128])
     record_flat = pl.reshape(record_lines, [1, 2048])
     pack_lanes = pl.tile.arange(0, [1, 64], dtype=pl.INT32)
     pack_cohorts = pl.tile.shrs(pack_lanes, 2)
     pack_local = pl.sub(pack_lanes, pl.mul(pack_cohorts, 4))
     cutoff_offsets = pl.add(pl.mul(pack_cohorts, 128), pack_local)
     quota_offsets = pl.add(cutoff_offsets, 4)
     cutoff_scratch = pl.tile.create([1, 2048], dtype=pl.INT32)
     quota_scratch = pl.tile.create([1, 2048], dtype=pl.INT32)
     cutoff_packed = pl.tile.gather(record_flat, cutoff_offsets, cutoff_scratch)
     quota_encoded = pl.tile.gather(record_flat, quota_offsets, quota_scratch)
     quota_packed = pl.cast(quota_encoded, pl.INT32)
     pl.store(cutoff_packed, [0, output_row], cutoff_output)
     pl.store(quota_packed, [0, output_row], quota_output)
   return indices
 elif NL == 3:

  @pl.jit
  def dsa41_topk_guarded_root(scores: pl.Tensor[[M, N], pl.FP32], lengths: pl.Tensor[[M], pl.INT32], block_table: pl.Tensor[[1, P], pl.INT32], wpairs: pl.InOut[pl.Tensor[[M, WORKSPACE_WIDTH], pl.FP32]], cutoffs: pl.InOut[pl.Tensor[[M, 1], pl.FP32]], quotas: pl.InOut[pl.Tensor[[M, 1], pl.INT32]], indices: pl.Out[pl.Tensor[[M, K], pl.INT32]]):
   aux_discarded = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_cutoffs = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_quotas = pl.create_tensor([M, 16], dtype=pl.INT32)
   aux_front = pl.create_tensor([M, 16], dtype=pl.INT32)
   leaf_discarded_bank = pl.create_tensor([M, NL * 16], dtype=pl.FP32)
   leaf_front_bank = pl.create_tensor([M, NL * 16], dtype=pl.INT32)
   for leaf_job in pl.spmd(LEAF_COHORTS * NL, name_hint='canonical_leaf_plane'):
    leaf = leaf_job // LEAF_COHORTS
    leaf_cohort_base = leaf_job % LEAF_COHORTS * LEAF_ROWS
    for t in pl.range(leaf_cohort_base, pl.min(leaf_cohort_base + LEAF_ROWS, M), 1):
     zero_pairs = pl.tile.full([1, PAIR_W], dtype=pl.FP32, value=0.0)
     if leaf == 0:
      for clear_offset in pl.range(NL * PAIR_W, WORKSPACE_WIDTH, PAIR_W):
       pl.store(zero_pairs, [t, clear_offset], wpairs)
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     begin = leaf * LEAF
     if begin < wv:
      valid = pl.min(LEAF, wv - begin)
      if valid <= 512:
       narrow_raw = pl.load(scores, [t, begin], [1, 512], valid_shape=[1, valid])
       narrow_padded = pl.tile.fillpad(narrow_raw, pad_value=pl.PadValue.min)
       narrow_values = pl.maximum(narrow_padded, SORT_FLOOR)
       narrow_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       narrow_pairs = pl.tile.sort32(narrow_values, pl.reinterpret_view(narrow_positions, pl.UINT32))
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=64)
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=256)
       pl.store(narrow_pairs, [t, leaf * PAIR_W], wpairs)
      elif valid <= 1024:
       left_valid = pl.min(valid, 512)
       right_valid = pl.max(valid - 512, 0)
       left_raw = pl.load(scores, [t, begin], [1, 512], valid_shape=[1, left_valid])
       right_raw = pl.load(scores, [t, begin + 512], [1, 512], valid_shape=[1, right_valid])
       left_values = pl.maximum(pl.tile.fillpad(left_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       right_values = pl.maximum(pl.tile.fillpad(right_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       left_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       right_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin + 512, pl.INT32))
       left_pairs = pl.tile.sort32(left_values, pl.reinterpret_view(left_positions, pl.UINT32))
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=64)
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=256)
       right_pairs = pl.tile.sort32(right_values, pl.reinterpret_view(right_positions, pl.UINT32))
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=64)
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=256)
       native_scratch = pl.tile.create([1, 2048], dtype=pl.FP32)
       native_pairs = pl.tile.mrgsort(left_pairs, right_pairs, tmp=native_scratch)
       native_discarded = pl.tile.read(native_pairs, [0, PAIR_W])
       if native_discarded > discarded:
        discarded = native_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if native_discarded == discarded:
        native_front_boundary = pl.tile.read(native_pairs, [0, PAIR_W - 2])
        native_front_bound = native_front_boundary
        if native_front_bound < FIN_FLOOR:
         native_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if native_discarded >= native_front_bound:
         native_arity_third = pl.tile.read(native_pairs, [0, PAIR_W + 4])
         if native_arity_third < native_discarded:
          native_arity_second = pl.tile.read(native_pairs, [0, PAIR_W + 2])
          native_arity_bits = pl.reinterpret_view(native_pairs, pl.INT32)
          native_front_result = pl.tile.read(native_arity_bits, [0, PAIR_W + 1])
          if native_arity_second == native_discarded:
           native_arity_second_position = pl.tile.read(native_arity_bits, [0, PAIR_W + 3])
           native_front_result = pl.min(native_front_result, native_arity_second_position)
         else:
          native_front_tail = pl.tile.slice(native_pairs, [1, 1024], [0, PAIR_W])
          native_front_values = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
          native_front_positions = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          native_front_candidates = pl.tile.select(pl.tile.cmps(native_front_values, native_discarded, cmp_type=0), native_front_positions, 2147483647)
          native_front_wide = pl.col_min(pl.reshape(native_front_candidates, [8, 64]))
          native_front_eight = pl.col_min(pl.reshape(native_front_wide, [8, 8]))
          native_front_result = pl.tile.read(native_front_eight, [0, 0])
          native_front_lane_1 = pl.tile.read(native_front_eight, [0, 1])
          if native_front_lane_1 < native_front_result:
           native_front_result = native_front_lane_1
          native_front_lane_2 = pl.tile.read(native_front_eight, [0, 2])
          if native_front_lane_2 < native_front_result:
           native_front_result = native_front_lane_2
          native_front_lane_3 = pl.tile.read(native_front_eight, [0, 3])
          if native_front_lane_3 < native_front_result:
           native_front_result = native_front_lane_3
          native_front_lane_4 = pl.tile.read(native_front_eight, [0, 4])
          if native_front_lane_4 < native_front_result:
           native_front_result = native_front_lane_4
          native_front_lane_5 = pl.tile.read(native_front_eight, [0, 5])
          if native_front_lane_5 < native_front_result:
           native_front_result = native_front_lane_5
          native_front_lane_6 = pl.tile.read(native_front_eight, [0, 6])
          if native_front_lane_6 < native_front_result:
           native_front_result = native_front_lane_6
          native_front_lane_7 = pl.tile.read(native_front_eight, [0, 7])
          if native_front_lane_7 < native_front_result:
           native_front_result = native_front_lane_7
         if native_front_result < frontier_position:
          frontier_position = native_front_result
       pl.store(pl.tile.slice(native_pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
      else:
       raw = pl.load(scores, [t, begin], [1, LEAF], valid_shape=[1, valid])
       padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
       flr = pl.tile.full([1, LEAF], dtype=pl.FP32, value=SORT_FLOOR)
       vals = pl.maximum(padded, flr)
       idx_i32 = pl.add(pl.tile.arange(0, [1, LEAF], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       pairs = pl.tile.sort32(vals, pl.reinterpret_view(idx_i32, pl.UINT32))
       pairs = pl.tile.mrgsort(pairs, block_len=64)
       pairs = pl.tile.mrgsort(pairs, block_len=256)
       pairs = pl.tile.mrgsort(pairs, block_len=1024)
       leaf_discarded = pl.tile.read(pairs, [0, PAIR_W])
       if leaf_discarded > discarded:
        discarded = leaf_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if leaf_discarded == discarded:
        leaf_front_boundary = pl.tile.read(pairs, [0, PAIR_W - 2])
        leaf_front_bound = leaf_front_boundary
        if leaf_front_bound < FIN_FLOOR:
         leaf_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if leaf_discarded >= leaf_front_bound:
         leaf_arity_third = pl.tile.read(pairs, [0, PAIR_W + 4])
         if leaf_arity_third < leaf_discarded:
          leaf_arity_second = pl.tile.read(pairs, [0, PAIR_W + 2])
          leaf_arity_bits = pl.reinterpret_view(pairs, pl.INT32)
          leaf_front_result = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 1])
          if leaf_arity_second == leaf_discarded:
           leaf_arity_second_position = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 3])
           leaf_front_result = pl.min(leaf_front_result, leaf_arity_second_position)
         else:
          leaf_front_tail = pl.tile.slice(pairs, [1, 3072], [0, PAIR_W])
          leaf_front_values = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
          leaf_front_positions = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          leaf_front_candidates = pl.tile.select(pl.tile.cmps(leaf_front_values, leaf_discarded, cmp_type=0), leaf_front_positions, 2147483647)
          leaf_front_wide = pl.col_min(pl.reshape(leaf_front_candidates, [24, 64]))
          leaf_front_eight = pl.col_min(pl.reshape(leaf_front_wide, [8, 8]))
          leaf_front_result = pl.tile.read(leaf_front_eight, [0, 0])
          leaf_front_lane_1 = pl.tile.read(leaf_front_eight, [0, 1])
          if leaf_front_lane_1 < leaf_front_result:
           leaf_front_result = leaf_front_lane_1
          leaf_front_lane_2 = pl.tile.read(leaf_front_eight, [0, 2])
          if leaf_front_lane_2 < leaf_front_result:
           leaf_front_result = leaf_front_lane_2
          leaf_front_lane_3 = pl.tile.read(leaf_front_eight, [0, 3])
          if leaf_front_lane_3 < leaf_front_result:
           leaf_front_result = leaf_front_lane_3
          leaf_front_lane_4 = pl.tile.read(leaf_front_eight, [0, 4])
          if leaf_front_lane_4 < leaf_front_result:
           leaf_front_result = leaf_front_lane_4
          leaf_front_lane_5 = pl.tile.read(leaf_front_eight, [0, 5])
          if leaf_front_lane_5 < leaf_front_result:
           leaf_front_result = leaf_front_lane_5
          leaf_front_lane_6 = pl.tile.read(leaf_front_eight, [0, 6])
          if leaf_front_lane_6 < leaf_front_result:
           leaf_front_result = leaf_front_lane_6
          leaf_front_lane_7 = pl.tile.read(leaf_front_eight, [0, 7])
          if leaf_front_lane_7 < leaf_front_result:
           leaf_front_result = leaf_front_lane_7
         if leaf_front_result < frontier_position:
          frontier_position = leaf_front_result
       pl.store(pl.tile.slice(pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
     else:
      pl.store(zero_pairs, [t, leaf * PAIR_W], wpairs)
     leaf_score_line = pl.tile.full([1, 16], dtype=pl.FP32, value=SORT_FLOOR)
     leaf_position_line = pl.tile.full([1, 16], dtype=pl.INT32, value=2147483647)
     pl.tile.write(leaf_score_line, [0, 0], discarded)
     pl.tile.write(leaf_position_line, [0, 0], frontier_position)
     pl.store(leaf_score_line, [t, leaf * 16], leaf_discarded_bank)
     pl.store(leaf_position_line, [t, leaf * 16], leaf_front_bank)
   for merge_cohort_job in pl.spmd((M + 4 - 1) // 4, name_hint='bounded_root_handoff'):
    cohort_base = merge_cohort_job * 4
    if P == 1:
     materialize_single_page = pl.read(block_table, [0, 0])
     materialize_pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), materialize_single_page), pad_value=pl.PadValue.zero)
    else:
     materialize_pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
    materialize_page_lanes = pl.tile.arange(0, [1, P8], dtype=pl.INT32)
    materialize_displacements = pl.mul(pl.sub(materialize_pages, materialize_page_lanes), PAGE)
    for t in pl.range(cohort_base, pl.min(cohort_base + 4, M), 1):
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     lc = (wv + LEAF - 1) // LEAF
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     for proof_leaf in pl.range(NL):
      proof_score = pl.read(leaf_discarded_bank, [t, proof_leaf * 16])
      proof_position = pl.read(leaf_front_bank, [t, proof_leaf * 16])
      if proof_score > discarded:
       discarded = proof_score
       frontier_position = pl.cast(2147483647, pl.INT32)
      if proof_score == discarded:
       if proof_position < frontier_position:
        frontier_position = proof_position
     if lc == 1:
      fin = pl.load(wpairs, [t, 0], [1, PAIR_W])
      one_values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
      one_finite = pl.maximum(one_values, FIN_FLOOR)
      one_cutoff = pl.tile.read(one_finite, [0, K - 1])
      one_ones_c = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
      one_flags = pl.tile.select(pl.tile.cmps(one_values, one_cutoff, cmp_type=4), one_ones_c, 0.0)
      one_sums64 = pl.col_sum(pl.reshape(one_flags, [8, 64]))
      one_sums8 = pl.col_sum(pl.reshape(one_sums64, [8, 8]))
      one_above = pl.cast(0, pl.INDEX)
      for one_lane in pl.unroll(8):
       one_part = pl.cast(pl.tile.read(one_sums8, [0, one_lane]), pl.INT32)
       one_above = one_above + pl.cast(one_part, pl.INDEX)
      one_cutoff_stage_line = pl.tile.full([1, 16], dtype=pl.FP32, value=0)
      pl.tile.write(one_cutoff_stage_line, [0, 0], one_cutoff)
      pl.store(one_cutoff_stage_line, [t, 0], aux_cutoffs)
      one_quota_stage_line = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
      pl.tile.write(one_quota_stage_line, [0, 0], pl.cast(K - one_above, pl.INT32))
      pl.store(one_quota_stage_line, [t, 0], aux_quotas)
      one_replay_safe = discarded < one_cutoff
      if discarded == one_cutoff:
       one_cert_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       one_cert_candidates = pl.tile.select(pl.tile.cmps(one_values, one_cutoff, cmp_type=0), one_cert_positions, -1)
       one_cert_wide = pl.col_max(pl.reshape(one_cert_candidates, [8, 64]))
       one_cert_eight = pl.col_max(pl.reshape(one_cert_wide, [8, 8]))
       one_cert_result = pl.tile.read(one_cert_eight, [0, 0])
       one_cert_lane_1 = pl.tile.read(one_cert_eight, [0, 1])
       if one_cert_lane_1 > one_cert_result:
        one_cert_result = one_cert_lane_1
       one_cert_lane_2 = pl.tile.read(one_cert_eight, [0, 2])
       if one_cert_lane_2 > one_cert_result:
        one_cert_result = one_cert_lane_2
       one_cert_lane_3 = pl.tile.read(one_cert_eight, [0, 3])
       if one_cert_lane_3 > one_cert_result:
        one_cert_result = one_cert_lane_3
       one_cert_lane_4 = pl.tile.read(one_cert_eight, [0, 4])
       if one_cert_lane_4 > one_cert_result:
        one_cert_result = one_cert_lane_4
       one_cert_lane_5 = pl.tile.read(one_cert_eight, [0, 5])
       if one_cert_lane_5 > one_cert_result:
        one_cert_result = one_cert_lane_5
       one_cert_lane_6 = pl.tile.read(one_cert_eight, [0, 6])
       if one_cert_lane_6 > one_cert_result:
        one_cert_result = one_cert_lane_6
       one_cert_lane_7 = pl.tile.read(one_cert_eight, [0, 7])
       if one_cert_lane_7 > one_cert_result:
        one_cert_result = one_cert_lane_7
       if frontier_position > one_cert_result:
        one_replay_safe = True
      if one_replay_safe:
       if one_cutoff > FIN_FLOOR:
        one_root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        one_selected = pl.cast(K, pl.INDEX)
        if one_selected > 0:
         one_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(one_root_positions, pl.FP32))
         one_root_keys = one_root_priority
         one_root_order = pl.tile.sort32(one_root_keys, pl.reinterpret_view(one_root_positions, pl.UINT32))
         one_root_order = pl.tile.mrgsort(one_root_order, block_len=64)
         one_root_order = pl.tile.mrgsort(one_root_order, block_len=256)
         one_root_logical = pl.tile.gather_mask(one_root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         one_root_valid = pl.tile.set_validshape(one_root_logical, 1, one_selected)
         one_root_safe = pl.tile.fillpad(one_root_valid, pad_value=pl.PadValue.zero)
         one_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for one_map_start in pl.range(0, one_selected, MAP):
          one_map_count = pl.min(MAP, one_selected - one_map_start)
          one_map_bits = pl.tile.slice(pl.reinterpret_view(one_root_safe, pl.FP32), [1, MAP], [0, one_map_start])
          one_map_logical = pl.reinterpret_view(one_map_bits, pl.INT32)
          one_root_page_idx = pl.tile.shrs(one_map_logical, 7)
          one_root_page_ids = pl.tile.gather(materialize_displacements, one_root_page_idx, one_root_tmp)
          one_root_physical = pl.add(one_map_logical, one_root_page_ids)
          pl.store(pl.tile.set_validshape(one_root_physical, 1, one_map_count), [t, one_map_start], indices)
       else:
        one_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
        one_root = fin
        one_root_values = pl.tile.gather_mask(one_root, mask_pattern=pl.tile.MaskPattern.P0101)
        one_root_positions = pl.tile.gather_mask(one_root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        one_ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
        one_root_strict = pl.tile.select(pl.tile.cmps(one_root_values, one_cutoff, cmp_type=4), one_ones_root, 0.0)
        one_root_equal = pl.tile.select(pl.tile.cmps(one_root_values, one_cutoff, cmp_type=0), one_ones_root, 0.0)
        one_root_eligible = pl.maximum(one_root_strict, one_root_equal)
        one_root_visible = pl.tile.cmps(pl.cast(one_root_positions, pl.FP32), pl.cast(pl.cast(one_visible, pl.INT32), pl.FP32), cmp_type=2)
        one_root_keep = pl.tile.select(one_root_visible, one_root_eligible, 0.0)
        one_root_sums64 = pl.col_sum(pl.reshape(one_root_keep, [8, 64]))
        one_root_sums8 = pl.col_sum(pl.reshape(one_root_sums64, [8, 8]))
        one_selected = pl.cast(0, pl.INDEX)
        for one_root_lane in pl.unroll(8):
         one_root_part = pl.cast(pl.tile.read(one_root_sums8, [0, one_root_lane]), pl.INT32)
         one_selected = one_selected + pl.cast(one_root_part, pl.INDEX)
        pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
        if one_selected > 0:
         one_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(one_root_positions, pl.FP32))
         one_root_keys = pl.tile.select(pl.tile.cmps(one_root_keep, 0.0, cmp_type=4), one_root_priority, -1.0)
         one_root_order = pl.tile.sort32(one_root_keys, pl.reinterpret_view(one_root_positions, pl.UINT32))
         one_root_order = pl.tile.mrgsort(one_root_order, block_len=64)
         one_root_order = pl.tile.mrgsort(one_root_order, block_len=256)
         one_root_sorted_keys = pl.tile.gather_mask(one_root_order, mask_pattern=pl.tile.MaskPattern.P0101)
         one_root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), one_root_sorted_keys), pl.INT32)
         one_root_valid = pl.tile.set_validshape(one_root_logical, 1, one_selected)
         one_root_safe = pl.tile.fillpad(one_root_valid, pad_value=pl.PadValue.zero)
         one_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for one_map_start in pl.range(0, one_selected, MAP):
          one_map_count = pl.min(MAP, one_selected - one_map_start)
          one_map_bits = pl.tile.slice(pl.reinterpret_view(one_root_safe, pl.FP32), [1, MAP], [0, one_map_start])
          one_map_logical = pl.reinterpret_view(one_map_bits, pl.INT32)
          one_root_page_idx = pl.tile.shrs(one_map_logical, 7)
          one_root_page_ids = pl.tile.gather(materialize_displacements, one_root_page_idx, one_root_tmp)
          one_root_physical = pl.add(one_map_logical, one_root_page_ids)
          pl.store(pl.tile.set_validshape(one_root_physical, 1, one_map_count), [t, one_map_start], indices)
      else:
       one_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
       one_remaining = pl.cast(K - one_above, pl.INDEX)
       pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
       one_ones_m = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
       one_lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
       one_lanes_f = pl.cast(one_lanes, pl.FP32)
       one_lane_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
       one_priority = pl.cast(pl.sub(one_lane_origin, one_lanes), pl.FP32)
       one_gtmp = pl.tile.create([1, SCAN], dtype=pl.INT32)
       one_count = pl.cast(0, pl.INDEX)
       for one_start in pl.range(0, one_visible, SCAN):
        if one_count < K:
         one_valid = pl.min(SCAN, one_visible - one_start)
         one_scan_raw = pl.load(scores, [t, one_start], [1, SCAN], valid_shape=[1, one_valid])
         one_scan_vals = pl.tile.fillpad(one_scan_raw, pad_value=pl.PadValue.min)
         one_strict = pl.tile.select(pl.tile.cmps(one_scan_vals, one_cutoff, cmp_type=4), one_ones_m, 0.0)
         one_ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
         if one_remaining > 0:
          one_eq = pl.tile.select(pl.tile.cmps(one_scan_vals, one_cutoff, cmp_type=0), one_ones_m, 0.0)
          one_eq_valid = pl.tile.cmps(one_lanes_f, pl.cast(pl.cast(one_valid, pl.INT32), pl.FP32), cmp_type=2)
          one_eq = pl.tile.select(one_eq_valid, one_eq, 0.0)
          one_eq_sums64 = pl.col_sum(pl.reshape(one_eq, [SCAN // 64, 64]))
          one_eq_sums8 = pl.col_sum(pl.reshape(one_eq_sums64, [8, 8]))
          one_available = pl.cast(0, pl.INDEX)
          for one_eq_lane in pl.unroll(8):
           one_eq_part = pl.cast(pl.tile.read(one_eq_sums8, [0, one_eq_lane]), pl.INT32)
           one_available = one_available + pl.cast(one_eq_part, pl.INDEX)
          one_take = pl.min(one_available, one_remaining)
          if one_take > 0:
           if one_take == one_available:
            one_ties = one_eq
           else:
            one_tie_keys = pl.mul(one_eq, one_priority)
            one_tie_pairs = pl.tile.sort32(one_tie_keys, pl.reinterpret_view(one_lanes, pl.UINT32))
            one_tie_pairs = pl.tile.mrgsort(one_tie_pairs, block_len=64)
            one_tie_merged = one_tie_pairs
            one_equal_keys = pl.tile.gather_mask(one_tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
            one_last_key_f = pl.tile.read(one_equal_keys, [0, one_take - 1])
            one_boundary_f = 128.0 - one_last_key_f
            one_ties = pl.tile.select(pl.tile.cmps(one_lanes_f, one_boundary_f, cmp_type=3), one_eq, 0.0)
           one_remaining = one_remaining - one_take
         one_keep = pl.maximum(one_strict, one_ties)
         one_keep_valid = pl.tile.set_validshape(one_keep, 1, one_valid)
         one_keep_padded = pl.tile.fillpad(one_keep_valid, pad_value=pl.PadValue.zero)
         one_keep_sums64 = pl.col_sum(pl.reshape(one_keep_padded, [SCAN // 64, 64]))
         one_keep_sums8 = pl.col_sum(pl.reshape(one_keep_sums64, [8, 8]))
         one_keep_count = pl.cast(0, pl.INDEX)
         for one_keep_lane in pl.unroll(8):
          one_keep_part = pl.cast(pl.tile.read(one_keep_sums8, [0, one_keep_lane]), pl.INT32)
          one_keep_count = one_keep_count + pl.cast(one_keep_part, pl.INDEX)
         one_selected = pl.min(one_keep_count, K - one_count)
         if one_selected > 0:
          one_order_keys = pl.mul(one_keep_padded, one_priority)
          one_order_pairs = pl.tile.sort32(one_order_keys, pl.reinterpret_view(one_lanes, pl.UINT32))
          one_order_pairs = pl.tile.mrgsort(one_order_pairs, block_len=64)
          one_order_merged = one_order_pairs
          one_order_vals = pl.tile.gather_mask(one_order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
          one_offs = pl.sub(one_lane_origin, pl.cast(one_order_vals, pl.INT32))
          one_compact_valid = pl.tile.set_validshape(one_offs, 1, one_selected)
          one_compact_safe = pl.tile.fillpad(one_compact_valid, pad_value=pl.PadValue.zero)
          one_logical = pl.add(one_compact_safe, pl.cast(one_start, pl.INT32))
          one_page_idx = pl.tile.shrs(one_logical, 7)
          one_page_ids = pl.tile.gather(materialize_displacements, one_page_idx, one_gtmp)
          one_physical = pl.add(one_logical, one_page_ids)
          pl.store(pl.tile.set_validshape(one_physical, 1, one_selected), [t, one_count], indices)
          one_count = one_count + one_selected
     elif lc == 2:
      handoff_scratch = pl.tile.create([1, 2 * PAIR_W], dtype=pl.FP32)
      handoff_left = pl.load(wpairs, [t, 0], [1, PAIR_W])
      handoff_right = pl.load(wpairs, [t, PAIR_W], [1, PAIR_W])
      mboth = pl.tile.mrgsort(handoff_left, handoff_right, tmp=handoff_scratch)
      merge_discarded = pl.tile.read(mboth, [0, PAIR_W])
      if merge_discarded > discarded:
       discarded = merge_discarded
       frontier_position = pl.cast(2147483647, pl.INT32)
      if merge_discarded == discarded:
       merge_front_boundary = pl.tile.read(mboth, [0, PAIR_W - 2])
       merge_front_bound = merge_front_boundary
       if merge_front_bound < FIN_FLOOR:
        merge_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
       if merge_discarded >= merge_front_bound:
        merge_front_tail = pl.tile.slice(mboth, [1, 1024], [0, PAIR_W])
        merge_front_values = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
        merge_front_positions = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        merge_front_candidates = pl.tile.select(pl.tile.cmps(merge_front_values, merge_discarded, cmp_type=0), merge_front_positions, 2147483647)
        merge_front_wide = pl.col_min(pl.reshape(merge_front_candidates, [8, 64]))
        merge_front_eight = pl.col_min(pl.reshape(merge_front_wide, [8, 8]))
        merge_front_result = pl.tile.read(merge_front_eight, [0, 0])
        merge_front_lane_1 = pl.tile.read(merge_front_eight, [0, 1])
        if merge_front_lane_1 < merge_front_result:
         merge_front_result = merge_front_lane_1
        merge_front_lane_2 = pl.tile.read(merge_front_eight, [0, 2])
        if merge_front_lane_2 < merge_front_result:
         merge_front_result = merge_front_lane_2
        merge_front_lane_3 = pl.tile.read(merge_front_eight, [0, 3])
        if merge_front_lane_3 < merge_front_result:
         merge_front_result = merge_front_lane_3
        merge_front_lane_4 = pl.tile.read(merge_front_eight, [0, 4])
        if merge_front_lane_4 < merge_front_result:
         merge_front_result = merge_front_lane_4
        merge_front_lane_5 = pl.tile.read(merge_front_eight, [0, 5])
        if merge_front_lane_5 < merge_front_result:
         merge_front_result = merge_front_lane_5
        merge_front_lane_6 = pl.tile.read(merge_front_eight, [0, 6])
        if merge_front_lane_6 < merge_front_result:
         merge_front_result = merge_front_lane_6
        merge_front_lane_7 = pl.tile.read(merge_front_eight, [0, 7])
        if merge_front_lane_7 < merge_front_result:
         merge_front_result = merge_front_lane_7
        if merge_front_result < frontier_position:
         frontier_position = merge_front_result
      pl.store(pl.tile.slice(mboth, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
      fin = pl.tile.slice(mboth, [1, PAIR_W], [0, 0])
      two_values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
      two_finite = pl.maximum(two_values, FIN_FLOOR)
      two_cutoff = pl.tile.read(two_finite, [0, K - 1])
      two_ones_c = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
      two_flags = pl.tile.select(pl.tile.cmps(two_values, two_cutoff, cmp_type=4), two_ones_c, 0.0)
      two_sums64 = pl.col_sum(pl.reshape(two_flags, [8, 64]))
      two_sums8 = pl.col_sum(pl.reshape(two_sums64, [8, 8]))
      two_above = pl.cast(0, pl.INDEX)
      for two_lane in pl.unroll(8):
       two_part = pl.cast(pl.tile.read(two_sums8, [0, two_lane]), pl.INT32)
       two_above = two_above + pl.cast(two_part, pl.INDEX)
      two_cutoff_stage_line = pl.tile.full([1, 16], dtype=pl.FP32, value=0)
      pl.tile.write(two_cutoff_stage_line, [0, 0], two_cutoff)
      pl.store(two_cutoff_stage_line, [t, 0], aux_cutoffs)
      two_quota_stage_line = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
      pl.tile.write(two_quota_stage_line, [0, 0], pl.cast(K - two_above, pl.INT32))
      pl.store(two_quota_stage_line, [t, 0], aux_quotas)
      two_replay_safe = discarded < two_cutoff
      if discarded == two_cutoff:
       two_cert_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       two_cert_candidates = pl.tile.select(pl.tile.cmps(two_values, two_cutoff, cmp_type=0), two_cert_positions, -1)
       two_cert_wide = pl.col_max(pl.reshape(two_cert_candidates, [8, 64]))
       two_cert_eight = pl.col_max(pl.reshape(two_cert_wide, [8, 8]))
       two_cert_result = pl.tile.read(two_cert_eight, [0, 0])
       two_cert_lane_1 = pl.tile.read(two_cert_eight, [0, 1])
       if two_cert_lane_1 > two_cert_result:
        two_cert_result = two_cert_lane_1
       two_cert_lane_2 = pl.tile.read(two_cert_eight, [0, 2])
       if two_cert_lane_2 > two_cert_result:
        two_cert_result = two_cert_lane_2
       two_cert_lane_3 = pl.tile.read(two_cert_eight, [0, 3])
       if two_cert_lane_3 > two_cert_result:
        two_cert_result = two_cert_lane_3
       two_cert_lane_4 = pl.tile.read(two_cert_eight, [0, 4])
       if two_cert_lane_4 > two_cert_result:
        two_cert_result = two_cert_lane_4
       two_cert_lane_5 = pl.tile.read(two_cert_eight, [0, 5])
       if two_cert_lane_5 > two_cert_result:
        two_cert_result = two_cert_lane_5
       two_cert_lane_6 = pl.tile.read(two_cert_eight, [0, 6])
       if two_cert_lane_6 > two_cert_result:
        two_cert_result = two_cert_lane_6
       two_cert_lane_7 = pl.tile.read(two_cert_eight, [0, 7])
       if two_cert_lane_7 > two_cert_result:
        two_cert_result = two_cert_lane_7
       if frontier_position > two_cert_result:
        two_replay_safe = True
      if two_replay_safe:
       if two_cutoff > FIN_FLOOR:
        two_root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        two_selected = pl.cast(K, pl.INDEX)
        if two_selected > 0:
         two_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(two_root_positions, pl.FP32))
         two_root_keys = two_root_priority
         two_root_order = pl.tile.sort32(two_root_keys, pl.reinterpret_view(two_root_positions, pl.UINT32))
         two_root_order = pl.tile.mrgsort(two_root_order, block_len=64)
         two_root_order = pl.tile.mrgsort(two_root_order, block_len=256)
         two_root_logical = pl.tile.gather_mask(two_root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         two_root_valid = pl.tile.set_validshape(two_root_logical, 1, two_selected)
         two_root_safe = pl.tile.fillpad(two_root_valid, pad_value=pl.PadValue.zero)
         two_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for two_map_start in pl.range(0, two_selected, MAP):
          two_map_count = pl.min(MAP, two_selected - two_map_start)
          two_map_bits = pl.tile.slice(pl.reinterpret_view(two_root_safe, pl.FP32), [1, MAP], [0, two_map_start])
          two_map_logical = pl.reinterpret_view(two_map_bits, pl.INT32)
          two_root_page_idx = pl.tile.shrs(two_map_logical, 7)
          two_root_page_ids = pl.tile.gather(materialize_displacements, two_root_page_idx, two_root_tmp)
          two_root_physical = pl.add(two_map_logical, two_root_page_ids)
          pl.store(pl.tile.set_validshape(two_root_physical, 1, two_map_count), [t, two_map_start], indices)
       else:
        two_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
        two_root = fin
        two_root_values = pl.tile.gather_mask(two_root, mask_pattern=pl.tile.MaskPattern.P0101)
        two_root_positions = pl.tile.gather_mask(two_root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        two_ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
        two_root_strict = pl.tile.select(pl.tile.cmps(two_root_values, two_cutoff, cmp_type=4), two_ones_root, 0.0)
        two_root_equal = pl.tile.select(pl.tile.cmps(two_root_values, two_cutoff, cmp_type=0), two_ones_root, 0.0)
        two_root_eligible = pl.maximum(two_root_strict, two_root_equal)
        two_root_visible = pl.tile.cmps(pl.cast(two_root_positions, pl.FP32), pl.cast(pl.cast(two_visible, pl.INT32), pl.FP32), cmp_type=2)
        two_root_keep = pl.tile.select(two_root_visible, two_root_eligible, 0.0)
        two_root_sums64 = pl.col_sum(pl.reshape(two_root_keep, [8, 64]))
        two_root_sums8 = pl.col_sum(pl.reshape(two_root_sums64, [8, 8]))
        two_selected = pl.cast(0, pl.INDEX)
        for two_root_lane in pl.unroll(8):
         two_root_part = pl.cast(pl.tile.read(two_root_sums8, [0, two_root_lane]), pl.INT32)
         two_selected = two_selected + pl.cast(two_root_part, pl.INDEX)
        pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
        if two_selected > 0:
         two_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(two_root_positions, pl.FP32))
         two_root_keys = pl.tile.select(pl.tile.cmps(two_root_keep, 0.0, cmp_type=4), two_root_priority, -1.0)
         two_root_order = pl.tile.sort32(two_root_keys, pl.reinterpret_view(two_root_positions, pl.UINT32))
         two_root_order = pl.tile.mrgsort(two_root_order, block_len=64)
         two_root_order = pl.tile.mrgsort(two_root_order, block_len=256)
         two_root_sorted_keys = pl.tile.gather_mask(two_root_order, mask_pattern=pl.tile.MaskPattern.P0101)
         two_root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), two_root_sorted_keys), pl.INT32)
         two_root_valid = pl.tile.set_validshape(two_root_logical, 1, two_selected)
         two_root_safe = pl.tile.fillpad(two_root_valid, pad_value=pl.PadValue.zero)
         two_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for two_map_start in pl.range(0, two_selected, MAP):
          two_map_count = pl.min(MAP, two_selected - two_map_start)
          two_map_bits = pl.tile.slice(pl.reinterpret_view(two_root_safe, pl.FP32), [1, MAP], [0, two_map_start])
          two_map_logical = pl.reinterpret_view(two_map_bits, pl.INT32)
          two_root_page_idx = pl.tile.shrs(two_map_logical, 7)
          two_root_page_ids = pl.tile.gather(materialize_displacements, two_root_page_idx, two_root_tmp)
          two_root_physical = pl.add(two_map_logical, two_root_page_ids)
          pl.store(pl.tile.set_validshape(two_root_physical, 1, two_map_count), [t, two_map_start], indices)
      else:
       two_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
       two_remaining = pl.cast(K - two_above, pl.INDEX)
       pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
       two_ones_m = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
       two_lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
       two_lanes_f = pl.cast(two_lanes, pl.FP32)
       two_lane_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
       two_priority = pl.cast(pl.sub(two_lane_origin, two_lanes), pl.FP32)
       two_gtmp = pl.tile.create([1, SCAN], dtype=pl.INT32)
       two_count = pl.cast(0, pl.INDEX)
       for two_start in pl.range(0, two_visible, SCAN):
        if two_count < K:
         two_valid = pl.min(SCAN, two_visible - two_start)
         two_scan_raw = pl.load(scores, [t, two_start], [1, SCAN], valid_shape=[1, two_valid])
         two_scan_vals = pl.tile.fillpad(two_scan_raw, pad_value=pl.PadValue.min)
         two_strict = pl.tile.select(pl.tile.cmps(two_scan_vals, two_cutoff, cmp_type=4), two_ones_m, 0.0)
         two_ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
         if two_remaining > 0:
          two_eq = pl.tile.select(pl.tile.cmps(two_scan_vals, two_cutoff, cmp_type=0), two_ones_m, 0.0)
          two_eq_valid = pl.tile.cmps(two_lanes_f, pl.cast(pl.cast(two_valid, pl.INT32), pl.FP32), cmp_type=2)
          two_eq = pl.tile.select(two_eq_valid, two_eq, 0.0)
          two_eq_sums64 = pl.col_sum(pl.reshape(two_eq, [SCAN // 64, 64]))
          two_eq_sums8 = pl.col_sum(pl.reshape(two_eq_sums64, [8, 8]))
          two_available = pl.cast(0, pl.INDEX)
          for two_eq_lane in pl.unroll(8):
           two_eq_part = pl.cast(pl.tile.read(two_eq_sums8, [0, two_eq_lane]), pl.INT32)
           two_available = two_available + pl.cast(two_eq_part, pl.INDEX)
          two_take = pl.min(two_available, two_remaining)
          if two_take > 0:
           if two_take == two_available:
            two_ties = two_eq
           else:
            two_tie_keys = pl.mul(two_eq, two_priority)
            two_tie_pairs = pl.tile.sort32(two_tie_keys, pl.reinterpret_view(two_lanes, pl.UINT32))
            two_tie_pairs = pl.tile.mrgsort(two_tie_pairs, block_len=64)
            two_tie_merged = two_tie_pairs
            two_equal_keys = pl.tile.gather_mask(two_tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
            two_last_key_f = pl.tile.read(two_equal_keys, [0, two_take - 1])
            two_boundary_f = 128.0 - two_last_key_f
            two_ties = pl.tile.select(pl.tile.cmps(two_lanes_f, two_boundary_f, cmp_type=3), two_eq, 0.0)
           two_remaining = two_remaining - two_take
         two_keep = pl.maximum(two_strict, two_ties)
         two_keep_valid = pl.tile.set_validshape(two_keep, 1, two_valid)
         two_keep_padded = pl.tile.fillpad(two_keep_valid, pad_value=pl.PadValue.zero)
         two_keep_sums64 = pl.col_sum(pl.reshape(two_keep_padded, [SCAN // 64, 64]))
         two_keep_sums8 = pl.col_sum(pl.reshape(two_keep_sums64, [8, 8]))
         two_keep_count = pl.cast(0, pl.INDEX)
         for two_keep_lane in pl.unroll(8):
          two_keep_part = pl.cast(pl.tile.read(two_keep_sums8, [0, two_keep_lane]), pl.INT32)
          two_keep_count = two_keep_count + pl.cast(two_keep_part, pl.INDEX)
         two_selected = pl.min(two_keep_count, K - two_count)
         if two_selected > 0:
          two_order_keys = pl.mul(two_keep_padded, two_priority)
          two_order_pairs = pl.tile.sort32(two_order_keys, pl.reinterpret_view(two_lanes, pl.UINT32))
          two_order_pairs = pl.tile.mrgsort(two_order_pairs, block_len=64)
          two_order_merged = two_order_pairs
          two_order_vals = pl.tile.gather_mask(two_order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
          two_offs = pl.sub(two_lane_origin, pl.cast(two_order_vals, pl.INT32))
          two_compact_valid = pl.tile.set_validshape(two_offs, 1, two_selected)
          two_compact_safe = pl.tile.fillpad(two_compact_valid, pad_value=pl.PadValue.zero)
          two_logical = pl.add(two_compact_safe, pl.cast(two_start, pl.INT32))
          two_page_idx = pl.tile.shrs(two_logical, 7)
          two_page_ids = pl.tile.gather(materialize_displacements, two_page_idx, two_gtmp)
          two_physical = pl.add(two_logical, two_page_ids)
          pl.store(pl.tile.set_validshape(two_physical, 1, two_selected), [t, two_count], indices)
          two_count = two_count + two_selected
     else:
      handoff_scratch = pl.tile.create([1, 2 * PAIR_W], dtype=pl.FP32)
      handoff_left = pl.load(wpairs, [t, 0], [1, PAIR_W])
      handoff_right = pl.load(wpairs, [t, PAIR_W], [1, PAIR_W])
      mboth = pl.tile.mrgsort(handoff_left, handoff_right, tmp=handoff_scratch)
      merge_discarded = pl.tile.read(mboth, [0, PAIR_W])
      if merge_discarded > discarded:
       discarded = merge_discarded
       frontier_position = pl.cast(2147483647, pl.INT32)
      if merge_discarded == discarded:
       merge_front_boundary = pl.tile.read(mboth, [0, PAIR_W - 2])
       merge_front_bound = merge_front_boundary
       if merge_front_bound < FIN_FLOOR:
        merge_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
       if merge_discarded >= merge_front_bound:
        merge_front_tail = pl.tile.slice(mboth, [1, 1024], [0, PAIR_W])
        merge_front_values = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
        merge_front_positions = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        merge_front_candidates = pl.tile.select(pl.tile.cmps(merge_front_values, merge_discarded, cmp_type=0), merge_front_positions, 2147483647)
        merge_front_wide = pl.col_min(pl.reshape(merge_front_candidates, [8, 64]))
        merge_front_eight = pl.col_min(pl.reshape(merge_front_wide, [8, 8]))
        merge_front_result = pl.tile.read(merge_front_eight, [0, 0])
        merge_front_lane_1 = pl.tile.read(merge_front_eight, [0, 1])
        if merge_front_lane_1 < merge_front_result:
         merge_front_result = merge_front_lane_1
        merge_front_lane_2 = pl.tile.read(merge_front_eight, [0, 2])
        if merge_front_lane_2 < merge_front_result:
         merge_front_result = merge_front_lane_2
        merge_front_lane_3 = pl.tile.read(merge_front_eight, [0, 3])
        if merge_front_lane_3 < merge_front_result:
         merge_front_result = merge_front_lane_3
        merge_front_lane_4 = pl.tile.read(merge_front_eight, [0, 4])
        if merge_front_lane_4 < merge_front_result:
         merge_front_result = merge_front_lane_4
        merge_front_lane_5 = pl.tile.read(merge_front_eight, [0, 5])
        if merge_front_lane_5 < merge_front_result:
         merge_front_result = merge_front_lane_5
        merge_front_lane_6 = pl.tile.read(merge_front_eight, [0, 6])
        if merge_front_lane_6 < merge_front_result:
         merge_front_result = merge_front_lane_6
        merge_front_lane_7 = pl.tile.read(merge_front_eight, [0, 7])
        if merge_front_lane_7 < merge_front_result:
         merge_front_result = merge_front_lane_7
        if merge_front_result < frontier_position:
         frontier_position = merge_front_result
      pl.store(pl.tile.slice(mboth, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
      handoff_offsets = pl.tile.arange(0, [1, PAIR_W], dtype=pl.INT32)
      handoff_gather_scratch = pl.tile.create([1, 2 * PAIR_W], dtype=pl.INT32)
      handoff_resident = pl.tile.gather(mboth, handoff_offsets, handoff_gather_scratch)
      handoff_third = pl.load(wpairs, [t, 2 * PAIR_W], [1, PAIR_W])
      mboth = pl.tile.mrgsort(handoff_resident, handoff_third, tmp=handoff_scratch)
      merge_discarded = pl.tile.read(mboth, [0, PAIR_W])
      if merge_discarded > discarded:
       discarded = merge_discarded
       frontier_position = pl.cast(2147483647, pl.INT32)
      if merge_discarded == discarded:
       merge_front_boundary = pl.tile.read(mboth, [0, PAIR_W - 2])
       merge_front_bound = merge_front_boundary
       if merge_front_bound < FIN_FLOOR:
        merge_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
       if merge_discarded >= merge_front_bound:
        merge_front_tail = pl.tile.slice(mboth, [1, 1024], [0, PAIR_W])
        merge_front_values = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
        merge_front_positions = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        merge_front_candidates = pl.tile.select(pl.tile.cmps(merge_front_values, merge_discarded, cmp_type=0), merge_front_positions, 2147483647)
        merge_front_wide = pl.col_min(pl.reshape(merge_front_candidates, [8, 64]))
        merge_front_eight = pl.col_min(pl.reshape(merge_front_wide, [8, 8]))
        merge_front_result = pl.tile.read(merge_front_eight, [0, 0])
        merge_front_lane_1 = pl.tile.read(merge_front_eight, [0, 1])
        if merge_front_lane_1 < merge_front_result:
         merge_front_result = merge_front_lane_1
        merge_front_lane_2 = pl.tile.read(merge_front_eight, [0, 2])
        if merge_front_lane_2 < merge_front_result:
         merge_front_result = merge_front_lane_2
        merge_front_lane_3 = pl.tile.read(merge_front_eight, [0, 3])
        if merge_front_lane_3 < merge_front_result:
         merge_front_result = merge_front_lane_3
        merge_front_lane_4 = pl.tile.read(merge_front_eight, [0, 4])
        if merge_front_lane_4 < merge_front_result:
         merge_front_result = merge_front_lane_4
        merge_front_lane_5 = pl.tile.read(merge_front_eight, [0, 5])
        if merge_front_lane_5 < merge_front_result:
         merge_front_result = merge_front_lane_5
        merge_front_lane_6 = pl.tile.read(merge_front_eight, [0, 6])
        if merge_front_lane_6 < merge_front_result:
         merge_front_result = merge_front_lane_6
        merge_front_lane_7 = pl.tile.read(merge_front_eight, [0, 7])
        if merge_front_lane_7 < merge_front_result:
         merge_front_result = merge_front_lane_7
        if merge_front_result < frontier_position:
         frontier_position = merge_front_result
      pl.store(pl.tile.slice(mboth, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
      fin = pl.tile.slice(mboth, [1, PAIR_W], [0, 0])
      three_values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
      three_finite = pl.maximum(three_values, FIN_FLOOR)
      three_cutoff = pl.tile.read(three_finite, [0, K - 1])
      three_ones_c = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
      three_flags = pl.tile.select(pl.tile.cmps(three_values, three_cutoff, cmp_type=4), three_ones_c, 0.0)
      three_sums64 = pl.col_sum(pl.reshape(three_flags, [8, 64]))
      three_sums8 = pl.col_sum(pl.reshape(three_sums64, [8, 8]))
      three_above = pl.cast(0, pl.INDEX)
      for three_lane in pl.unroll(8):
       three_part = pl.cast(pl.tile.read(three_sums8, [0, three_lane]), pl.INT32)
       three_above = three_above + pl.cast(three_part, pl.INDEX)
      three_cutoff_stage_line = pl.tile.full([1, 16], dtype=pl.FP32, value=0)
      pl.tile.write(three_cutoff_stage_line, [0, 0], three_cutoff)
      pl.store(three_cutoff_stage_line, [t, 0], aux_cutoffs)
      three_quota_stage_line = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
      pl.tile.write(three_quota_stage_line, [0, 0], pl.cast(K - three_above, pl.INT32))
      pl.store(three_quota_stage_line, [t, 0], aux_quotas)
      three_replay_safe = discarded < three_cutoff
      if discarded == three_cutoff:
       three_cert_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       three_cert_candidates = pl.tile.select(pl.tile.cmps(three_values, three_cutoff, cmp_type=0), three_cert_positions, -1)
       three_cert_wide = pl.col_max(pl.reshape(three_cert_candidates, [8, 64]))
       three_cert_eight = pl.col_max(pl.reshape(three_cert_wide, [8, 8]))
       three_cert_result = pl.tile.read(three_cert_eight, [0, 0])
       three_cert_lane_1 = pl.tile.read(three_cert_eight, [0, 1])
       if three_cert_lane_1 > three_cert_result:
        three_cert_result = three_cert_lane_1
       three_cert_lane_2 = pl.tile.read(three_cert_eight, [0, 2])
       if three_cert_lane_2 > three_cert_result:
        three_cert_result = three_cert_lane_2
       three_cert_lane_3 = pl.tile.read(three_cert_eight, [0, 3])
       if three_cert_lane_3 > three_cert_result:
        three_cert_result = three_cert_lane_3
       three_cert_lane_4 = pl.tile.read(three_cert_eight, [0, 4])
       if three_cert_lane_4 > three_cert_result:
        three_cert_result = three_cert_lane_4
       three_cert_lane_5 = pl.tile.read(three_cert_eight, [0, 5])
       if three_cert_lane_5 > three_cert_result:
        three_cert_result = three_cert_lane_5
       three_cert_lane_6 = pl.tile.read(three_cert_eight, [0, 6])
       if three_cert_lane_6 > three_cert_result:
        three_cert_result = three_cert_lane_6
       three_cert_lane_7 = pl.tile.read(three_cert_eight, [0, 7])
       if three_cert_lane_7 > three_cert_result:
        three_cert_result = three_cert_lane_7
       if frontier_position > three_cert_result:
        three_replay_safe = True
      if three_replay_safe:
       if three_cutoff > FIN_FLOOR:
        three_root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        three_selected = pl.cast(K, pl.INDEX)
        if three_selected > 0:
         three_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(three_root_positions, pl.FP32))
         three_root_keys = three_root_priority
         three_root_order = pl.tile.sort32(three_root_keys, pl.reinterpret_view(three_root_positions, pl.UINT32))
         three_root_order = pl.tile.mrgsort(three_root_order, block_len=64)
         three_root_order = pl.tile.mrgsort(three_root_order, block_len=256)
         three_root_logical = pl.tile.gather_mask(three_root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         three_root_valid = pl.tile.set_validshape(three_root_logical, 1, three_selected)
         three_root_safe = pl.tile.fillpad(three_root_valid, pad_value=pl.PadValue.zero)
         three_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for three_map_start in pl.range(0, three_selected, MAP):
          three_map_count = pl.min(MAP, three_selected - three_map_start)
          three_map_bits = pl.tile.slice(pl.reinterpret_view(three_root_safe, pl.FP32), [1, MAP], [0, three_map_start])
          three_map_logical = pl.reinterpret_view(three_map_bits, pl.INT32)
          three_root_page_idx = pl.tile.shrs(three_map_logical, 7)
          three_root_page_ids = pl.tile.gather(materialize_displacements, three_root_page_idx, three_root_tmp)
          three_root_physical = pl.add(three_map_logical, three_root_page_ids)
          pl.store(pl.tile.set_validshape(three_root_physical, 1, three_map_count), [t, three_map_start], indices)
       else:
        three_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
        three_root = fin
        three_root_values = pl.tile.gather_mask(three_root, mask_pattern=pl.tile.MaskPattern.P0101)
        three_root_positions = pl.tile.gather_mask(three_root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        three_ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
        three_root_strict = pl.tile.select(pl.tile.cmps(three_root_values, three_cutoff, cmp_type=4), three_ones_root, 0.0)
        three_root_equal = pl.tile.select(pl.tile.cmps(three_root_values, three_cutoff, cmp_type=0), three_ones_root, 0.0)
        three_root_eligible = pl.maximum(three_root_strict, three_root_equal)
        three_root_visible = pl.tile.cmps(pl.cast(three_root_positions, pl.FP32), pl.cast(pl.cast(three_visible, pl.INT32), pl.FP32), cmp_type=2)
        three_root_keep = pl.tile.select(three_root_visible, three_root_eligible, 0.0)
        three_root_sums64 = pl.col_sum(pl.reshape(three_root_keep, [8, 64]))
        three_root_sums8 = pl.col_sum(pl.reshape(three_root_sums64, [8, 8]))
        three_selected = pl.cast(0, pl.INDEX)
        for three_root_lane in pl.unroll(8):
         three_root_part = pl.cast(pl.tile.read(three_root_sums8, [0, three_root_lane]), pl.INT32)
         three_selected = three_selected + pl.cast(three_root_part, pl.INDEX)
        pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
        if three_selected > 0:
         three_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(three_root_positions, pl.FP32))
         three_root_keys = pl.tile.select(pl.tile.cmps(three_root_keep, 0.0, cmp_type=4), three_root_priority, -1.0)
         three_root_order = pl.tile.sort32(three_root_keys, pl.reinterpret_view(three_root_positions, pl.UINT32))
         three_root_order = pl.tile.mrgsort(three_root_order, block_len=64)
         three_root_order = pl.tile.mrgsort(three_root_order, block_len=256)
         three_root_sorted_keys = pl.tile.gather_mask(three_root_order, mask_pattern=pl.tile.MaskPattern.P0101)
         three_root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), three_root_sorted_keys), pl.INT32)
         three_root_valid = pl.tile.set_validshape(three_root_logical, 1, three_selected)
         three_root_safe = pl.tile.fillpad(three_root_valid, pad_value=pl.PadValue.zero)
         three_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for three_map_start in pl.range(0, three_selected, MAP):
          three_map_count = pl.min(MAP, three_selected - three_map_start)
          three_map_bits = pl.tile.slice(pl.reinterpret_view(three_root_safe, pl.FP32), [1, MAP], [0, three_map_start])
          three_map_logical = pl.reinterpret_view(three_map_bits, pl.INT32)
          three_root_page_idx = pl.tile.shrs(three_map_logical, 7)
          three_root_page_ids = pl.tile.gather(materialize_displacements, three_root_page_idx, three_root_tmp)
          three_root_physical = pl.add(three_map_logical, three_root_page_ids)
          pl.store(pl.tile.set_validshape(three_root_physical, 1, three_map_count), [t, three_map_start], indices)
      else:
       three_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
       three_remaining = pl.cast(K - three_above, pl.INDEX)
       pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
       three_ones_m = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
       three_lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
       three_lanes_f = pl.cast(three_lanes, pl.FP32)
       three_lane_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
       three_priority = pl.cast(pl.sub(three_lane_origin, three_lanes), pl.FP32)
       three_gtmp = pl.tile.create([1, SCAN], dtype=pl.INT32)
       three_count = pl.cast(0, pl.INDEX)
       for three_start in pl.range(0, three_visible, SCAN):
        if three_count < K:
         three_valid = pl.min(SCAN, three_visible - three_start)
         three_scan_raw = pl.load(scores, [t, three_start], [1, SCAN], valid_shape=[1, three_valid])
         three_scan_vals = pl.tile.fillpad(three_scan_raw, pad_value=pl.PadValue.min)
         three_strict = pl.tile.select(pl.tile.cmps(three_scan_vals, three_cutoff, cmp_type=4), three_ones_m, 0.0)
         three_ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
         if three_remaining > 0:
          three_eq = pl.tile.select(pl.tile.cmps(three_scan_vals, three_cutoff, cmp_type=0), three_ones_m, 0.0)
          three_eq_valid = pl.tile.cmps(three_lanes_f, pl.cast(pl.cast(three_valid, pl.INT32), pl.FP32), cmp_type=2)
          three_eq = pl.tile.select(three_eq_valid, three_eq, 0.0)
          three_eq_sums64 = pl.col_sum(pl.reshape(three_eq, [SCAN // 64, 64]))
          three_eq_sums8 = pl.col_sum(pl.reshape(three_eq_sums64, [8, 8]))
          three_available = pl.cast(0, pl.INDEX)
          for three_eq_lane in pl.unroll(8):
           three_eq_part = pl.cast(pl.tile.read(three_eq_sums8, [0, three_eq_lane]), pl.INT32)
           three_available = three_available + pl.cast(three_eq_part, pl.INDEX)
          three_take = pl.min(three_available, three_remaining)
          if three_take > 0:
           if three_take == three_available:
            three_ties = three_eq
           else:
            three_tie_keys = pl.mul(three_eq, three_priority)
            three_tie_pairs = pl.tile.sort32(three_tie_keys, pl.reinterpret_view(three_lanes, pl.UINT32))
            three_tie_pairs = pl.tile.mrgsort(three_tie_pairs, block_len=64)
            three_tie_merged = three_tie_pairs
            three_equal_keys = pl.tile.gather_mask(three_tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
            three_last_key_f = pl.tile.read(three_equal_keys, [0, three_take - 1])
            three_boundary_f = 128.0 - three_last_key_f
            three_ties = pl.tile.select(pl.tile.cmps(three_lanes_f, three_boundary_f, cmp_type=3), three_eq, 0.0)
           three_remaining = three_remaining - three_take
         three_keep = pl.maximum(three_strict, three_ties)
         three_keep_valid = pl.tile.set_validshape(three_keep, 1, three_valid)
         three_keep_padded = pl.tile.fillpad(three_keep_valid, pad_value=pl.PadValue.zero)
         three_keep_sums64 = pl.col_sum(pl.reshape(three_keep_padded, [SCAN // 64, 64]))
         three_keep_sums8 = pl.col_sum(pl.reshape(three_keep_sums64, [8, 8]))
         three_keep_count = pl.cast(0, pl.INDEX)
         for three_keep_lane in pl.unroll(8):
          three_keep_part = pl.cast(pl.tile.read(three_keep_sums8, [0, three_keep_lane]), pl.INT32)
          three_keep_count = three_keep_count + pl.cast(three_keep_part, pl.INDEX)
         three_selected = pl.min(three_keep_count, K - three_count)
         if three_selected > 0:
          three_order_keys = pl.mul(three_keep_padded, three_priority)
          three_order_pairs = pl.tile.sort32(three_order_keys, pl.reinterpret_view(three_lanes, pl.UINT32))
          three_order_pairs = pl.tile.mrgsort(three_order_pairs, block_len=64)
          three_order_merged = three_order_pairs
          three_order_vals = pl.tile.gather_mask(three_order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
          three_offs = pl.sub(three_lane_origin, pl.cast(three_order_vals, pl.INT32))
          three_compact_valid = pl.tile.set_validshape(three_offs, 1, three_selected)
          three_compact_safe = pl.tile.fillpad(three_compact_valid, pad_value=pl.PadValue.zero)
          three_logical = pl.add(three_compact_safe, pl.cast(three_start, pl.INT32))
          three_page_idx = pl.tile.shrs(three_logical, 7)
          three_page_ids = pl.tile.gather(materialize_displacements, three_page_idx, three_gtmp)
          three_physical = pl.add(three_logical, three_page_ids)
          pl.store(pl.tile.set_validshape(three_physical, 1, three_selected), [t, three_count], indices)
          three_count = three_count + three_selected
   for pack_task in pl.spmd(1, name_hint='pack_auxiliary'):
    for output_row in pl.range(0, M, 16):
     cutoff_lines = pl.load(aux_cutoffs, [output_row, 0], [16, 16])
     quota_lines = pl.load(aux_quotas, [output_row, 0], [16, 16])
     cutoff_flat = pl.reshape(cutoff_lines, [1, 256])
     quota_flat = pl.reshape(quota_lines, [1, 256])
     pack_offsets = pl.mul(pl.tile.arange(0, [1, 16], dtype=pl.INT32), 16)
     cutoff_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     quota_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     cutoff_packed = pl.tile.gather(cutoff_flat, pack_offsets, cutoff_scratch)
     quota_packed = pl.tile.gather(quota_flat, pack_offsets, quota_scratch)
     pl.store(pl.reshape(cutoff_packed, [16, 1]), [output_row, 0], cutoffs)
     pl.store(pl.reshape(quota_packed, [16, 1]), [output_row, 0], quotas)
   return indices
 else:

  @pl.jit
  def dsa41_topk_guarded_root(scores: pl.Tensor[[M, N], pl.FP32], lengths: pl.Tensor[[M], pl.INT32], block_table: pl.Tensor[[1, P], pl.INT32], wpairs: pl.InOut[pl.Tensor[[M, WORKSPACE_WIDTH], pl.FP32]], cutoffs: pl.InOut[pl.Tensor[[M, 1], pl.FP32]], quotas: pl.InOut[pl.Tensor[[M, 1], pl.INT32]], indices: pl.Out[pl.Tensor[[M, K], pl.INT32]]):
   aux_discarded = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_cutoffs = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_quotas = pl.create_tensor([M, 16], dtype=pl.INT32)
   aux_front = pl.create_tensor([M, 16], dtype=pl.INT32)
   leaf_discarded_bank = pl.create_tensor([M, NL * 16], dtype=pl.FP32)
   leaf_front_bank = pl.create_tensor([M, NL * 16], dtype=pl.INT32)
   for leaf_job in pl.spmd(LEAF_COHORTS * NL, name_hint='canonical_leaf_plane'):
    leaf = leaf_job // LEAF_COHORTS
    leaf_cohort_base = leaf_job % LEAF_COHORTS * LEAF_ROWS
    for t in pl.range(leaf_cohort_base, pl.min(leaf_cohort_base + LEAF_ROWS, M), 1):
     zero_pairs = pl.tile.full([1, PAIR_W], dtype=pl.FP32, value=0.0)
     if leaf == 0:
      for clear_offset in pl.range(NL * PAIR_W, WORKSPACE_WIDTH, PAIR_W):
       pl.store(zero_pairs, [t, clear_offset], wpairs)
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     begin = leaf * LEAF
     if begin < wv:
      valid = pl.min(LEAF, wv - begin)
      if valid <= 512:
       narrow_raw = pl.load(scores, [t, begin], [1, 512], valid_shape=[1, valid])
       narrow_padded = pl.tile.fillpad(narrow_raw, pad_value=pl.PadValue.min)
       narrow_values = pl.maximum(narrow_padded, SORT_FLOOR)
       narrow_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       narrow_pairs = pl.tile.sort32(narrow_values, pl.reinterpret_view(narrow_positions, pl.UINT32))
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=64)
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=256)
       pl.store(narrow_pairs, [t, leaf * PAIR_W], wpairs)
      elif valid <= 1024:
       left_valid = pl.min(valid, 512)
       right_valid = pl.max(valid - 512, 0)
       left_raw = pl.load(scores, [t, begin], [1, 512], valid_shape=[1, left_valid])
       right_raw = pl.load(scores, [t, begin + 512], [1, 512], valid_shape=[1, right_valid])
       left_values = pl.maximum(pl.tile.fillpad(left_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       right_values = pl.maximum(pl.tile.fillpad(right_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       left_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       right_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin + 512, pl.INT32))
       left_pairs = pl.tile.sort32(left_values, pl.reinterpret_view(left_positions, pl.UINT32))
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=64)
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=256)
       right_pairs = pl.tile.sort32(right_values, pl.reinterpret_view(right_positions, pl.UINT32))
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=64)
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=256)
       native_scratch = pl.tile.create([1, 2048], dtype=pl.FP32)
       native_pairs = pl.tile.mrgsort(left_pairs, right_pairs, tmp=native_scratch)
       native_discarded = pl.tile.read(native_pairs, [0, PAIR_W])
       if native_discarded > discarded:
        discarded = native_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if native_discarded == discarded:
        native_front_boundary = pl.tile.read(native_pairs, [0, PAIR_W - 2])
        native_front_bound = native_front_boundary
        if native_front_bound < FIN_FLOOR:
         native_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if native_discarded >= native_front_bound:
         native_arity_third = pl.tile.read(native_pairs, [0, PAIR_W + 4])
         if native_arity_third < native_discarded:
          native_arity_second = pl.tile.read(native_pairs, [0, PAIR_W + 2])
          native_arity_bits = pl.reinterpret_view(native_pairs, pl.INT32)
          native_front_result = pl.tile.read(native_arity_bits, [0, PAIR_W + 1])
          if native_arity_second == native_discarded:
           native_arity_second_position = pl.tile.read(native_arity_bits, [0, PAIR_W + 3])
           native_front_result = pl.min(native_front_result, native_arity_second_position)
         else:
          native_front_tail = pl.tile.slice(native_pairs, [1, 1024], [0, PAIR_W])
          native_front_values = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
          native_front_positions = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          native_front_candidates = pl.tile.select(pl.tile.cmps(native_front_values, native_discarded, cmp_type=0), native_front_positions, 2147483647)
          native_front_wide = pl.col_min(pl.reshape(native_front_candidates, [8, 64]))
          native_front_eight = pl.col_min(pl.reshape(native_front_wide, [8, 8]))
          native_front_result = pl.tile.read(native_front_eight, [0, 0])
          native_front_lane_1 = pl.tile.read(native_front_eight, [0, 1])
          if native_front_lane_1 < native_front_result:
           native_front_result = native_front_lane_1
          native_front_lane_2 = pl.tile.read(native_front_eight, [0, 2])
          if native_front_lane_2 < native_front_result:
           native_front_result = native_front_lane_2
          native_front_lane_3 = pl.tile.read(native_front_eight, [0, 3])
          if native_front_lane_3 < native_front_result:
           native_front_result = native_front_lane_3
          native_front_lane_4 = pl.tile.read(native_front_eight, [0, 4])
          if native_front_lane_4 < native_front_result:
           native_front_result = native_front_lane_4
          native_front_lane_5 = pl.tile.read(native_front_eight, [0, 5])
          if native_front_lane_5 < native_front_result:
           native_front_result = native_front_lane_5
          native_front_lane_6 = pl.tile.read(native_front_eight, [0, 6])
          if native_front_lane_6 < native_front_result:
           native_front_result = native_front_lane_6
          native_front_lane_7 = pl.tile.read(native_front_eight, [0, 7])
          if native_front_lane_7 < native_front_result:
           native_front_result = native_front_lane_7
         if native_front_result < frontier_position:
          frontier_position = native_front_result
       pl.store(pl.tile.slice(native_pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
      else:
       raw = pl.load(scores, [t, begin], [1, LEAF], valid_shape=[1, valid])
       padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
       flr = pl.tile.full([1, LEAF], dtype=pl.FP32, value=SORT_FLOOR)
       vals = pl.maximum(padded, flr)
       idx_i32 = pl.add(pl.tile.arange(0, [1, LEAF], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       pairs = pl.tile.sort32(vals, pl.reinterpret_view(idx_i32, pl.UINT32))
       pairs = pl.tile.mrgsort(pairs, block_len=64)
       pairs = pl.tile.mrgsort(pairs, block_len=256)
       pairs = pl.tile.mrgsort(pairs, block_len=1024)
       leaf_discarded = pl.tile.read(pairs, [0, PAIR_W])
       if leaf_discarded > discarded:
        discarded = leaf_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if leaf_discarded == discarded:
        leaf_front_boundary = pl.tile.read(pairs, [0, PAIR_W - 2])
        leaf_front_bound = leaf_front_boundary
        if leaf_front_bound < FIN_FLOOR:
         leaf_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if leaf_discarded >= leaf_front_bound:
         leaf_arity_third = pl.tile.read(pairs, [0, PAIR_W + 4])
         if leaf_arity_third < leaf_discarded:
          leaf_arity_second = pl.tile.read(pairs, [0, PAIR_W + 2])
          leaf_arity_bits = pl.reinterpret_view(pairs, pl.INT32)
          leaf_front_result = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 1])
          if leaf_arity_second == leaf_discarded:
           leaf_arity_second_position = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 3])
           leaf_front_result = pl.min(leaf_front_result, leaf_arity_second_position)
         else:
          leaf_front_tail = pl.tile.slice(pairs, [1, 3072], [0, PAIR_W])
          leaf_front_values = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
          leaf_front_positions = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          leaf_front_candidates = pl.tile.select(pl.tile.cmps(leaf_front_values, leaf_discarded, cmp_type=0), leaf_front_positions, 2147483647)
          leaf_front_wide = pl.col_min(pl.reshape(leaf_front_candidates, [24, 64]))
          leaf_front_eight = pl.col_min(pl.reshape(leaf_front_wide, [8, 8]))
          leaf_front_result = pl.tile.read(leaf_front_eight, [0, 0])
          leaf_front_lane_1 = pl.tile.read(leaf_front_eight, [0, 1])
          if leaf_front_lane_1 < leaf_front_result:
           leaf_front_result = leaf_front_lane_1
          leaf_front_lane_2 = pl.tile.read(leaf_front_eight, [0, 2])
          if leaf_front_lane_2 < leaf_front_result:
           leaf_front_result = leaf_front_lane_2
          leaf_front_lane_3 = pl.tile.read(leaf_front_eight, [0, 3])
          if leaf_front_lane_3 < leaf_front_result:
           leaf_front_result = leaf_front_lane_3
          leaf_front_lane_4 = pl.tile.read(leaf_front_eight, [0, 4])
          if leaf_front_lane_4 < leaf_front_result:
           leaf_front_result = leaf_front_lane_4
          leaf_front_lane_5 = pl.tile.read(leaf_front_eight, [0, 5])
          if leaf_front_lane_5 < leaf_front_result:
           leaf_front_result = leaf_front_lane_5
          leaf_front_lane_6 = pl.tile.read(leaf_front_eight, [0, 6])
          if leaf_front_lane_6 < leaf_front_result:
           leaf_front_result = leaf_front_lane_6
          leaf_front_lane_7 = pl.tile.read(leaf_front_eight, [0, 7])
          if leaf_front_lane_7 < leaf_front_result:
           leaf_front_result = leaf_front_lane_7
         if leaf_front_result < frontier_position:
          frontier_position = leaf_front_result
       pl.store(pl.tile.slice(pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
     else:
      pl.store(zero_pairs, [t, leaf * PAIR_W], wpairs)
     leaf_score_line = pl.tile.full([1, 16], dtype=pl.FP32, value=SORT_FLOOR)
     leaf_position_line = pl.tile.full([1, 16], dtype=pl.INT32, value=2147483647)
     pl.tile.write(leaf_score_line, [0, 0], discarded)
     pl.tile.write(leaf_position_line, [0, 0], frontier_position)
     pl.store(leaf_score_line, [t, leaf * 16], leaf_discarded_bank)
     pl.store(leaf_position_line, [t, leaf * 16], leaf_front_bank)
   for merge_cohort_job in pl.spmd((M + 4 - 1) // 4, name_hint='decomposed_merge'):
    cohort_base = merge_cohort_job * 4
    for t in pl.range(cohort_base, pl.min(cohort_base + 4, M), 1):
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     lc = (wv + LEAF - 1) // LEAF
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     for proof_leaf in pl.range(NL):
      proof_score = pl.read(leaf_discarded_bank, [t, proof_leaf * 16])
      proof_position = pl.read(leaf_front_bank, [t, proof_leaf * 16])
      if proof_score > discarded:
       discarded = proof_score
       frontier_position = pl.cast(2147483647, pl.INT32)
      if proof_score == discarded:
       if proof_position < frontier_position:
        frontier_position = proof_position
     mtmp = pl.tile.create([1, 2 * PAIR_W], dtype=pl.FP32)
     for cb in pl.range(1, lc):
      mleft = pl.load(wpairs, [t, 0], [1, PAIR_W])
      mright = pl.load(wpairs, [t, cb * PAIR_W], [1, PAIR_W])
      mboth = pl.tile.mrgsort(mleft, mright, tmp=mtmp)
      merge_discarded = pl.tile.read(mboth, [0, PAIR_W])
      if merge_discarded > discarded:
       discarded = merge_discarded
       frontier_position = pl.cast(2147483647, pl.INT32)
      if merge_discarded == discarded:
       merge_front_boundary = pl.tile.read(mboth, [0, PAIR_W - 2])
       merge_front_bound = merge_front_boundary
       if merge_front_bound < FIN_FLOOR:
        merge_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
       if merge_discarded >= merge_front_bound:
        merge_front_tail = pl.tile.slice(mboth, [1, 1024], [0, PAIR_W])
        merge_front_values = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
        merge_front_positions = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        merge_front_candidates = pl.tile.select(pl.tile.cmps(merge_front_values, merge_discarded, cmp_type=0), merge_front_positions, 2147483647)
        merge_front_wide = pl.col_min(pl.reshape(merge_front_candidates, [8, 64]))
        merge_front_eight = pl.col_min(pl.reshape(merge_front_wide, [8, 8]))
        merge_front_result = pl.tile.read(merge_front_eight, [0, 0])
        merge_front_lane_1 = pl.tile.read(merge_front_eight, [0, 1])
        if merge_front_lane_1 < merge_front_result:
         merge_front_result = merge_front_lane_1
        merge_front_lane_2 = pl.tile.read(merge_front_eight, [0, 2])
        if merge_front_lane_2 < merge_front_result:
         merge_front_result = merge_front_lane_2
        merge_front_lane_3 = pl.tile.read(merge_front_eight, [0, 3])
        if merge_front_lane_3 < merge_front_result:
         merge_front_result = merge_front_lane_3
        merge_front_lane_4 = pl.tile.read(merge_front_eight, [0, 4])
        if merge_front_lane_4 < merge_front_result:
         merge_front_result = merge_front_lane_4
        merge_front_lane_5 = pl.tile.read(merge_front_eight, [0, 5])
        if merge_front_lane_5 < merge_front_result:
         merge_front_result = merge_front_lane_5
        merge_front_lane_6 = pl.tile.read(merge_front_eight, [0, 6])
        if merge_front_lane_6 < merge_front_result:
         merge_front_result = merge_front_lane_6
        merge_front_lane_7 = pl.tile.read(merge_front_eight, [0, 7])
        if merge_front_lane_7 < merge_front_result:
         merge_front_result = merge_front_lane_7
        if merge_front_result < frontier_position:
         frontier_position = merge_front_result
      pl.store(pl.tile.slice(mboth, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
     pl.write(aux_discarded, [t, 0], discarded)
     pl.write(aux_front, [t, 0], frontier_position)
   for materialize_row_job in pl.spmd((M + 4 - 1) // 4, name_hint='decomposed_materialize'):
    root_row_base = materialize_row_job * 4
    if P == 1:
     materialize_single_page = pl.read(block_table, [0, 0])
     materialize_pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), materialize_single_page), pad_value=pl.PadValue.zero)
    else:
     materialize_pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
    materialize_page_lanes = pl.tile.arange(0, [1, P8], dtype=pl.INT32)
    materialize_displacements = pl.mul(pl.sub(materialize_pages, materialize_page_lanes), PAGE)
    for t in pl.range(root_row_base, pl.min(root_row_base + 4, M), 1):
     fin = pl.load(wpairs, [t, 0], [1, PAIR_W])
     values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
     finite = pl.maximum(values, FIN_FLOOR)
     cutoff = pl.tile.read(finite, [0, K - 1])
     ones_c = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
     flags = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=4), ones_c, 0.0)
     sums64 = pl.col_sum(pl.reshape(flags, [8, 64]))
     sums8 = pl.col_sum(pl.reshape(sums64, [8, 8]))
     above = pl.cast(0, pl.INDEX)
     for lane in pl.unroll(8):
      part = pl.cast(pl.tile.read(sums8, [0, lane]), pl.INT32)
      above = above + pl.cast(part, pl.INDEX)
     cutoff_stage_line = pl.tile.full([1, 16], dtype=pl.FP32, value=0)
     pl.tile.write(cutoff_stage_line, [0, 0], cutoff)
     pl.store(cutoff_stage_line, [t, 0], aux_cutoffs)
     quota_stage_line = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
     pl.tile.write(quota_stage_line, [0, 0], pl.cast(K - above, pl.INT32))
     pl.store(quota_stage_line, [t, 0], aux_quotas)
     discarded = pl.read(aux_discarded, [t, 0])
     frontier_position = pl.read(aux_front, [t, 0])
     replay_safe = discarded < cutoff
     if discarded == cutoff:
      certificate_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
      certificate_candidates = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=0), certificate_positions, -1)
      certificate_wide = pl.col_max(pl.reshape(certificate_candidates, [8, 64]))
      certificate_eight = pl.col_max(pl.reshape(certificate_wide, [8, 8]))
      certificate_result = pl.tile.read(certificate_eight, [0, 0])
      certificate_lane_1 = pl.tile.read(certificate_eight, [0, 1])
      if certificate_lane_1 > certificate_result:
       certificate_result = certificate_lane_1
      certificate_lane_2 = pl.tile.read(certificate_eight, [0, 2])
      if certificate_lane_2 > certificate_result:
       certificate_result = certificate_lane_2
      certificate_lane_3 = pl.tile.read(certificate_eight, [0, 3])
      if certificate_lane_3 > certificate_result:
       certificate_result = certificate_lane_3
      certificate_lane_4 = pl.tile.read(certificate_eight, [0, 4])
      if certificate_lane_4 > certificate_result:
       certificate_result = certificate_lane_4
      certificate_lane_5 = pl.tile.read(certificate_eight, [0, 5])
      if certificate_lane_5 > certificate_result:
       certificate_result = certificate_lane_5
      certificate_lane_6 = pl.tile.read(certificate_eight, [0, 6])
      if certificate_lane_6 > certificate_result:
       certificate_result = certificate_lane_6
      certificate_lane_7 = pl.tile.read(certificate_eight, [0, 7])
      if certificate_lane_7 > certificate_result:
       certificate_result = certificate_lane_7
      if frontier_position > certificate_result:
       replay_safe = True
     if replay_safe:
      if cutoff > FIN_FLOOR:
       root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       selected = pl.cast(K, pl.INDEX)
       if selected > 0:
        root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(root_positions, pl.FP32))
        root_keys = root_priority
        root_order = pl.tile.sort32(root_keys, pl.reinterpret_view(root_positions, pl.UINT32))
        root_order = pl.tile.mrgsort(root_order, block_len=64)
        root_order = pl.tile.mrgsort(root_order, block_len=256)
        root_logical = pl.tile.gather_mask(root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        root_valid = pl.tile.set_validshape(root_logical, 1, selected)
        root_safe = pl.tile.fillpad(root_valid, pad_value=pl.PadValue.zero)
        root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
        for map_start in pl.range(0, selected, MAP):
         map_count = pl.min(MAP, selected - map_start)
         map_bits = pl.tile.slice(pl.reinterpret_view(root_safe, pl.FP32), [1, MAP], [0, map_start])
         map_logical = pl.reinterpret_view(map_bits, pl.INT32)
         root_page_idx = pl.tile.shrs(map_logical, 7)
         root_page_ids = pl.tile.gather(materialize_displacements, root_page_idx, root_tmp)
         root_physical = pl.add(map_logical, root_page_ids)
         pl.store(pl.tile.set_validshape(root_physical, 1, map_count), [t, map_start], indices)
      else:
       visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
       root = fin
       root_values = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P0101)
       root_positions = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
       root_strict = pl.tile.select(pl.tile.cmps(root_values, cutoff, cmp_type=4), ones_root, 0.0)
       root_equal = pl.tile.select(pl.tile.cmps(root_values, cutoff, cmp_type=0), ones_root, 0.0)
       root_eligible = pl.maximum(root_strict, root_equal)
       root_visible = pl.tile.cmps(pl.cast(root_positions, pl.FP32), pl.cast(pl.cast(visible, pl.INT32), pl.FP32), cmp_type=2)
       root_keep = pl.tile.select(root_visible, root_eligible, 0.0)
       root_sums64 = pl.col_sum(pl.reshape(root_keep, [8, 64]))
       root_sums8 = pl.col_sum(pl.reshape(root_sums64, [8, 8]))
       selected = pl.cast(0, pl.INDEX)
       for root_lane in pl.unroll(8):
        root_part = pl.cast(pl.tile.read(root_sums8, [0, root_lane]), pl.INT32)
        selected = selected + pl.cast(root_part, pl.INDEX)
       pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
       if selected > 0:
        root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(root_positions, pl.FP32))
        root_keys = pl.tile.select(pl.tile.cmps(root_keep, 0.0, cmp_type=4), root_priority, -1.0)
        root_order = pl.tile.sort32(root_keys, pl.reinterpret_view(root_positions, pl.UINT32))
        root_order = pl.tile.mrgsort(root_order, block_len=64)
        root_order = pl.tile.mrgsort(root_order, block_len=256)
        root_sorted_keys = pl.tile.gather_mask(root_order, mask_pattern=pl.tile.MaskPattern.P0101)
        root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), root_sorted_keys), pl.INT32)
        root_valid = pl.tile.set_validshape(root_logical, 1, selected)
        root_safe = pl.tile.fillpad(root_valid, pad_value=pl.PadValue.zero)
        root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
        for map_start in pl.range(0, selected, MAP):
         map_count = pl.min(MAP, selected - map_start)
         map_bits = pl.tile.slice(pl.reinterpret_view(root_safe, pl.FP32), [1, MAP], [0, map_start])
         map_logical = pl.reinterpret_view(map_bits, pl.INT32)
         root_page_idx = pl.tile.shrs(map_logical, 7)
         root_page_ids = pl.tile.gather(materialize_displacements, root_page_idx, root_tmp)
         root_physical = pl.add(map_logical, root_page_ids)
         pl.store(pl.tile.set_validshape(root_physical, 1, map_count), [t, map_start], indices)
     else:
      visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
      remaining = pl.cast(K - above, pl.INDEX)
      pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
      ones_m = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
      lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
      lanes_f = pl.cast(lanes, pl.FP32)
      lane_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
      priority = pl.cast(pl.sub(lane_origin, lanes), pl.FP32)
      gtmp = pl.tile.create([1, SCAN], dtype=pl.INT32)
      count = pl.cast(0, pl.INDEX)
      for start in pl.range(0, visible, SCAN):
       if count < K:
        valid = pl.min(SCAN, visible - start)
        scan_raw = pl.load(scores, [t, start], [1, SCAN], valid_shape=[1, valid])
        scan_vals = pl.tile.fillpad(scan_raw, pad_value=pl.PadValue.min)
        strict = pl.tile.select(pl.tile.cmps(scan_vals, cutoff, cmp_type=4), ones_m, 0.0)
        ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
        if remaining > 0:
         eq = pl.tile.select(pl.tile.cmps(scan_vals, cutoff, cmp_type=0), ones_m, 0.0)
         eq_valid = pl.tile.cmps(lanes_f, pl.cast(pl.cast(valid, pl.INT32), pl.FP32), cmp_type=2)
         eq = pl.tile.select(eq_valid, eq, 0.0)
         eq_sums64 = pl.col_sum(pl.reshape(eq, [SCAN // 64, 64]))
         eq_sums8 = pl.col_sum(pl.reshape(eq_sums64, [8, 8]))
         available = pl.cast(0, pl.INDEX)
         for eq_lane in pl.unroll(8):
          eq_part = pl.cast(pl.tile.read(eq_sums8, [0, eq_lane]), pl.INT32)
          available = available + pl.cast(eq_part, pl.INDEX)
         take = pl.min(available, remaining)
         if take > 0:
          if take == available:
           ties = eq
          else:
           tie_keys = pl.mul(eq, priority)
           tie_pairs = pl.tile.sort32(tie_keys, pl.reinterpret_view(lanes, pl.UINT32))
           tie_pairs = pl.tile.mrgsort(tie_pairs, block_len=64)
           tie_merged = tie_pairs
           equal_keys = pl.tile.gather_mask(tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
           last_key_f = pl.tile.read(equal_keys, [0, take - 1])
           boundary_f = 128.0 - last_key_f
           ties = pl.tile.select(pl.tile.cmps(lanes_f, boundary_f, cmp_type=3), eq, 0.0)
          remaining = remaining - take
        keep = pl.maximum(strict, ties)
        keep_valid = pl.tile.set_validshape(keep, 1, valid)
        keep_padded = pl.tile.fillpad(keep_valid, pad_value=pl.PadValue.zero)
        keep_sums64 = pl.col_sum(pl.reshape(keep_padded, [SCAN // 64, 64]))
        keep_sums8 = pl.col_sum(pl.reshape(keep_sums64, [8, 8]))
        keep_count = pl.cast(0, pl.INDEX)
        for keep_lane in pl.unroll(8):
         keep_part = pl.cast(pl.tile.read(keep_sums8, [0, keep_lane]), pl.INT32)
         keep_count = keep_count + pl.cast(keep_part, pl.INDEX)
        selected = pl.min(keep_count, K - count)
        if selected > 0:
         order_keys = pl.mul(keep_padded, priority)
         order_pairs = pl.tile.sort32(order_keys, pl.reinterpret_view(lanes, pl.UINT32))
         order_pairs = pl.tile.mrgsort(order_pairs, block_len=64)
         order_merged = order_pairs
         order_vals = pl.tile.gather_mask(order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
         offs = pl.sub(lane_origin, pl.cast(order_vals, pl.INT32))
         compact_valid = pl.tile.set_validshape(offs, 1, selected)
         compact_safe = pl.tile.fillpad(compact_valid, pad_value=pl.PadValue.zero)
         logical = pl.add(compact_safe, pl.cast(start, pl.INT32))
         page_idx = pl.tile.shrs(logical, 7)
         page_ids = pl.tile.gather(materialize_displacements, page_idx, gtmp)
         physical = pl.add(logical, page_ids)
         pl.store(pl.tile.set_validshape(physical, 1, selected), [t, count], indices)
         count = count + selected
   for pack_task in pl.spmd(1, name_hint='pack_auxiliary'):
    for output_row in pl.range(0, M, 16):
     cutoff_lines = pl.load(aux_cutoffs, [output_row, 0], [16, 16])
     quota_lines = pl.load(aux_quotas, [output_row, 0], [16, 16])
     cutoff_flat = pl.reshape(cutoff_lines, [1, 256])
     quota_flat = pl.reshape(quota_lines, [1, 256])
     pack_offsets = pl.mul(pl.tile.arange(0, [1, 16], dtype=pl.INT32), 16)
     cutoff_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     quota_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     cutoff_packed = pl.tile.gather(cutoff_flat, pack_offsets, cutoff_scratch)
     quota_packed = pl.tile.gather(quota_flat, pack_offsets, quota_scratch)
     pl.store(pl.reshape(cutoff_packed, [16, 1]), [output_row, 0], cutoffs)
     pl.store(pl.reshape(quota_packed, [16, 1]), [output_row, 0], quotas)
   return indices

 def specs():
  return [TensorSpec('scores', [M, N], torch.float32, init_value=torch.randn), TensorSpec('lengths', [M], torch.int32, init_value=lambda: torch.randint(1, N + 1, (M,), dtype=torch.int32)), TensorSpec('block_table', [1, P], torch.int32, init_value=lambda: torch.randperm(P, dtype=torch.int64).to(torch.int32).reshape(1, P)), TensorSpec('wpairs', [M, WORKSPACE_WIDTH], torch.float32, init_value=lambda: torch.full((M, WORKSPACE_WIDTH), 17.0)), TensorSpec('cutoffs', [M, 1], torch.float32, init_value=lambda: torch.full((M, 1), 23.0)), TensorSpec('quotas', [M, 1], torch.int32, init_value=lambda: torch.full((M, 1), 29, dtype=torch.int32)), TensorSpec('indices', [M, K], torch.int32)]

 def golden(t):
  vis = t['lengths'].long().clamp(0, N)
  bt = t['block_table'][0].long()
  wp = torch.zeros(M, WORKSPACE_WIDTH)
  cut = torch.zeros(M, 1)
  quo = torch.zeros(M, 1, dtype=torch.int32)
  out_idx = torch.full((M, K), -1, dtype=torch.int32)
  allpos = torch.arange(NL * LEAF, dtype=torch.int64)
  for m in range(M):
   v = int(vis[m])
   w = max(v, 1)
   padded = torch.full((NL * LEAF,), SORT_FLOOR, dtype=torch.float32)
   padded[:w] = t['scores'][m, :w].float().clamp_min(SORT_FLOOR)
   lc = (w + LEAF - 1) // LEAF
   for leaf in range(lc):
    seg = padded[leaf * LEAF:(leaf + 1) * LEAF]
    order = torch.argsort(seg, descending=True, stable=True)
    top = order[:K]
    wp[m, leaf * PAIR_W:(leaf + 1) * PAIR_W] = torch.stack([seg[top], top.to(torch.int32).view(torch.float32)], dim=-1).reshape(-1)
   order = torch.argsort(padded, descending=True, stable=True)
   top = order[:K]
   vals = padded[top]
   wp[m, 0:PAIR_W] = torch.stack([vals, top.to(torch.int32).view(torch.float32)], dim=-1).reshape(-1)
   cutoff = torch.clamp(vals, min=FIN_FLOOR)[K - 1]
   above = int((padded > cutoff).sum())
   quota = K - above
   cut[m, 0] = cutoff
   quo[m, 0] = quota
   sel = []
   rem = quota
   for p in range(v):
    sp = padded[p]
    if sp > cutoff:
     sel.append(p)
    elif sp == cutoff and rem > 0:
     sel.append(p)
     rem -= 1
    if len(sel) == K:
     break
   if sel:
    selt = torch.tensor(sel, dtype=torch.int64)
    out_idx[m, :len(sel)] = (bt[selt // PAGE] * PAGE + selt % PAGE).to(torch.int32)
  t['wpairs'][:] = wp
  t['cutoffs'][:] = cut
  t['quotas'][:] = quo
  t['indices'][:] = out_idx
 return (dsa41_topk_guarded_root, specs(), golden, RTOL, ATOL)

def _build_bf16(shape):
 M, N = (shape['M'], shape['N'])
 P = N // PAGE
 P8 = (P + 7) // 8 * 8
 NL = (N + LEAF - 1) // LEAF
 LEAF_COHORTS = (M + LEAF_ROWS - 1) // LEAF_ROWS
 WORKSPACE_WIDTH = (N + 1023) // 1024 * 1024
 SHORT_LIMIT = min(K, N - 1)
 if N <= 512:

  @pl.jit
  def dsa41_topk_guarded_root(scores: pl.Tensor[[M, N], pl.BF16], lengths: pl.Tensor[[M], pl.INT32], block_table: pl.Tensor[[1, P], pl.INT32], wpairs: pl.InOut[pl.Tensor[[M, WORKSPACE_WIDTH], pl.FP32]], cutoffs: pl.InOut[pl.Tensor[[M, 1], pl.FP32]], quotas: pl.InOut[pl.Tensor[[M, 1], pl.INT32]], indices: pl.Out[pl.Tensor[[M, K], pl.INT32]]):
   aux_discarded = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_cutoffs = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_quotas = pl.create_tensor([M, 16], dtype=pl.INT32)
   aux_front = pl.create_tensor([M, 16], dtype=pl.INT32)
   for cohort_base in pl.parallel(0, M, ROW_COHORT):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='decomposed_leaf'):
     for t in pl.range(cohort_base, pl.min(cohort_base + ROW_COHORT, M), 1):
      discarded = pl.cast(SORT_FLOOR, pl.FP32)
      frontier_position = pl.cast(2147483647, pl.INT32)
      zero_pairs = pl.tile.full([1, PAIR_W], dtype=pl.FP32, value=0.0)
      for clear_offset in pl.range(0, WORKSPACE_WIDTH, PAIR_W):
       pl.store(zero_pairs, [t, clear_offset], wpairs)
      visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
      wv = pl.max(visible, 1)
      for leaf in pl.range(NL):
       begin = leaf * LEAF
       if begin < wv:
        valid = pl.min(LEAF, wv - begin)
        narrow_raw = pl.cast(pl.load(scores, [t, begin], [1, 512], valid_shape=[1, valid]), pl.FP32)
        narrow_padded = pl.tile.fillpad(narrow_raw, pad_value=pl.PadValue.min)
        narrow_values = pl.maximum(narrow_padded, SORT_FLOOR)
        narrow_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
        narrow_pairs = pl.tile.sort32(narrow_values, pl.reinterpret_view(narrow_positions, pl.UINT32))
        narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=64)
        narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=256)
        pl.store(narrow_pairs, [t, leaf * PAIR_W], wpairs)
      pl.write(aux_discarded, [t, 0], discarded)
      pl.write(aux_front, [t, 0], frontier_position)
   for cohort_base in pl.parallel(0, M, ROW_COHORT):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='decomposed_merge'):
     for t in pl.range(cohort_base, pl.min(cohort_base + ROW_COHORT, M), 1):
      visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
      wv = pl.max(visible, 1)
      lc = (wv + LEAF - 1) // LEAF
      discarded = pl.read(aux_discarded, [t, 0])
      frontier_position = pl.read(aux_front, [t, 0])
      mtmp = pl.tile.create([1, 2 * PAIR_W], dtype=pl.FP32)
      for cb in pl.range(1, lc):
       mleft = pl.load(wpairs, [t, 0], [1, PAIR_W])
       mright = pl.load(wpairs, [t, cb * PAIR_W], [1, PAIR_W])
       mboth = pl.tile.mrgsort(mleft, mright, tmp=mtmp)
       merge_discarded = pl.tile.read(mboth, [0, PAIR_W])
       if merge_discarded > discarded:
        discarded = merge_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if merge_discarded == discarded:
        merge_front_boundary = pl.tile.read(mboth, [0, PAIR_W - 2])
        merge_front_bound = merge_front_boundary
        if merge_front_bound < FIN_FLOOR:
         merge_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if merge_discarded >= merge_front_bound:
         merge_front_tail = pl.tile.slice(mboth, [1, 1024], [0, PAIR_W])
         merge_front_values = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
         merge_front_positions = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         merge_front_candidates = pl.tile.select(pl.tile.cmps(merge_front_values, merge_discarded, cmp_type=0), merge_front_positions, 2147483647)
         merge_front_wide = pl.col_min(pl.reshape(merge_front_candidates, [8, 64]))
         merge_front_eight = pl.col_min(pl.reshape(merge_front_wide, [8, 8]))
         merge_front_result = pl.tile.read(merge_front_eight, [0, 0])
         merge_front_lane_1 = pl.tile.read(merge_front_eight, [0, 1])
         if merge_front_lane_1 < merge_front_result:
          merge_front_result = merge_front_lane_1
         merge_front_lane_2 = pl.tile.read(merge_front_eight, [0, 2])
         if merge_front_lane_2 < merge_front_result:
          merge_front_result = merge_front_lane_2
         merge_front_lane_3 = pl.tile.read(merge_front_eight, [0, 3])
         if merge_front_lane_3 < merge_front_result:
          merge_front_result = merge_front_lane_3
         merge_front_lane_4 = pl.tile.read(merge_front_eight, [0, 4])
         if merge_front_lane_4 < merge_front_result:
          merge_front_result = merge_front_lane_4
         merge_front_lane_5 = pl.tile.read(merge_front_eight, [0, 5])
         if merge_front_lane_5 < merge_front_result:
          merge_front_result = merge_front_lane_5
         merge_front_lane_6 = pl.tile.read(merge_front_eight, [0, 6])
         if merge_front_lane_6 < merge_front_result:
          merge_front_result = merge_front_lane_6
         merge_front_lane_7 = pl.tile.read(merge_front_eight, [0, 7])
         if merge_front_lane_7 < merge_front_result:
          merge_front_result = merge_front_lane_7
         if merge_front_result < frontier_position:
          frontier_position = merge_front_result
       pl.store(pl.tile.slice(mboth, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
      pl.write(aux_discarded, [t, 0], discarded)
      pl.write(aux_front, [t, 0], frontier_position)
   for cohort_base in pl.parallel(0, M, ROW_COHORT):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='decomposed_materialize'):
     for t in pl.range(cohort_base, pl.min(cohort_base + ROW_COHORT, M), 1):
      fin = pl.load(wpairs, [t, 0], [1, PAIR_W])
      values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
      finite = pl.maximum(values, FIN_FLOOR)
      cutoff = pl.tile.read(finite, [0, K - 1])
      ones_c = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
      flags = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=4), ones_c, 0.0)
      sums64 = pl.col_sum(pl.reshape(flags, [8, 64]))
      sums8 = pl.col_sum(pl.reshape(sums64, [8, 8]))
      above = pl.cast(0, pl.INDEX)
      for lane in pl.unroll(8):
       part = pl.cast(pl.tile.read(sums8, [0, lane]), pl.INT32)
       above = above + pl.cast(part, pl.INDEX)
      cutoff_stage_line = pl.tile.full([1, 16], dtype=pl.FP32, value=0)
      pl.tile.write(cutoff_stage_line, [0, 0], cutoff)
      pl.store(cutoff_stage_line, [t, 0], aux_cutoffs)
      quota_stage_line = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
      pl.tile.write(quota_stage_line, [0, 0], pl.cast(K - above, pl.INT32))
      pl.store(quota_stage_line, [t, 0], aux_quotas)
      discarded = pl.read(aux_discarded, [t, 0])
      frontier_position = pl.read(aux_front, [t, 0])
      replay_safe = discarded < cutoff
      if discarded == cutoff:
       certificate_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       certificate_candidates = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=0), certificate_positions, -1)
       certificate_wide = pl.col_max(pl.reshape(certificate_candidates, [8, 64]))
       certificate_eight = pl.col_max(pl.reshape(certificate_wide, [8, 8]))
       certificate_result = pl.tile.read(certificate_eight, [0, 0])
       certificate_lane_1 = pl.tile.read(certificate_eight, [0, 1])
       if certificate_lane_1 > certificate_result:
        certificate_result = certificate_lane_1
       certificate_lane_2 = pl.tile.read(certificate_eight, [0, 2])
       if certificate_lane_2 > certificate_result:
        certificate_result = certificate_lane_2
       certificate_lane_3 = pl.tile.read(certificate_eight, [0, 3])
       if certificate_lane_3 > certificate_result:
        certificate_result = certificate_lane_3
       certificate_lane_4 = pl.tile.read(certificate_eight, [0, 4])
       if certificate_lane_4 > certificate_result:
        certificate_result = certificate_lane_4
       certificate_lane_5 = pl.tile.read(certificate_eight, [0, 5])
       if certificate_lane_5 > certificate_result:
        certificate_result = certificate_lane_5
       certificate_lane_6 = pl.tile.read(certificate_eight, [0, 6])
       if certificate_lane_6 > certificate_result:
        certificate_result = certificate_lane_6
       certificate_lane_7 = pl.tile.read(certificate_eight, [0, 7])
       if certificate_lane_7 > certificate_result:
        certificate_result = certificate_lane_7
       if frontier_position > certificate_result:
        replay_safe = True
      if replay_safe:
       if cutoff > FIN_FLOOR:
        root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        selected = pl.cast(K, pl.INDEX)
        if selected > 0:
         root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(root_positions, pl.FP32))
         root_keys = root_priority
         root_order = pl.tile.sort32(root_keys, pl.reinterpret_view(root_positions, pl.UINT32))
         root_order = pl.tile.mrgsort(root_order, block_len=64)
         root_order = pl.tile.mrgsort(root_order, block_len=256)
         root_logical = pl.tile.gather_mask(root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         root_valid = pl.tile.set_validshape(root_logical, 1, selected)
         root_safe = pl.tile.fillpad(root_valid, pad_value=pl.PadValue.zero)
         if P == 1:
          single_page = pl.read(block_table, [0, 0])
          pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), single_page), pad_value=pl.PadValue.zero)
         else:
          pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
         root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for map_start in pl.range(0, selected, MAP):
          map_count = pl.min(MAP, selected - map_start)
          map_bits = pl.tile.slice(pl.reinterpret_view(root_safe, pl.FP32), [1, MAP], [0, map_start])
          map_logical = pl.reinterpret_view(map_bits, pl.INT32)
          root_page_idx = pl.tile.shrs(map_logical, 7)
          root_page_ids = pl.tile.gather(pages, root_page_idx, root_tmp)
          root_offsets = pl.sub(map_logical, pl.mul(root_page_idx, PAGE))
          root_physical = pl.add(pl.mul(root_page_ids, PAGE), root_offsets)
          pl.store(pl.tile.set_validshape(root_physical, 1, map_count), [t, map_start], indices)
       else:
        visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
        root = fin
        root_values = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P0101)
        root_positions = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
        root_strict = pl.tile.select(pl.tile.cmps(root_values, cutoff, cmp_type=4), ones_root, 0.0)
        root_equal = pl.tile.select(pl.tile.cmps(root_values, cutoff, cmp_type=0), ones_root, 0.0)
        root_eligible = pl.maximum(root_strict, root_equal)
        root_visible = pl.tile.cmps(pl.cast(root_positions, pl.FP32), pl.cast(pl.cast(visible, pl.INT32), pl.FP32), cmp_type=2)
        root_keep = pl.tile.select(root_visible, root_eligible, 0.0)
        root_sums64 = pl.col_sum(pl.reshape(root_keep, [8, 64]))
        root_sums8 = pl.col_sum(pl.reshape(root_sums64, [8, 8]))
        selected = pl.cast(0, pl.INDEX)
        for root_lane in pl.unroll(8):
         root_part = pl.cast(pl.tile.read(root_sums8, [0, root_lane]), pl.INT32)
         selected = selected + pl.cast(root_part, pl.INDEX)
        pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
        if selected > 0:
         root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(root_positions, pl.FP32))
         root_keys = pl.tile.select(pl.tile.cmps(root_keep, 0.0, cmp_type=4), root_priority, -1.0)
         root_order = pl.tile.sort32(root_keys, pl.reinterpret_view(root_positions, pl.UINT32))
         root_order = pl.tile.mrgsort(root_order, block_len=64)
         root_order = pl.tile.mrgsort(root_order, block_len=256)
         root_sorted_keys = pl.tile.gather_mask(root_order, mask_pattern=pl.tile.MaskPattern.P0101)
         root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), root_sorted_keys), pl.INT32)
         root_valid = pl.tile.set_validshape(root_logical, 1, selected)
         root_safe = pl.tile.fillpad(root_valid, pad_value=pl.PadValue.zero)
         if P == 1:
          single_page = pl.read(block_table, [0, 0])
          pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), single_page), pad_value=pl.PadValue.zero)
         else:
          pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
         root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for map_start in pl.range(0, selected, MAP):
          map_count = pl.min(MAP, selected - map_start)
          map_bits = pl.tile.slice(pl.reinterpret_view(root_safe, pl.FP32), [1, MAP], [0, map_start])
          map_logical = pl.reinterpret_view(map_bits, pl.INT32)
          root_page_idx = pl.tile.shrs(map_logical, 7)
          root_page_ids = pl.tile.gather(pages, root_page_idx, root_tmp)
          root_offsets = pl.sub(map_logical, pl.mul(root_page_idx, PAGE))
          root_physical = pl.add(pl.mul(root_page_ids, PAGE), root_offsets)
          pl.store(pl.tile.set_validshape(root_physical, 1, map_count), [t, map_start], indices)
      else:
       visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
       remaining = pl.cast(K - above, pl.INDEX)
       if P == 1:
        single_page = pl.read(block_table, [0, 0])
        pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), single_page), pad_value=pl.PadValue.zero)
       else:
        pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
       pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
       ones_m = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
       lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
       lanes_f = pl.cast(lanes, pl.FP32)
       lane_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
       priority = pl.cast(pl.sub(lane_origin, lanes), pl.FP32)
       gtmp = pl.tile.create([1, SCAN], dtype=pl.INT32)
       count = pl.cast(0, pl.INDEX)
       for start in pl.range(0, visible, SCAN):
        if count < K:
         valid = pl.min(SCAN, visible - start)
         scan_raw = pl.cast(pl.load(scores, [t, start], [1, SCAN], valid_shape=[1, valid]), pl.FP32)
         scan_vals = pl.tile.fillpad(scan_raw, pad_value=pl.PadValue.min)
         strict = pl.tile.select(pl.tile.cmps(scan_vals, cutoff, cmp_type=4), ones_m, 0.0)
         ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
         if remaining > 0:
          eq = pl.tile.select(pl.tile.cmps(scan_vals, cutoff, cmp_type=0), ones_m, 0.0)
          eq_valid = pl.tile.cmps(lanes_f, pl.cast(pl.cast(valid, pl.INT32), pl.FP32), cmp_type=2)
          eq = pl.tile.select(eq_valid, eq, 0.0)
          eq_sums64 = pl.col_sum(pl.reshape(eq, [SCAN // 64, 64]))
          eq_sums8 = pl.col_sum(pl.reshape(eq_sums64, [8, 8]))
          available = pl.cast(0, pl.INDEX)
          for eq_lane in pl.unroll(8):
           eq_part = pl.cast(pl.tile.read(eq_sums8, [0, eq_lane]), pl.INT32)
           available = available + pl.cast(eq_part, pl.INDEX)
          take = pl.min(available, remaining)
          if take > 0:
           if take == available:
            ties = eq
           else:
            tie_keys = pl.mul(eq, priority)
            tie_pairs = pl.tile.sort32(tie_keys, pl.reinterpret_view(lanes, pl.UINT32))
            tie_pairs = pl.tile.mrgsort(tie_pairs, block_len=64)
            tie_merged = tie_pairs
            equal_keys = pl.tile.gather_mask(tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
            last_key_f = pl.tile.read(equal_keys, [0, take - 1])
            boundary_f = 128.0 - last_key_f
            ties = pl.tile.select(pl.tile.cmps(lanes_f, boundary_f, cmp_type=3), eq, 0.0)
           remaining = remaining - take
         keep = pl.maximum(strict, ties)
         keep_valid = pl.tile.set_validshape(keep, 1, valid)
         keep_padded = pl.tile.fillpad(keep_valid, pad_value=pl.PadValue.zero)
         keep_sums64 = pl.col_sum(pl.reshape(keep_padded, [SCAN // 64, 64]))
         keep_sums8 = pl.col_sum(pl.reshape(keep_sums64, [8, 8]))
         keep_count = pl.cast(0, pl.INDEX)
         for keep_lane in pl.unroll(8):
          keep_part = pl.cast(pl.tile.read(keep_sums8, [0, keep_lane]), pl.INT32)
          keep_count = keep_count + pl.cast(keep_part, pl.INDEX)
         selected = pl.min(keep_count, K - count)
         if selected > 0:
          order_keys = pl.mul(keep_padded, priority)
          order_pairs = pl.tile.sort32(order_keys, pl.reinterpret_view(lanes, pl.UINT32))
          order_pairs = pl.tile.mrgsort(order_pairs, block_len=64)
          order_merged = order_pairs
          order_vals = pl.tile.gather_mask(order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
          offs = pl.sub(lane_origin, pl.cast(order_vals, pl.INT32))
          compact_valid = pl.tile.set_validshape(offs, 1, selected)
          compact_safe = pl.tile.fillpad(compact_valid, pad_value=pl.PadValue.zero)
          logical = pl.add(compact_safe, pl.cast(start, pl.INT32))
          page_idx = pl.tile.shrs(logical, 7)
          page_ids = pl.tile.gather(pages, page_idx, gtmp)
          offs_in_page = pl.sub(logical, pl.mul(page_idx, 128))
          physical = pl.add(pl.mul(page_ids, 128), offs_in_page)
          pl.store(pl.tile.set_validshape(physical, 1, selected), [t, count], indices)
          count = count + selected
   for pack_task in pl.spmd(1, name_hint='pack_auxiliary'):
    for output_row in pl.range(0, M, 16):
     cutoff_lines = pl.load(aux_cutoffs, [output_row, 0], [16, 16])
     quota_lines = pl.load(aux_quotas, [output_row, 0], [16, 16])
     cutoff_flat = pl.reshape(cutoff_lines, [1, 256])
     quota_flat = pl.reshape(quota_lines, [1, 256])
     pack_offsets = pl.mul(pl.tile.arange(0, [1, 16], dtype=pl.INT32), 16)
     cutoff_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     quota_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     cutoff_packed = pl.tile.gather(cutoff_flat, pack_offsets, cutoff_scratch)
     quota_packed = pl.tile.gather(quota_flat, pack_offsets, quota_scratch)
     pl.store(pl.reshape(cutoff_packed, [16, 1]), [output_row, 0], cutoffs)
     pl.store(pl.reshape(quota_packed, [16, 1]), [output_row, 0], quotas)
   return indices
 elif N <= 1024:

  @pl.jit
  def dsa41_topk_guarded_root(scores: pl.Tensor[[M, N], pl.BF16], lengths: pl.Tensor[[M], pl.INT32], block_table: pl.Tensor[[1, P], pl.INT32], wpairs: pl.InOut[pl.Tensor[[M, WORKSPACE_WIDTH], pl.FP32]], cutoffs: pl.InOut[pl.Tensor[[M, 1], pl.FP32]], quotas: pl.InOut[pl.Tensor[[M, 1], pl.INT32]], indices: pl.Out[pl.Tensor[[M, K], pl.INT32]]):
   aux_discarded = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_cutoffs = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_quotas = pl.create_tensor([M, 16], dtype=pl.INT32)
   aux_front = pl.create_tensor([M, 16], dtype=pl.INT32)
   for cohort_base in pl.parallel(0, M, ROW_COHORT):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='decomposed_leaf'):
     for t in pl.range(cohort_base, pl.min(cohort_base + ROW_COHORT, M), 1):
      discarded = pl.cast(SORT_FLOOR, pl.FP32)
      frontier_position = pl.cast(2147483647, pl.INT32)
      zero_pairs = pl.tile.full([1, PAIR_W], dtype=pl.FP32, value=0.0)
      for clear_offset in pl.range(0, WORKSPACE_WIDTH, PAIR_W):
       pl.store(zero_pairs, [t, clear_offset], wpairs)
      visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
      wv = pl.max(visible, 1)
      for leaf in pl.range(NL):
       begin = leaf * LEAF
       if begin < wv:
        valid = pl.min(LEAF, wv - begin)
        if valid <= 512:
         narrow_raw = pl.cast(pl.load(scores, [t, begin], [1, 512], valid_shape=[1, valid]), pl.FP32)
         narrow_padded = pl.tile.fillpad(narrow_raw, pad_value=pl.PadValue.min)
         narrow_values = pl.maximum(narrow_padded, SORT_FLOOR)
         narrow_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
         narrow_pairs = pl.tile.sort32(narrow_values, pl.reinterpret_view(narrow_positions, pl.UINT32))
         narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=64)
         narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=256)
         pl.store(narrow_pairs, [t, leaf * PAIR_W], wpairs)
        else:
         left_valid = pl.min(valid, 512)
         right_valid = pl.max(valid - 512, 0)
         left_raw = pl.cast(pl.load(scores, [t, begin], [1, 512], valid_shape=[1, left_valid]), pl.FP32)
         right_raw = pl.cast(pl.load(scores, [t, begin + 512], [1, 512], valid_shape=[1, right_valid]), pl.FP32)
         left_values = pl.maximum(pl.tile.fillpad(left_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
         right_values = pl.maximum(pl.tile.fillpad(right_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
         left_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
         right_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin + 512, pl.INT32))
         left_pairs = pl.tile.sort32(left_values, pl.reinterpret_view(left_positions, pl.UINT32))
         left_pairs = pl.tile.mrgsort(left_pairs, block_len=64)
         left_pairs = pl.tile.mrgsort(left_pairs, block_len=256)
         right_pairs = pl.tile.sort32(right_values, pl.reinterpret_view(right_positions, pl.UINT32))
         right_pairs = pl.tile.mrgsort(right_pairs, block_len=64)
         right_pairs = pl.tile.mrgsort(right_pairs, block_len=256)
         native_scratch = pl.tile.create([1, 2048], dtype=pl.FP32)
         native_pairs = pl.tile.mrgsort(left_pairs, right_pairs, tmp=native_scratch)
         native_discarded = pl.tile.read(native_pairs, [0, PAIR_W])
         if native_discarded > discarded:
          discarded = native_discarded
          frontier_position = pl.cast(2147483647, pl.INT32)
         if native_discarded == discarded:
          native_front_boundary = pl.tile.read(native_pairs, [0, PAIR_W - 2])
          native_front_bound = native_front_boundary
          if native_front_bound < FIN_FLOOR:
           native_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
          if native_discarded >= native_front_bound:
           native_arity_third = pl.tile.read(native_pairs, [0, PAIR_W + 4])
           if native_arity_third < native_discarded:
            native_arity_second = pl.tile.read(native_pairs, [0, PAIR_W + 2])
            native_arity_bits = pl.reinterpret_view(native_pairs, pl.INT32)
            native_front_result = pl.tile.read(native_arity_bits, [0, PAIR_W + 1])
            if native_arity_second == native_discarded:
             native_arity_second_position = pl.tile.read(native_arity_bits, [0, PAIR_W + 3])
             native_front_result = pl.min(native_front_result, native_arity_second_position)
           else:
            native_front_tail = pl.tile.slice(native_pairs, [1, 1024], [0, PAIR_W])
            native_front_values = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
            native_front_positions = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            native_front_candidates = pl.tile.select(pl.tile.cmps(native_front_values, native_discarded, cmp_type=0), native_front_positions, 2147483647)
            native_front_wide = pl.col_min(pl.reshape(native_front_candidates, [8, 64]))
            native_front_eight = pl.col_min(pl.reshape(native_front_wide, [8, 8]))
            native_front_result = pl.tile.read(native_front_eight, [0, 0])
            native_front_lane_1 = pl.tile.read(native_front_eight, [0, 1])
            if native_front_lane_1 < native_front_result:
             native_front_result = native_front_lane_1
            native_front_lane_2 = pl.tile.read(native_front_eight, [0, 2])
            if native_front_lane_2 < native_front_result:
             native_front_result = native_front_lane_2
            native_front_lane_3 = pl.tile.read(native_front_eight, [0, 3])
            if native_front_lane_3 < native_front_result:
             native_front_result = native_front_lane_3
            native_front_lane_4 = pl.tile.read(native_front_eight, [0, 4])
            if native_front_lane_4 < native_front_result:
             native_front_result = native_front_lane_4
            native_front_lane_5 = pl.tile.read(native_front_eight, [0, 5])
            if native_front_lane_5 < native_front_result:
             native_front_result = native_front_lane_5
            native_front_lane_6 = pl.tile.read(native_front_eight, [0, 6])
            if native_front_lane_6 < native_front_result:
             native_front_result = native_front_lane_6
            native_front_lane_7 = pl.tile.read(native_front_eight, [0, 7])
            if native_front_lane_7 < native_front_result:
             native_front_result = native_front_lane_7
           if native_front_result < frontier_position:
            frontier_position = native_front_result
         pl.store(pl.tile.slice(native_pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
      pl.write(aux_discarded, [t, 0], discarded)
      pl.write(aux_front, [t, 0], frontier_position)
   for cohort_base in pl.parallel(0, M, ROW_COHORT):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='decomposed_merge'):
     for t in pl.range(cohort_base, pl.min(cohort_base + ROW_COHORT, M), 1):
      visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
      wv = pl.max(visible, 1)
      lc = (wv + LEAF - 1) // LEAF
      discarded = pl.read(aux_discarded, [t, 0])
      frontier_position = pl.read(aux_front, [t, 0])
      mtmp = pl.tile.create([1, 2 * PAIR_W], dtype=pl.FP32)
      for cb in pl.range(1, lc):
       mleft = pl.load(wpairs, [t, 0], [1, PAIR_W])
       mright = pl.load(wpairs, [t, cb * PAIR_W], [1, PAIR_W])
       mboth = pl.tile.mrgsort(mleft, mright, tmp=mtmp)
       merge_discarded = pl.tile.read(mboth, [0, PAIR_W])
       if merge_discarded > discarded:
        discarded = merge_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if merge_discarded == discarded:
        merge_front_boundary = pl.tile.read(mboth, [0, PAIR_W - 2])
        merge_front_bound = merge_front_boundary
        if merge_front_bound < FIN_FLOOR:
         merge_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if merge_discarded >= merge_front_bound:
         merge_front_tail = pl.tile.slice(mboth, [1, 1024], [0, PAIR_W])
         merge_front_values = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
         merge_front_positions = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         merge_front_candidates = pl.tile.select(pl.tile.cmps(merge_front_values, merge_discarded, cmp_type=0), merge_front_positions, 2147483647)
         merge_front_wide = pl.col_min(pl.reshape(merge_front_candidates, [8, 64]))
         merge_front_eight = pl.col_min(pl.reshape(merge_front_wide, [8, 8]))
         merge_front_result = pl.tile.read(merge_front_eight, [0, 0])
         merge_front_lane_1 = pl.tile.read(merge_front_eight, [0, 1])
         if merge_front_lane_1 < merge_front_result:
          merge_front_result = merge_front_lane_1
         merge_front_lane_2 = pl.tile.read(merge_front_eight, [0, 2])
         if merge_front_lane_2 < merge_front_result:
          merge_front_result = merge_front_lane_2
         merge_front_lane_3 = pl.tile.read(merge_front_eight, [0, 3])
         if merge_front_lane_3 < merge_front_result:
          merge_front_result = merge_front_lane_3
         merge_front_lane_4 = pl.tile.read(merge_front_eight, [0, 4])
         if merge_front_lane_4 < merge_front_result:
          merge_front_result = merge_front_lane_4
         merge_front_lane_5 = pl.tile.read(merge_front_eight, [0, 5])
         if merge_front_lane_5 < merge_front_result:
          merge_front_result = merge_front_lane_5
         merge_front_lane_6 = pl.tile.read(merge_front_eight, [0, 6])
         if merge_front_lane_6 < merge_front_result:
          merge_front_result = merge_front_lane_6
         merge_front_lane_7 = pl.tile.read(merge_front_eight, [0, 7])
         if merge_front_lane_7 < merge_front_result:
          merge_front_result = merge_front_lane_7
         if merge_front_result < frontier_position:
          frontier_position = merge_front_result
       pl.store(pl.tile.slice(mboth, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
      pl.write(aux_discarded, [t, 0], discarded)
      pl.write(aux_front, [t, 0], frontier_position)
   for cohort_base in pl.parallel(0, M, ROW_COHORT):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='decomposed_materialize'):
     for t in pl.range(cohort_base, pl.min(cohort_base + ROW_COHORT, M), 1):
      fin = pl.load(wpairs, [t, 0], [1, PAIR_W])
      values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
      finite = pl.maximum(values, FIN_FLOOR)
      cutoff = pl.tile.read(finite, [0, K - 1])
      ones_c = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
      flags = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=4), ones_c, 0.0)
      sums64 = pl.col_sum(pl.reshape(flags, [8, 64]))
      sums8 = pl.col_sum(pl.reshape(sums64, [8, 8]))
      above = pl.cast(0, pl.INDEX)
      for lane in pl.unroll(8):
       part = pl.cast(pl.tile.read(sums8, [0, lane]), pl.INT32)
       above = above + pl.cast(part, pl.INDEX)
      cutoff_stage_line = pl.tile.full([1, 16], dtype=pl.FP32, value=0)
      pl.tile.write(cutoff_stage_line, [0, 0], cutoff)
      pl.store(cutoff_stage_line, [t, 0], aux_cutoffs)
      quota_stage_line = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
      pl.tile.write(quota_stage_line, [0, 0], pl.cast(K - above, pl.INT32))
      pl.store(quota_stage_line, [t, 0], aux_quotas)
      discarded = pl.read(aux_discarded, [t, 0])
      frontier_position = pl.read(aux_front, [t, 0])
      replay_safe = discarded < cutoff
      if discarded == cutoff:
       certificate_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       certificate_candidates = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=0), certificate_positions, -1)
       certificate_wide = pl.col_max(pl.reshape(certificate_candidates, [8, 64]))
       certificate_eight = pl.col_max(pl.reshape(certificate_wide, [8, 8]))
       certificate_result = pl.tile.read(certificate_eight, [0, 0])
       certificate_lane_1 = pl.tile.read(certificate_eight, [0, 1])
       if certificate_lane_1 > certificate_result:
        certificate_result = certificate_lane_1
       certificate_lane_2 = pl.tile.read(certificate_eight, [0, 2])
       if certificate_lane_2 > certificate_result:
        certificate_result = certificate_lane_2
       certificate_lane_3 = pl.tile.read(certificate_eight, [0, 3])
       if certificate_lane_3 > certificate_result:
        certificate_result = certificate_lane_3
       certificate_lane_4 = pl.tile.read(certificate_eight, [0, 4])
       if certificate_lane_4 > certificate_result:
        certificate_result = certificate_lane_4
       certificate_lane_5 = pl.tile.read(certificate_eight, [0, 5])
       if certificate_lane_5 > certificate_result:
        certificate_result = certificate_lane_5
       certificate_lane_6 = pl.tile.read(certificate_eight, [0, 6])
       if certificate_lane_6 > certificate_result:
        certificate_result = certificate_lane_6
       certificate_lane_7 = pl.tile.read(certificate_eight, [0, 7])
       if certificate_lane_7 > certificate_result:
        certificate_result = certificate_lane_7
       if frontier_position > certificate_result:
        replay_safe = True
      if replay_safe:
       if cutoff > FIN_FLOOR:
        root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        selected = pl.cast(K, pl.INDEX)
        if selected > 0:
         root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(root_positions, pl.FP32))
         root_keys = root_priority
         root_order = pl.tile.sort32(root_keys, pl.reinterpret_view(root_positions, pl.UINT32))
         root_order = pl.tile.mrgsort(root_order, block_len=64)
         root_order = pl.tile.mrgsort(root_order, block_len=256)
         root_logical = pl.tile.gather_mask(root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         root_valid = pl.tile.set_validshape(root_logical, 1, selected)
         root_safe = pl.tile.fillpad(root_valid, pad_value=pl.PadValue.zero)
         if P == 1:
          single_page = pl.read(block_table, [0, 0])
          pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), single_page), pad_value=pl.PadValue.zero)
         else:
          pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
         root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for map_start in pl.range(0, selected, MAP):
          map_count = pl.min(MAP, selected - map_start)
          map_bits = pl.tile.slice(pl.reinterpret_view(root_safe, pl.FP32), [1, MAP], [0, map_start])
          map_logical = pl.reinterpret_view(map_bits, pl.INT32)
          root_page_idx = pl.tile.shrs(map_logical, 7)
          root_page_ids = pl.tile.gather(pages, root_page_idx, root_tmp)
          root_offsets = pl.sub(map_logical, pl.mul(root_page_idx, PAGE))
          root_physical = pl.add(pl.mul(root_page_ids, PAGE), root_offsets)
          pl.store(pl.tile.set_validshape(root_physical, 1, map_count), [t, map_start], indices)
       else:
        visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
        root = fin
        root_values = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P0101)
        root_positions = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
        root_strict = pl.tile.select(pl.tile.cmps(root_values, cutoff, cmp_type=4), ones_root, 0.0)
        root_equal = pl.tile.select(pl.tile.cmps(root_values, cutoff, cmp_type=0), ones_root, 0.0)
        root_eligible = pl.maximum(root_strict, root_equal)
        root_visible = pl.tile.cmps(pl.cast(root_positions, pl.FP32), pl.cast(pl.cast(visible, pl.INT32), pl.FP32), cmp_type=2)
        root_keep = pl.tile.select(root_visible, root_eligible, 0.0)
        root_sums64 = pl.col_sum(pl.reshape(root_keep, [8, 64]))
        root_sums8 = pl.col_sum(pl.reshape(root_sums64, [8, 8]))
        selected = pl.cast(0, pl.INDEX)
        for root_lane in pl.unroll(8):
         root_part = pl.cast(pl.tile.read(root_sums8, [0, root_lane]), pl.INT32)
         selected = selected + pl.cast(root_part, pl.INDEX)
        pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
        if selected > 0:
         root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(root_positions, pl.FP32))
         root_keys = pl.tile.select(pl.tile.cmps(root_keep, 0.0, cmp_type=4), root_priority, -1.0)
         root_order = pl.tile.sort32(root_keys, pl.reinterpret_view(root_positions, pl.UINT32))
         root_order = pl.tile.mrgsort(root_order, block_len=64)
         root_order = pl.tile.mrgsort(root_order, block_len=256)
         root_sorted_keys = pl.tile.gather_mask(root_order, mask_pattern=pl.tile.MaskPattern.P0101)
         root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), root_sorted_keys), pl.INT32)
         root_valid = pl.tile.set_validshape(root_logical, 1, selected)
         root_safe = pl.tile.fillpad(root_valid, pad_value=pl.PadValue.zero)
         if P == 1:
          single_page = pl.read(block_table, [0, 0])
          pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), single_page), pad_value=pl.PadValue.zero)
         else:
          pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
         root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for map_start in pl.range(0, selected, MAP):
          map_count = pl.min(MAP, selected - map_start)
          map_bits = pl.tile.slice(pl.reinterpret_view(root_safe, pl.FP32), [1, MAP], [0, map_start])
          map_logical = pl.reinterpret_view(map_bits, pl.INT32)
          root_page_idx = pl.tile.shrs(map_logical, 7)
          root_page_ids = pl.tile.gather(pages, root_page_idx, root_tmp)
          root_offsets = pl.sub(map_logical, pl.mul(root_page_idx, PAGE))
          root_physical = pl.add(pl.mul(root_page_ids, PAGE), root_offsets)
          pl.store(pl.tile.set_validshape(root_physical, 1, map_count), [t, map_start], indices)
      else:
       visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
       remaining = pl.cast(K - above, pl.INDEX)
       if P == 1:
        single_page = pl.read(block_table, [0, 0])
        pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), single_page), pad_value=pl.PadValue.zero)
       else:
        pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
       pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
       ones_m = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
       lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
       lanes_f = pl.cast(lanes, pl.FP32)
       lane_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
       priority = pl.cast(pl.sub(lane_origin, lanes), pl.FP32)
       gtmp = pl.tile.create([1, SCAN], dtype=pl.INT32)
       count = pl.cast(0, pl.INDEX)
       for start in pl.range(0, visible, SCAN):
        if count < K:
         valid = pl.min(SCAN, visible - start)
         scan_raw = pl.cast(pl.load(scores, [t, start], [1, SCAN], valid_shape=[1, valid]), pl.FP32)
         scan_vals = pl.tile.fillpad(scan_raw, pad_value=pl.PadValue.min)
         strict = pl.tile.select(pl.tile.cmps(scan_vals, cutoff, cmp_type=4), ones_m, 0.0)
         ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
         if remaining > 0:
          eq = pl.tile.select(pl.tile.cmps(scan_vals, cutoff, cmp_type=0), ones_m, 0.0)
          eq_valid = pl.tile.cmps(lanes_f, pl.cast(pl.cast(valid, pl.INT32), pl.FP32), cmp_type=2)
          eq = pl.tile.select(eq_valid, eq, 0.0)
          eq_sums64 = pl.col_sum(pl.reshape(eq, [SCAN // 64, 64]))
          eq_sums8 = pl.col_sum(pl.reshape(eq_sums64, [8, 8]))
          available = pl.cast(0, pl.INDEX)
          for eq_lane in pl.unroll(8):
           eq_part = pl.cast(pl.tile.read(eq_sums8, [0, eq_lane]), pl.INT32)
           available = available + pl.cast(eq_part, pl.INDEX)
          take = pl.min(available, remaining)
          if take > 0:
           if take == available:
            ties = eq
           else:
            tie_keys = pl.mul(eq, priority)
            tie_pairs = pl.tile.sort32(tie_keys, pl.reinterpret_view(lanes, pl.UINT32))
            tie_pairs = pl.tile.mrgsort(tie_pairs, block_len=64)
            tie_merged = tie_pairs
            equal_keys = pl.tile.gather_mask(tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
            last_key_f = pl.tile.read(equal_keys, [0, take - 1])
            boundary_f = 128.0 - last_key_f
            ties = pl.tile.select(pl.tile.cmps(lanes_f, boundary_f, cmp_type=3), eq, 0.0)
           remaining = remaining - take
         keep = pl.maximum(strict, ties)
         keep_valid = pl.tile.set_validshape(keep, 1, valid)
         keep_padded = pl.tile.fillpad(keep_valid, pad_value=pl.PadValue.zero)
         keep_sums64 = pl.col_sum(pl.reshape(keep_padded, [SCAN // 64, 64]))
         keep_sums8 = pl.col_sum(pl.reshape(keep_sums64, [8, 8]))
         keep_count = pl.cast(0, pl.INDEX)
         for keep_lane in pl.unroll(8):
          keep_part = pl.cast(pl.tile.read(keep_sums8, [0, keep_lane]), pl.INT32)
          keep_count = keep_count + pl.cast(keep_part, pl.INDEX)
         selected = pl.min(keep_count, K - count)
         if selected > 0:
          order_keys = pl.mul(keep_padded, priority)
          order_pairs = pl.tile.sort32(order_keys, pl.reinterpret_view(lanes, pl.UINT32))
          order_pairs = pl.tile.mrgsort(order_pairs, block_len=64)
          order_merged = order_pairs
          order_vals = pl.tile.gather_mask(order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
          offs = pl.sub(lane_origin, pl.cast(order_vals, pl.INT32))
          compact_valid = pl.tile.set_validshape(offs, 1, selected)
          compact_safe = pl.tile.fillpad(compact_valid, pad_value=pl.PadValue.zero)
          logical = pl.add(compact_safe, pl.cast(start, pl.INT32))
          page_idx = pl.tile.shrs(logical, 7)
          page_ids = pl.tile.gather(pages, page_idx, gtmp)
          offs_in_page = pl.sub(logical, pl.mul(page_idx, 128))
          physical = pl.add(pl.mul(page_ids, 128), offs_in_page)
          pl.store(pl.tile.set_validshape(physical, 1, selected), [t, count], indices)
          count = count + selected
   for pack_task in pl.spmd(1, name_hint='pack_auxiliary'):
    for output_row in pl.range(0, M, 16):
     cutoff_lines = pl.load(aux_cutoffs, [output_row, 0], [16, 16])
     quota_lines = pl.load(aux_quotas, [output_row, 0], [16, 16])
     cutoff_flat = pl.reshape(cutoff_lines, [1, 256])
     quota_flat = pl.reshape(quota_lines, [1, 256])
     pack_offsets = pl.mul(pl.tile.arange(0, [1, 16], dtype=pl.INT32), 16)
     cutoff_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     quota_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     cutoff_packed = pl.tile.gather(cutoff_flat, pack_offsets, cutoff_scratch)
     quota_packed = pl.tile.gather(quota_flat, pack_offsets, quota_scratch)
     pl.store(pl.reshape(cutoff_packed, [16, 1]), [output_row, 0], cutoffs)
     pl.store(pl.reshape(quota_packed, [16, 1]), [output_row, 0], quotas)
   return indices
 elif 8192 <= N <= 32768:

  @pl.jit
  def dsa41_topk_guarded_root(scores: pl.Tensor[[M, N], pl.BF16], lengths: pl.Tensor[[M], pl.INT32], block_table: pl.Tensor[[1, P], pl.INT32], wpairs: pl.InOut[pl.Tensor[[M, WORKSPACE_WIDTH], pl.FP32]], cutoffs: pl.InOut[pl.Tensor[[M, 1], pl.FP32]], quotas: pl.InOut[pl.Tensor[[M, 1], pl.INT32]], indices: pl.Out[pl.Tensor[[M, K], pl.INT32]]):
   aux_discarded = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_cutoffs = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_quotas = pl.create_tensor([M, 16], dtype=pl.INT32)
   aux_front = pl.create_tensor([M, 16], dtype=pl.INT32)
   leaf_discarded_bank = pl.create_tensor([M, NL * 16], dtype=pl.FP32)
   leaf_front_bank = pl.create_tensor([M, NL * 16], dtype=pl.INT32)
   for leaf_job in pl.spmd(LEAF_COHORTS * NL, name_hint='canonical_leaf_plane'):
    leaf = leaf_job // LEAF_COHORTS
    leaf_cohort_base = leaf_job % LEAF_COHORTS * LEAF_ROWS
    for t in pl.range(leaf_cohort_base, pl.min(leaf_cohort_base + LEAF_ROWS, M), 1):
     zero_pairs = pl.tile.full([1, PAIR_W], dtype=pl.FP32, value=0.0)
     if leaf == 0:
      for clear_offset in pl.range(NL * PAIR_W, WORKSPACE_WIDTH, PAIR_W):
       pl.store(zero_pairs, [t, clear_offset], wpairs)
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     begin = leaf * LEAF
     if begin < wv:
      valid = pl.min(LEAF, wv - begin)
      if valid <= 512:
       narrow_raw = pl.cast(pl.load(scores, [t, begin], [1, 512], valid_shape=[1, valid]), pl.FP32)
       narrow_padded = pl.tile.fillpad(narrow_raw, pad_value=pl.PadValue.min)
       narrow_values = pl.maximum(narrow_padded, SORT_FLOOR)
       narrow_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       narrow_pairs = pl.tile.sort32(narrow_values, pl.reinterpret_view(narrow_positions, pl.UINT32))
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=64)
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=256)
       pl.store(narrow_pairs, [t, leaf * PAIR_W], wpairs)
      elif valid <= 1024:
       left_valid = pl.min(valid, 512)
       right_valid = pl.max(valid - 512, 0)
       left_raw = pl.cast(pl.load(scores, [t, begin], [1, 512], valid_shape=[1, left_valid]), pl.FP32)
       right_raw = pl.cast(pl.load(scores, [t, begin + 512], [1, 512], valid_shape=[1, right_valid]), pl.FP32)
       left_values = pl.maximum(pl.tile.fillpad(left_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       right_values = pl.maximum(pl.tile.fillpad(right_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       left_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       right_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin + 512, pl.INT32))
       left_pairs = pl.tile.sort32(left_values, pl.reinterpret_view(left_positions, pl.UINT32))
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=64)
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=256)
       right_pairs = pl.tile.sort32(right_values, pl.reinterpret_view(right_positions, pl.UINT32))
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=64)
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=256)
       native_scratch = pl.tile.create([1, 2048], dtype=pl.FP32)
       native_pairs = pl.tile.mrgsort(left_pairs, right_pairs, tmp=native_scratch)
       native_discarded = pl.tile.read(native_pairs, [0, PAIR_W])
       if native_discarded > discarded:
        discarded = native_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if native_discarded == discarded:
        native_front_boundary = pl.tile.read(native_pairs, [0, PAIR_W - 2])
        native_front_bound = native_front_boundary
        if native_front_bound < FIN_FLOOR:
         native_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if native_discarded >= native_front_bound:
         native_arity_third = pl.tile.read(native_pairs, [0, PAIR_W + 4])
         if native_arity_third < native_discarded:
          native_arity_second = pl.tile.read(native_pairs, [0, PAIR_W + 2])
          native_arity_bits = pl.reinterpret_view(native_pairs, pl.INT32)
          native_front_result = pl.tile.read(native_arity_bits, [0, PAIR_W + 1])
          if native_arity_second == native_discarded:
           native_arity_second_position = pl.tile.read(native_arity_bits, [0, PAIR_W + 3])
           native_front_result = pl.min(native_front_result, native_arity_second_position)
         else:
          native_front_tail = pl.tile.slice(native_pairs, [1, 1024], [0, PAIR_W])
          native_front_values = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
          native_front_positions = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          native_front_candidates = pl.tile.select(pl.tile.cmps(native_front_values, native_discarded, cmp_type=0), native_front_positions, 2147483647)
          native_front_wide = pl.col_min(pl.reshape(native_front_candidates, [8, 64]))
          native_front_eight = pl.col_min(pl.reshape(native_front_wide, [8, 8]))
          native_front_result = pl.tile.read(native_front_eight, [0, 0])
          native_front_lane_1 = pl.tile.read(native_front_eight, [0, 1])
          if native_front_lane_1 < native_front_result:
           native_front_result = native_front_lane_1
          native_front_lane_2 = pl.tile.read(native_front_eight, [0, 2])
          if native_front_lane_2 < native_front_result:
           native_front_result = native_front_lane_2
          native_front_lane_3 = pl.tile.read(native_front_eight, [0, 3])
          if native_front_lane_3 < native_front_result:
           native_front_result = native_front_lane_3
          native_front_lane_4 = pl.tile.read(native_front_eight, [0, 4])
          if native_front_lane_4 < native_front_result:
           native_front_result = native_front_lane_4
          native_front_lane_5 = pl.tile.read(native_front_eight, [0, 5])
          if native_front_lane_5 < native_front_result:
           native_front_result = native_front_lane_5
          native_front_lane_6 = pl.tile.read(native_front_eight, [0, 6])
          if native_front_lane_6 < native_front_result:
           native_front_result = native_front_lane_6
          native_front_lane_7 = pl.tile.read(native_front_eight, [0, 7])
          if native_front_lane_7 < native_front_result:
           native_front_result = native_front_lane_7
         if native_front_result < frontier_position:
          frontier_position = native_front_result
       pl.store(pl.tile.slice(native_pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
      else:
       raw = pl.cast(pl.load(scores, [t, begin], [1, LEAF], valid_shape=[1, valid]), pl.FP32)
       padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
       flr = pl.tile.full([1, LEAF], dtype=pl.FP32, value=SORT_FLOOR)
       vals = pl.maximum(padded, flr)
       idx_i32 = pl.add(pl.tile.arange(0, [1, LEAF], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       pairs = pl.tile.sort32(vals, pl.reinterpret_view(idx_i32, pl.UINT32))
       pairs = pl.tile.mrgsort(pairs, block_len=64)
       pairs = pl.tile.mrgsort(pairs, block_len=256)
       pairs = pl.tile.mrgsort(pairs, block_len=1024)
       leaf_discarded = pl.tile.read(pairs, [0, PAIR_W])
       if leaf_discarded > discarded:
        discarded = leaf_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if leaf_discarded == discarded:
        leaf_front_boundary = pl.tile.read(pairs, [0, PAIR_W - 2])
        leaf_front_bound = leaf_front_boundary
        if leaf_front_bound < FIN_FLOOR:
         leaf_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if leaf_discarded >= leaf_front_bound:
         leaf_arity_third = pl.tile.read(pairs, [0, PAIR_W + 4])
         if leaf_arity_third < leaf_discarded:
          leaf_arity_second = pl.tile.read(pairs, [0, PAIR_W + 2])
          leaf_arity_bits = pl.reinterpret_view(pairs, pl.INT32)
          leaf_front_result = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 1])
          if leaf_arity_second == leaf_discarded:
           leaf_arity_second_position = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 3])
           leaf_front_result = pl.min(leaf_front_result, leaf_arity_second_position)
         else:
          leaf_front_tail = pl.tile.slice(pairs, [1, 3072], [0, PAIR_W])
          leaf_front_values = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
          leaf_front_positions = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          leaf_front_candidates = pl.tile.select(pl.tile.cmps(leaf_front_values, leaf_discarded, cmp_type=0), leaf_front_positions, 2147483647)
          leaf_front_wide = pl.col_min(pl.reshape(leaf_front_candidates, [24, 64]))
          leaf_front_eight = pl.col_min(pl.reshape(leaf_front_wide, [8, 8]))
          leaf_front_result = pl.tile.read(leaf_front_eight, [0, 0])
          leaf_front_lane_1 = pl.tile.read(leaf_front_eight, [0, 1])
          if leaf_front_lane_1 < leaf_front_result:
           leaf_front_result = leaf_front_lane_1
          leaf_front_lane_2 = pl.tile.read(leaf_front_eight, [0, 2])
          if leaf_front_lane_2 < leaf_front_result:
           leaf_front_result = leaf_front_lane_2
          leaf_front_lane_3 = pl.tile.read(leaf_front_eight, [0, 3])
          if leaf_front_lane_3 < leaf_front_result:
           leaf_front_result = leaf_front_lane_3
          leaf_front_lane_4 = pl.tile.read(leaf_front_eight, [0, 4])
          if leaf_front_lane_4 < leaf_front_result:
           leaf_front_result = leaf_front_lane_4
          leaf_front_lane_5 = pl.tile.read(leaf_front_eight, [0, 5])
          if leaf_front_lane_5 < leaf_front_result:
           leaf_front_result = leaf_front_lane_5
          leaf_front_lane_6 = pl.tile.read(leaf_front_eight, [0, 6])
          if leaf_front_lane_6 < leaf_front_result:
           leaf_front_result = leaf_front_lane_6
          leaf_front_lane_7 = pl.tile.read(leaf_front_eight, [0, 7])
          if leaf_front_lane_7 < leaf_front_result:
           leaf_front_result = leaf_front_lane_7
         if leaf_front_result < frontier_position:
          frontier_position = leaf_front_result
       pl.store(pl.tile.slice(pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
     else:
      pl.store(zero_pairs, [t, leaf * PAIR_W], wpairs)
     leaf_score_line = pl.tile.full([1, 16], dtype=pl.FP32, value=SORT_FLOOR)
     leaf_position_line = pl.tile.full([1, 16], dtype=pl.INT32, value=2147483647)
     pl.tile.write(leaf_score_line, [0, 0], discarded)
     pl.tile.write(leaf_position_line, [0, 0], frontier_position)
     pl.store(leaf_score_line, [t, leaf * 16], leaf_discarded_bank)
     pl.store(leaf_position_line, [t, leaf * 16], leaf_front_bank)
   for merge_cohort_job in pl.spmd((M + ROW_COHORT - 1) // ROW_COHORT, name_hint='decomposed_merge'):
    cohort_base = merge_cohort_job * ROW_COHORT
    for t in pl.range(cohort_base, pl.min(cohort_base + ROW_COHORT, M), 1):
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     lc = (wv + LEAF - 1) // LEAF
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     for proof_leaf in pl.range(NL):
      proof_score = pl.read(leaf_discarded_bank, [t, proof_leaf * 16])
      proof_position = pl.read(leaf_front_bank, [t, proof_leaf * 16])
      if proof_score > discarded:
       discarded = proof_score
       frontier_position = pl.cast(2147483647, pl.INT32)
      if proof_score == discarded:
       if proof_position < frontier_position:
        frontier_position = proof_position
     pool_accepted = False
     if lc >= 14:
      pool64_outside = discarded
      pool64_pairs = pl.tile.full([1, 2048], dtype=pl.FP32, value=SORT_FLOOR)
      for pool64_leaf in pl.unroll(16):
       pool64_piece = pl.tile.full([1, 128], dtype=pl.FP32, value=SORT_FLOOR)
       if pool64_leaf < lc:
        pool64_safe_leaf = pl.min(pool64_leaf, NL - 1)
        pool64_piece = pl.load(wpairs, [t, pool64_safe_leaf * PAIR_W], [1, 128])
        pool64_excluded = pl.read(wpairs, [t, pool64_safe_leaf * PAIR_W + 128])
        if pool64_excluded > pool64_outside:
         pool64_outside = pool64_excluded
       pool64_pairs = pl.tile.assemble(pool64_pairs, pool64_piece, [0, pool64_leaf * 128])
      pool64_pairs = pl.tile.mrgsort(pool64_pairs, block_len=128)
      pool64_pairs = pl.tile.mrgsort(pool64_pairs, block_len=512)
      pool64_raw_cutoff = pl.tile.read(pool64_pairs, [0, PAIR_W - 2])
      pool64_rank513 = pl.tile.read(pool64_pairs, [0, PAIR_W])
      if pool64_outside < pool64_raw_cutoff and pool64_raw_cutoff > FIN_FLOOR:
       pool_accepted = True
       discarded = pool64_outside
       frontier_position = pl.cast(2147483647, pl.INT32)
       if pool64_rank513 > discarded:
        discarded = pool64_rank513
       if pool64_rank513 == pool64_raw_cutoff:
        for pool64_tail_chunk in pl.unroll(1):
         pool64_front_tail = pl.tile.slice(pool64_pairs, [1, 1024], [0, (pool64_tail_chunk + 1) * PAIR_W])
         pool64_front_values = pl.tile.gather_mask(pool64_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
         pool64_front_positions = pl.tile.gather_mask(pool64_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         pool64_front_candidates = pl.tile.select(pl.tile.cmps(pool64_front_values, pool64_rank513, cmp_type=0), pool64_front_positions, 2147483647)
         pool64_front_wide = pl.col_min(pl.reshape(pool64_front_candidates, [8, 64]))
         pool64_front_eight = pl.col_min(pl.reshape(pool64_front_wide, [8, 8]))
         pool64_front_result = pl.tile.read(pool64_front_eight, [0, 0])
         pool64_front_lane_1 = pl.tile.read(pool64_front_eight, [0, 1])
         if pool64_front_lane_1 < pool64_front_result:
          pool64_front_result = pool64_front_lane_1
         pool64_front_lane_2 = pl.tile.read(pool64_front_eight, [0, 2])
         if pool64_front_lane_2 < pool64_front_result:
          pool64_front_result = pool64_front_lane_2
         pool64_front_lane_3 = pl.tile.read(pool64_front_eight, [0, 3])
         if pool64_front_lane_3 < pool64_front_result:
          pool64_front_result = pool64_front_lane_3
         pool64_front_lane_4 = pl.tile.read(pool64_front_eight, [0, 4])
         if pool64_front_lane_4 < pool64_front_result:
          pool64_front_result = pool64_front_lane_4
         pool64_front_lane_5 = pl.tile.read(pool64_front_eight, [0, 5])
         if pool64_front_lane_5 < pool64_front_result:
          pool64_front_result = pool64_front_lane_5
         pool64_front_lane_6 = pl.tile.read(pool64_front_eight, [0, 6])
         if pool64_front_lane_6 < pool64_front_result:
          pool64_front_result = pool64_front_lane_6
         pool64_front_lane_7 = pl.tile.read(pool64_front_eight, [0, 7])
         if pool64_front_lane_7 < pool64_front_result:
          pool64_front_result = pool64_front_lane_7
         if pool64_front_result < frontier_position:
          frontier_position = pool64_front_result
       pl.store(pl.tile.slice(pool64_pairs, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
     elif lc >= 8:
      pool128_outside = discarded
      pool128_pairs = pl.tile.full([1, 4096], dtype=pl.FP32, value=SORT_FLOOR)
      for pool128_leaf in pl.unroll(16):
       pool128_piece = pl.tile.full([1, 256], dtype=pl.FP32, value=SORT_FLOOR)
       if pool128_leaf < lc:
        pool128_safe_leaf = pl.min(pool128_leaf, NL - 1)
        pool128_piece = pl.load(wpairs, [t, pool128_safe_leaf * PAIR_W], [1, 256])
        pool128_excluded = pl.read(wpairs, [t, pool128_safe_leaf * PAIR_W + 256])
        if pool128_excluded > pool128_outside:
         pool128_outside = pool128_excluded
       pool128_pairs = pl.tile.assemble(pool128_pairs, pool128_piece, [0, pool128_leaf * 256])
      pool128_pairs = pl.tile.mrgsort(pool128_pairs, block_len=256)
      pool128_pairs = pl.tile.mrgsort(pool128_pairs, block_len=1024)
      pool128_raw_cutoff = pl.tile.read(pool128_pairs, [0, PAIR_W - 2])
      pool128_rank513 = pl.tile.read(pool128_pairs, [0, PAIR_W])
      if pool128_outside < pool128_raw_cutoff and pool128_raw_cutoff > FIN_FLOOR:
       pool_accepted = True
       discarded = pool128_outside
       frontier_position = pl.cast(2147483647, pl.INT32)
       if pool128_rank513 > discarded:
        discarded = pool128_rank513
       if pool128_rank513 == pool128_raw_cutoff:
        for pool128_tail_chunk in pl.unroll(3):
         pool128_front_tail = pl.tile.slice(pool128_pairs, [1, 1024], [0, (pool128_tail_chunk + 1) * PAIR_W])
         pool128_front_values = pl.tile.gather_mask(pool128_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
         pool128_front_positions = pl.tile.gather_mask(pool128_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         pool128_front_candidates = pl.tile.select(pl.tile.cmps(pool128_front_values, pool128_rank513, cmp_type=0), pool128_front_positions, 2147483647)
         pool128_front_wide = pl.col_min(pl.reshape(pool128_front_candidates, [8, 64]))
         pool128_front_eight = pl.col_min(pl.reshape(pool128_front_wide, [8, 8]))
         pool128_front_result = pl.tile.read(pool128_front_eight, [0, 0])
         pool128_front_lane_1 = pl.tile.read(pool128_front_eight, [0, 1])
         if pool128_front_lane_1 < pool128_front_result:
          pool128_front_result = pool128_front_lane_1
         pool128_front_lane_2 = pl.tile.read(pool128_front_eight, [0, 2])
         if pool128_front_lane_2 < pool128_front_result:
          pool128_front_result = pool128_front_lane_2
         pool128_front_lane_3 = pl.tile.read(pool128_front_eight, [0, 3])
         if pool128_front_lane_3 < pool128_front_result:
          pool128_front_result = pool128_front_lane_3
         pool128_front_lane_4 = pl.tile.read(pool128_front_eight, [0, 4])
         if pool128_front_lane_4 < pool128_front_result:
          pool128_front_result = pool128_front_lane_4
         pool128_front_lane_5 = pl.tile.read(pool128_front_eight, [0, 5])
         if pool128_front_lane_5 < pool128_front_result:
          pool128_front_result = pool128_front_lane_5
         pool128_front_lane_6 = pl.tile.read(pool128_front_eight, [0, 6])
         if pool128_front_lane_6 < pool128_front_result:
          pool128_front_result = pool128_front_lane_6
         pool128_front_lane_7 = pl.tile.read(pool128_front_eight, [0, 7])
         if pool128_front_lane_7 < pool128_front_result:
          pool128_front_result = pool128_front_lane_7
         if pool128_front_result < frontier_position:
          frontier_position = pool128_front_result
       pl.store(pl.tile.slice(pool128_pairs, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
     if not pool_accepted:
      mtmp = pl.tile.create([1, 2 * PAIR_W], dtype=pl.FP32)
      for cb in pl.range(1, lc):
       mleft = pl.load(wpairs, [t, 0], [1, PAIR_W])
       mright = pl.load(wpairs, [t, cb * PAIR_W], [1, PAIR_W])
       mboth = pl.tile.mrgsort(mleft, mright, tmp=mtmp)
       merge_discarded = pl.tile.read(mboth, [0, PAIR_W])
       if merge_discarded > discarded:
        discarded = merge_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if merge_discarded == discarded:
        merge_front_boundary = pl.tile.read(mboth, [0, PAIR_W - 2])
        merge_front_bound = merge_front_boundary
        if merge_front_bound < FIN_FLOOR:
         merge_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if merge_discarded >= merge_front_bound:
         merge_front_tail = pl.tile.slice(mboth, [1, 1024], [0, PAIR_W])
         merge_front_values = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
         merge_front_positions = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         merge_front_candidates = pl.tile.select(pl.tile.cmps(merge_front_values, merge_discarded, cmp_type=0), merge_front_positions, 2147483647)
         merge_front_wide = pl.col_min(pl.reshape(merge_front_candidates, [8, 64]))
         merge_front_eight = pl.col_min(pl.reshape(merge_front_wide, [8, 8]))
         merge_front_result = pl.tile.read(merge_front_eight, [0, 0])
         merge_front_lane_1 = pl.tile.read(merge_front_eight, [0, 1])
         if merge_front_lane_1 < merge_front_result:
          merge_front_result = merge_front_lane_1
         merge_front_lane_2 = pl.tile.read(merge_front_eight, [0, 2])
         if merge_front_lane_2 < merge_front_result:
          merge_front_result = merge_front_lane_2
         merge_front_lane_3 = pl.tile.read(merge_front_eight, [0, 3])
         if merge_front_lane_3 < merge_front_result:
          merge_front_result = merge_front_lane_3
         merge_front_lane_4 = pl.tile.read(merge_front_eight, [0, 4])
         if merge_front_lane_4 < merge_front_result:
          merge_front_result = merge_front_lane_4
         merge_front_lane_5 = pl.tile.read(merge_front_eight, [0, 5])
         if merge_front_lane_5 < merge_front_result:
          merge_front_result = merge_front_lane_5
         merge_front_lane_6 = pl.tile.read(merge_front_eight, [0, 6])
         if merge_front_lane_6 < merge_front_result:
          merge_front_result = merge_front_lane_6
         merge_front_lane_7 = pl.tile.read(merge_front_eight, [0, 7])
         if merge_front_lane_7 < merge_front_result:
          merge_front_result = merge_front_lane_7
         if merge_front_result < frontier_position:
          frontier_position = merge_front_result
       pl.store(pl.tile.slice(mboth, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
     pl.write(aux_discarded, [t, 0], discarded)
     pl.write(aux_front, [t, 0], frontier_position)
   for materialize_row_job in pl.spmd((M + 4 - 1) // 4, name_hint='decomposed_materialize'):
    root_row_base = materialize_row_job * 4
    if P == 1:
     materialize_single_page = pl.read(block_table, [0, 0])
     materialize_pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), materialize_single_page), pad_value=pl.PadValue.zero)
    else:
     materialize_pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
    materialize_page_lanes = pl.tile.arange(0, [1, P8], dtype=pl.INT32)
    materialize_displacements = pl.mul(pl.sub(materialize_pages, materialize_page_lanes), PAGE)
    for t in pl.range(root_row_base, pl.min(root_row_base + 4, M), 1):
     fin = pl.load(wpairs, [t, 0], [1, PAIR_W])
     values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
     finite = pl.maximum(values, FIN_FLOOR)
     cutoff = pl.tile.read(finite, [0, K - 1])
     ones_c = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
     flags = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=4), ones_c, 0.0)
     sums64 = pl.col_sum(pl.reshape(flags, [8, 64]))
     sums8 = pl.col_sum(pl.reshape(sums64, [8, 8]))
     above = pl.cast(0, pl.INDEX)
     for lane in pl.unroll(8):
      part = pl.cast(pl.tile.read(sums8, [0, lane]), pl.INT32)
      above = above + pl.cast(part, pl.INDEX)
     cutoff_stage_line = pl.tile.full([1, 16], dtype=pl.FP32, value=0)
     pl.tile.write(cutoff_stage_line, [0, 0], cutoff)
     pl.store(cutoff_stage_line, [t, 0], aux_cutoffs)
     quota_stage_line = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
     pl.tile.write(quota_stage_line, [0, 0], pl.cast(K - above, pl.INT32))
     pl.store(quota_stage_line, [t, 0], aux_quotas)
     discarded = pl.read(aux_discarded, [t, 0])
     frontier_position = pl.read(aux_front, [t, 0])
     replay_safe = discarded < cutoff
     if discarded == cutoff:
      certificate_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
      certificate_candidates = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=0), certificate_positions, -1)
      certificate_wide = pl.col_max(pl.reshape(certificate_candidates, [8, 64]))
      certificate_eight = pl.col_max(pl.reshape(certificate_wide, [8, 8]))
      certificate_result = pl.tile.read(certificate_eight, [0, 0])
      certificate_lane_1 = pl.tile.read(certificate_eight, [0, 1])
      if certificate_lane_1 > certificate_result:
       certificate_result = certificate_lane_1
      certificate_lane_2 = pl.tile.read(certificate_eight, [0, 2])
      if certificate_lane_2 > certificate_result:
       certificate_result = certificate_lane_2
      certificate_lane_3 = pl.tile.read(certificate_eight, [0, 3])
      if certificate_lane_3 > certificate_result:
       certificate_result = certificate_lane_3
      certificate_lane_4 = pl.tile.read(certificate_eight, [0, 4])
      if certificate_lane_4 > certificate_result:
       certificate_result = certificate_lane_4
      certificate_lane_5 = pl.tile.read(certificate_eight, [0, 5])
      if certificate_lane_5 > certificate_result:
       certificate_result = certificate_lane_5
      certificate_lane_6 = pl.tile.read(certificate_eight, [0, 6])
      if certificate_lane_6 > certificate_result:
       certificate_result = certificate_lane_6
      certificate_lane_7 = pl.tile.read(certificate_eight, [0, 7])
      if certificate_lane_7 > certificate_result:
       certificate_result = certificate_lane_7
      if frontier_position > certificate_result:
       replay_safe = True
     if replay_safe:
      if cutoff > FIN_FLOOR:
       root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       selected = pl.cast(K, pl.INDEX)
       if selected > 0:
        root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(root_positions, pl.FP32))
        root_keys = root_priority
        root_order = pl.tile.sort32(root_keys, pl.reinterpret_view(root_positions, pl.UINT32))
        root_order = pl.tile.mrgsort(root_order, block_len=64)
        root_order = pl.tile.mrgsort(root_order, block_len=256)
        root_logical = pl.tile.gather_mask(root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        root_valid = pl.tile.set_validshape(root_logical, 1, selected)
        root_safe = pl.tile.fillpad(root_valid, pad_value=pl.PadValue.zero)
        root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
        for map_start in pl.range(0, selected, MAP):
         map_count = pl.min(MAP, selected - map_start)
         map_bits = pl.tile.slice(pl.reinterpret_view(root_safe, pl.FP32), [1, MAP], [0, map_start])
         map_logical = pl.reinterpret_view(map_bits, pl.INT32)
         root_page_idx = pl.tile.shrs(map_logical, 7)
         root_page_ids = pl.tile.gather(materialize_displacements, root_page_idx, root_tmp)
         root_physical = pl.add(map_logical, root_page_ids)
         pl.store(pl.tile.set_validshape(root_physical, 1, map_count), [t, map_start], indices)
      else:
       visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
       root = fin
       root_values = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P0101)
       root_positions = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
       root_strict = pl.tile.select(pl.tile.cmps(root_values, cutoff, cmp_type=4), ones_root, 0.0)
       root_equal = pl.tile.select(pl.tile.cmps(root_values, cutoff, cmp_type=0), ones_root, 0.0)
       root_eligible = pl.maximum(root_strict, root_equal)
       root_visible = pl.tile.cmps(pl.cast(root_positions, pl.FP32), pl.cast(pl.cast(visible, pl.INT32), pl.FP32), cmp_type=2)
       root_keep = pl.tile.select(root_visible, root_eligible, 0.0)
       root_sums64 = pl.col_sum(pl.reshape(root_keep, [8, 64]))
       root_sums8 = pl.col_sum(pl.reshape(root_sums64, [8, 8]))
       selected = pl.cast(0, pl.INDEX)
       for root_lane in pl.unroll(8):
        root_part = pl.cast(pl.tile.read(root_sums8, [0, root_lane]), pl.INT32)
        selected = selected + pl.cast(root_part, pl.INDEX)
       pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
       if selected > 0:
        root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(root_positions, pl.FP32))
        root_keys = pl.tile.select(pl.tile.cmps(root_keep, 0.0, cmp_type=4), root_priority, -1.0)
        root_order = pl.tile.sort32(root_keys, pl.reinterpret_view(root_positions, pl.UINT32))
        root_order = pl.tile.mrgsort(root_order, block_len=64)
        root_order = pl.tile.mrgsort(root_order, block_len=256)
        root_sorted_keys = pl.tile.gather_mask(root_order, mask_pattern=pl.tile.MaskPattern.P0101)
        root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), root_sorted_keys), pl.INT32)
        root_valid = pl.tile.set_validshape(root_logical, 1, selected)
        root_safe = pl.tile.fillpad(root_valid, pad_value=pl.PadValue.zero)
        root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
        for map_start in pl.range(0, selected, MAP):
         map_count = pl.min(MAP, selected - map_start)
         map_bits = pl.tile.slice(pl.reinterpret_view(root_safe, pl.FP32), [1, MAP], [0, map_start])
         map_logical = pl.reinterpret_view(map_bits, pl.INT32)
         root_page_idx = pl.tile.shrs(map_logical, 7)
         root_page_ids = pl.tile.gather(materialize_displacements, root_page_idx, root_tmp)
         root_physical = pl.add(map_logical, root_page_ids)
         pl.store(pl.tile.set_validshape(root_physical, 1, map_count), [t, map_start], indices)
     else:
      visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
      remaining = pl.cast(K - above, pl.INDEX)
      pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
      ones_m = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
      lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
      lanes_f = pl.cast(lanes, pl.FP32)
      lane_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
      priority = pl.cast(pl.sub(lane_origin, lanes), pl.FP32)
      gtmp = pl.tile.create([1, SCAN], dtype=pl.INT32)
      count = pl.cast(0, pl.INDEX)
      for start in pl.range(0, visible, SCAN):
       if count < K:
        valid = pl.min(SCAN, visible - start)
        scan_raw = pl.cast(pl.load(scores, [t, start], [1, SCAN], valid_shape=[1, valid]), pl.FP32)
        scan_vals = pl.tile.fillpad(scan_raw, pad_value=pl.PadValue.min)
        strict = pl.tile.select(pl.tile.cmps(scan_vals, cutoff, cmp_type=4), ones_m, 0.0)
        ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
        if remaining > 0:
         eq = pl.tile.select(pl.tile.cmps(scan_vals, cutoff, cmp_type=0), ones_m, 0.0)
         eq_valid = pl.tile.cmps(lanes_f, pl.cast(pl.cast(valid, pl.INT32), pl.FP32), cmp_type=2)
         eq = pl.tile.select(eq_valid, eq, 0.0)
         eq_sums64 = pl.col_sum(pl.reshape(eq, [SCAN // 64, 64]))
         eq_sums8 = pl.col_sum(pl.reshape(eq_sums64, [8, 8]))
         available = pl.cast(0, pl.INDEX)
         for eq_lane in pl.unroll(8):
          eq_part = pl.cast(pl.tile.read(eq_sums8, [0, eq_lane]), pl.INT32)
          available = available + pl.cast(eq_part, pl.INDEX)
         take = pl.min(available, remaining)
         if take > 0:
          if take == available:
           ties = eq
          else:
           tie_keys = pl.mul(eq, priority)
           tie_pairs = pl.tile.sort32(tie_keys, pl.reinterpret_view(lanes, pl.UINT32))
           tie_pairs = pl.tile.mrgsort(tie_pairs, block_len=64)
           tie_merged = tie_pairs
           equal_keys = pl.tile.gather_mask(tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
           last_key_f = pl.tile.read(equal_keys, [0, take - 1])
           boundary_f = 128.0 - last_key_f
           ties = pl.tile.select(pl.tile.cmps(lanes_f, boundary_f, cmp_type=3), eq, 0.0)
          remaining = remaining - take
        keep = pl.maximum(strict, ties)
        keep_valid = pl.tile.set_validshape(keep, 1, valid)
        keep_padded = pl.tile.fillpad(keep_valid, pad_value=pl.PadValue.zero)
        keep_sums64 = pl.col_sum(pl.reshape(keep_padded, [SCAN // 64, 64]))
        keep_sums8 = pl.col_sum(pl.reshape(keep_sums64, [8, 8]))
        keep_count = pl.cast(0, pl.INDEX)
        for keep_lane in pl.unroll(8):
         keep_part = pl.cast(pl.tile.read(keep_sums8, [0, keep_lane]), pl.INT32)
         keep_count = keep_count + pl.cast(keep_part, pl.INDEX)
        selected = pl.min(keep_count, K - count)
        if selected > 0:
         order_keys = pl.mul(keep_padded, priority)
         order_pairs = pl.tile.sort32(order_keys, pl.reinterpret_view(lanes, pl.UINT32))
         order_pairs = pl.tile.mrgsort(order_pairs, block_len=64)
         order_merged = order_pairs
         order_vals = pl.tile.gather_mask(order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
         offs = pl.sub(lane_origin, pl.cast(order_vals, pl.INT32))
         compact_valid = pl.tile.set_validshape(offs, 1, selected)
         compact_safe = pl.tile.fillpad(compact_valid, pad_value=pl.PadValue.zero)
         logical = pl.add(compact_safe, pl.cast(start, pl.INT32))
         page_idx = pl.tile.shrs(logical, 7)
         page_ids = pl.tile.gather(materialize_displacements, page_idx, gtmp)
         physical = pl.add(logical, page_ids)
         pl.store(pl.tile.set_validshape(physical, 1, selected), [t, count], indices)
         count = count + selected
   for pack_task in pl.spmd(1, name_hint='pack_auxiliary'):
    for output_row in pl.range(0, M, 16):
     cutoff_lines = pl.load(aux_cutoffs, [output_row, 0], [16, 16])
     quota_lines = pl.load(aux_quotas, [output_row, 0], [16, 16])
     cutoff_flat = pl.reshape(cutoff_lines, [1, 256])
     quota_flat = pl.reshape(quota_lines, [1, 256])
     pack_offsets = pl.mul(pl.tile.arange(0, [1, 16], dtype=pl.INT32), 16)
     cutoff_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     quota_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     cutoff_packed = pl.tile.gather(cutoff_flat, pack_offsets, cutoff_scratch)
     quota_packed = pl.tile.gather(quota_flat, pack_offsets, quota_scratch)
     pl.store(pl.reshape(cutoff_packed, [16, 1]), [output_row, 0], cutoffs)
     pl.store(pl.reshape(quota_packed, [16, 1]), [output_row, 0], quotas)
   return indices
 elif NL == 1:

  @pl.jit
  def dsa41_topk_guarded_root(scores: pl.Tensor[[M, N], pl.BF16], lengths: pl.Tensor[[M], pl.INT32], block_table: pl.Tensor[[1, P], pl.INT32], wpairs: pl.InOut[pl.Tensor[[M, WORKSPACE_WIDTH], pl.FP32]], cutoffs: pl.InOut[pl.Tensor[[M, 1], pl.FP32]], quotas: pl.InOut[pl.Tensor[[M, 1], pl.INT32]], indices: pl.Out[pl.Tensor[[M, K], pl.INT32]]):
   aux_discarded = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_cutoffs = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_quotas = pl.create_tensor([M, 16], dtype=pl.INT32)
   aux_front = pl.create_tensor([M, 16], dtype=pl.INT32)
   leaf_discarded_bank = pl.create_tensor([M, NL * 16], dtype=pl.FP32)
   leaf_front_bank = pl.create_tensor([M, NL * 16], dtype=pl.INT32)
   for leaf_job in pl.spmd(LEAF_COHORTS * NL, name_hint='canonical_leaf_plane'):
    leaf = leaf_job // LEAF_COHORTS
    leaf_cohort_base = leaf_job % LEAF_COHORTS * LEAF_ROWS
    for t in pl.range(leaf_cohort_base, pl.min(leaf_cohort_base + LEAF_ROWS, M), 1):
     zero_pairs = pl.tile.full([1, PAIR_W], dtype=pl.FP32, value=0.0)
     if leaf == 0:
      for clear_offset in pl.range(NL * PAIR_W, WORKSPACE_WIDTH, PAIR_W):
       pl.store(zero_pairs, [t, clear_offset], wpairs)
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     begin = leaf * LEAF
     if begin < wv:
      valid = pl.min(LEAF, wv - begin)
      if valid <= 512:
       narrow_raw = pl.cast(pl.load(scores, [t, begin], [1, 512], valid_shape=[1, valid]), pl.FP32)
       narrow_padded = pl.tile.fillpad(narrow_raw, pad_value=pl.PadValue.min)
       narrow_values = pl.maximum(narrow_padded, SORT_FLOOR)
       narrow_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       narrow_pairs = pl.tile.sort32(narrow_values, pl.reinterpret_view(narrow_positions, pl.UINT32))
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=64)
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=256)
       pl.store(narrow_pairs, [t, leaf * PAIR_W], wpairs)
      elif valid <= 1024:
       left_valid = pl.min(valid, 512)
       right_valid = pl.max(valid - 512, 0)
       left_raw = pl.cast(pl.load(scores, [t, begin], [1, 512], valid_shape=[1, left_valid]), pl.FP32)
       right_raw = pl.cast(pl.load(scores, [t, begin + 512], [1, 512], valid_shape=[1, right_valid]), pl.FP32)
       left_values = pl.maximum(pl.tile.fillpad(left_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       right_values = pl.maximum(pl.tile.fillpad(right_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       left_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       right_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin + 512, pl.INT32))
       left_pairs = pl.tile.sort32(left_values, pl.reinterpret_view(left_positions, pl.UINT32))
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=64)
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=256)
       right_pairs = pl.tile.sort32(right_values, pl.reinterpret_view(right_positions, pl.UINT32))
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=64)
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=256)
       native_scratch = pl.tile.create([1, 2048], dtype=pl.FP32)
       native_pairs = pl.tile.mrgsort(left_pairs, right_pairs, tmp=native_scratch)
       native_discarded = pl.tile.read(native_pairs, [0, PAIR_W])
       if native_discarded > discarded:
        discarded = native_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if native_discarded == discarded:
        native_front_boundary = pl.tile.read(native_pairs, [0, PAIR_W - 2])
        native_front_bound = native_front_boundary
        if native_front_bound < FIN_FLOOR:
         native_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if native_discarded >= native_front_bound:
         native_arity_third = pl.tile.read(native_pairs, [0, PAIR_W + 4])
         if native_arity_third < native_discarded:
          native_arity_second = pl.tile.read(native_pairs, [0, PAIR_W + 2])
          native_arity_bits = pl.reinterpret_view(native_pairs, pl.INT32)
          native_front_result = pl.tile.read(native_arity_bits, [0, PAIR_W + 1])
          if native_arity_second == native_discarded:
           native_arity_second_position = pl.tile.read(native_arity_bits, [0, PAIR_W + 3])
           native_front_result = pl.min(native_front_result, native_arity_second_position)
         else:
          native_front_tail = pl.tile.slice(native_pairs, [1, 1024], [0, PAIR_W])
          native_front_values = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
          native_front_positions = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          native_front_candidates = pl.tile.select(pl.tile.cmps(native_front_values, native_discarded, cmp_type=0), native_front_positions, 2147483647)
          native_front_wide = pl.col_min(pl.reshape(native_front_candidates, [8, 64]))
          native_front_eight = pl.col_min(pl.reshape(native_front_wide, [8, 8]))
          native_front_result = pl.tile.read(native_front_eight, [0, 0])
          native_front_lane_1 = pl.tile.read(native_front_eight, [0, 1])
          if native_front_lane_1 < native_front_result:
           native_front_result = native_front_lane_1
          native_front_lane_2 = pl.tile.read(native_front_eight, [0, 2])
          if native_front_lane_2 < native_front_result:
           native_front_result = native_front_lane_2
          native_front_lane_3 = pl.tile.read(native_front_eight, [0, 3])
          if native_front_lane_3 < native_front_result:
           native_front_result = native_front_lane_3
          native_front_lane_4 = pl.tile.read(native_front_eight, [0, 4])
          if native_front_lane_4 < native_front_result:
           native_front_result = native_front_lane_4
          native_front_lane_5 = pl.tile.read(native_front_eight, [0, 5])
          if native_front_lane_5 < native_front_result:
           native_front_result = native_front_lane_5
          native_front_lane_6 = pl.tile.read(native_front_eight, [0, 6])
          if native_front_lane_6 < native_front_result:
           native_front_result = native_front_lane_6
          native_front_lane_7 = pl.tile.read(native_front_eight, [0, 7])
          if native_front_lane_7 < native_front_result:
           native_front_result = native_front_lane_7
         if native_front_result < frontier_position:
          frontier_position = native_front_result
       pl.store(pl.tile.slice(native_pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
      else:
       raw = pl.cast(pl.load(scores, [t, begin], [1, LEAF], valid_shape=[1, valid]), pl.FP32)
       padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
       flr = pl.tile.full([1, LEAF], dtype=pl.FP32, value=SORT_FLOOR)
       vals = pl.maximum(padded, flr)
       idx_i32 = pl.add(pl.tile.arange(0, [1, LEAF], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       pairs = pl.tile.sort32(vals, pl.reinterpret_view(idx_i32, pl.UINT32))
       pairs = pl.tile.mrgsort(pairs, block_len=64)
       pairs = pl.tile.mrgsort(pairs, block_len=256)
       pairs = pl.tile.mrgsort(pairs, block_len=1024)
       leaf_discarded = pl.tile.read(pairs, [0, PAIR_W])
       if leaf_discarded > discarded:
        discarded = leaf_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if leaf_discarded == discarded:
        leaf_front_boundary = pl.tile.read(pairs, [0, PAIR_W - 2])
        leaf_front_bound = leaf_front_boundary
        if leaf_front_bound < FIN_FLOOR:
         leaf_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if leaf_discarded >= leaf_front_bound:
         leaf_arity_third = pl.tile.read(pairs, [0, PAIR_W + 4])
         if leaf_arity_third < leaf_discarded:
          leaf_arity_second = pl.tile.read(pairs, [0, PAIR_W + 2])
          leaf_arity_bits = pl.reinterpret_view(pairs, pl.INT32)
          leaf_front_result = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 1])
          if leaf_arity_second == leaf_discarded:
           leaf_arity_second_position = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 3])
           leaf_front_result = pl.min(leaf_front_result, leaf_arity_second_position)
         else:
          leaf_front_tail = pl.tile.slice(pairs, [1, 3072], [0, PAIR_W])
          leaf_front_values = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
          leaf_front_positions = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          leaf_front_candidates = pl.tile.select(pl.tile.cmps(leaf_front_values, leaf_discarded, cmp_type=0), leaf_front_positions, 2147483647)
          leaf_front_wide = pl.col_min(pl.reshape(leaf_front_candidates, [24, 64]))
          leaf_front_eight = pl.col_min(pl.reshape(leaf_front_wide, [8, 8]))
          leaf_front_result = pl.tile.read(leaf_front_eight, [0, 0])
          leaf_front_lane_1 = pl.tile.read(leaf_front_eight, [0, 1])
          if leaf_front_lane_1 < leaf_front_result:
           leaf_front_result = leaf_front_lane_1
          leaf_front_lane_2 = pl.tile.read(leaf_front_eight, [0, 2])
          if leaf_front_lane_2 < leaf_front_result:
           leaf_front_result = leaf_front_lane_2
          leaf_front_lane_3 = pl.tile.read(leaf_front_eight, [0, 3])
          if leaf_front_lane_3 < leaf_front_result:
           leaf_front_result = leaf_front_lane_3
          leaf_front_lane_4 = pl.tile.read(leaf_front_eight, [0, 4])
          if leaf_front_lane_4 < leaf_front_result:
           leaf_front_result = leaf_front_lane_4
          leaf_front_lane_5 = pl.tile.read(leaf_front_eight, [0, 5])
          if leaf_front_lane_5 < leaf_front_result:
           leaf_front_result = leaf_front_lane_5
          leaf_front_lane_6 = pl.tile.read(leaf_front_eight, [0, 6])
          if leaf_front_lane_6 < leaf_front_result:
           leaf_front_result = leaf_front_lane_6
          leaf_front_lane_7 = pl.tile.read(leaf_front_eight, [0, 7])
          if leaf_front_lane_7 < leaf_front_result:
           leaf_front_result = leaf_front_lane_7
         if leaf_front_result < frontier_position:
          frontier_position = leaf_front_result
       pl.store(pl.tile.slice(pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
     else:
      pl.store(zero_pairs, [t, leaf * PAIR_W], wpairs)
     leaf_score_line = pl.tile.full([1, 16], dtype=pl.FP32, value=SORT_FLOOR)
     leaf_position_line = pl.tile.full([1, 16], dtype=pl.INT32, value=2147483647)
     pl.tile.write(leaf_score_line, [0, 0], discarded)
     pl.tile.write(leaf_position_line, [0, 0], frontier_position)
     pl.store(leaf_score_line, [t, leaf * 16], leaf_discarded_bank)
     pl.store(leaf_position_line, [t, leaf * 16], leaf_front_bank)
   for merge_cohort_job in pl.spmd((M + 4 - 1) // 4, name_hint='bounded_root_handoff'):
    cohort_base = merge_cohort_job * 4
    if P == 1:
     materialize_single_page = pl.read(block_table, [0, 0])
     materialize_pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), materialize_single_page), pad_value=pl.PadValue.zero)
    else:
     materialize_pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
    materialize_page_lanes = pl.tile.arange(0, [1, P8], dtype=pl.INT32)
    materialize_displacements = pl.mul(pl.sub(materialize_pages, materialize_page_lanes), PAGE)
    for t in pl.range(cohort_base, pl.min(cohort_base + 4, M), 1):
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     lc = (wv + LEAF - 1) // LEAF
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     for proof_leaf in pl.range(NL):
      proof_score = pl.read(leaf_discarded_bank, [t, proof_leaf * 16])
      proof_position = pl.read(leaf_front_bank, [t, proof_leaf * 16])
      if proof_score > discarded:
       discarded = proof_score
       frontier_position = pl.cast(2147483647, pl.INT32)
      if proof_score == discarded:
       if proof_position < frontier_position:
        frontier_position = proof_position
     fin = pl.load(wpairs, [t, 0], [1, PAIR_W])
     one_values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
     one_finite = pl.maximum(one_values, FIN_FLOOR)
     one_cutoff = pl.tile.read(one_finite, [0, K - 1])
     one_ones_c = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
     one_flags = pl.tile.select(pl.tile.cmps(one_values, one_cutoff, cmp_type=4), one_ones_c, 0.0)
     one_sums64 = pl.col_sum(pl.reshape(one_flags, [8, 64]))
     one_sums8 = pl.col_sum(pl.reshape(one_sums64, [8, 8]))
     one_above = pl.cast(0, pl.INDEX)
     for one_lane in pl.unroll(8):
      one_part = pl.cast(pl.tile.read(one_sums8, [0, one_lane]), pl.INT32)
      one_above = one_above + pl.cast(one_part, pl.INDEX)
     one_cutoff_stage_line = pl.tile.full([1, 16], dtype=pl.FP32, value=0)
     pl.tile.write(one_cutoff_stage_line, [0, 0], one_cutoff)
     pl.store(one_cutoff_stage_line, [t, 0], aux_cutoffs)
     one_quota_stage_line = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
     pl.tile.write(one_quota_stage_line, [0, 0], pl.cast(K - one_above, pl.INT32))
     pl.store(one_quota_stage_line, [t, 0], aux_quotas)
     one_replay_safe = discarded < one_cutoff
     if discarded == one_cutoff:
      one_cert_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
      one_cert_candidates = pl.tile.select(pl.tile.cmps(one_values, one_cutoff, cmp_type=0), one_cert_positions, -1)
      one_cert_wide = pl.col_max(pl.reshape(one_cert_candidates, [8, 64]))
      one_cert_eight = pl.col_max(pl.reshape(one_cert_wide, [8, 8]))
      one_cert_result = pl.tile.read(one_cert_eight, [0, 0])
      one_cert_lane_1 = pl.tile.read(one_cert_eight, [0, 1])
      if one_cert_lane_1 > one_cert_result:
       one_cert_result = one_cert_lane_1
      one_cert_lane_2 = pl.tile.read(one_cert_eight, [0, 2])
      if one_cert_lane_2 > one_cert_result:
       one_cert_result = one_cert_lane_2
      one_cert_lane_3 = pl.tile.read(one_cert_eight, [0, 3])
      if one_cert_lane_3 > one_cert_result:
       one_cert_result = one_cert_lane_3
      one_cert_lane_4 = pl.tile.read(one_cert_eight, [0, 4])
      if one_cert_lane_4 > one_cert_result:
       one_cert_result = one_cert_lane_4
      one_cert_lane_5 = pl.tile.read(one_cert_eight, [0, 5])
      if one_cert_lane_5 > one_cert_result:
       one_cert_result = one_cert_lane_5
      one_cert_lane_6 = pl.tile.read(one_cert_eight, [0, 6])
      if one_cert_lane_6 > one_cert_result:
       one_cert_result = one_cert_lane_6
      one_cert_lane_7 = pl.tile.read(one_cert_eight, [0, 7])
      if one_cert_lane_7 > one_cert_result:
       one_cert_result = one_cert_lane_7
      if frontier_position > one_cert_result:
       one_replay_safe = True
     if one_replay_safe:
      if one_cutoff > FIN_FLOOR:
       one_root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       one_selected = pl.cast(K, pl.INDEX)
       if one_selected > 0:
        one_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(one_root_positions, pl.FP32))
        one_root_keys = one_root_priority
        one_root_order = pl.tile.sort32(one_root_keys, pl.reinterpret_view(one_root_positions, pl.UINT32))
        one_root_order = pl.tile.mrgsort(one_root_order, block_len=64)
        one_root_order = pl.tile.mrgsort(one_root_order, block_len=256)
        one_root_logical = pl.tile.gather_mask(one_root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        one_root_valid = pl.tile.set_validshape(one_root_logical, 1, one_selected)
        one_root_safe = pl.tile.fillpad(one_root_valid, pad_value=pl.PadValue.zero)
        one_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
        for one_map_start in pl.range(0, one_selected, MAP):
         one_map_count = pl.min(MAP, one_selected - one_map_start)
         one_map_bits = pl.tile.slice(pl.reinterpret_view(one_root_safe, pl.FP32), [1, MAP], [0, one_map_start])
         one_map_logical = pl.reinterpret_view(one_map_bits, pl.INT32)
         one_root_page_idx = pl.tile.shrs(one_map_logical, 7)
         one_root_page_ids = pl.tile.gather(materialize_displacements, one_root_page_idx, one_root_tmp)
         one_root_physical = pl.add(one_map_logical, one_root_page_ids)
         pl.store(pl.tile.set_validshape(one_root_physical, 1, one_map_count), [t, one_map_start], indices)
      else:
       one_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
       one_root = fin
       one_root_values = pl.tile.gather_mask(one_root, mask_pattern=pl.tile.MaskPattern.P0101)
       one_root_positions = pl.tile.gather_mask(one_root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       one_ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
       one_root_strict = pl.tile.select(pl.tile.cmps(one_root_values, one_cutoff, cmp_type=4), one_ones_root, 0.0)
       one_root_equal = pl.tile.select(pl.tile.cmps(one_root_values, one_cutoff, cmp_type=0), one_ones_root, 0.0)
       one_root_eligible = pl.maximum(one_root_strict, one_root_equal)
       one_root_visible = pl.tile.cmps(pl.cast(one_root_positions, pl.FP32), pl.cast(pl.cast(one_visible, pl.INT32), pl.FP32), cmp_type=2)
       one_root_keep = pl.tile.select(one_root_visible, one_root_eligible, 0.0)
       one_root_sums64 = pl.col_sum(pl.reshape(one_root_keep, [8, 64]))
       one_root_sums8 = pl.col_sum(pl.reshape(one_root_sums64, [8, 8]))
       one_selected = pl.cast(0, pl.INDEX)
       for one_root_lane in pl.unroll(8):
        one_root_part = pl.cast(pl.tile.read(one_root_sums8, [0, one_root_lane]), pl.INT32)
        one_selected = one_selected + pl.cast(one_root_part, pl.INDEX)
       pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
       if one_selected > 0:
        one_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(one_root_positions, pl.FP32))
        one_root_keys = pl.tile.select(pl.tile.cmps(one_root_keep, 0.0, cmp_type=4), one_root_priority, -1.0)
        one_root_order = pl.tile.sort32(one_root_keys, pl.reinterpret_view(one_root_positions, pl.UINT32))
        one_root_order = pl.tile.mrgsort(one_root_order, block_len=64)
        one_root_order = pl.tile.mrgsort(one_root_order, block_len=256)
        one_root_sorted_keys = pl.tile.gather_mask(one_root_order, mask_pattern=pl.tile.MaskPattern.P0101)
        one_root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), one_root_sorted_keys), pl.INT32)
        one_root_valid = pl.tile.set_validshape(one_root_logical, 1, one_selected)
        one_root_safe = pl.tile.fillpad(one_root_valid, pad_value=pl.PadValue.zero)
        one_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
        for one_map_start in pl.range(0, one_selected, MAP):
         one_map_count = pl.min(MAP, one_selected - one_map_start)
         one_map_bits = pl.tile.slice(pl.reinterpret_view(one_root_safe, pl.FP32), [1, MAP], [0, one_map_start])
         one_map_logical = pl.reinterpret_view(one_map_bits, pl.INT32)
         one_root_page_idx = pl.tile.shrs(one_map_logical, 7)
         one_root_page_ids = pl.tile.gather(materialize_displacements, one_root_page_idx, one_root_tmp)
         one_root_physical = pl.add(one_map_logical, one_root_page_ids)
         pl.store(pl.tile.set_validshape(one_root_physical, 1, one_map_count), [t, one_map_start], indices)
     else:
      one_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
      one_remaining = pl.cast(K - one_above, pl.INDEX)
      pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
      one_ones_m = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
      one_lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
      one_lanes_f = pl.cast(one_lanes, pl.FP32)
      one_lane_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
      one_priority = pl.cast(pl.sub(one_lane_origin, one_lanes), pl.FP32)
      one_gtmp = pl.tile.create([1, SCAN], dtype=pl.INT32)
      one_count = pl.cast(0, pl.INDEX)
      for one_start in pl.range(0, one_visible, SCAN):
       if one_count < K:
        one_valid = pl.min(SCAN, one_visible - one_start)
        one_scan_raw = pl.cast(pl.load(scores, [t, one_start], [1, SCAN], valid_shape=[1, one_valid]), pl.FP32)
        one_scan_vals = pl.tile.fillpad(one_scan_raw, pad_value=pl.PadValue.min)
        one_strict = pl.tile.select(pl.tile.cmps(one_scan_vals, one_cutoff, cmp_type=4), one_ones_m, 0.0)
        one_ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
        if one_remaining > 0:
         one_eq = pl.tile.select(pl.tile.cmps(one_scan_vals, one_cutoff, cmp_type=0), one_ones_m, 0.0)
         one_eq_valid = pl.tile.cmps(one_lanes_f, pl.cast(pl.cast(one_valid, pl.INT32), pl.FP32), cmp_type=2)
         one_eq = pl.tile.select(one_eq_valid, one_eq, 0.0)
         one_eq_sums64 = pl.col_sum(pl.reshape(one_eq, [SCAN // 64, 64]))
         one_eq_sums8 = pl.col_sum(pl.reshape(one_eq_sums64, [8, 8]))
         one_available = pl.cast(0, pl.INDEX)
         for one_eq_lane in pl.unroll(8):
          one_eq_part = pl.cast(pl.tile.read(one_eq_sums8, [0, one_eq_lane]), pl.INT32)
          one_available = one_available + pl.cast(one_eq_part, pl.INDEX)
         one_take = pl.min(one_available, one_remaining)
         if one_take > 0:
          if one_take == one_available:
           one_ties = one_eq
          else:
           one_tie_keys = pl.mul(one_eq, one_priority)
           one_tie_pairs = pl.tile.sort32(one_tie_keys, pl.reinterpret_view(one_lanes, pl.UINT32))
           one_tie_pairs = pl.tile.mrgsort(one_tie_pairs, block_len=64)
           one_tie_merged = one_tie_pairs
           one_equal_keys = pl.tile.gather_mask(one_tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
           one_last_key_f = pl.tile.read(one_equal_keys, [0, one_take - 1])
           one_boundary_f = 128.0 - one_last_key_f
           one_ties = pl.tile.select(pl.tile.cmps(one_lanes_f, one_boundary_f, cmp_type=3), one_eq, 0.0)
          one_remaining = one_remaining - one_take
        one_keep = pl.maximum(one_strict, one_ties)
        one_keep_valid = pl.tile.set_validshape(one_keep, 1, one_valid)
        one_keep_padded = pl.tile.fillpad(one_keep_valid, pad_value=pl.PadValue.zero)
        one_keep_sums64 = pl.col_sum(pl.reshape(one_keep_padded, [SCAN // 64, 64]))
        one_keep_sums8 = pl.col_sum(pl.reshape(one_keep_sums64, [8, 8]))
        one_keep_count = pl.cast(0, pl.INDEX)
        for one_keep_lane in pl.unroll(8):
         one_keep_part = pl.cast(pl.tile.read(one_keep_sums8, [0, one_keep_lane]), pl.INT32)
         one_keep_count = one_keep_count + pl.cast(one_keep_part, pl.INDEX)
        one_selected = pl.min(one_keep_count, K - one_count)
        if one_selected > 0:
         one_order_keys = pl.mul(one_keep_padded, one_priority)
         one_order_pairs = pl.tile.sort32(one_order_keys, pl.reinterpret_view(one_lanes, pl.UINT32))
         one_order_pairs = pl.tile.mrgsort(one_order_pairs, block_len=64)
         one_order_merged = one_order_pairs
         one_order_vals = pl.tile.gather_mask(one_order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
         one_offs = pl.sub(one_lane_origin, pl.cast(one_order_vals, pl.INT32))
         one_compact_valid = pl.tile.set_validshape(one_offs, 1, one_selected)
         one_compact_safe = pl.tile.fillpad(one_compact_valid, pad_value=pl.PadValue.zero)
         one_logical = pl.add(one_compact_safe, pl.cast(one_start, pl.INT32))
         one_page_idx = pl.tile.shrs(one_logical, 7)
         one_page_ids = pl.tile.gather(materialize_displacements, one_page_idx, one_gtmp)
         one_physical = pl.add(one_logical, one_page_ids)
         pl.store(pl.tile.set_validshape(one_physical, 1, one_selected), [t, one_count], indices)
         one_count = one_count + one_selected
   for pack_task in pl.spmd(1, name_hint='pack_auxiliary'):
    for output_row in pl.range(0, M, 16):
     cutoff_lines = pl.load(aux_cutoffs, [output_row, 0], [16, 16])
     quota_lines = pl.load(aux_quotas, [output_row, 0], [16, 16])
     cutoff_flat = pl.reshape(cutoff_lines, [1, 256])
     quota_flat = pl.reshape(quota_lines, [1, 256])
     pack_offsets = pl.mul(pl.tile.arange(0, [1, 16], dtype=pl.INT32), 16)
     cutoff_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     quota_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     cutoff_packed = pl.tile.gather(cutoff_flat, pack_offsets, cutoff_scratch)
     quota_packed = pl.tile.gather(quota_flat, pack_offsets, quota_scratch)
     pl.store(pl.reshape(cutoff_packed, [16, 1]), [output_row, 0], cutoffs)
     pl.store(pl.reshape(quota_packed, [16, 1]), [output_row, 0], quotas)
   return indices
 elif NL == 2:

  @pl.jit
  def dsa41_topk_guarded_root(scores: pl.Tensor[[M, N], pl.BF16], lengths: pl.Tensor[[M], pl.INT32], block_table: pl.Tensor[[1, P], pl.INT32], wpairs: pl.InOut[pl.Tensor[[M, WORKSPACE_WIDTH], pl.FP32]], cutoffs: pl.InOut[pl.Tensor[[M, 1], pl.FP32]], quotas: pl.InOut[pl.Tensor[[M, 1], pl.INT32]], indices: pl.Out[pl.Tensor[[M, K], pl.INT32]]):
   aux_discarded = pl.create_tensor([M, 16], dtype=pl.FP32)
   publication_bank = pl.create_tensor([M // 4, 128], dtype=pl.FP32)
   aux_front = pl.create_tensor([M, 16], dtype=pl.INT32)
   leaf_discarded_bank = pl.create_tensor([M, NL * 16], dtype=pl.FP32)
   leaf_front_bank = pl.create_tensor([M, NL * 16], dtype=pl.INT32)
   for leaf_job in pl.spmd(LEAF_COHORTS * NL, name_hint='canonical_leaf_plane', allow_early_resolve=True):
    leaf = leaf_job // LEAF_COHORTS
    leaf_cohort_base = leaf_job % LEAF_COHORTS * LEAF_ROWS
    for t in pl.range(leaf_cohort_base, pl.min(leaf_cohort_base + LEAF_ROWS, M), 1):
     zero_pairs = pl.tile.full([1, PAIR_W], dtype=pl.FP32, value=0.0)
     if leaf == 0:
      for clear_offset in pl.range(NL * PAIR_W, WORKSPACE_WIDTH, PAIR_W):
       pl.store(zero_pairs, [t, clear_offset], wpairs)
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     begin = leaf * LEAF
     if begin < wv:
      valid = pl.min(LEAF, wv - begin)
      if valid <= 512:
       narrow_raw = pl.cast(pl.load(scores, [t, begin], [1, 512], valid_shape=[1, valid]), pl.FP32)
       narrow_padded = pl.tile.fillpad(narrow_raw, pad_value=pl.PadValue.min)
       narrow_values = pl.maximum(narrow_padded, SORT_FLOOR)
       narrow_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       narrow_pairs = pl.tile.sort32(narrow_values, pl.reinterpret_view(narrow_positions, pl.UINT32))
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=64)
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=256)
       pl.store(narrow_pairs, [t, leaf * PAIR_W], wpairs)
      elif valid <= 1024:
       left_valid = pl.min(valid, 512)
       right_valid = pl.max(valid - 512, 0)
       left_raw = pl.cast(pl.load(scores, [t, begin], [1, 512], valid_shape=[1, left_valid]), pl.FP32)
       right_raw = pl.cast(pl.load(scores, [t, begin + 512], [1, 512], valid_shape=[1, right_valid]), pl.FP32)
       left_values = pl.maximum(pl.tile.fillpad(left_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       right_values = pl.maximum(pl.tile.fillpad(right_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       left_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       right_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin + 512, pl.INT32))
       left_pairs = pl.tile.sort32(left_values, pl.reinterpret_view(left_positions, pl.UINT32))
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=64)
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=256)
       right_pairs = pl.tile.sort32(right_values, pl.reinterpret_view(right_positions, pl.UINT32))
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=64)
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=256)
       native_scratch = pl.tile.create([1, 2048], dtype=pl.FP32)
       native_pairs = pl.tile.mrgsort(left_pairs, right_pairs, tmp=native_scratch)
       native_discarded = pl.tile.read(native_pairs, [0, PAIR_W])
       if native_discarded > discarded:
        discarded = native_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if native_discarded == discarded:
        native_front_boundary = pl.tile.read(native_pairs, [0, PAIR_W - 2])
        native_front_bound = native_front_boundary
        if native_front_bound < FIN_FLOOR:
         native_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if native_discarded >= native_front_bound:
         native_arity_third = pl.tile.read(native_pairs, [0, PAIR_W + 4])
         if native_arity_third < native_discarded:
          native_arity_second = pl.tile.read(native_pairs, [0, PAIR_W + 2])
          native_arity_bits = pl.reinterpret_view(native_pairs, pl.INT32)
          native_front_result = pl.tile.read(native_arity_bits, [0, PAIR_W + 1])
          if native_arity_second == native_discarded:
           native_arity_second_position = pl.tile.read(native_arity_bits, [0, PAIR_W + 3])
           native_front_result = pl.min(native_front_result, native_arity_second_position)
         else:
          native_front_lookahead = pl.tile.read(native_pairs, [0, PAIR_W + 64])
          if native_front_lookahead < native_discarded:
           native_front_compact_lanes = pl.add(pl.tile.arange(0, [1, 64], dtype=pl.INT32), PAIR_W)
           native_front_compact_scratch = pl.tile.create([1, 2048], dtype=pl.INT32)
           native_front_compact = pl.tile.gather(native_pairs, native_front_compact_lanes, native_front_compact_scratch)
           native_front_compact_values = pl.tile.gather_mask(native_front_compact, mask_pattern=pl.tile.MaskPattern.P0101)
           native_front_compact_positions = pl.tile.gather_mask(native_front_compact, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
           native_front_compact_candidates = pl.tile.select(pl.tile.cmps(native_front_compact_values, native_discarded, cmp_type=0), native_front_compact_positions, 2147483647)
           native_front_compact_carrier = pl.add(pl.tile.full([8, 32], dtype=pl.INT32, value=0), native_front_compact_candidates)
           native_front_compact_reduce_scratch = pl.tile.create([8, 32], dtype=pl.INT32)
           native_front_compact_min = pl.row_min(native_front_compact_carrier, native_front_compact_reduce_scratch)
           native_front_result = pl.tile.read(native_front_compact_min, [0, 0])
          else:
           native_front_tail = pl.tile.slice(native_pairs, [1, 1024], [0, PAIR_W])
           native_front_values = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
           native_front_positions = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
           native_front_candidates = pl.tile.select(pl.tile.cmps(native_front_values, native_discarded, cmp_type=0), native_front_positions, 2147483647)
           native_front_wide = pl.col_min(pl.reshape(native_front_candidates, [8, 64]))
           native_front_eight = pl.col_min(pl.reshape(native_front_wide, [8, 8]))
           native_front_result = pl.tile.read(native_front_eight, [0, 0])
           native_front_lane_1 = pl.tile.read(native_front_eight, [0, 1])
           if native_front_lane_1 < native_front_result:
            native_front_result = native_front_lane_1
           native_front_lane_2 = pl.tile.read(native_front_eight, [0, 2])
           if native_front_lane_2 < native_front_result:
            native_front_result = native_front_lane_2
           native_front_lane_3 = pl.tile.read(native_front_eight, [0, 3])
           if native_front_lane_3 < native_front_result:
            native_front_result = native_front_lane_3
           native_front_lane_4 = pl.tile.read(native_front_eight, [0, 4])
           if native_front_lane_4 < native_front_result:
            native_front_result = native_front_lane_4
           native_front_lane_5 = pl.tile.read(native_front_eight, [0, 5])
           if native_front_lane_5 < native_front_result:
            native_front_result = native_front_lane_5
           native_front_lane_6 = pl.tile.read(native_front_eight, [0, 6])
           if native_front_lane_6 < native_front_result:
            native_front_result = native_front_lane_6
           native_front_lane_7 = pl.tile.read(native_front_eight, [0, 7])
           if native_front_lane_7 < native_front_result:
            native_front_result = native_front_lane_7
         if native_front_result < frontier_position:
          frontier_position = native_front_result
       pl.store(pl.tile.slice(native_pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
      else:
       raw = pl.cast(pl.load(scores, [t, begin], [1, LEAF], valid_shape=[1, valid]), pl.FP32)
       padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
       flr = pl.tile.full([1, LEAF], dtype=pl.FP32, value=SORT_FLOOR)
       vals = pl.maximum(padded, flr)
       idx_i32 = pl.add(pl.tile.arange(0, [1, LEAF], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       pairs = pl.tile.sort32(vals, pl.reinterpret_view(idx_i32, pl.UINT32))
       pairs = pl.tile.mrgsort(pairs, block_len=64)
       pairs = pl.tile.mrgsort(pairs, block_len=256)
       pairs = pl.tile.mrgsort(pairs, block_len=1024)
       leaf_discarded = pl.tile.read(pairs, [0, PAIR_W])
       if leaf_discarded > discarded:
        discarded = leaf_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if leaf_discarded == discarded:
        leaf_front_boundary = pl.tile.read(pairs, [0, PAIR_W - 2])
        leaf_front_bound = leaf_front_boundary
        if leaf_front_bound < FIN_FLOOR:
         leaf_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if leaf_discarded >= leaf_front_bound:
         leaf_arity_third = pl.tile.read(pairs, [0, PAIR_W + 4])
         if leaf_arity_third < leaf_discarded:
          leaf_arity_second = pl.tile.read(pairs, [0, PAIR_W + 2])
          leaf_arity_bits = pl.reinterpret_view(pairs, pl.INT32)
          leaf_front_result = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 1])
          if leaf_arity_second == leaf_discarded:
           leaf_arity_second_position = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 3])
           leaf_front_result = pl.min(leaf_front_result, leaf_arity_second_position)
         else:
          leaf_front_lookahead = pl.tile.read(pairs, [0, PAIR_W + 64])
          if leaf_front_lookahead < leaf_discarded:
           leaf_front_compact_lanes = pl.add(pl.tile.arange(0, [1, 64], dtype=pl.INT32), PAIR_W)
           leaf_front_compact_scratch = pl.tile.create([1, 4096], dtype=pl.INT32)
           leaf_front_compact = pl.tile.gather(pairs, leaf_front_compact_lanes, leaf_front_compact_scratch)
           leaf_front_compact_values = pl.tile.gather_mask(leaf_front_compact, mask_pattern=pl.tile.MaskPattern.P0101)
           leaf_front_compact_positions = pl.tile.gather_mask(leaf_front_compact, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
           leaf_front_compact_candidates = pl.tile.select(pl.tile.cmps(leaf_front_compact_values, leaf_discarded, cmp_type=0), leaf_front_compact_positions, 2147483647)
           leaf_front_compact_carrier = pl.add(pl.tile.full([8, 32], dtype=pl.INT32, value=0), leaf_front_compact_candidates)
           leaf_front_compact_reduce_scratch = pl.tile.create([8, 32], dtype=pl.INT32)
           leaf_front_compact_min = pl.row_min(leaf_front_compact_carrier, leaf_front_compact_reduce_scratch)
           leaf_front_result = pl.tile.read(leaf_front_compact_min, [0, 0])
          else:
           leaf_front_tail = pl.tile.slice(pairs, [1, 3072], [0, PAIR_W])
           leaf_front_values = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
           leaf_front_positions = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
           leaf_front_candidates = pl.tile.select(pl.tile.cmps(leaf_front_values, leaf_discarded, cmp_type=0), leaf_front_positions, 2147483647)
           leaf_front_wide = pl.col_min(pl.reshape(leaf_front_candidates, [24, 64]))
           leaf_front_eight = pl.col_min(pl.reshape(leaf_front_wide, [8, 8]))
           leaf_front_result = pl.tile.read(leaf_front_eight, [0, 0])
           leaf_front_lane_1 = pl.tile.read(leaf_front_eight, [0, 1])
           if leaf_front_lane_1 < leaf_front_result:
            leaf_front_result = leaf_front_lane_1
           leaf_front_lane_2 = pl.tile.read(leaf_front_eight, [0, 2])
           if leaf_front_lane_2 < leaf_front_result:
            leaf_front_result = leaf_front_lane_2
           leaf_front_lane_3 = pl.tile.read(leaf_front_eight, [0, 3])
           if leaf_front_lane_3 < leaf_front_result:
            leaf_front_result = leaf_front_lane_3
           leaf_front_lane_4 = pl.tile.read(leaf_front_eight, [0, 4])
           if leaf_front_lane_4 < leaf_front_result:
            leaf_front_result = leaf_front_lane_4
           leaf_front_lane_5 = pl.tile.read(leaf_front_eight, [0, 5])
           if leaf_front_lane_5 < leaf_front_result:
            leaf_front_result = leaf_front_lane_5
           leaf_front_lane_6 = pl.tile.read(leaf_front_eight, [0, 6])
           if leaf_front_lane_6 < leaf_front_result:
            leaf_front_result = leaf_front_lane_6
           leaf_front_lane_7 = pl.tile.read(leaf_front_eight, [0, 7])
           if leaf_front_lane_7 < leaf_front_result:
            leaf_front_result = leaf_front_lane_7
         if leaf_front_result < frontier_position:
          frontier_position = leaf_front_result
       pl.store(pl.tile.slice(pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
     else:
      pl.store(zero_pairs, [t, leaf * PAIR_W], wpairs)
     leaf_score_line = pl.tile.full([1, 16], dtype=pl.FP32, value=SORT_FLOOR)
     leaf_position_line = pl.tile.full([1, 16], dtype=pl.INT32, value=2147483647)
     pl.tile.write(leaf_score_line, [0, 0], discarded)
     pl.tile.write(leaf_position_line, [0, 0], frontier_position)
     pl.store(leaf_score_line, [t, leaf * 16], leaf_discarded_bank)
     pl.store(leaf_position_line, [t, leaf * 16], leaf_front_bank)
   for merge_cohort_job in pl.spmd((M + 4 - 1) // 4, name_hint='bounded_root_handoff', allow_early_resolve=True):
    cohort_base = merge_cohort_job * 4
    publication_record = pl.tile.full([1, 128], dtype=pl.FP32, value=0.0)
    if P == 1:
     materialize_single_page = pl.read(block_table, [0, 0])
     materialize_pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), materialize_single_page), pad_value=pl.PadValue.zero)
    else:
     materialize_pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
    materialize_page_lanes = pl.tile.arange(0, [1, P8], dtype=pl.INT32)
    materialize_displacements = pl.mul(pl.sub(materialize_pages, materialize_page_lanes), PAGE)
    for t in pl.range(cohort_base, pl.min(cohort_base + 4, M), 1):
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     lc = (wv + LEAF - 1) // LEAF
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     for proof_leaf in pl.range(NL):
      proof_score = pl.read(leaf_discarded_bank, [t, proof_leaf * 16])
      proof_position = pl.read(leaf_front_bank, [t, proof_leaf * 16])
      if proof_score > discarded:
       discarded = proof_score
       frontier_position = pl.cast(2147483647, pl.INT32)
      if proof_score == discarded:
       if proof_position < frontier_position:
        frontier_position = proof_position
     if lc == 1:
      fin = pl.load(wpairs, [t, 0], [1, PAIR_W])
      short_complete = False
      if visible > 0:
       if visible < K:
        short_minimum = pl.tile.read(fin, [0, 2 * (visible - 1)])
        if short_minimum > FIN_FLOOR:
         short_complete = True
      if short_complete:
       short_cutoff = pl.cast(FIN_FLOOR, pl.FP32)
       pl.tile.write(publication_record, [0, t - cohort_base], short_cutoff)
       pl.tile.write(publication_record, [0, 4 + t - cohort_base], pl.cast(pl.cast(K - visible, pl.INT32), pl.FP32))
       short_lanes = pl.tile.arange(0, [1, K], dtype=pl.INT32)
       short_keep = pl.tile.cmps(pl.cast(short_lanes, pl.FP32), pl.cast(pl.cast(visible, pl.INT32), pl.FP32), cmp_type=2)
       short_safe = pl.tile.select(short_keep, short_lanes, 0)
       short_pages = pl.tile.shrs(short_safe, 7)
       short_scratch = pl.tile.create([1, MAP], dtype=pl.INT32)
       short_displacements = pl.tile.gather(materialize_displacements, short_pages, short_scratch)
       short_physical = pl.add(short_safe, short_displacements)
       short_output = pl.tile.select(short_keep, short_physical, -1)
       pl.store(short_output, [t, 0], indices)
      else:
       one_cutoff = pl.tile.read(fin, [0, 2 * (K - 1)])
       if one_cutoff < FIN_FLOOR:
        one_cutoff = pl.cast(FIN_FLOOR, pl.FP32)
       one_above = pl.cast(0, pl.INDEX)
       one_neighbor = pl.tile.read(fin, [0, 2 * (K - 2)])
       one_witness = False
       if visible >= K:
        if one_cutoff > FIN_FLOOR:
         if one_neighbor > one_cutoff:
          one_witness = True
       if one_witness:
        one_above = pl.cast(K - 1, pl.INDEX)
       else:
        one_rank_probe_256 = pl.tile.read(fin, [0, 2 * (one_above + 256 - 1)])
        one_rank_bit_256 = pl.cast(pl.cast(one_rank_probe_256 > one_cutoff, pl.INT32), pl.INDEX)
        one_above = one_above + 256 * one_rank_bit_256
        one_rank_probe_128 = pl.tile.read(fin, [0, 2 * (one_above + 128 - 1)])
        one_rank_bit_128 = pl.cast(pl.cast(one_rank_probe_128 > one_cutoff, pl.INT32), pl.INDEX)
        one_above = one_above + 128 * one_rank_bit_128
        one_rank_probe_64 = pl.tile.read(fin, [0, 2 * (one_above + 64 - 1)])
        one_rank_bit_64 = pl.cast(pl.cast(one_rank_probe_64 > one_cutoff, pl.INT32), pl.INDEX)
        one_above = one_above + 64 * one_rank_bit_64
        one_rank_probe_32 = pl.tile.read(fin, [0, 2 * (one_above + 32 - 1)])
        one_rank_bit_32 = pl.cast(pl.cast(one_rank_probe_32 > one_cutoff, pl.INT32), pl.INDEX)
        one_above = one_above + 32 * one_rank_bit_32
        one_rank_probe_16 = pl.tile.read(fin, [0, 2 * (one_above + 16 - 1)])
        one_rank_bit_16 = pl.cast(pl.cast(one_rank_probe_16 > one_cutoff, pl.INT32), pl.INDEX)
        one_above = one_above + 16 * one_rank_bit_16
        one_rank_probe_8 = pl.tile.read(fin, [0, 2 * (one_above + 8 - 1)])
        one_rank_bit_8 = pl.cast(pl.cast(one_rank_probe_8 > one_cutoff, pl.INT32), pl.INDEX)
        one_above = one_above + 8 * one_rank_bit_8
        one_rank_probe_4 = pl.tile.read(fin, [0, 2 * (one_above + 4 - 1)])
        one_rank_bit_4 = pl.cast(pl.cast(one_rank_probe_4 > one_cutoff, pl.INT32), pl.INDEX)
        one_above = one_above + 4 * one_rank_bit_4
        one_rank_probe_2 = pl.tile.read(fin, [0, 2 * (one_above + 2 - 1)])
        one_rank_bit_2 = pl.cast(pl.cast(one_rank_probe_2 > one_cutoff, pl.INT32), pl.INDEX)
        one_above = one_above + 2 * one_rank_bit_2
        one_rank_probe_1 = pl.tile.read(fin, [0, 2 * (one_above + 1 - 1)])
        one_rank_bit_1 = pl.cast(pl.cast(one_rank_probe_1 > one_cutoff, pl.INT32), pl.INDEX)
        one_above = one_above + 1 * one_rank_bit_1
       pl.tile.write(publication_record, [0, t - cohort_base], one_cutoff)
       pl.tile.write(publication_record, [0, 4 + t - cohort_base], pl.cast(pl.cast(K - one_above, pl.INT32), pl.FP32))
       one_replay_safe = discarded < one_cutoff
       one_single = one_witness
       if discarded > one_cutoff:
        one_single = False
       if one_single:
        one_replay_safe = True
       else:
        if discarded == one_cutoff:
         if K - one_above <= 8:
          one_cert_result = pl.cast(-1, pl.INT32)
          one_tbits = pl.reinterpret_view(fin, pl.INT32)
          for one_tidx in pl.range(one_above, K):
           one_tpos = pl.tile.read(one_tbits, [0, 2 * one_tidx + 1])
           if pl.tile.read(fin, [0, 2 * one_tidx]) == one_cutoff:
            one_cert_result = pl.max(one_cert_result, one_tpos)
         else:
          one_cert_values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
          one_cert_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          one_cert_candidates = pl.tile.select(pl.tile.cmps(one_cert_values, one_cutoff, cmp_type=0), one_cert_positions, -1)
          one_cert_wide = pl.col_max(pl.reshape(one_cert_candidates, [8, 64]))
          one_cert_eight = pl.col_max(pl.reshape(one_cert_wide, [8, 8]))
          one_cert_result = pl.tile.read(one_cert_eight, [0, 0])
          one_cert_lane_1 = pl.tile.read(one_cert_eight, [0, 1])
          if one_cert_lane_1 > one_cert_result:
           one_cert_result = one_cert_lane_1
          one_cert_lane_2 = pl.tile.read(one_cert_eight, [0, 2])
          if one_cert_lane_2 > one_cert_result:
           one_cert_result = one_cert_lane_2
          one_cert_lane_3 = pl.tile.read(one_cert_eight, [0, 3])
          if one_cert_lane_3 > one_cert_result:
           one_cert_result = one_cert_lane_3
          one_cert_lane_4 = pl.tile.read(one_cert_eight, [0, 4])
          if one_cert_lane_4 > one_cert_result:
           one_cert_result = one_cert_lane_4
          one_cert_lane_5 = pl.tile.read(one_cert_eight, [0, 5])
          if one_cert_lane_5 > one_cert_result:
           one_cert_result = one_cert_lane_5
          one_cert_lane_6 = pl.tile.read(one_cert_eight, [0, 6])
          if one_cert_lane_6 > one_cert_result:
           one_cert_result = one_cert_lane_6
          one_cert_lane_7 = pl.tile.read(one_cert_eight, [0, 7])
          if one_cert_lane_7 > one_cert_result:
           one_cert_result = one_cert_lane_7
         if frontier_position > one_cert_result:
          one_replay_safe = True
       if one_replay_safe:
        if one_cutoff > FIN_FLOOR:
         one_root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         if one_single:
          one_endpoint = pl.tile.read(one_root_positions, [0, K - 1])
          if discarded == one_cutoff:
           one_endpoint = pl.min(one_endpoint, frontier_position)
          pl.tile.write(one_root_positions, [0, K - 1], one_endpoint)
         one_selected = pl.cast(K, pl.INDEX)
         if one_selected > 0:
          one_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(one_root_positions, pl.FP32))
          one_root_keys = one_root_priority
          one_root_order = pl.tile.sort32(one_root_keys, pl.reinterpret_view(one_root_positions, pl.UINT32))
          one_root_order = pl.tile.mrgsort(one_root_order, block_len=64)
          one_root_order = pl.tile.mrgsort(one_root_order, block_len=256)
          one_root_logical = pl.tile.gather_mask(one_root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          one_root_valid = pl.tile.set_validshape(one_root_logical, 1, one_selected)
          one_root_safe = pl.tile.fillpad(one_root_valid, pad_value=pl.PadValue.zero)
          one_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
          for one_map_start in pl.range(0, one_selected, MAP):
           one_map_count = pl.min(MAP, one_selected - one_map_start)
           one_map_bits = pl.tile.slice(pl.reinterpret_view(one_root_safe, pl.FP32), [1, MAP], [0, one_map_start])
           one_map_logical = pl.reinterpret_view(one_map_bits, pl.INT32)
           one_root_page_idx = pl.tile.shrs(one_map_logical, 7)
           one_root_page_ids = pl.tile.gather(materialize_displacements, one_root_page_idx, one_root_tmp)
           one_root_physical = pl.add(one_map_logical, one_root_page_ids)
           pl.store(pl.tile.set_validshape(one_root_physical, 1, one_map_count), [t, one_map_start], indices)
        else:
         one_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
         one_root = fin
         one_root_values = pl.tile.gather_mask(one_root, mask_pattern=pl.tile.MaskPattern.P0101)
         one_root_positions = pl.tile.gather_mask(one_root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         one_ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
         one_root_strict = pl.tile.select(pl.tile.cmps(one_root_values, one_cutoff, cmp_type=4), one_ones_root, 0.0)
         one_root_equal = pl.tile.select(pl.tile.cmps(one_root_values, one_cutoff, cmp_type=0), one_ones_root, 0.0)
         one_root_eligible = pl.maximum(one_root_strict, one_root_equal)
         one_root_visible = pl.tile.cmps(pl.cast(one_root_positions, pl.FP32), pl.cast(pl.cast(one_visible, pl.INT32), pl.FP32), cmp_type=2)
         one_root_keep = pl.tile.select(one_root_visible, one_root_eligible, 0.0)
         one_root_sums64 = pl.col_sum(pl.reshape(one_root_keep, [8, 64]))
         one_root_sums8 = pl.col_sum(pl.reshape(one_root_sums64, [8, 8]))
         one_selected = pl.cast(0, pl.INDEX)
         for one_root_lane in pl.unroll(8):
          one_root_part = pl.cast(pl.tile.read(one_root_sums8, [0, one_root_lane]), pl.INT32)
          one_selected = one_selected + pl.cast(one_root_part, pl.INDEX)
         pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
         if one_selected > 0:
          one_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(one_root_positions, pl.FP32))
          one_root_keys = pl.tile.select(pl.tile.cmps(one_root_keep, 0.0, cmp_type=4), one_root_priority, -1.0)
          one_root_order = pl.tile.sort32(one_root_keys, pl.reinterpret_view(one_root_positions, pl.UINT32))
          one_root_order = pl.tile.mrgsort(one_root_order, block_len=64)
          one_root_order = pl.tile.mrgsort(one_root_order, block_len=256)
          one_root_sorted_keys = pl.tile.gather_mask(one_root_order, mask_pattern=pl.tile.MaskPattern.P0101)
          one_root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), one_root_sorted_keys), pl.INT32)
          one_root_valid = pl.tile.set_validshape(one_root_logical, 1, one_selected)
          one_root_safe = pl.tile.fillpad(one_root_valid, pad_value=pl.PadValue.zero)
          one_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
          for one_map_start in pl.range(0, one_selected, MAP):
           one_map_count = pl.min(MAP, one_selected - one_map_start)
           one_map_bits = pl.tile.slice(pl.reinterpret_view(one_root_safe, pl.FP32), [1, MAP], [0, one_map_start])
           one_map_logical = pl.reinterpret_view(one_map_bits, pl.INT32)
           one_root_page_idx = pl.tile.shrs(one_map_logical, 7)
           one_root_page_ids = pl.tile.gather(materialize_displacements, one_root_page_idx, one_root_tmp)
           one_root_physical = pl.add(one_map_logical, one_root_page_ids)
           pl.store(pl.tile.set_validshape(one_root_physical, 1, one_map_count), [t, one_map_start], indices)
       else:
        one_bulk_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
        one_bulk_quota = pl.cast(K - one_above, pl.INT32)
        pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
        one_bulk_raw = pl.load(scores, [t, 0], [1, 4096], valid_shape=[1, N])
        one_bulk_values = pl.reshape(pl.tile.fillpad(pl.cast(one_bulk_raw, pl.FP32), pad_value=pl.PadValue.min), [32, SCAN])
        one_bulk_positions = pl.reshape(pl.tile.arange(0, [1, 4096], dtype=pl.INT32), [32, SCAN])
        one_bulk_valid = pl.tile.cmps(pl.cast(one_bulk_positions, pl.FP32), pl.cast(pl.cast(one_bulk_visible, pl.INT32), pl.FP32), cmp_type=2)
        one_bulk_strict = pl.tile.select(one_bulk_valid, pl.tile.select(pl.tile.cmps(one_bulk_values, one_cutoff, cmp_type=4), pl.tile.full([32, SCAN], dtype=pl.FP32, value=1.0), 0.0), 0.0)
        one_bulk_strict_scratch = pl.tile.create([32, SCAN], dtype=pl.FP32)
        one_bulk_count_offsets = pl.mul(pl.tile.arange(0, [1, 32], dtype=pl.INT32), 8)
        one_bulk_count_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
        one_bulk_equal = pl.tile.select(one_bulk_valid, pl.tile.select(pl.tile.cmps(one_bulk_values, one_cutoff, cmp_type=0), pl.tile.full([32, SCAN], dtype=pl.FP32, value=1.0), 0.0), 0.0)
        one_bulk_packed_flags = pl.add(pl.mul(one_bulk_strict, 8192.0), one_bulk_equal)
        one_bulk_packed_reduced = pl.row_sum(one_bulk_packed_flags, one_bulk_strict_scratch)
        one_bulk_packed_carrier = pl.row_expand_add(pl.tile.full([32, 8], dtype=pl.FP32, value=0.0), one_bulk_packed_reduced)
        one_bulk_packed_counts = pl.tile.gather(pl.reshape(one_bulk_packed_carrier, [1, 256]), one_bulk_count_offsets, one_bulk_count_scratch)
        one_bulk_blocks = pl.tile.arange(0, [1, 32], dtype=pl.INT32)
        one_bulk_prefix_scratch = pl.tile.create([1, 32], dtype=pl.INT32)
        one_bulk_packed_prefix = one_bulk_packed_counts
        one_bulk_shift_index = pl.maximum(pl.sub(one_bulk_blocks, 1), 0)
        one_bulk_shift_valid = pl.tile.cmps(pl.cast(one_bulk_blocks, pl.FP32), 1.0, cmp_type=5)
        one_bulk_packed_shift = pl.tile.gather(one_bulk_packed_prefix, one_bulk_shift_index, one_bulk_prefix_scratch)
        one_bulk_packed_prefix = pl.add(one_bulk_packed_prefix, pl.tile.select(one_bulk_shift_valid, one_bulk_packed_shift, 0.0))
        one_bulk_shift_index = pl.maximum(pl.sub(one_bulk_blocks, 2), 0)
        one_bulk_shift_valid = pl.tile.cmps(pl.cast(one_bulk_blocks, pl.FP32), 2.0, cmp_type=5)
        one_bulk_packed_shift = pl.tile.gather(one_bulk_packed_prefix, one_bulk_shift_index, one_bulk_prefix_scratch)
        one_bulk_packed_prefix = pl.add(one_bulk_packed_prefix, pl.tile.select(one_bulk_shift_valid, one_bulk_packed_shift, 0.0))
        one_bulk_shift_index = pl.maximum(pl.sub(one_bulk_blocks, 4), 0)
        one_bulk_shift_valid = pl.tile.cmps(pl.cast(one_bulk_blocks, pl.FP32), 4.0, cmp_type=5)
        one_bulk_packed_shift = pl.tile.gather(one_bulk_packed_prefix, one_bulk_shift_index, one_bulk_prefix_scratch)
        one_bulk_packed_prefix = pl.add(one_bulk_packed_prefix, pl.tile.select(one_bulk_shift_valid, one_bulk_packed_shift, 0.0))
        one_bulk_shift_index = pl.maximum(pl.sub(one_bulk_blocks, 8), 0)
        one_bulk_shift_valid = pl.tile.cmps(pl.cast(one_bulk_blocks, pl.FP32), 8.0, cmp_type=5)
        one_bulk_packed_shift = pl.tile.gather(one_bulk_packed_prefix, one_bulk_shift_index, one_bulk_prefix_scratch)
        one_bulk_packed_prefix = pl.add(one_bulk_packed_prefix, pl.tile.select(one_bulk_shift_valid, one_bulk_packed_shift, 0.0))
        one_bulk_shift_index = pl.maximum(pl.sub(one_bulk_blocks, 16), 0)
        one_bulk_shift_valid = pl.tile.cmps(pl.cast(one_bulk_blocks, pl.FP32), 16.0, cmp_type=5)
        one_bulk_packed_shift = pl.tile.gather(one_bulk_packed_prefix, one_bulk_shift_index, one_bulk_prefix_scratch)
        one_bulk_packed_prefix = pl.add(one_bulk_packed_prefix, pl.tile.select(one_bulk_shift_valid, one_bulk_packed_shift, 0.0))
        one_bulk_strict_counts = pl.cast(pl.cast(pl.mul(one_bulk_packed_counts, 0.0001220703125), pl.INT32, mode='floor'), pl.FP32)
        one_bulk_equal_counts = pl.sub(one_bulk_packed_counts, pl.mul(one_bulk_strict_counts, 8192.0))
        one_bulk_strict_prefix = pl.cast(pl.cast(pl.mul(one_bulk_packed_prefix, 0.0001220703125), pl.INT32, mode='floor'), pl.FP32)
        one_bulk_equal_prefix = pl.sub(one_bulk_packed_prefix, pl.mul(one_bulk_strict_prefix, 8192.0))
        one_bulk_strict_before = pl.sub(one_bulk_strict_prefix, one_bulk_strict_counts)
        one_bulk_equal_before = pl.sub(one_bulk_equal_prefix, one_bulk_equal_counts)
        one_bulk_tie_take = pl.minimum(one_bulk_equal_counts, pl.maximum(pl.sub(pl.add(pl.tile.full([1, 32], dtype=pl.FP32, value=0.0), pl.cast(one_bulk_quota, pl.FP32)), one_bulk_equal_before), 0.0))
        one_bulk_output_begin = pl.add(one_bulk_strict_before, pl.minimum(one_bulk_equal_before, pl.cast(one_bulk_quota, pl.FP32)))
        one_bulk_output_count = pl.add(one_bulk_strict_counts, one_bulk_tie_take)
        one_bulk_lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
        one_bulk_lanes_f = pl.cast(one_bulk_lanes, pl.FP32)
        one_bulk_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
        one_bulk_priority = pl.cast(pl.sub(one_bulk_origin, one_bulk_lanes), pl.FP32)
        one_bulk_map_scratch = pl.tile.create([1, SCAN], dtype=pl.INT32)
        for one_bulk_block in pl.range(0, (one_bulk_visible + SCAN - 1) // SCAN):
          one_bulk_selected = pl.cast(pl.cast(pl.tile.read(one_bulk_output_count, [0, one_bulk_block]), pl.INT32), pl.INDEX)
          if one_bulk_selected > 0:
            one_bulk_start = one_bulk_block * SCAN
            one_bulk_count = pl.cast(pl.cast(pl.tile.read(one_bulk_output_begin, [0, one_bulk_block]), pl.INT32), pl.INDEX)
            one_bulk_take = pl.cast(pl.cast(pl.tile.read(one_bulk_tie_take, [0, one_bulk_block]), pl.INT32), pl.INDEX)
            one_bulk_available = pl.cast(pl.cast(pl.tile.read(one_bulk_equal_counts, [0, one_bulk_block]), pl.INT32), pl.INDEX)
            one_bulk_block_flags = pl.tile.slice(pl.reshape(one_bulk_packed_flags, [1, 4096]), [1, SCAN], [0, one_bulk_start])
            one_bulk_block_ones = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
            one_bulk_block_strict = pl.tile.select(pl.tile.cmps(one_bulk_block_flags, 8192.0, cmp_type=0), one_bulk_block_ones, 0.0)
            one_bulk_block_equal = pl.tile.select(pl.tile.cmps(one_bulk_block_flags, 1.0, cmp_type=0), one_bulk_block_ones, 0.0)
            one_bulk_ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
            if one_bulk_take > 0:
              if one_bulk_take == one_bulk_available:
                one_bulk_ties = one_bulk_block_equal
              else:
                one_bulk_tie_keys = pl.mul(one_bulk_block_equal, one_bulk_priority)
                one_bulk_tie_pairs = pl.tile.sort32(one_bulk_tie_keys, pl.reinterpret_view(one_bulk_lanes, pl.UINT32))
                one_bulk_tie_pairs = pl.tile.mrgsort(one_bulk_tie_pairs, block_len=64)
                one_bulk_equal_keys = pl.tile.gather_mask(one_bulk_tie_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
                one_bulk_last_key = pl.tile.read(one_bulk_equal_keys, [0, one_bulk_take - 1])
                one_bulk_boundary = 128.0 - one_bulk_last_key
                one_bulk_ties = pl.tile.select(pl.tile.cmps(one_bulk_lanes_f, one_bulk_boundary, cmp_type=3), one_bulk_block_equal, 0.0)
            one_bulk_keep = pl.maximum(one_bulk_block_strict, one_bulk_ties)
            one_bulk_order_keys = pl.mul(one_bulk_keep, one_bulk_priority)
            one_bulk_order_pairs = pl.tile.sort32(one_bulk_order_keys, pl.reinterpret_view(one_bulk_lanes, pl.UINT32))
            one_bulk_order_pairs = pl.tile.mrgsort(one_bulk_order_pairs, block_len=64)
            one_bulk_order_values = pl.tile.gather_mask(one_bulk_order_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
            one_bulk_offsets = pl.sub(one_bulk_origin, pl.cast(one_bulk_order_values, pl.INT32))
            one_bulk_compact_valid = pl.tile.set_validshape(one_bulk_offsets, 1, one_bulk_selected)
            one_bulk_compact_safe = pl.tile.fillpad(one_bulk_compact_valid, pad_value=pl.PadValue.zero)
            one_bulk_logical = pl.add(one_bulk_compact_safe, pl.cast(one_bulk_start, pl.INT32))
            one_bulk_page_index = pl.tile.shrs(one_bulk_logical, 7)
            one_bulk_page_ids = pl.tile.gather(materialize_displacements, one_bulk_page_index, one_bulk_map_scratch)
            one_bulk_physical = pl.add(one_bulk_logical, one_bulk_page_ids)
            pl.store(pl.tile.set_validshape(one_bulk_physical, 1, one_bulk_selected), [t, one_bulk_count], indices)
     else:
      handoff_scratch = pl.tile.create([1, 2 * PAIR_W], dtype=pl.FP32)
      handoff_left = pl.load(wpairs, [t, 0], [1, PAIR_W])
      handoff_right = pl.load(wpairs, [t, PAIR_W], [1, PAIR_W])
      mboth = pl.tile.mrgsort(handoff_left, handoff_right, tmp=handoff_scratch)
      merge_discarded = pl.tile.read(mboth, [0, PAIR_W])
      if merge_discarded > discarded:
       discarded = merge_discarded
       frontier_position = pl.cast(2147483647, pl.INT32)
      if merge_discarded == discarded:
       merge_front_boundary = pl.tile.read(mboth, [0, PAIR_W - 2])
       merge_front_bound = merge_front_boundary
       if merge_front_bound < FIN_FLOOR:
        merge_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
       if merge_discarded >= merge_front_bound:
        merge_arity_third = pl.tile.read(mboth, [0, PAIR_W + 4])
        if merge_arity_third < merge_discarded:
         merge_arity_second = pl.tile.read(mboth, [0, PAIR_W + 2])
         merge_arity_bits = pl.reinterpret_view(mboth, pl.INT32)
         merge_front_result = pl.tile.read(merge_arity_bits, [0, PAIR_W + 1])
         if merge_arity_second == merge_discarded:
          merge_arity_second_position = pl.tile.read(merge_arity_bits, [0, PAIR_W + 3])
          merge_front_result = pl.min(merge_front_result, merge_arity_second_position)
        else:
         tie_edge = pl.tile.read(mboth, [0, PAIR_W + 16])
         if tie_edge != merge_discarded:
          merge_front_result = pl.cast(2147483647, pl.INT32)
          merge_tbits = pl.reinterpret_view(mboth, pl.INT32)
          for merge_tidx in pl.range(8):
           merge_tpos = pl.tile.read(merge_tbits, [0, PAIR_W + 2 * merge_tidx + 1])
           if pl.tile.read(mboth, [0, PAIR_W + 2 * merge_tidx]) == merge_discarded:
            merge_front_result = pl.min(merge_front_result, merge_tpos)
         else:
          merge_front_tail = pl.tile.slice(mboth, [1, 1024], [0, PAIR_W])
          merge_front_values = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
          merge_front_positions = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          merge_front_candidates = pl.tile.select(pl.tile.cmps(merge_front_values, merge_discarded, cmp_type=0), merge_front_positions, 2147483647)
          merge_front_wide = pl.col_min(pl.reshape(merge_front_candidates, [8, 64]))
          merge_front_eight = pl.col_min(pl.reshape(merge_front_wide, [8, 8]))
          merge_front_result = pl.tile.read(merge_front_eight, [0, 0])
          merge_front_lane_1 = pl.tile.read(merge_front_eight, [0, 1])
          if merge_front_lane_1 < merge_front_result:
           merge_front_result = merge_front_lane_1
          merge_front_lane_2 = pl.tile.read(merge_front_eight, [0, 2])
          if merge_front_lane_2 < merge_front_result:
           merge_front_result = merge_front_lane_2
          merge_front_lane_3 = pl.tile.read(merge_front_eight, [0, 3])
          if merge_front_lane_3 < merge_front_result:
           merge_front_result = merge_front_lane_3
          merge_front_lane_4 = pl.tile.read(merge_front_eight, [0, 4])
          if merge_front_lane_4 < merge_front_result:
           merge_front_result = merge_front_lane_4
          merge_front_lane_5 = pl.tile.read(merge_front_eight, [0, 5])
          if merge_front_lane_5 < merge_front_result:
           merge_front_result = merge_front_lane_5
          merge_front_lane_6 = pl.tile.read(merge_front_eight, [0, 6])
          if merge_front_lane_6 < merge_front_result:
           merge_front_result = merge_front_lane_6
          merge_front_lane_7 = pl.tile.read(merge_front_eight, [0, 7])
          if merge_front_lane_7 < merge_front_result:
           merge_front_result = merge_front_lane_7
        if merge_front_result < frontier_position:
         frontier_position = merge_front_result
      pl.store(pl.tile.slice(mboth, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
      fin = pl.tile.slice(mboth, [1, PAIR_W], [0, 0])
      two_cutoff = pl.tile.read(fin, [0, 2 * (K - 1)])
      if two_cutoff < FIN_FLOOR:
       two_cutoff = pl.cast(FIN_FLOOR, pl.FP32)
      two_above = pl.cast(0, pl.INDEX)
      two_neighbor = pl.tile.read(fin, [0, 2 * (K - 2)])
      two_witness = False
      if visible >= K:
       if two_cutoff > FIN_FLOOR:
        if two_neighbor > two_cutoff:
         two_witness = True
      if two_witness:
       two_above = pl.cast(K - 1, pl.INDEX)
      else:
       two_rank_probe_256 = pl.tile.read(fin, [0, 2 * (two_above + 256 - 1)])
       two_rank_bit_256 = pl.cast(pl.cast(two_rank_probe_256 > two_cutoff, pl.INT32), pl.INDEX)
       two_above = two_above + 256 * two_rank_bit_256
       two_rank_probe_128 = pl.tile.read(fin, [0, 2 * (two_above + 128 - 1)])
       two_rank_bit_128 = pl.cast(pl.cast(two_rank_probe_128 > two_cutoff, pl.INT32), pl.INDEX)
       two_above = two_above + 128 * two_rank_bit_128
       two_rank_probe_64 = pl.tile.read(fin, [0, 2 * (two_above + 64 - 1)])
       two_rank_bit_64 = pl.cast(pl.cast(two_rank_probe_64 > two_cutoff, pl.INT32), pl.INDEX)
       two_above = two_above + 64 * two_rank_bit_64
       two_rank_probe_32 = pl.tile.read(fin, [0, 2 * (two_above + 32 - 1)])
       two_rank_bit_32 = pl.cast(pl.cast(two_rank_probe_32 > two_cutoff, pl.INT32), pl.INDEX)
       two_above = two_above + 32 * two_rank_bit_32
       two_rank_probe_16 = pl.tile.read(fin, [0, 2 * (two_above + 16 - 1)])
       two_rank_bit_16 = pl.cast(pl.cast(two_rank_probe_16 > two_cutoff, pl.INT32), pl.INDEX)
       two_above = two_above + 16 * two_rank_bit_16
       two_rank_probe_8 = pl.tile.read(fin, [0, 2 * (two_above + 8 - 1)])
       two_rank_bit_8 = pl.cast(pl.cast(two_rank_probe_8 > two_cutoff, pl.INT32), pl.INDEX)
       two_above = two_above + 8 * two_rank_bit_8
       two_rank_probe_4 = pl.tile.read(fin, [0, 2 * (two_above + 4 - 1)])
       two_rank_bit_4 = pl.cast(pl.cast(two_rank_probe_4 > two_cutoff, pl.INT32), pl.INDEX)
       two_above = two_above + 4 * two_rank_bit_4
       two_rank_probe_2 = pl.tile.read(fin, [0, 2 * (two_above + 2 - 1)])
       two_rank_bit_2 = pl.cast(pl.cast(two_rank_probe_2 > two_cutoff, pl.INT32), pl.INDEX)
       two_above = two_above + 2 * two_rank_bit_2
       two_rank_probe_1 = pl.tile.read(fin, [0, 2 * (two_above + 1 - 1)])
       two_rank_bit_1 = pl.cast(pl.cast(two_rank_probe_1 > two_cutoff, pl.INT32), pl.INDEX)
       two_above = two_above + 1 * two_rank_bit_1
      pl.tile.write(publication_record, [0, t - cohort_base], two_cutoff)
      pl.tile.write(publication_record, [0, 4 + t - cohort_base], pl.cast(pl.cast(K - two_above, pl.INT32), pl.FP32))
      two_replay_safe = discarded < two_cutoff
      two_single = two_witness
      if discarded > two_cutoff:
       two_single = False
      if two_single:
       two_replay_safe = True
      else:
       if discarded == two_cutoff:
        if K - two_above <= 8:
         two_cert_result = pl.cast(-1, pl.INT32)
         two_tbits = pl.reinterpret_view(fin, pl.INT32)
         for two_tidx in pl.range(two_above, K):
          two_tpos = pl.tile.read(two_tbits, [0, 2 * two_tidx + 1])
          if pl.tile.read(fin, [0, 2 * two_tidx]) == two_cutoff:
           two_cert_result = pl.max(two_cert_result, two_tpos)
        else:
         two_cert_values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
         two_cert_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         two_cert_candidates = pl.tile.select(pl.tile.cmps(two_cert_values, two_cutoff, cmp_type=0), two_cert_positions, -1)
         two_cert_wide = pl.col_max(pl.reshape(two_cert_candidates, [8, 64]))
         two_cert_eight = pl.col_max(pl.reshape(two_cert_wide, [8, 8]))
         two_cert_result = pl.tile.read(two_cert_eight, [0, 0])
         two_cert_lane_1 = pl.tile.read(two_cert_eight, [0, 1])
         if two_cert_lane_1 > two_cert_result:
          two_cert_result = two_cert_lane_1
         two_cert_lane_2 = pl.tile.read(two_cert_eight, [0, 2])
         if two_cert_lane_2 > two_cert_result:
          two_cert_result = two_cert_lane_2
         two_cert_lane_3 = pl.tile.read(two_cert_eight, [0, 3])
         if two_cert_lane_3 > two_cert_result:
          two_cert_result = two_cert_lane_3
         two_cert_lane_4 = pl.tile.read(two_cert_eight, [0, 4])
         if two_cert_lane_4 > two_cert_result:
          two_cert_result = two_cert_lane_4
         two_cert_lane_5 = pl.tile.read(two_cert_eight, [0, 5])
         if two_cert_lane_5 > two_cert_result:
          two_cert_result = two_cert_lane_5
         two_cert_lane_6 = pl.tile.read(two_cert_eight, [0, 6])
         if two_cert_lane_6 > two_cert_result:
          two_cert_result = two_cert_lane_6
         two_cert_lane_7 = pl.tile.read(two_cert_eight, [0, 7])
         if two_cert_lane_7 > two_cert_result:
          two_cert_result = two_cert_lane_7
        if frontier_position > two_cert_result:
         two_replay_safe = True
      if two_replay_safe:
       if two_cutoff > FIN_FLOOR:
        two_root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        if two_single:
         two_endpoint = pl.tile.read(two_root_positions, [0, K - 1])
         if discarded == two_cutoff:
          two_endpoint = pl.min(two_endpoint, frontier_position)
         pl.tile.write(two_root_positions, [0, K - 1], two_endpoint)
        two_selected = pl.cast(K, pl.INDEX)
        if two_selected > 0:
         two_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(two_root_positions, pl.FP32))
         two_root_keys = two_root_priority
         two_root_order = pl.tile.sort32(two_root_keys, pl.reinterpret_view(two_root_positions, pl.UINT32))
         two_root_order = pl.tile.mrgsort(two_root_order, block_len=64)
         two_root_order = pl.tile.mrgsort(two_root_order, block_len=256)
         two_root_logical = pl.tile.gather_mask(two_root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         two_root_valid = pl.tile.set_validshape(two_root_logical, 1, two_selected)
         two_root_safe = pl.tile.fillpad(two_root_valid, pad_value=pl.PadValue.zero)
         two_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for two_map_start in pl.range(0, two_selected, MAP):
          two_map_count = pl.min(MAP, two_selected - two_map_start)
          two_map_bits = pl.tile.slice(pl.reinterpret_view(two_root_safe, pl.FP32), [1, MAP], [0, two_map_start])
          two_map_logical = pl.reinterpret_view(two_map_bits, pl.INT32)
          two_root_page_idx = pl.tile.shrs(two_map_logical, 7)
          two_root_page_ids = pl.tile.gather(materialize_displacements, two_root_page_idx, two_root_tmp)
          two_root_physical = pl.add(two_map_logical, two_root_page_ids)
          pl.store(pl.tile.set_validshape(two_root_physical, 1, two_map_count), [t, two_map_start], indices)
       else:
        two_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
        two_root = fin
        two_root_values = pl.tile.gather_mask(two_root, mask_pattern=pl.tile.MaskPattern.P0101)
        two_root_positions = pl.tile.gather_mask(two_root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        two_ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
        two_root_strict = pl.tile.select(pl.tile.cmps(two_root_values, two_cutoff, cmp_type=4), two_ones_root, 0.0)
        two_root_equal = pl.tile.select(pl.tile.cmps(two_root_values, two_cutoff, cmp_type=0), two_ones_root, 0.0)
        two_root_eligible = pl.maximum(two_root_strict, two_root_equal)
        two_root_visible = pl.tile.cmps(pl.cast(two_root_positions, pl.FP32), pl.cast(pl.cast(two_visible, pl.INT32), pl.FP32), cmp_type=2)
        two_root_keep = pl.tile.select(two_root_visible, two_root_eligible, 0.0)
        two_root_sums64 = pl.col_sum(pl.reshape(two_root_keep, [8, 64]))
        two_root_sums8 = pl.col_sum(pl.reshape(two_root_sums64, [8, 8]))
        two_selected = pl.cast(0, pl.INDEX)
        for two_root_lane in pl.unroll(8):
         two_root_part = pl.cast(pl.tile.read(two_root_sums8, [0, two_root_lane]), pl.INT32)
         two_selected = two_selected + pl.cast(two_root_part, pl.INDEX)
        pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
        if two_selected > 0:
         two_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(two_root_positions, pl.FP32))
         two_root_keys = pl.tile.select(pl.tile.cmps(two_root_keep, 0.0, cmp_type=4), two_root_priority, -1.0)
         two_root_order = pl.tile.sort32(two_root_keys, pl.reinterpret_view(two_root_positions, pl.UINT32))
         two_root_order = pl.tile.mrgsort(two_root_order, block_len=64)
         two_root_order = pl.tile.mrgsort(two_root_order, block_len=256)
         two_root_sorted_keys = pl.tile.gather_mask(two_root_order, mask_pattern=pl.tile.MaskPattern.P0101)
         two_root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), two_root_sorted_keys), pl.INT32)
         two_root_valid = pl.tile.set_validshape(two_root_logical, 1, two_selected)
         two_root_safe = pl.tile.fillpad(two_root_valid, pad_value=pl.PadValue.zero)
         two_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for two_map_start in pl.range(0, two_selected, MAP):
          two_map_count = pl.min(MAP, two_selected - two_map_start)
          two_map_bits = pl.tile.slice(pl.reinterpret_view(two_root_safe, pl.FP32), [1, MAP], [0, two_map_start])
          two_map_logical = pl.reinterpret_view(two_map_bits, pl.INT32)
          two_root_page_idx = pl.tile.shrs(two_map_logical, 7)
          two_root_page_ids = pl.tile.gather(materialize_displacements, two_root_page_idx, two_root_tmp)
          two_root_physical = pl.add(two_map_logical, two_root_page_ids)
          pl.store(pl.tile.set_validshape(two_root_physical, 1, two_map_count), [t, two_map_start], indices)
      else:
       two_bulk_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
       two_bulk_quota = pl.cast(K - two_above, pl.INT32)
       pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
       two_bulk_raw = pl.load(scores, [t, 0], [1, 4096], valid_shape=[1, N])
       two_bulk_values = pl.reshape(pl.tile.fillpad(pl.cast(two_bulk_raw, pl.FP32), pad_value=pl.PadValue.min), [32, SCAN])
       two_bulk_positions = pl.reshape(pl.tile.arange(0, [1, 4096], dtype=pl.INT32), [32, SCAN])
       two_bulk_valid = pl.tile.cmps(pl.cast(two_bulk_positions, pl.FP32), pl.cast(pl.cast(two_bulk_visible, pl.INT32), pl.FP32), cmp_type=2)
       two_bulk_strict = pl.tile.select(two_bulk_valid, pl.tile.select(pl.tile.cmps(two_bulk_values, two_cutoff, cmp_type=4), pl.tile.full([32, SCAN], dtype=pl.FP32, value=1.0), 0.0), 0.0)
       two_bulk_strict_scratch = pl.tile.create([32, SCAN], dtype=pl.FP32)
       two_bulk_count_offsets = pl.mul(pl.tile.arange(0, [1, 32], dtype=pl.INT32), 8)
       two_bulk_count_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
       two_bulk_equal = pl.tile.select(two_bulk_valid, pl.tile.select(pl.tile.cmps(two_bulk_values, two_cutoff, cmp_type=0), pl.tile.full([32, SCAN], dtype=pl.FP32, value=1.0), 0.0), 0.0)
       two_bulk_packed_flags = pl.add(pl.mul(two_bulk_strict, 8192.0), two_bulk_equal)
       two_bulk_packed_reduced = pl.row_sum(two_bulk_packed_flags, two_bulk_strict_scratch)
       two_bulk_packed_carrier = pl.row_expand_add(pl.tile.full([32, 8], dtype=pl.FP32, value=0.0), two_bulk_packed_reduced)
       two_bulk_packed_counts = pl.tile.gather(pl.reshape(two_bulk_packed_carrier, [1, 256]), two_bulk_count_offsets, two_bulk_count_scratch)
       two_bulk_blocks = pl.tile.arange(0, [1, 32], dtype=pl.INT32)
       two_bulk_prefix_scratch = pl.tile.create([1, 32], dtype=pl.INT32)
       two_bulk_packed_prefix = two_bulk_packed_counts
       two_bulk_shift_index = pl.maximum(pl.sub(two_bulk_blocks, 1), 0)
       two_bulk_shift_valid = pl.tile.cmps(pl.cast(two_bulk_blocks, pl.FP32), 1.0, cmp_type=5)
       two_bulk_packed_shift = pl.tile.gather(two_bulk_packed_prefix, two_bulk_shift_index, two_bulk_prefix_scratch)
       two_bulk_packed_prefix = pl.add(two_bulk_packed_prefix, pl.tile.select(two_bulk_shift_valid, two_bulk_packed_shift, 0.0))
       two_bulk_shift_index = pl.maximum(pl.sub(two_bulk_blocks, 2), 0)
       two_bulk_shift_valid = pl.tile.cmps(pl.cast(two_bulk_blocks, pl.FP32), 2.0, cmp_type=5)
       two_bulk_packed_shift = pl.tile.gather(two_bulk_packed_prefix, two_bulk_shift_index, two_bulk_prefix_scratch)
       two_bulk_packed_prefix = pl.add(two_bulk_packed_prefix, pl.tile.select(two_bulk_shift_valid, two_bulk_packed_shift, 0.0))
       two_bulk_shift_index = pl.maximum(pl.sub(two_bulk_blocks, 4), 0)
       two_bulk_shift_valid = pl.tile.cmps(pl.cast(two_bulk_blocks, pl.FP32), 4.0, cmp_type=5)
       two_bulk_packed_shift = pl.tile.gather(two_bulk_packed_prefix, two_bulk_shift_index, two_bulk_prefix_scratch)
       two_bulk_packed_prefix = pl.add(two_bulk_packed_prefix, pl.tile.select(two_bulk_shift_valid, two_bulk_packed_shift, 0.0))
       two_bulk_shift_index = pl.maximum(pl.sub(two_bulk_blocks, 8), 0)
       two_bulk_shift_valid = pl.tile.cmps(pl.cast(two_bulk_blocks, pl.FP32), 8.0, cmp_type=5)
       two_bulk_packed_shift = pl.tile.gather(two_bulk_packed_prefix, two_bulk_shift_index, two_bulk_prefix_scratch)
       two_bulk_packed_prefix = pl.add(two_bulk_packed_prefix, pl.tile.select(two_bulk_shift_valid, two_bulk_packed_shift, 0.0))
       two_bulk_shift_index = pl.maximum(pl.sub(two_bulk_blocks, 16), 0)
       two_bulk_shift_valid = pl.tile.cmps(pl.cast(two_bulk_blocks, pl.FP32), 16.0, cmp_type=5)
       two_bulk_packed_shift = pl.tile.gather(two_bulk_packed_prefix, two_bulk_shift_index, two_bulk_prefix_scratch)
       two_bulk_packed_prefix = pl.add(two_bulk_packed_prefix, pl.tile.select(two_bulk_shift_valid, two_bulk_packed_shift, 0.0))
       two_bulk_strict_counts = pl.cast(pl.cast(pl.mul(two_bulk_packed_counts, 0.0001220703125), pl.INT32, mode='floor'), pl.FP32)
       two_bulk_equal_counts = pl.sub(two_bulk_packed_counts, pl.mul(two_bulk_strict_counts, 8192.0))
       two_bulk_strict_prefix = pl.cast(pl.cast(pl.mul(two_bulk_packed_prefix, 0.0001220703125), pl.INT32, mode='floor'), pl.FP32)
       two_bulk_equal_prefix = pl.sub(two_bulk_packed_prefix, pl.mul(two_bulk_strict_prefix, 8192.0))
       two_bulk_strict_before = pl.sub(two_bulk_strict_prefix, two_bulk_strict_counts)
       two_bulk_equal_before = pl.sub(two_bulk_equal_prefix, two_bulk_equal_counts)
       two_bulk_tie_take = pl.minimum(two_bulk_equal_counts, pl.maximum(pl.sub(pl.add(pl.tile.full([1, 32], dtype=pl.FP32, value=0.0), pl.cast(two_bulk_quota, pl.FP32)), two_bulk_equal_before), 0.0))
       two_bulk_output_begin = pl.add(two_bulk_strict_before, pl.minimum(two_bulk_equal_before, pl.cast(two_bulk_quota, pl.FP32)))
       two_bulk_output_count = pl.add(two_bulk_strict_counts, two_bulk_tie_take)
       two_bulk_lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
       two_bulk_lanes_f = pl.cast(two_bulk_lanes, pl.FP32)
       two_bulk_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
       two_bulk_priority = pl.cast(pl.sub(two_bulk_origin, two_bulk_lanes), pl.FP32)
       two_bulk_map_scratch = pl.tile.create([1, SCAN], dtype=pl.INT32)
       for two_bulk_block in pl.range(0, (two_bulk_visible + SCAN - 1) // SCAN):
         two_bulk_selected = pl.cast(pl.cast(pl.tile.read(two_bulk_output_count, [0, two_bulk_block]), pl.INT32), pl.INDEX)
         if two_bulk_selected > 0:
           two_bulk_start = two_bulk_block * SCAN
           two_bulk_count = pl.cast(pl.cast(pl.tile.read(two_bulk_output_begin, [0, two_bulk_block]), pl.INT32), pl.INDEX)
           two_bulk_take = pl.cast(pl.cast(pl.tile.read(two_bulk_tie_take, [0, two_bulk_block]), pl.INT32), pl.INDEX)
           two_bulk_available = pl.cast(pl.cast(pl.tile.read(two_bulk_equal_counts, [0, two_bulk_block]), pl.INT32), pl.INDEX)
           two_bulk_block_flags = pl.tile.slice(pl.reshape(two_bulk_packed_flags, [1, 4096]), [1, SCAN], [0, two_bulk_start])
           two_bulk_block_ones = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
           two_bulk_block_strict = pl.tile.select(pl.tile.cmps(two_bulk_block_flags, 8192.0, cmp_type=0), two_bulk_block_ones, 0.0)
           two_bulk_block_equal = pl.tile.select(pl.tile.cmps(two_bulk_block_flags, 1.0, cmp_type=0), two_bulk_block_ones, 0.0)
           two_bulk_ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
           if two_bulk_take > 0:
             if two_bulk_take == two_bulk_available:
               two_bulk_ties = two_bulk_block_equal
             else:
               two_bulk_tie_keys = pl.mul(two_bulk_block_equal, two_bulk_priority)
               two_bulk_tie_pairs = pl.tile.sort32(two_bulk_tie_keys, pl.reinterpret_view(two_bulk_lanes, pl.UINT32))
               two_bulk_tie_pairs = pl.tile.mrgsort(two_bulk_tie_pairs, block_len=64)
               two_bulk_equal_keys = pl.tile.gather_mask(two_bulk_tie_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
               two_bulk_last_key = pl.tile.read(two_bulk_equal_keys, [0, two_bulk_take - 1])
               two_bulk_boundary = 128.0 - two_bulk_last_key
               two_bulk_ties = pl.tile.select(pl.tile.cmps(two_bulk_lanes_f, two_bulk_boundary, cmp_type=3), two_bulk_block_equal, 0.0)
           two_bulk_keep = pl.maximum(two_bulk_block_strict, two_bulk_ties)
           two_bulk_order_keys = pl.mul(two_bulk_keep, two_bulk_priority)
           two_bulk_order_pairs = pl.tile.sort32(two_bulk_order_keys, pl.reinterpret_view(two_bulk_lanes, pl.UINT32))
           two_bulk_order_pairs = pl.tile.mrgsort(two_bulk_order_pairs, block_len=64)
           two_bulk_order_values = pl.tile.gather_mask(two_bulk_order_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
           two_bulk_offsets = pl.sub(two_bulk_origin, pl.cast(two_bulk_order_values, pl.INT32))
           two_bulk_compact_valid = pl.tile.set_validshape(two_bulk_offsets, 1, two_bulk_selected)
           two_bulk_compact_safe = pl.tile.fillpad(two_bulk_compact_valid, pad_value=pl.PadValue.zero)
           two_bulk_logical = pl.add(two_bulk_compact_safe, pl.cast(two_bulk_start, pl.INT32))
           two_bulk_page_index = pl.tile.shrs(two_bulk_logical, 7)
           two_bulk_page_ids = pl.tile.gather(materialize_displacements, two_bulk_page_index, two_bulk_map_scratch)
           two_bulk_physical = pl.add(two_bulk_logical, two_bulk_page_ids)
           pl.store(pl.tile.set_validshape(two_bulk_physical, 1, two_bulk_selected), [t, two_bulk_count], indices)
    pl.store(publication_record, [merge_cohort_job, 0], publication_bank)
   cutoff_output = pl.reshape(cutoffs, [1, M])
   quota_output = pl.reshape(quotas, [1, M])
   for pack_task in pl.spmd(1, name_hint='pack_auxiliary'):
    for output_row in pl.range(0, M, 64):
     record_lines = pl.load(publication_bank, [output_row // 4, 0], [16, 128])
     record_flat = pl.reshape(record_lines, [1, 2048])
     pack_lanes = pl.tile.arange(0, [1, 64], dtype=pl.INT32)
     pack_cohorts = pl.tile.shrs(pack_lanes, 2)
     pack_local = pl.sub(pack_lanes, pl.mul(pack_cohorts, 4))
     cutoff_offsets = pl.add(pl.mul(pack_cohorts, 128), pack_local)
     quota_offsets = pl.add(cutoff_offsets, 4)
     cutoff_scratch = pl.tile.create([1, 2048], dtype=pl.INT32)
     quota_scratch = pl.tile.create([1, 2048], dtype=pl.INT32)
     cutoff_packed = pl.tile.gather(record_flat, cutoff_offsets, cutoff_scratch)
     quota_encoded = pl.tile.gather(record_flat, quota_offsets, quota_scratch)
     quota_packed = pl.cast(quota_encoded, pl.INT32)
     pl.store(cutoff_packed, [0, output_row], cutoff_output)
     pl.store(quota_packed, [0, output_row], quota_output)
   return indices
 elif NL == 3:

  @pl.jit
  def dsa41_topk_guarded_root(scores: pl.Tensor[[M, N], pl.BF16], lengths: pl.Tensor[[M], pl.INT32], block_table: pl.Tensor[[1, P], pl.INT32], wpairs: pl.InOut[pl.Tensor[[M, WORKSPACE_WIDTH], pl.FP32]], cutoffs: pl.InOut[pl.Tensor[[M, 1], pl.FP32]], quotas: pl.InOut[pl.Tensor[[M, 1], pl.INT32]], indices: pl.Out[pl.Tensor[[M, K], pl.INT32]]):
   aux_discarded = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_cutoffs = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_quotas = pl.create_tensor([M, 16], dtype=pl.INT32)
   aux_front = pl.create_tensor([M, 16], dtype=pl.INT32)
   leaf_discarded_bank = pl.create_tensor([M, NL * 16], dtype=pl.FP32)
   leaf_front_bank = pl.create_tensor([M, NL * 16], dtype=pl.INT32)
   for leaf_job in pl.spmd(LEAF_COHORTS * NL, name_hint='canonical_leaf_plane'):
    leaf = leaf_job // LEAF_COHORTS
    leaf_cohort_base = leaf_job % LEAF_COHORTS * LEAF_ROWS
    for t in pl.range(leaf_cohort_base, pl.min(leaf_cohort_base + LEAF_ROWS, M), 1):
     zero_pairs = pl.tile.full([1, PAIR_W], dtype=pl.FP32, value=0.0)
     if leaf == 0:
      for clear_offset in pl.range(NL * PAIR_W, WORKSPACE_WIDTH, PAIR_W):
       pl.store(zero_pairs, [t, clear_offset], wpairs)
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     begin = leaf * LEAF
     if begin < wv:
      valid = pl.min(LEAF, wv - begin)
      if valid <= 512:
       narrow_raw = pl.cast(pl.load(scores, [t, begin], [1, 512], valid_shape=[1, valid]), pl.FP32)
       narrow_padded = pl.tile.fillpad(narrow_raw, pad_value=pl.PadValue.min)
       narrow_values = pl.maximum(narrow_padded, SORT_FLOOR)
       narrow_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       narrow_pairs = pl.tile.sort32(narrow_values, pl.reinterpret_view(narrow_positions, pl.UINT32))
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=64)
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=256)
       pl.store(narrow_pairs, [t, leaf * PAIR_W], wpairs)
      elif valid <= 1024:
       left_valid = pl.min(valid, 512)
       right_valid = pl.max(valid - 512, 0)
       left_raw = pl.cast(pl.load(scores, [t, begin], [1, 512], valid_shape=[1, left_valid]), pl.FP32)
       right_raw = pl.cast(pl.load(scores, [t, begin + 512], [1, 512], valid_shape=[1, right_valid]), pl.FP32)
       left_values = pl.maximum(pl.tile.fillpad(left_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       right_values = pl.maximum(pl.tile.fillpad(right_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       left_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       right_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin + 512, pl.INT32))
       left_pairs = pl.tile.sort32(left_values, pl.reinterpret_view(left_positions, pl.UINT32))
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=64)
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=256)
       right_pairs = pl.tile.sort32(right_values, pl.reinterpret_view(right_positions, pl.UINT32))
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=64)
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=256)
       native_scratch = pl.tile.create([1, 2048], dtype=pl.FP32)
       native_pairs = pl.tile.mrgsort(left_pairs, right_pairs, tmp=native_scratch)
       native_discarded = pl.tile.read(native_pairs, [0, PAIR_W])
       if native_discarded > discarded:
        discarded = native_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if native_discarded == discarded:
        native_front_boundary = pl.tile.read(native_pairs, [0, PAIR_W - 2])
        native_front_bound = native_front_boundary
        if native_front_bound < FIN_FLOOR:
         native_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if native_discarded >= native_front_bound:
         native_arity_third = pl.tile.read(native_pairs, [0, PAIR_W + 4])
         if native_arity_third < native_discarded:
          native_arity_second = pl.tile.read(native_pairs, [0, PAIR_W + 2])
          native_arity_bits = pl.reinterpret_view(native_pairs, pl.INT32)
          native_front_result = pl.tile.read(native_arity_bits, [0, PAIR_W + 1])
          if native_arity_second == native_discarded:
           native_arity_second_position = pl.tile.read(native_arity_bits, [0, PAIR_W + 3])
           native_front_result = pl.min(native_front_result, native_arity_second_position)
         else:
          native_front_tail = pl.tile.slice(native_pairs, [1, 1024], [0, PAIR_W])
          native_front_values = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
          native_front_positions = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          native_front_candidates = pl.tile.select(pl.tile.cmps(native_front_values, native_discarded, cmp_type=0), native_front_positions, 2147483647)
          native_front_wide = pl.col_min(pl.reshape(native_front_candidates, [8, 64]))
          native_front_eight = pl.col_min(pl.reshape(native_front_wide, [8, 8]))
          native_front_result = pl.tile.read(native_front_eight, [0, 0])
          native_front_lane_1 = pl.tile.read(native_front_eight, [0, 1])
          if native_front_lane_1 < native_front_result:
           native_front_result = native_front_lane_1
          native_front_lane_2 = pl.tile.read(native_front_eight, [0, 2])
          if native_front_lane_2 < native_front_result:
           native_front_result = native_front_lane_2
          native_front_lane_3 = pl.tile.read(native_front_eight, [0, 3])
          if native_front_lane_3 < native_front_result:
           native_front_result = native_front_lane_3
          native_front_lane_4 = pl.tile.read(native_front_eight, [0, 4])
          if native_front_lane_4 < native_front_result:
           native_front_result = native_front_lane_4
          native_front_lane_5 = pl.tile.read(native_front_eight, [0, 5])
          if native_front_lane_5 < native_front_result:
           native_front_result = native_front_lane_5
          native_front_lane_6 = pl.tile.read(native_front_eight, [0, 6])
          if native_front_lane_6 < native_front_result:
           native_front_result = native_front_lane_6
          native_front_lane_7 = pl.tile.read(native_front_eight, [0, 7])
          if native_front_lane_7 < native_front_result:
           native_front_result = native_front_lane_7
         if native_front_result < frontier_position:
          frontier_position = native_front_result
       pl.store(pl.tile.slice(native_pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
      else:
       raw = pl.cast(pl.load(scores, [t, begin], [1, LEAF], valid_shape=[1, valid]), pl.FP32)
       padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
       flr = pl.tile.full([1, LEAF], dtype=pl.FP32, value=SORT_FLOOR)
       vals = pl.maximum(padded, flr)
       idx_i32 = pl.add(pl.tile.arange(0, [1, LEAF], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       pairs = pl.tile.sort32(vals, pl.reinterpret_view(idx_i32, pl.UINT32))
       pairs = pl.tile.mrgsort(pairs, block_len=64)
       pairs = pl.tile.mrgsort(pairs, block_len=256)
       pairs = pl.tile.mrgsort(pairs, block_len=1024)
       leaf_discarded = pl.tile.read(pairs, [0, PAIR_W])
       if leaf_discarded > discarded:
        discarded = leaf_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if leaf_discarded == discarded:
        leaf_front_boundary = pl.tile.read(pairs, [0, PAIR_W - 2])
        leaf_front_bound = leaf_front_boundary
        if leaf_front_bound < FIN_FLOOR:
         leaf_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if leaf_discarded >= leaf_front_bound:
         leaf_arity_third = pl.tile.read(pairs, [0, PAIR_W + 4])
         if leaf_arity_third < leaf_discarded:
          leaf_arity_second = pl.tile.read(pairs, [0, PAIR_W + 2])
          leaf_arity_bits = pl.reinterpret_view(pairs, pl.INT32)
          leaf_front_result = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 1])
          if leaf_arity_second == leaf_discarded:
           leaf_arity_second_position = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 3])
           leaf_front_result = pl.min(leaf_front_result, leaf_arity_second_position)
         else:
          leaf_front_tail = pl.tile.slice(pairs, [1, 3072], [0, PAIR_W])
          leaf_front_values = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
          leaf_front_positions = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          leaf_front_candidates = pl.tile.select(pl.tile.cmps(leaf_front_values, leaf_discarded, cmp_type=0), leaf_front_positions, 2147483647)
          leaf_front_wide = pl.col_min(pl.reshape(leaf_front_candidates, [24, 64]))
          leaf_front_eight = pl.col_min(pl.reshape(leaf_front_wide, [8, 8]))
          leaf_front_result = pl.tile.read(leaf_front_eight, [0, 0])
          leaf_front_lane_1 = pl.tile.read(leaf_front_eight, [0, 1])
          if leaf_front_lane_1 < leaf_front_result:
           leaf_front_result = leaf_front_lane_1
          leaf_front_lane_2 = pl.tile.read(leaf_front_eight, [0, 2])
          if leaf_front_lane_2 < leaf_front_result:
           leaf_front_result = leaf_front_lane_2
          leaf_front_lane_3 = pl.tile.read(leaf_front_eight, [0, 3])
          if leaf_front_lane_3 < leaf_front_result:
           leaf_front_result = leaf_front_lane_3
          leaf_front_lane_4 = pl.tile.read(leaf_front_eight, [0, 4])
          if leaf_front_lane_4 < leaf_front_result:
           leaf_front_result = leaf_front_lane_4
          leaf_front_lane_5 = pl.tile.read(leaf_front_eight, [0, 5])
          if leaf_front_lane_5 < leaf_front_result:
           leaf_front_result = leaf_front_lane_5
          leaf_front_lane_6 = pl.tile.read(leaf_front_eight, [0, 6])
          if leaf_front_lane_6 < leaf_front_result:
           leaf_front_result = leaf_front_lane_6
          leaf_front_lane_7 = pl.tile.read(leaf_front_eight, [0, 7])
          if leaf_front_lane_7 < leaf_front_result:
           leaf_front_result = leaf_front_lane_7
         if leaf_front_result < frontier_position:
          frontier_position = leaf_front_result
       pl.store(pl.tile.slice(pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
     else:
      pl.store(zero_pairs, [t, leaf * PAIR_W], wpairs)
     leaf_score_line = pl.tile.full([1, 16], dtype=pl.FP32, value=SORT_FLOOR)
     leaf_position_line = pl.tile.full([1, 16], dtype=pl.INT32, value=2147483647)
     pl.tile.write(leaf_score_line, [0, 0], discarded)
     pl.tile.write(leaf_position_line, [0, 0], frontier_position)
     pl.store(leaf_score_line, [t, leaf * 16], leaf_discarded_bank)
     pl.store(leaf_position_line, [t, leaf * 16], leaf_front_bank)
   for merge_cohort_job in pl.spmd((M + 4 - 1) // 4, name_hint='bounded_root_handoff'):
    cohort_base = merge_cohort_job * 4
    if P == 1:
     materialize_single_page = pl.read(block_table, [0, 0])
     materialize_pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), materialize_single_page), pad_value=pl.PadValue.zero)
    else:
     materialize_pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
    materialize_page_lanes = pl.tile.arange(0, [1, P8], dtype=pl.INT32)
    materialize_displacements = pl.mul(pl.sub(materialize_pages, materialize_page_lanes), PAGE)
    for t in pl.range(cohort_base, pl.min(cohort_base + 4, M), 1):
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     lc = (wv + LEAF - 1) // LEAF
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     for proof_leaf in pl.range(NL):
      proof_score = pl.read(leaf_discarded_bank, [t, proof_leaf * 16])
      proof_position = pl.read(leaf_front_bank, [t, proof_leaf * 16])
      if proof_score > discarded:
       discarded = proof_score
       frontier_position = pl.cast(2147483647, pl.INT32)
      if proof_score == discarded:
       if proof_position < frontier_position:
        frontier_position = proof_position
     if lc == 1:
      fin = pl.load(wpairs, [t, 0], [1, PAIR_W])
      one_values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
      one_finite = pl.maximum(one_values, FIN_FLOOR)
      one_cutoff = pl.tile.read(one_finite, [0, K - 1])
      one_ones_c = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
      one_flags = pl.tile.select(pl.tile.cmps(one_values, one_cutoff, cmp_type=4), one_ones_c, 0.0)
      one_sums64 = pl.col_sum(pl.reshape(one_flags, [8, 64]))
      one_sums8 = pl.col_sum(pl.reshape(one_sums64, [8, 8]))
      one_above = pl.cast(0, pl.INDEX)
      for one_lane in pl.unroll(8):
       one_part = pl.cast(pl.tile.read(one_sums8, [0, one_lane]), pl.INT32)
       one_above = one_above + pl.cast(one_part, pl.INDEX)
      one_cutoff_stage_line = pl.tile.full([1, 16], dtype=pl.FP32, value=0)
      pl.tile.write(one_cutoff_stage_line, [0, 0], one_cutoff)
      pl.store(one_cutoff_stage_line, [t, 0], aux_cutoffs)
      one_quota_stage_line = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
      pl.tile.write(one_quota_stage_line, [0, 0], pl.cast(K - one_above, pl.INT32))
      pl.store(one_quota_stage_line, [t, 0], aux_quotas)
      one_replay_safe = discarded < one_cutoff
      if discarded == one_cutoff:
       one_cert_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       one_cert_candidates = pl.tile.select(pl.tile.cmps(one_values, one_cutoff, cmp_type=0), one_cert_positions, -1)
       one_cert_wide = pl.col_max(pl.reshape(one_cert_candidates, [8, 64]))
       one_cert_eight = pl.col_max(pl.reshape(one_cert_wide, [8, 8]))
       one_cert_result = pl.tile.read(one_cert_eight, [0, 0])
       one_cert_lane_1 = pl.tile.read(one_cert_eight, [0, 1])
       if one_cert_lane_1 > one_cert_result:
        one_cert_result = one_cert_lane_1
       one_cert_lane_2 = pl.tile.read(one_cert_eight, [0, 2])
       if one_cert_lane_2 > one_cert_result:
        one_cert_result = one_cert_lane_2
       one_cert_lane_3 = pl.tile.read(one_cert_eight, [0, 3])
       if one_cert_lane_3 > one_cert_result:
        one_cert_result = one_cert_lane_3
       one_cert_lane_4 = pl.tile.read(one_cert_eight, [0, 4])
       if one_cert_lane_4 > one_cert_result:
        one_cert_result = one_cert_lane_4
       one_cert_lane_5 = pl.tile.read(one_cert_eight, [0, 5])
       if one_cert_lane_5 > one_cert_result:
        one_cert_result = one_cert_lane_5
       one_cert_lane_6 = pl.tile.read(one_cert_eight, [0, 6])
       if one_cert_lane_6 > one_cert_result:
        one_cert_result = one_cert_lane_6
       one_cert_lane_7 = pl.tile.read(one_cert_eight, [0, 7])
       if one_cert_lane_7 > one_cert_result:
        one_cert_result = one_cert_lane_7
       if frontier_position > one_cert_result:
        one_replay_safe = True
      if one_replay_safe:
       if one_cutoff > FIN_FLOOR:
        one_root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        one_selected = pl.cast(K, pl.INDEX)
        if one_selected > 0:
         one_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(one_root_positions, pl.FP32))
         one_root_keys = one_root_priority
         one_root_order = pl.tile.sort32(one_root_keys, pl.reinterpret_view(one_root_positions, pl.UINT32))
         one_root_order = pl.tile.mrgsort(one_root_order, block_len=64)
         one_root_order = pl.tile.mrgsort(one_root_order, block_len=256)
         one_root_logical = pl.tile.gather_mask(one_root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         one_root_valid = pl.tile.set_validshape(one_root_logical, 1, one_selected)
         one_root_safe = pl.tile.fillpad(one_root_valid, pad_value=pl.PadValue.zero)
         one_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for one_map_start in pl.range(0, one_selected, MAP):
          one_map_count = pl.min(MAP, one_selected - one_map_start)
          one_map_bits = pl.tile.slice(pl.reinterpret_view(one_root_safe, pl.FP32), [1, MAP], [0, one_map_start])
          one_map_logical = pl.reinterpret_view(one_map_bits, pl.INT32)
          one_root_page_idx = pl.tile.shrs(one_map_logical, 7)
          one_root_page_ids = pl.tile.gather(materialize_displacements, one_root_page_idx, one_root_tmp)
          one_root_physical = pl.add(one_map_logical, one_root_page_ids)
          pl.store(pl.tile.set_validshape(one_root_physical, 1, one_map_count), [t, one_map_start], indices)
       else:
        one_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
        one_root = fin
        one_root_values = pl.tile.gather_mask(one_root, mask_pattern=pl.tile.MaskPattern.P0101)
        one_root_positions = pl.tile.gather_mask(one_root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        one_ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
        one_root_strict = pl.tile.select(pl.tile.cmps(one_root_values, one_cutoff, cmp_type=4), one_ones_root, 0.0)
        one_root_equal = pl.tile.select(pl.tile.cmps(one_root_values, one_cutoff, cmp_type=0), one_ones_root, 0.0)
        one_root_eligible = pl.maximum(one_root_strict, one_root_equal)
        one_root_visible = pl.tile.cmps(pl.cast(one_root_positions, pl.FP32), pl.cast(pl.cast(one_visible, pl.INT32), pl.FP32), cmp_type=2)
        one_root_keep = pl.tile.select(one_root_visible, one_root_eligible, 0.0)
        one_root_sums64 = pl.col_sum(pl.reshape(one_root_keep, [8, 64]))
        one_root_sums8 = pl.col_sum(pl.reshape(one_root_sums64, [8, 8]))
        one_selected = pl.cast(0, pl.INDEX)
        for one_root_lane in pl.unroll(8):
         one_root_part = pl.cast(pl.tile.read(one_root_sums8, [0, one_root_lane]), pl.INT32)
         one_selected = one_selected + pl.cast(one_root_part, pl.INDEX)
        pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
        if one_selected > 0:
         one_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(one_root_positions, pl.FP32))
         one_root_keys = pl.tile.select(pl.tile.cmps(one_root_keep, 0.0, cmp_type=4), one_root_priority, -1.0)
         one_root_order = pl.tile.sort32(one_root_keys, pl.reinterpret_view(one_root_positions, pl.UINT32))
         one_root_order = pl.tile.mrgsort(one_root_order, block_len=64)
         one_root_order = pl.tile.mrgsort(one_root_order, block_len=256)
         one_root_sorted_keys = pl.tile.gather_mask(one_root_order, mask_pattern=pl.tile.MaskPattern.P0101)
         one_root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), one_root_sorted_keys), pl.INT32)
         one_root_valid = pl.tile.set_validshape(one_root_logical, 1, one_selected)
         one_root_safe = pl.tile.fillpad(one_root_valid, pad_value=pl.PadValue.zero)
         one_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for one_map_start in pl.range(0, one_selected, MAP):
          one_map_count = pl.min(MAP, one_selected - one_map_start)
          one_map_bits = pl.tile.slice(pl.reinterpret_view(one_root_safe, pl.FP32), [1, MAP], [0, one_map_start])
          one_map_logical = pl.reinterpret_view(one_map_bits, pl.INT32)
          one_root_page_idx = pl.tile.shrs(one_map_logical, 7)
          one_root_page_ids = pl.tile.gather(materialize_displacements, one_root_page_idx, one_root_tmp)
          one_root_physical = pl.add(one_map_logical, one_root_page_ids)
          pl.store(pl.tile.set_validshape(one_root_physical, 1, one_map_count), [t, one_map_start], indices)
      else:
       one_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
       one_remaining = pl.cast(K - one_above, pl.INDEX)
       pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
       one_ones_m = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
       one_lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
       one_lanes_f = pl.cast(one_lanes, pl.FP32)
       one_lane_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
       one_priority = pl.cast(pl.sub(one_lane_origin, one_lanes), pl.FP32)
       one_gtmp = pl.tile.create([1, SCAN], dtype=pl.INT32)
       one_count = pl.cast(0, pl.INDEX)
       for one_start in pl.range(0, one_visible, SCAN):
        if one_count < K:
         one_valid = pl.min(SCAN, one_visible - one_start)
         one_scan_raw = pl.cast(pl.load(scores, [t, one_start], [1, SCAN], valid_shape=[1, one_valid]), pl.FP32)
         one_scan_vals = pl.tile.fillpad(one_scan_raw, pad_value=pl.PadValue.min)
         one_strict = pl.tile.select(pl.tile.cmps(one_scan_vals, one_cutoff, cmp_type=4), one_ones_m, 0.0)
         one_ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
         if one_remaining > 0:
          one_eq = pl.tile.select(pl.tile.cmps(one_scan_vals, one_cutoff, cmp_type=0), one_ones_m, 0.0)
          one_eq_valid = pl.tile.cmps(one_lanes_f, pl.cast(pl.cast(one_valid, pl.INT32), pl.FP32), cmp_type=2)
          one_eq = pl.tile.select(one_eq_valid, one_eq, 0.0)
          one_eq_sums64 = pl.col_sum(pl.reshape(one_eq, [SCAN // 64, 64]))
          one_eq_sums8 = pl.col_sum(pl.reshape(one_eq_sums64, [8, 8]))
          one_available = pl.cast(0, pl.INDEX)
          for one_eq_lane in pl.unroll(8):
           one_eq_part = pl.cast(pl.tile.read(one_eq_sums8, [0, one_eq_lane]), pl.INT32)
           one_available = one_available + pl.cast(one_eq_part, pl.INDEX)
          one_take = pl.min(one_available, one_remaining)
          if one_take > 0:
           if one_take == one_available:
            one_ties = one_eq
           else:
            one_tie_keys = pl.mul(one_eq, one_priority)
            one_tie_pairs = pl.tile.sort32(one_tie_keys, pl.reinterpret_view(one_lanes, pl.UINT32))
            one_tie_pairs = pl.tile.mrgsort(one_tie_pairs, block_len=64)
            one_tie_merged = one_tie_pairs
            one_equal_keys = pl.tile.gather_mask(one_tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
            one_last_key_f = pl.tile.read(one_equal_keys, [0, one_take - 1])
            one_boundary_f = 128.0 - one_last_key_f
            one_ties = pl.tile.select(pl.tile.cmps(one_lanes_f, one_boundary_f, cmp_type=3), one_eq, 0.0)
           one_remaining = one_remaining - one_take
         one_keep = pl.maximum(one_strict, one_ties)
         one_keep_valid = pl.tile.set_validshape(one_keep, 1, one_valid)
         one_keep_padded = pl.tile.fillpad(one_keep_valid, pad_value=pl.PadValue.zero)
         one_keep_sums64 = pl.col_sum(pl.reshape(one_keep_padded, [SCAN // 64, 64]))
         one_keep_sums8 = pl.col_sum(pl.reshape(one_keep_sums64, [8, 8]))
         one_keep_count = pl.cast(0, pl.INDEX)
         for one_keep_lane in pl.unroll(8):
          one_keep_part = pl.cast(pl.tile.read(one_keep_sums8, [0, one_keep_lane]), pl.INT32)
          one_keep_count = one_keep_count + pl.cast(one_keep_part, pl.INDEX)
         one_selected = pl.min(one_keep_count, K - one_count)
         if one_selected > 0:
          one_order_keys = pl.mul(one_keep_padded, one_priority)
          one_order_pairs = pl.tile.sort32(one_order_keys, pl.reinterpret_view(one_lanes, pl.UINT32))
          one_order_pairs = pl.tile.mrgsort(one_order_pairs, block_len=64)
          one_order_merged = one_order_pairs
          one_order_vals = pl.tile.gather_mask(one_order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
          one_offs = pl.sub(one_lane_origin, pl.cast(one_order_vals, pl.INT32))
          one_compact_valid = pl.tile.set_validshape(one_offs, 1, one_selected)
          one_compact_safe = pl.tile.fillpad(one_compact_valid, pad_value=pl.PadValue.zero)
          one_logical = pl.add(one_compact_safe, pl.cast(one_start, pl.INT32))
          one_page_idx = pl.tile.shrs(one_logical, 7)
          one_page_ids = pl.tile.gather(materialize_displacements, one_page_idx, one_gtmp)
          one_physical = pl.add(one_logical, one_page_ids)
          pl.store(pl.tile.set_validshape(one_physical, 1, one_selected), [t, one_count], indices)
          one_count = one_count + one_selected
     elif lc == 2:
      handoff_scratch = pl.tile.create([1, 2 * PAIR_W], dtype=pl.FP32)
      handoff_left = pl.load(wpairs, [t, 0], [1, PAIR_W])
      handoff_right = pl.load(wpairs, [t, PAIR_W], [1, PAIR_W])
      mboth = pl.tile.mrgsort(handoff_left, handoff_right, tmp=handoff_scratch)
      merge_discarded = pl.tile.read(mboth, [0, PAIR_W])
      if merge_discarded > discarded:
       discarded = merge_discarded
       frontier_position = pl.cast(2147483647, pl.INT32)
      if merge_discarded == discarded:
       merge_front_boundary = pl.tile.read(mboth, [0, PAIR_W - 2])
       merge_front_bound = merge_front_boundary
       if merge_front_bound < FIN_FLOOR:
        merge_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
       if merge_discarded >= merge_front_bound:
        merge_front_tail = pl.tile.slice(mboth, [1, 1024], [0, PAIR_W])
        merge_front_values = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
        merge_front_positions = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        merge_front_candidates = pl.tile.select(pl.tile.cmps(merge_front_values, merge_discarded, cmp_type=0), merge_front_positions, 2147483647)
        merge_front_wide = pl.col_min(pl.reshape(merge_front_candidates, [8, 64]))
        merge_front_eight = pl.col_min(pl.reshape(merge_front_wide, [8, 8]))
        merge_front_result = pl.tile.read(merge_front_eight, [0, 0])
        merge_front_lane_1 = pl.tile.read(merge_front_eight, [0, 1])
        if merge_front_lane_1 < merge_front_result:
         merge_front_result = merge_front_lane_1
        merge_front_lane_2 = pl.tile.read(merge_front_eight, [0, 2])
        if merge_front_lane_2 < merge_front_result:
         merge_front_result = merge_front_lane_2
        merge_front_lane_3 = pl.tile.read(merge_front_eight, [0, 3])
        if merge_front_lane_3 < merge_front_result:
         merge_front_result = merge_front_lane_3
        merge_front_lane_4 = pl.tile.read(merge_front_eight, [0, 4])
        if merge_front_lane_4 < merge_front_result:
         merge_front_result = merge_front_lane_4
        merge_front_lane_5 = pl.tile.read(merge_front_eight, [0, 5])
        if merge_front_lane_5 < merge_front_result:
         merge_front_result = merge_front_lane_5
        merge_front_lane_6 = pl.tile.read(merge_front_eight, [0, 6])
        if merge_front_lane_6 < merge_front_result:
         merge_front_result = merge_front_lane_6
        merge_front_lane_7 = pl.tile.read(merge_front_eight, [0, 7])
        if merge_front_lane_7 < merge_front_result:
         merge_front_result = merge_front_lane_7
        if merge_front_result < frontier_position:
         frontier_position = merge_front_result
      pl.store(pl.tile.slice(mboth, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
      fin = pl.tile.slice(mboth, [1, PAIR_W], [0, 0])
      two_values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
      two_finite = pl.maximum(two_values, FIN_FLOOR)
      two_cutoff = pl.tile.read(two_finite, [0, K - 1])
      two_ones_c = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
      two_flags = pl.tile.select(pl.tile.cmps(two_values, two_cutoff, cmp_type=4), two_ones_c, 0.0)
      two_sums64 = pl.col_sum(pl.reshape(two_flags, [8, 64]))
      two_sums8 = pl.col_sum(pl.reshape(two_sums64, [8, 8]))
      two_above = pl.cast(0, pl.INDEX)
      for two_lane in pl.unroll(8):
       two_part = pl.cast(pl.tile.read(two_sums8, [0, two_lane]), pl.INT32)
       two_above = two_above + pl.cast(two_part, pl.INDEX)
      two_cutoff_stage_line = pl.tile.full([1, 16], dtype=pl.FP32, value=0)
      pl.tile.write(two_cutoff_stage_line, [0, 0], two_cutoff)
      pl.store(two_cutoff_stage_line, [t, 0], aux_cutoffs)
      two_quota_stage_line = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
      pl.tile.write(two_quota_stage_line, [0, 0], pl.cast(K - two_above, pl.INT32))
      pl.store(two_quota_stage_line, [t, 0], aux_quotas)
      two_replay_safe = discarded < two_cutoff
      if discarded == two_cutoff:
       two_cert_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       two_cert_candidates = pl.tile.select(pl.tile.cmps(two_values, two_cutoff, cmp_type=0), two_cert_positions, -1)
       two_cert_wide = pl.col_max(pl.reshape(two_cert_candidates, [8, 64]))
       two_cert_eight = pl.col_max(pl.reshape(two_cert_wide, [8, 8]))
       two_cert_result = pl.tile.read(two_cert_eight, [0, 0])
       two_cert_lane_1 = pl.tile.read(two_cert_eight, [0, 1])
       if two_cert_lane_1 > two_cert_result:
        two_cert_result = two_cert_lane_1
       two_cert_lane_2 = pl.tile.read(two_cert_eight, [0, 2])
       if two_cert_lane_2 > two_cert_result:
        two_cert_result = two_cert_lane_2
       two_cert_lane_3 = pl.tile.read(two_cert_eight, [0, 3])
       if two_cert_lane_3 > two_cert_result:
        two_cert_result = two_cert_lane_3
       two_cert_lane_4 = pl.tile.read(two_cert_eight, [0, 4])
       if two_cert_lane_4 > two_cert_result:
        two_cert_result = two_cert_lane_4
       two_cert_lane_5 = pl.tile.read(two_cert_eight, [0, 5])
       if two_cert_lane_5 > two_cert_result:
        two_cert_result = two_cert_lane_5
       two_cert_lane_6 = pl.tile.read(two_cert_eight, [0, 6])
       if two_cert_lane_6 > two_cert_result:
        two_cert_result = two_cert_lane_6
       two_cert_lane_7 = pl.tile.read(two_cert_eight, [0, 7])
       if two_cert_lane_7 > two_cert_result:
        two_cert_result = two_cert_lane_7
       if frontier_position > two_cert_result:
        two_replay_safe = True
      if two_replay_safe:
       if two_cutoff > FIN_FLOOR:
        two_root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        two_selected = pl.cast(K, pl.INDEX)
        if two_selected > 0:
         two_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(two_root_positions, pl.FP32))
         two_root_keys = two_root_priority
         two_root_order = pl.tile.sort32(two_root_keys, pl.reinterpret_view(two_root_positions, pl.UINT32))
         two_root_order = pl.tile.mrgsort(two_root_order, block_len=64)
         two_root_order = pl.tile.mrgsort(two_root_order, block_len=256)
         two_root_logical = pl.tile.gather_mask(two_root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         two_root_valid = pl.tile.set_validshape(two_root_logical, 1, two_selected)
         two_root_safe = pl.tile.fillpad(two_root_valid, pad_value=pl.PadValue.zero)
         two_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for two_map_start in pl.range(0, two_selected, MAP):
          two_map_count = pl.min(MAP, two_selected - two_map_start)
          two_map_bits = pl.tile.slice(pl.reinterpret_view(two_root_safe, pl.FP32), [1, MAP], [0, two_map_start])
          two_map_logical = pl.reinterpret_view(two_map_bits, pl.INT32)
          two_root_page_idx = pl.tile.shrs(two_map_logical, 7)
          two_root_page_ids = pl.tile.gather(materialize_displacements, two_root_page_idx, two_root_tmp)
          two_root_physical = pl.add(two_map_logical, two_root_page_ids)
          pl.store(pl.tile.set_validshape(two_root_physical, 1, two_map_count), [t, two_map_start], indices)
       else:
        two_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
        two_root = fin
        two_root_values = pl.tile.gather_mask(two_root, mask_pattern=pl.tile.MaskPattern.P0101)
        two_root_positions = pl.tile.gather_mask(two_root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        two_ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
        two_root_strict = pl.tile.select(pl.tile.cmps(two_root_values, two_cutoff, cmp_type=4), two_ones_root, 0.0)
        two_root_equal = pl.tile.select(pl.tile.cmps(two_root_values, two_cutoff, cmp_type=0), two_ones_root, 0.0)
        two_root_eligible = pl.maximum(two_root_strict, two_root_equal)
        two_root_visible = pl.tile.cmps(pl.cast(two_root_positions, pl.FP32), pl.cast(pl.cast(two_visible, pl.INT32), pl.FP32), cmp_type=2)
        two_root_keep = pl.tile.select(two_root_visible, two_root_eligible, 0.0)
        two_root_sums64 = pl.col_sum(pl.reshape(two_root_keep, [8, 64]))
        two_root_sums8 = pl.col_sum(pl.reshape(two_root_sums64, [8, 8]))
        two_selected = pl.cast(0, pl.INDEX)
        for two_root_lane in pl.unroll(8):
         two_root_part = pl.cast(pl.tile.read(two_root_sums8, [0, two_root_lane]), pl.INT32)
         two_selected = two_selected + pl.cast(two_root_part, pl.INDEX)
        pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
        if two_selected > 0:
         two_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(two_root_positions, pl.FP32))
         two_root_keys = pl.tile.select(pl.tile.cmps(two_root_keep, 0.0, cmp_type=4), two_root_priority, -1.0)
         two_root_order = pl.tile.sort32(two_root_keys, pl.reinterpret_view(two_root_positions, pl.UINT32))
         two_root_order = pl.tile.mrgsort(two_root_order, block_len=64)
         two_root_order = pl.tile.mrgsort(two_root_order, block_len=256)
         two_root_sorted_keys = pl.tile.gather_mask(two_root_order, mask_pattern=pl.tile.MaskPattern.P0101)
         two_root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), two_root_sorted_keys), pl.INT32)
         two_root_valid = pl.tile.set_validshape(two_root_logical, 1, two_selected)
         two_root_safe = pl.tile.fillpad(two_root_valid, pad_value=pl.PadValue.zero)
         two_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for two_map_start in pl.range(0, two_selected, MAP):
          two_map_count = pl.min(MAP, two_selected - two_map_start)
          two_map_bits = pl.tile.slice(pl.reinterpret_view(two_root_safe, pl.FP32), [1, MAP], [0, two_map_start])
          two_map_logical = pl.reinterpret_view(two_map_bits, pl.INT32)
          two_root_page_idx = pl.tile.shrs(two_map_logical, 7)
          two_root_page_ids = pl.tile.gather(materialize_displacements, two_root_page_idx, two_root_tmp)
          two_root_physical = pl.add(two_map_logical, two_root_page_ids)
          pl.store(pl.tile.set_validshape(two_root_physical, 1, two_map_count), [t, two_map_start], indices)
      else:
       two_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
       two_remaining = pl.cast(K - two_above, pl.INDEX)
       pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
       two_ones_m = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
       two_lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
       two_lanes_f = pl.cast(two_lanes, pl.FP32)
       two_lane_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
       two_priority = pl.cast(pl.sub(two_lane_origin, two_lanes), pl.FP32)
       two_gtmp = pl.tile.create([1, SCAN], dtype=pl.INT32)
       two_count = pl.cast(0, pl.INDEX)
       for two_start in pl.range(0, two_visible, SCAN):
        if two_count < K:
         two_valid = pl.min(SCAN, two_visible - two_start)
         two_scan_raw = pl.cast(pl.load(scores, [t, two_start], [1, SCAN], valid_shape=[1, two_valid]), pl.FP32)
         two_scan_vals = pl.tile.fillpad(two_scan_raw, pad_value=pl.PadValue.min)
         two_strict = pl.tile.select(pl.tile.cmps(two_scan_vals, two_cutoff, cmp_type=4), two_ones_m, 0.0)
         two_ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
         if two_remaining > 0:
          two_eq = pl.tile.select(pl.tile.cmps(two_scan_vals, two_cutoff, cmp_type=0), two_ones_m, 0.0)
          two_eq_valid = pl.tile.cmps(two_lanes_f, pl.cast(pl.cast(two_valid, pl.INT32), pl.FP32), cmp_type=2)
          two_eq = pl.tile.select(two_eq_valid, two_eq, 0.0)
          two_eq_sums64 = pl.col_sum(pl.reshape(two_eq, [SCAN // 64, 64]))
          two_eq_sums8 = pl.col_sum(pl.reshape(two_eq_sums64, [8, 8]))
          two_available = pl.cast(0, pl.INDEX)
          for two_eq_lane in pl.unroll(8):
           two_eq_part = pl.cast(pl.tile.read(two_eq_sums8, [0, two_eq_lane]), pl.INT32)
           two_available = two_available + pl.cast(two_eq_part, pl.INDEX)
          two_take = pl.min(two_available, two_remaining)
          if two_take > 0:
           if two_take == two_available:
            two_ties = two_eq
           else:
            two_tie_keys = pl.mul(two_eq, two_priority)
            two_tie_pairs = pl.tile.sort32(two_tie_keys, pl.reinterpret_view(two_lanes, pl.UINT32))
            two_tie_pairs = pl.tile.mrgsort(two_tie_pairs, block_len=64)
            two_tie_merged = two_tie_pairs
            two_equal_keys = pl.tile.gather_mask(two_tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
            two_last_key_f = pl.tile.read(two_equal_keys, [0, two_take - 1])
            two_boundary_f = 128.0 - two_last_key_f
            two_ties = pl.tile.select(pl.tile.cmps(two_lanes_f, two_boundary_f, cmp_type=3), two_eq, 0.0)
           two_remaining = two_remaining - two_take
         two_keep = pl.maximum(two_strict, two_ties)
         two_keep_valid = pl.tile.set_validshape(two_keep, 1, two_valid)
         two_keep_padded = pl.tile.fillpad(two_keep_valid, pad_value=pl.PadValue.zero)
         two_keep_sums64 = pl.col_sum(pl.reshape(two_keep_padded, [SCAN // 64, 64]))
         two_keep_sums8 = pl.col_sum(pl.reshape(two_keep_sums64, [8, 8]))
         two_keep_count = pl.cast(0, pl.INDEX)
         for two_keep_lane in pl.unroll(8):
          two_keep_part = pl.cast(pl.tile.read(two_keep_sums8, [0, two_keep_lane]), pl.INT32)
          two_keep_count = two_keep_count + pl.cast(two_keep_part, pl.INDEX)
         two_selected = pl.min(two_keep_count, K - two_count)
         if two_selected > 0:
          two_order_keys = pl.mul(two_keep_padded, two_priority)
          two_order_pairs = pl.tile.sort32(two_order_keys, pl.reinterpret_view(two_lanes, pl.UINT32))
          two_order_pairs = pl.tile.mrgsort(two_order_pairs, block_len=64)
          two_order_merged = two_order_pairs
          two_order_vals = pl.tile.gather_mask(two_order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
          two_offs = pl.sub(two_lane_origin, pl.cast(two_order_vals, pl.INT32))
          two_compact_valid = pl.tile.set_validshape(two_offs, 1, two_selected)
          two_compact_safe = pl.tile.fillpad(two_compact_valid, pad_value=pl.PadValue.zero)
          two_logical = pl.add(two_compact_safe, pl.cast(two_start, pl.INT32))
          two_page_idx = pl.tile.shrs(two_logical, 7)
          two_page_ids = pl.tile.gather(materialize_displacements, two_page_idx, two_gtmp)
          two_physical = pl.add(two_logical, two_page_ids)
          pl.store(pl.tile.set_validshape(two_physical, 1, two_selected), [t, two_count], indices)
          two_count = two_count + two_selected
     else:
      handoff_scratch = pl.tile.create([1, 2 * PAIR_W], dtype=pl.FP32)
      handoff_left = pl.load(wpairs, [t, 0], [1, PAIR_W])
      handoff_right = pl.load(wpairs, [t, PAIR_W], [1, PAIR_W])
      mboth = pl.tile.mrgsort(handoff_left, handoff_right, tmp=handoff_scratch)
      merge_discarded = pl.tile.read(mboth, [0, PAIR_W])
      if merge_discarded > discarded:
       discarded = merge_discarded
       frontier_position = pl.cast(2147483647, pl.INT32)
      if merge_discarded == discarded:
       merge_front_boundary = pl.tile.read(mboth, [0, PAIR_W - 2])
       merge_front_bound = merge_front_boundary
       if merge_front_bound < FIN_FLOOR:
        merge_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
       if merge_discarded >= merge_front_bound:
        merge_front_tail = pl.tile.slice(mboth, [1, 1024], [0, PAIR_W])
        merge_front_values = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
        merge_front_positions = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        merge_front_candidates = pl.tile.select(pl.tile.cmps(merge_front_values, merge_discarded, cmp_type=0), merge_front_positions, 2147483647)
        merge_front_wide = pl.col_min(pl.reshape(merge_front_candidates, [8, 64]))
        merge_front_eight = pl.col_min(pl.reshape(merge_front_wide, [8, 8]))
        merge_front_result = pl.tile.read(merge_front_eight, [0, 0])
        merge_front_lane_1 = pl.tile.read(merge_front_eight, [0, 1])
        if merge_front_lane_1 < merge_front_result:
         merge_front_result = merge_front_lane_1
        merge_front_lane_2 = pl.tile.read(merge_front_eight, [0, 2])
        if merge_front_lane_2 < merge_front_result:
         merge_front_result = merge_front_lane_2
        merge_front_lane_3 = pl.tile.read(merge_front_eight, [0, 3])
        if merge_front_lane_3 < merge_front_result:
         merge_front_result = merge_front_lane_3
        merge_front_lane_4 = pl.tile.read(merge_front_eight, [0, 4])
        if merge_front_lane_4 < merge_front_result:
         merge_front_result = merge_front_lane_4
        merge_front_lane_5 = pl.tile.read(merge_front_eight, [0, 5])
        if merge_front_lane_5 < merge_front_result:
         merge_front_result = merge_front_lane_5
        merge_front_lane_6 = pl.tile.read(merge_front_eight, [0, 6])
        if merge_front_lane_6 < merge_front_result:
         merge_front_result = merge_front_lane_6
        merge_front_lane_7 = pl.tile.read(merge_front_eight, [0, 7])
        if merge_front_lane_7 < merge_front_result:
         merge_front_result = merge_front_lane_7
        if merge_front_result < frontier_position:
         frontier_position = merge_front_result
      pl.store(pl.tile.slice(mboth, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
      handoff_offsets = pl.tile.arange(0, [1, PAIR_W], dtype=pl.INT32)
      handoff_gather_scratch = pl.tile.create([1, 2 * PAIR_W], dtype=pl.INT32)
      handoff_resident = pl.tile.gather(mboth, handoff_offsets, handoff_gather_scratch)
      handoff_third = pl.load(wpairs, [t, 2 * PAIR_W], [1, PAIR_W])
      mboth = pl.tile.mrgsort(handoff_resident, handoff_third, tmp=handoff_scratch)
      merge_discarded = pl.tile.read(mboth, [0, PAIR_W])
      if merge_discarded > discarded:
       discarded = merge_discarded
       frontier_position = pl.cast(2147483647, pl.INT32)
      if merge_discarded == discarded:
       merge_front_boundary = pl.tile.read(mboth, [0, PAIR_W - 2])
       merge_front_bound = merge_front_boundary
       if merge_front_bound < FIN_FLOOR:
        merge_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
       if merge_discarded >= merge_front_bound:
        merge_front_tail = pl.tile.slice(mboth, [1, 1024], [0, PAIR_W])
        merge_front_values = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
        merge_front_positions = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        merge_front_candidates = pl.tile.select(pl.tile.cmps(merge_front_values, merge_discarded, cmp_type=0), merge_front_positions, 2147483647)
        merge_front_wide = pl.col_min(pl.reshape(merge_front_candidates, [8, 64]))
        merge_front_eight = pl.col_min(pl.reshape(merge_front_wide, [8, 8]))
        merge_front_result = pl.tile.read(merge_front_eight, [0, 0])
        merge_front_lane_1 = pl.tile.read(merge_front_eight, [0, 1])
        if merge_front_lane_1 < merge_front_result:
         merge_front_result = merge_front_lane_1
        merge_front_lane_2 = pl.tile.read(merge_front_eight, [0, 2])
        if merge_front_lane_2 < merge_front_result:
         merge_front_result = merge_front_lane_2
        merge_front_lane_3 = pl.tile.read(merge_front_eight, [0, 3])
        if merge_front_lane_3 < merge_front_result:
         merge_front_result = merge_front_lane_3
        merge_front_lane_4 = pl.tile.read(merge_front_eight, [0, 4])
        if merge_front_lane_4 < merge_front_result:
         merge_front_result = merge_front_lane_4
        merge_front_lane_5 = pl.tile.read(merge_front_eight, [0, 5])
        if merge_front_lane_5 < merge_front_result:
         merge_front_result = merge_front_lane_5
        merge_front_lane_6 = pl.tile.read(merge_front_eight, [0, 6])
        if merge_front_lane_6 < merge_front_result:
         merge_front_result = merge_front_lane_6
        merge_front_lane_7 = pl.tile.read(merge_front_eight, [0, 7])
        if merge_front_lane_7 < merge_front_result:
         merge_front_result = merge_front_lane_7
        if merge_front_result < frontier_position:
         frontier_position = merge_front_result
      pl.store(pl.tile.slice(mboth, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
      fin = pl.tile.slice(mboth, [1, PAIR_W], [0, 0])
      three_values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
      three_finite = pl.maximum(three_values, FIN_FLOOR)
      three_cutoff = pl.tile.read(three_finite, [0, K - 1])
      three_ones_c = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
      three_flags = pl.tile.select(pl.tile.cmps(three_values, three_cutoff, cmp_type=4), three_ones_c, 0.0)
      three_sums64 = pl.col_sum(pl.reshape(three_flags, [8, 64]))
      three_sums8 = pl.col_sum(pl.reshape(three_sums64, [8, 8]))
      three_above = pl.cast(0, pl.INDEX)
      for three_lane in pl.unroll(8):
       three_part = pl.cast(pl.tile.read(three_sums8, [0, three_lane]), pl.INT32)
       three_above = three_above + pl.cast(three_part, pl.INDEX)
      three_cutoff_stage_line = pl.tile.full([1, 16], dtype=pl.FP32, value=0)
      pl.tile.write(three_cutoff_stage_line, [0, 0], three_cutoff)
      pl.store(three_cutoff_stage_line, [t, 0], aux_cutoffs)
      three_quota_stage_line = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
      pl.tile.write(three_quota_stage_line, [0, 0], pl.cast(K - three_above, pl.INT32))
      pl.store(three_quota_stage_line, [t, 0], aux_quotas)
      three_replay_safe = discarded < three_cutoff
      if discarded == three_cutoff:
       three_cert_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       three_cert_candidates = pl.tile.select(pl.tile.cmps(three_values, three_cutoff, cmp_type=0), three_cert_positions, -1)
       three_cert_wide = pl.col_max(pl.reshape(three_cert_candidates, [8, 64]))
       three_cert_eight = pl.col_max(pl.reshape(three_cert_wide, [8, 8]))
       three_cert_result = pl.tile.read(three_cert_eight, [0, 0])
       three_cert_lane_1 = pl.tile.read(three_cert_eight, [0, 1])
       if three_cert_lane_1 > three_cert_result:
        three_cert_result = three_cert_lane_1
       three_cert_lane_2 = pl.tile.read(three_cert_eight, [0, 2])
       if three_cert_lane_2 > three_cert_result:
        three_cert_result = three_cert_lane_2
       three_cert_lane_3 = pl.tile.read(three_cert_eight, [0, 3])
       if three_cert_lane_3 > three_cert_result:
        three_cert_result = three_cert_lane_3
       three_cert_lane_4 = pl.tile.read(three_cert_eight, [0, 4])
       if three_cert_lane_4 > three_cert_result:
        three_cert_result = three_cert_lane_4
       three_cert_lane_5 = pl.tile.read(three_cert_eight, [0, 5])
       if three_cert_lane_5 > three_cert_result:
        three_cert_result = three_cert_lane_5
       three_cert_lane_6 = pl.tile.read(three_cert_eight, [0, 6])
       if three_cert_lane_6 > three_cert_result:
        three_cert_result = three_cert_lane_6
       three_cert_lane_7 = pl.tile.read(three_cert_eight, [0, 7])
       if three_cert_lane_7 > three_cert_result:
        three_cert_result = three_cert_lane_7
       if frontier_position > three_cert_result:
        three_replay_safe = True
      if three_replay_safe:
       if three_cutoff > FIN_FLOOR:
        three_root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        three_selected = pl.cast(K, pl.INDEX)
        if three_selected > 0:
         three_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(three_root_positions, pl.FP32))
         three_root_keys = three_root_priority
         three_root_order = pl.tile.sort32(three_root_keys, pl.reinterpret_view(three_root_positions, pl.UINT32))
         three_root_order = pl.tile.mrgsort(three_root_order, block_len=64)
         three_root_order = pl.tile.mrgsort(three_root_order, block_len=256)
         three_root_logical = pl.tile.gather_mask(three_root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
         three_root_valid = pl.tile.set_validshape(three_root_logical, 1, three_selected)
         three_root_safe = pl.tile.fillpad(three_root_valid, pad_value=pl.PadValue.zero)
         three_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for three_map_start in pl.range(0, three_selected, MAP):
          three_map_count = pl.min(MAP, three_selected - three_map_start)
          three_map_bits = pl.tile.slice(pl.reinterpret_view(three_root_safe, pl.FP32), [1, MAP], [0, three_map_start])
          three_map_logical = pl.reinterpret_view(three_map_bits, pl.INT32)
          three_root_page_idx = pl.tile.shrs(three_map_logical, 7)
          three_root_page_ids = pl.tile.gather(materialize_displacements, three_root_page_idx, three_root_tmp)
          three_root_physical = pl.add(three_map_logical, three_root_page_ids)
          pl.store(pl.tile.set_validshape(three_root_physical, 1, three_map_count), [t, three_map_start], indices)
       else:
        three_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
        three_root = fin
        three_root_values = pl.tile.gather_mask(three_root, mask_pattern=pl.tile.MaskPattern.P0101)
        three_root_positions = pl.tile.gather_mask(three_root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        three_ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
        three_root_strict = pl.tile.select(pl.tile.cmps(three_root_values, three_cutoff, cmp_type=4), three_ones_root, 0.0)
        three_root_equal = pl.tile.select(pl.tile.cmps(three_root_values, three_cutoff, cmp_type=0), three_ones_root, 0.0)
        three_root_eligible = pl.maximum(three_root_strict, three_root_equal)
        three_root_visible = pl.tile.cmps(pl.cast(three_root_positions, pl.FP32), pl.cast(pl.cast(three_visible, pl.INT32), pl.FP32), cmp_type=2)
        three_root_keep = pl.tile.select(three_root_visible, three_root_eligible, 0.0)
        three_root_sums64 = pl.col_sum(pl.reshape(three_root_keep, [8, 64]))
        three_root_sums8 = pl.col_sum(pl.reshape(three_root_sums64, [8, 8]))
        three_selected = pl.cast(0, pl.INDEX)
        for three_root_lane in pl.unroll(8):
         three_root_part = pl.cast(pl.tile.read(three_root_sums8, [0, three_root_lane]), pl.INT32)
         three_selected = three_selected + pl.cast(three_root_part, pl.INDEX)
        pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
        if three_selected > 0:
         three_root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(three_root_positions, pl.FP32))
         three_root_keys = pl.tile.select(pl.tile.cmps(three_root_keep, 0.0, cmp_type=4), three_root_priority, -1.0)
         three_root_order = pl.tile.sort32(three_root_keys, pl.reinterpret_view(three_root_positions, pl.UINT32))
         three_root_order = pl.tile.mrgsort(three_root_order, block_len=64)
         three_root_order = pl.tile.mrgsort(three_root_order, block_len=256)
         three_root_sorted_keys = pl.tile.gather_mask(three_root_order, mask_pattern=pl.tile.MaskPattern.P0101)
         three_root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), three_root_sorted_keys), pl.INT32)
         three_root_valid = pl.tile.set_validshape(three_root_logical, 1, three_selected)
         three_root_safe = pl.tile.fillpad(three_root_valid, pad_value=pl.PadValue.zero)
         three_root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
         for three_map_start in pl.range(0, three_selected, MAP):
          three_map_count = pl.min(MAP, three_selected - three_map_start)
          three_map_bits = pl.tile.slice(pl.reinterpret_view(three_root_safe, pl.FP32), [1, MAP], [0, three_map_start])
          three_map_logical = pl.reinterpret_view(three_map_bits, pl.INT32)
          three_root_page_idx = pl.tile.shrs(three_map_logical, 7)
          three_root_page_ids = pl.tile.gather(materialize_displacements, three_root_page_idx, three_root_tmp)
          three_root_physical = pl.add(three_map_logical, three_root_page_ids)
          pl.store(pl.tile.set_validshape(three_root_physical, 1, three_map_count), [t, three_map_start], indices)
      else:
       three_visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
       three_remaining = pl.cast(K - three_above, pl.INDEX)
       pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
       three_ones_m = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
       three_lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
       three_lanes_f = pl.cast(three_lanes, pl.FP32)
       three_lane_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
       three_priority = pl.cast(pl.sub(three_lane_origin, three_lanes), pl.FP32)
       three_gtmp = pl.tile.create([1, SCAN], dtype=pl.INT32)
       three_count = pl.cast(0, pl.INDEX)
       for three_start in pl.range(0, three_visible, SCAN):
        if three_count < K:
         three_valid = pl.min(SCAN, three_visible - three_start)
         three_scan_raw = pl.cast(pl.load(scores, [t, three_start], [1, SCAN], valid_shape=[1, three_valid]), pl.FP32)
         three_scan_vals = pl.tile.fillpad(three_scan_raw, pad_value=pl.PadValue.min)
         three_strict = pl.tile.select(pl.tile.cmps(three_scan_vals, three_cutoff, cmp_type=4), three_ones_m, 0.0)
         three_ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
         if three_remaining > 0:
          three_eq = pl.tile.select(pl.tile.cmps(three_scan_vals, three_cutoff, cmp_type=0), three_ones_m, 0.0)
          three_eq_valid = pl.tile.cmps(three_lanes_f, pl.cast(pl.cast(three_valid, pl.INT32), pl.FP32), cmp_type=2)
          three_eq = pl.tile.select(three_eq_valid, three_eq, 0.0)
          three_eq_sums64 = pl.col_sum(pl.reshape(three_eq, [SCAN // 64, 64]))
          three_eq_sums8 = pl.col_sum(pl.reshape(three_eq_sums64, [8, 8]))
          three_available = pl.cast(0, pl.INDEX)
          for three_eq_lane in pl.unroll(8):
           three_eq_part = pl.cast(pl.tile.read(three_eq_sums8, [0, three_eq_lane]), pl.INT32)
           three_available = three_available + pl.cast(three_eq_part, pl.INDEX)
          three_take = pl.min(three_available, three_remaining)
          if three_take > 0:
           if three_take == three_available:
            three_ties = three_eq
           else:
            three_tie_keys = pl.mul(three_eq, three_priority)
            three_tie_pairs = pl.tile.sort32(three_tie_keys, pl.reinterpret_view(three_lanes, pl.UINT32))
            three_tie_pairs = pl.tile.mrgsort(three_tie_pairs, block_len=64)
            three_tie_merged = three_tie_pairs
            three_equal_keys = pl.tile.gather_mask(three_tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
            three_last_key_f = pl.tile.read(three_equal_keys, [0, three_take - 1])
            three_boundary_f = 128.0 - three_last_key_f
            three_ties = pl.tile.select(pl.tile.cmps(three_lanes_f, three_boundary_f, cmp_type=3), three_eq, 0.0)
           three_remaining = three_remaining - three_take
         three_keep = pl.maximum(three_strict, three_ties)
         three_keep_valid = pl.tile.set_validshape(three_keep, 1, three_valid)
         three_keep_padded = pl.tile.fillpad(three_keep_valid, pad_value=pl.PadValue.zero)
         three_keep_sums64 = pl.col_sum(pl.reshape(three_keep_padded, [SCAN // 64, 64]))
         three_keep_sums8 = pl.col_sum(pl.reshape(three_keep_sums64, [8, 8]))
         three_keep_count = pl.cast(0, pl.INDEX)
         for three_keep_lane in pl.unroll(8):
          three_keep_part = pl.cast(pl.tile.read(three_keep_sums8, [0, three_keep_lane]), pl.INT32)
          three_keep_count = three_keep_count + pl.cast(three_keep_part, pl.INDEX)
         three_selected = pl.min(three_keep_count, K - three_count)
         if three_selected > 0:
          three_order_keys = pl.mul(three_keep_padded, three_priority)
          three_order_pairs = pl.tile.sort32(three_order_keys, pl.reinterpret_view(three_lanes, pl.UINT32))
          three_order_pairs = pl.tile.mrgsort(three_order_pairs, block_len=64)
          three_order_merged = three_order_pairs
          three_order_vals = pl.tile.gather_mask(three_order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
          three_offs = pl.sub(three_lane_origin, pl.cast(three_order_vals, pl.INT32))
          three_compact_valid = pl.tile.set_validshape(three_offs, 1, three_selected)
          three_compact_safe = pl.tile.fillpad(three_compact_valid, pad_value=pl.PadValue.zero)
          three_logical = pl.add(three_compact_safe, pl.cast(three_start, pl.INT32))
          three_page_idx = pl.tile.shrs(three_logical, 7)
          three_page_ids = pl.tile.gather(materialize_displacements, three_page_idx, three_gtmp)
          three_physical = pl.add(three_logical, three_page_ids)
          pl.store(pl.tile.set_validshape(three_physical, 1, three_selected), [t, three_count], indices)
          three_count = three_count + three_selected
   for pack_task in pl.spmd(1, name_hint='pack_auxiliary'):
    for output_row in pl.range(0, M, 16):
     cutoff_lines = pl.load(aux_cutoffs, [output_row, 0], [16, 16])
     quota_lines = pl.load(aux_quotas, [output_row, 0], [16, 16])
     cutoff_flat = pl.reshape(cutoff_lines, [1, 256])
     quota_flat = pl.reshape(quota_lines, [1, 256])
     pack_offsets = pl.mul(pl.tile.arange(0, [1, 16], dtype=pl.INT32), 16)
     cutoff_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     quota_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     cutoff_packed = pl.tile.gather(cutoff_flat, pack_offsets, cutoff_scratch)
     quota_packed = pl.tile.gather(quota_flat, pack_offsets, quota_scratch)
     pl.store(pl.reshape(cutoff_packed, [16, 1]), [output_row, 0], cutoffs)
     pl.store(pl.reshape(quota_packed, [16, 1]), [output_row, 0], quotas)
   return indices
 else:

  @pl.jit
  def dsa41_topk_guarded_root(scores: pl.Tensor[[M, N], pl.BF16], lengths: pl.Tensor[[M], pl.INT32], block_table: pl.Tensor[[1, P], pl.INT32], wpairs: pl.InOut[pl.Tensor[[M, WORKSPACE_WIDTH], pl.FP32]], cutoffs: pl.InOut[pl.Tensor[[M, 1], pl.FP32]], quotas: pl.InOut[pl.Tensor[[M, 1], pl.INT32]], indices: pl.Out[pl.Tensor[[M, K], pl.INT32]]):
   aux_discarded = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_cutoffs = pl.create_tensor([M, 16], dtype=pl.FP32)
   aux_quotas = pl.create_tensor([M, 16], dtype=pl.INT32)
   aux_front = pl.create_tensor([M, 16], dtype=pl.INT32)
   leaf_discarded_bank = pl.create_tensor([M, NL * 16], dtype=pl.FP32)
   leaf_front_bank = pl.create_tensor([M, NL * 16], dtype=pl.INT32)
   for leaf_job in pl.spmd(LEAF_COHORTS * NL, name_hint='canonical_leaf_plane'):
    leaf = leaf_job // LEAF_COHORTS
    leaf_cohort_base = leaf_job % LEAF_COHORTS * LEAF_ROWS
    for t in pl.range(leaf_cohort_base, pl.min(leaf_cohort_base + LEAF_ROWS, M), 1):
     zero_pairs = pl.tile.full([1, PAIR_W], dtype=pl.FP32, value=0.0)
     if leaf == 0:
      for clear_offset in pl.range(NL * PAIR_W, WORKSPACE_WIDTH, PAIR_W):
       pl.store(zero_pairs, [t, clear_offset], wpairs)
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     begin = leaf * LEAF
     if begin < wv:
      valid = pl.min(LEAF, wv - begin)
      if valid <= 512:
       narrow_raw = pl.cast(pl.load(scores, [t, begin], [1, 512], valid_shape=[1, valid]), pl.FP32)
       narrow_padded = pl.tile.fillpad(narrow_raw, pad_value=pl.PadValue.min)
       narrow_values = pl.maximum(narrow_padded, SORT_FLOOR)
       narrow_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       narrow_pairs = pl.tile.sort32(narrow_values, pl.reinterpret_view(narrow_positions, pl.UINT32))
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=64)
       narrow_pairs = pl.tile.mrgsort(narrow_pairs, block_len=256)
       pl.store(narrow_pairs, [t, leaf * PAIR_W], wpairs)
      elif valid <= 1024:
       left_valid = pl.min(valid, 512)
       right_valid = pl.max(valid - 512, 0)
       left_raw = pl.cast(pl.load(scores, [t, begin], [1, 512], valid_shape=[1, left_valid]), pl.FP32)
       right_raw = pl.cast(pl.load(scores, [t, begin + 512], [1, 512], valid_shape=[1, right_valid]), pl.FP32)
       left_values = pl.maximum(pl.tile.fillpad(left_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       right_values = pl.maximum(pl.tile.fillpad(right_raw, pad_value=pl.PadValue.min), SORT_FLOOR)
       left_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       right_positions = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), pl.cast(begin + 512, pl.INT32))
       left_pairs = pl.tile.sort32(left_values, pl.reinterpret_view(left_positions, pl.UINT32))
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=64)
       left_pairs = pl.tile.mrgsort(left_pairs, block_len=256)
       right_pairs = pl.tile.sort32(right_values, pl.reinterpret_view(right_positions, pl.UINT32))
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=64)
       right_pairs = pl.tile.mrgsort(right_pairs, block_len=256)
       native_scratch = pl.tile.create([1, 2048], dtype=pl.FP32)
       native_pairs = pl.tile.mrgsort(left_pairs, right_pairs, tmp=native_scratch)
       native_discarded = pl.tile.read(native_pairs, [0, PAIR_W])
       if native_discarded > discarded:
        discarded = native_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if native_discarded == discarded:
        native_front_boundary = pl.tile.read(native_pairs, [0, PAIR_W - 2])
        native_front_bound = native_front_boundary
        if native_front_bound < FIN_FLOOR:
         native_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if native_discarded >= native_front_bound:
         native_arity_third = pl.tile.read(native_pairs, [0, PAIR_W + 4])
         if native_arity_third < native_discarded:
          native_arity_second = pl.tile.read(native_pairs, [0, PAIR_W + 2])
          native_arity_bits = pl.reinterpret_view(native_pairs, pl.INT32)
          native_front_result = pl.tile.read(native_arity_bits, [0, PAIR_W + 1])
          if native_arity_second == native_discarded:
           native_arity_second_position = pl.tile.read(native_arity_bits, [0, PAIR_W + 3])
           native_front_result = pl.min(native_front_result, native_arity_second_position)
         else:
          native_front_tail = pl.tile.slice(native_pairs, [1, 1024], [0, PAIR_W])
          native_front_values = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
          native_front_positions = pl.tile.gather_mask(native_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          native_front_candidates = pl.tile.select(pl.tile.cmps(native_front_values, native_discarded, cmp_type=0), native_front_positions, 2147483647)
          native_front_wide = pl.col_min(pl.reshape(native_front_candidates, [8, 64]))
          native_front_eight = pl.col_min(pl.reshape(native_front_wide, [8, 8]))
          native_front_result = pl.tile.read(native_front_eight, [0, 0])
          native_front_lane_1 = pl.tile.read(native_front_eight, [0, 1])
          if native_front_lane_1 < native_front_result:
           native_front_result = native_front_lane_1
          native_front_lane_2 = pl.tile.read(native_front_eight, [0, 2])
          if native_front_lane_2 < native_front_result:
           native_front_result = native_front_lane_2
          native_front_lane_3 = pl.tile.read(native_front_eight, [0, 3])
          if native_front_lane_3 < native_front_result:
           native_front_result = native_front_lane_3
          native_front_lane_4 = pl.tile.read(native_front_eight, [0, 4])
          if native_front_lane_4 < native_front_result:
           native_front_result = native_front_lane_4
          native_front_lane_5 = pl.tile.read(native_front_eight, [0, 5])
          if native_front_lane_5 < native_front_result:
           native_front_result = native_front_lane_5
          native_front_lane_6 = pl.tile.read(native_front_eight, [0, 6])
          if native_front_lane_6 < native_front_result:
           native_front_result = native_front_lane_6
          native_front_lane_7 = pl.tile.read(native_front_eight, [0, 7])
          if native_front_lane_7 < native_front_result:
           native_front_result = native_front_lane_7
         if native_front_result < frontier_position:
          frontier_position = native_front_result
       pl.store(pl.tile.slice(native_pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
      else:
       raw = pl.cast(pl.load(scores, [t, begin], [1, LEAF], valid_shape=[1, valid]), pl.FP32)
       padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
       flr = pl.tile.full([1, LEAF], dtype=pl.FP32, value=SORT_FLOOR)
       vals = pl.maximum(padded, flr)
       idx_i32 = pl.add(pl.tile.arange(0, [1, LEAF], dtype=pl.INT32), pl.cast(begin, pl.INT32))
       pairs = pl.tile.sort32(vals, pl.reinterpret_view(idx_i32, pl.UINT32))
       pairs = pl.tile.mrgsort(pairs, block_len=64)
       pairs = pl.tile.mrgsort(pairs, block_len=256)
       pairs = pl.tile.mrgsort(pairs, block_len=1024)
       leaf_discarded = pl.tile.read(pairs, [0, PAIR_W])
       if leaf_discarded > discarded:
        discarded = leaf_discarded
        frontier_position = pl.cast(2147483647, pl.INT32)
       if leaf_discarded == discarded:
        leaf_front_boundary = pl.tile.read(pairs, [0, PAIR_W - 2])
        leaf_front_bound = leaf_front_boundary
        if leaf_front_bound < FIN_FLOOR:
         leaf_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
        if leaf_discarded >= leaf_front_bound:
         leaf_arity_third = pl.tile.read(pairs, [0, PAIR_W + 4])
         if leaf_arity_third < leaf_discarded:
          leaf_arity_second = pl.tile.read(pairs, [0, PAIR_W + 2])
          leaf_arity_bits = pl.reinterpret_view(pairs, pl.INT32)
          leaf_front_result = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 1])
          if leaf_arity_second == leaf_discarded:
           leaf_arity_second_position = pl.tile.read(leaf_arity_bits, [0, PAIR_W + 3])
           leaf_front_result = pl.min(leaf_front_result, leaf_arity_second_position)
         else:
          leaf_front_tail = pl.tile.slice(pairs, [1, 3072], [0, PAIR_W])
          leaf_front_values = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
          leaf_front_positions = pl.tile.gather_mask(leaf_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
          leaf_front_candidates = pl.tile.select(pl.tile.cmps(leaf_front_values, leaf_discarded, cmp_type=0), leaf_front_positions, 2147483647)
          leaf_front_wide = pl.col_min(pl.reshape(leaf_front_candidates, [24, 64]))
          leaf_front_eight = pl.col_min(pl.reshape(leaf_front_wide, [8, 8]))
          leaf_front_result = pl.tile.read(leaf_front_eight, [0, 0])
          leaf_front_lane_1 = pl.tile.read(leaf_front_eight, [0, 1])
          if leaf_front_lane_1 < leaf_front_result:
           leaf_front_result = leaf_front_lane_1
          leaf_front_lane_2 = pl.tile.read(leaf_front_eight, [0, 2])
          if leaf_front_lane_2 < leaf_front_result:
           leaf_front_result = leaf_front_lane_2
          leaf_front_lane_3 = pl.tile.read(leaf_front_eight, [0, 3])
          if leaf_front_lane_3 < leaf_front_result:
           leaf_front_result = leaf_front_lane_3
          leaf_front_lane_4 = pl.tile.read(leaf_front_eight, [0, 4])
          if leaf_front_lane_4 < leaf_front_result:
           leaf_front_result = leaf_front_lane_4
          leaf_front_lane_5 = pl.tile.read(leaf_front_eight, [0, 5])
          if leaf_front_lane_5 < leaf_front_result:
           leaf_front_result = leaf_front_lane_5
          leaf_front_lane_6 = pl.tile.read(leaf_front_eight, [0, 6])
          if leaf_front_lane_6 < leaf_front_result:
           leaf_front_result = leaf_front_lane_6
          leaf_front_lane_7 = pl.tile.read(leaf_front_eight, [0, 7])
          if leaf_front_lane_7 < leaf_front_result:
           leaf_front_result = leaf_front_lane_7
         if leaf_front_result < frontier_position:
          frontier_position = leaf_front_result
       pl.store(pl.tile.slice(pairs, [1, PAIR_W], [0, 0]), [t, leaf * PAIR_W], wpairs)
     else:
      pl.store(zero_pairs, [t, leaf * PAIR_W], wpairs)
     leaf_score_line = pl.tile.full([1, 16], dtype=pl.FP32, value=SORT_FLOOR)
     leaf_position_line = pl.tile.full([1, 16], dtype=pl.INT32, value=2147483647)
     pl.tile.write(leaf_score_line, [0, 0], discarded)
     pl.tile.write(leaf_position_line, [0, 0], frontier_position)
     pl.store(leaf_score_line, [t, leaf * 16], leaf_discarded_bank)
     pl.store(leaf_position_line, [t, leaf * 16], leaf_front_bank)
   for merge_cohort_job in pl.spmd((M + 4 - 1) // 4, name_hint='decomposed_merge'):
    cohort_base = merge_cohort_job * 4
    for t in pl.range(cohort_base, pl.min(cohort_base + 4, M), 1):
     visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
     wv = pl.max(visible, 1)
     lc = (wv + LEAF - 1) // LEAF
     discarded = pl.cast(SORT_FLOOR, pl.FP32)
     frontier_position = pl.cast(2147483647, pl.INT32)
     for proof_leaf in pl.range(NL):
      proof_score = pl.read(leaf_discarded_bank, [t, proof_leaf * 16])
      proof_position = pl.read(leaf_front_bank, [t, proof_leaf * 16])
      if proof_score > discarded:
       discarded = proof_score
       frontier_position = pl.cast(2147483647, pl.INT32)
      if proof_score == discarded:
       if proof_position < frontier_position:
        frontier_position = proof_position
     mtmp = pl.tile.create([1, 2 * PAIR_W], dtype=pl.FP32)
     for cb in pl.range(1, lc):
      mleft = pl.load(wpairs, [t, 0], [1, PAIR_W])
      mright = pl.load(wpairs, [t, cb * PAIR_W], [1, PAIR_W])
      mboth = pl.tile.mrgsort(mleft, mright, tmp=mtmp)
      merge_discarded = pl.tile.read(mboth, [0, PAIR_W])
      if merge_discarded > discarded:
       discarded = merge_discarded
       frontier_position = pl.cast(2147483647, pl.INT32)
      if merge_discarded == discarded:
       merge_front_boundary = pl.tile.read(mboth, [0, PAIR_W - 2])
       merge_front_bound = merge_front_boundary
       if merge_front_bound < FIN_FLOOR:
        merge_front_bound = pl.cast(FIN_FLOOR, pl.FP32)
       if merge_discarded >= merge_front_bound:
        merge_front_tail = pl.tile.slice(mboth, [1, 1024], [0, PAIR_W])
        merge_front_values = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P0101)
        merge_front_positions = pl.tile.gather_mask(merge_front_tail, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        merge_front_candidates = pl.tile.select(pl.tile.cmps(merge_front_values, merge_discarded, cmp_type=0), merge_front_positions, 2147483647)
        merge_front_wide = pl.col_min(pl.reshape(merge_front_candidates, [8, 64]))
        merge_front_eight = pl.col_min(pl.reshape(merge_front_wide, [8, 8]))
        merge_front_result = pl.tile.read(merge_front_eight, [0, 0])
        merge_front_lane_1 = pl.tile.read(merge_front_eight, [0, 1])
        if merge_front_lane_1 < merge_front_result:
         merge_front_result = merge_front_lane_1
        merge_front_lane_2 = pl.tile.read(merge_front_eight, [0, 2])
        if merge_front_lane_2 < merge_front_result:
         merge_front_result = merge_front_lane_2
        merge_front_lane_3 = pl.tile.read(merge_front_eight, [0, 3])
        if merge_front_lane_3 < merge_front_result:
         merge_front_result = merge_front_lane_3
        merge_front_lane_4 = pl.tile.read(merge_front_eight, [0, 4])
        if merge_front_lane_4 < merge_front_result:
         merge_front_result = merge_front_lane_4
        merge_front_lane_5 = pl.tile.read(merge_front_eight, [0, 5])
        if merge_front_lane_5 < merge_front_result:
         merge_front_result = merge_front_lane_5
        merge_front_lane_6 = pl.tile.read(merge_front_eight, [0, 6])
        if merge_front_lane_6 < merge_front_result:
         merge_front_result = merge_front_lane_6
        merge_front_lane_7 = pl.tile.read(merge_front_eight, [0, 7])
        if merge_front_lane_7 < merge_front_result:
         merge_front_result = merge_front_lane_7
        if merge_front_result < frontier_position:
         frontier_position = merge_front_result
      pl.store(pl.tile.slice(mboth, [1, PAIR_W], [0, 0]), [t, 0], wpairs)
     pl.write(aux_discarded, [t, 0], discarded)
     pl.write(aux_front, [t, 0], frontier_position)
   for materialize_row_job in pl.spmd((M + 4 - 1) // 4, name_hint='decomposed_materialize'):
    root_row_base = materialize_row_job * 4
    if P == 1:
     materialize_single_page = pl.read(block_table, [0, 0])
     materialize_pages = pl.tile.fillpad(pl.add(pl.tile.full([1, P8], dtype=pl.INT32, value=0), materialize_single_page), pad_value=pl.PadValue.zero)
    else:
     materialize_pages = pl.tile.fillpad(pl.load(block_table, [0, 0], [1, P8], valid_shape=[1, P]), pad_value=pl.PadValue.zero)
    materialize_page_lanes = pl.tile.arange(0, [1, P8], dtype=pl.INT32)
    materialize_displacements = pl.mul(pl.sub(materialize_pages, materialize_page_lanes), PAGE)
    for t in pl.range(root_row_base, pl.min(root_row_base + 4, M), 1):
     fin = pl.load(wpairs, [t, 0], [1, PAIR_W])
     values = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P0101)
     finite = pl.maximum(values, FIN_FLOOR)
     cutoff = pl.tile.read(finite, [0, K - 1])
     ones_c = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
     flags = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=4), ones_c, 0.0)
     sums64 = pl.col_sum(pl.reshape(flags, [8, 64]))
     sums8 = pl.col_sum(pl.reshape(sums64, [8, 8]))
     above = pl.cast(0, pl.INDEX)
     for lane in pl.unroll(8):
      part = pl.cast(pl.tile.read(sums8, [0, lane]), pl.INT32)
      above = above + pl.cast(part, pl.INDEX)
     cutoff_stage_line = pl.tile.full([1, 16], dtype=pl.FP32, value=0)
     pl.tile.write(cutoff_stage_line, [0, 0], cutoff)
     pl.store(cutoff_stage_line, [t, 0], aux_cutoffs)
     quota_stage_line = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
     pl.tile.write(quota_stage_line, [0, 0], pl.cast(K - above, pl.INT32))
     pl.store(quota_stage_line, [t, 0], aux_quotas)
     discarded = pl.read(aux_discarded, [t, 0])
     frontier_position = pl.read(aux_front, [t, 0])
     replay_safe = discarded < cutoff
     if discarded == cutoff:
      certificate_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
      certificate_candidates = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=0), certificate_positions, -1)
      certificate_wide = pl.col_max(pl.reshape(certificate_candidates, [8, 64]))
      certificate_eight = pl.col_max(pl.reshape(certificate_wide, [8, 8]))
      certificate_result = pl.tile.read(certificate_eight, [0, 0])
      certificate_lane_1 = pl.tile.read(certificate_eight, [0, 1])
      if certificate_lane_1 > certificate_result:
       certificate_result = certificate_lane_1
      certificate_lane_2 = pl.tile.read(certificate_eight, [0, 2])
      if certificate_lane_2 > certificate_result:
       certificate_result = certificate_lane_2
      certificate_lane_3 = pl.tile.read(certificate_eight, [0, 3])
      if certificate_lane_3 > certificate_result:
       certificate_result = certificate_lane_3
      certificate_lane_4 = pl.tile.read(certificate_eight, [0, 4])
      if certificate_lane_4 > certificate_result:
       certificate_result = certificate_lane_4
      certificate_lane_5 = pl.tile.read(certificate_eight, [0, 5])
      if certificate_lane_5 > certificate_result:
       certificate_result = certificate_lane_5
      certificate_lane_6 = pl.tile.read(certificate_eight, [0, 6])
      if certificate_lane_6 > certificate_result:
       certificate_result = certificate_lane_6
      certificate_lane_7 = pl.tile.read(certificate_eight, [0, 7])
      if certificate_lane_7 > certificate_result:
       certificate_result = certificate_lane_7
      if frontier_position > certificate_result:
       replay_safe = True
     if replay_safe:
      if cutoff > FIN_FLOOR:
       root_positions = pl.tile.gather_mask(fin, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       selected = pl.cast(K, pl.INDEX)
       if selected > 0:
        root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(root_positions, pl.FP32))
        root_keys = root_priority
        root_order = pl.tile.sort32(root_keys, pl.reinterpret_view(root_positions, pl.UINT32))
        root_order = pl.tile.mrgsort(root_order, block_len=64)
        root_order = pl.tile.mrgsort(root_order, block_len=256)
        root_logical = pl.tile.gather_mask(root_order, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        root_valid = pl.tile.set_validshape(root_logical, 1, selected)
        root_safe = pl.tile.fillpad(root_valid, pad_value=pl.PadValue.zero)
        root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
        for map_start in pl.range(0, selected, MAP):
         map_count = pl.min(MAP, selected - map_start)
         map_bits = pl.tile.slice(pl.reinterpret_view(root_safe, pl.FP32), [1, MAP], [0, map_start])
         map_logical = pl.reinterpret_view(map_bits, pl.INT32)
         root_page_idx = pl.tile.shrs(map_logical, 7)
         root_page_ids = pl.tile.gather(materialize_displacements, root_page_idx, root_tmp)
         root_physical = pl.add(map_logical, root_page_ids)
         pl.store(pl.tile.set_validshape(root_physical, 1, map_count), [t, map_start], indices)
      else:
       visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
       root = fin
       root_values = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P0101)
       root_positions = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
       ones_root = pl.tile.full([1, K], dtype=pl.FP32, value=1.0)
       root_strict = pl.tile.select(pl.tile.cmps(root_values, cutoff, cmp_type=4), ones_root, 0.0)
       root_equal = pl.tile.select(pl.tile.cmps(root_values, cutoff, cmp_type=0), ones_root, 0.0)
       root_eligible = pl.maximum(root_strict, root_equal)
       root_visible = pl.tile.cmps(pl.cast(root_positions, pl.FP32), pl.cast(pl.cast(visible, pl.INT32), pl.FP32), cmp_type=2)
       root_keep = pl.tile.select(root_visible, root_eligible, 0.0)
       root_sums64 = pl.col_sum(pl.reshape(root_keep, [8, 64]))
       root_sums8 = pl.col_sum(pl.reshape(root_sums64, [8, 8]))
       selected = pl.cast(0, pl.INDEX)
       for root_lane in pl.unroll(8):
        root_part = pl.cast(pl.tile.read(root_sums8, [0, root_lane]), pl.INT32)
        selected = selected + pl.cast(root_part, pl.INDEX)
       pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
       if selected > 0:
        root_priority = pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), pl.cast(root_positions, pl.FP32))
        root_keys = pl.tile.select(pl.tile.cmps(root_keep, 0.0, cmp_type=4), root_priority, -1.0)
        root_order = pl.tile.sort32(root_keys, pl.reinterpret_view(root_positions, pl.UINT32))
        root_order = pl.tile.mrgsort(root_order, block_len=64)
        root_order = pl.tile.mrgsort(root_order, block_len=256)
        root_sorted_keys = pl.tile.gather_mask(root_order, mask_pattern=pl.tile.MaskPattern.P0101)
        root_logical = pl.cast(pl.sub(pl.tile.full([1, K], dtype=pl.FP32, value=N * 1.0), root_sorted_keys), pl.INT32)
        root_valid = pl.tile.set_validshape(root_logical, 1, selected)
        root_safe = pl.tile.fillpad(root_valid, pad_value=pl.PadValue.zero)
        root_tmp = pl.tile.create([1, MAP], dtype=pl.INT32)
        for map_start in pl.range(0, selected, MAP):
         map_count = pl.min(MAP, selected - map_start)
         map_bits = pl.tile.slice(pl.reinterpret_view(root_safe, pl.FP32), [1, MAP], [0, map_start])
         map_logical = pl.reinterpret_view(map_bits, pl.INT32)
         root_page_idx = pl.tile.shrs(map_logical, 7)
         root_page_ids = pl.tile.gather(materialize_displacements, root_page_idx, root_tmp)
         root_physical = pl.add(map_logical, root_page_ids)
         pl.store(pl.tile.set_validshape(root_physical, 1, map_count), [t, map_start], indices)
     else:
      visible = pl.min(pl.max(pl.read(lengths, [t]), 0), N)
      remaining = pl.cast(K - above, pl.INDEX)
      pl.store(pl.tile.full([1, K], dtype=pl.INT32, value=-1), [t, 0], indices)
      ones_m = pl.tile.full([1, SCAN], dtype=pl.FP32, value=1.0)
      lanes = pl.tile.arange(0, [1, SCAN], dtype=pl.INT32)
      lanes_f = pl.cast(lanes, pl.FP32)
      lane_origin = pl.tile.full([1, SCAN], dtype=pl.INT32, value=SCAN)
      priority = pl.cast(pl.sub(lane_origin, lanes), pl.FP32)
      gtmp = pl.tile.create([1, SCAN], dtype=pl.INT32)
      count = pl.cast(0, pl.INDEX)
      for start in pl.range(0, visible, SCAN):
       if count < K:
        valid = pl.min(SCAN, visible - start)
        scan_raw = pl.cast(pl.load(scores, [t, start], [1, SCAN], valid_shape=[1, valid]), pl.FP32)
        scan_vals = pl.tile.fillpad(scan_raw, pad_value=pl.PadValue.min)
        strict = pl.tile.select(pl.tile.cmps(scan_vals, cutoff, cmp_type=4), ones_m, 0.0)
        ties = pl.tile.full([1, SCAN], dtype=pl.FP32, value=0.0)
        if remaining > 0:
         eq = pl.tile.select(pl.tile.cmps(scan_vals, cutoff, cmp_type=0), ones_m, 0.0)
         eq_valid = pl.tile.cmps(lanes_f, pl.cast(pl.cast(valid, pl.INT32), pl.FP32), cmp_type=2)
         eq = pl.tile.select(eq_valid, eq, 0.0)
         eq_sums64 = pl.col_sum(pl.reshape(eq, [SCAN // 64, 64]))
         eq_sums8 = pl.col_sum(pl.reshape(eq_sums64, [8, 8]))
         available = pl.cast(0, pl.INDEX)
         for eq_lane in pl.unroll(8):
          eq_part = pl.cast(pl.tile.read(eq_sums8, [0, eq_lane]), pl.INT32)
          available = available + pl.cast(eq_part, pl.INDEX)
         take = pl.min(available, remaining)
         if take > 0:
          if take == available:
           ties = eq
          else:
           tie_keys = pl.mul(eq, priority)
           tie_pairs = pl.tile.sort32(tie_keys, pl.reinterpret_view(lanes, pl.UINT32))
           tie_pairs = pl.tile.mrgsort(tie_pairs, block_len=64)
           tie_merged = tie_pairs
           equal_keys = pl.tile.gather_mask(tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
           last_key_f = pl.tile.read(equal_keys, [0, take - 1])
           boundary_f = 128.0 - last_key_f
           ties = pl.tile.select(pl.tile.cmps(lanes_f, boundary_f, cmp_type=3), eq, 0.0)
          remaining = remaining - take
        keep = pl.maximum(strict, ties)
        keep_valid = pl.tile.set_validshape(keep, 1, valid)
        keep_padded = pl.tile.fillpad(keep_valid, pad_value=pl.PadValue.zero)
        keep_sums64 = pl.col_sum(pl.reshape(keep_padded, [SCAN // 64, 64]))
        keep_sums8 = pl.col_sum(pl.reshape(keep_sums64, [8, 8]))
        keep_count = pl.cast(0, pl.INDEX)
        for keep_lane in pl.unroll(8):
         keep_part = pl.cast(pl.tile.read(keep_sums8, [0, keep_lane]), pl.INT32)
         keep_count = keep_count + pl.cast(keep_part, pl.INDEX)
        selected = pl.min(keep_count, K - count)
        if selected > 0:
         order_keys = pl.mul(keep_padded, priority)
         order_pairs = pl.tile.sort32(order_keys, pl.reinterpret_view(lanes, pl.UINT32))
         order_pairs = pl.tile.mrgsort(order_pairs, block_len=64)
         order_merged = order_pairs
         order_vals = pl.tile.gather_mask(order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
         offs = pl.sub(lane_origin, pl.cast(order_vals, pl.INT32))
         compact_valid = pl.tile.set_validshape(offs, 1, selected)
         compact_safe = pl.tile.fillpad(compact_valid, pad_value=pl.PadValue.zero)
         logical = pl.add(compact_safe, pl.cast(start, pl.INT32))
         page_idx = pl.tile.shrs(logical, 7)
         page_ids = pl.tile.gather(materialize_displacements, page_idx, gtmp)
         physical = pl.add(logical, page_ids)
         pl.store(pl.tile.set_validshape(physical, 1, selected), [t, count], indices)
         count = count + selected
   for pack_task in pl.spmd(1, name_hint='pack_auxiliary'):
    for output_row in pl.range(0, M, 16):
     cutoff_lines = pl.load(aux_cutoffs, [output_row, 0], [16, 16])
     quota_lines = pl.load(aux_quotas, [output_row, 0], [16, 16])
     cutoff_flat = pl.reshape(cutoff_lines, [1, 256])
     quota_flat = pl.reshape(quota_lines, [1, 256])
     pack_offsets = pl.mul(pl.tile.arange(0, [1, 16], dtype=pl.INT32), 16)
     cutoff_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     quota_scratch = pl.tile.create([1, 256], dtype=pl.INT32)
     cutoff_packed = pl.tile.gather(cutoff_flat, pack_offsets, cutoff_scratch)
     quota_packed = pl.tile.gather(quota_flat, pack_offsets, quota_scratch)
     pl.store(pl.reshape(cutoff_packed, [16, 1]), [output_row, 0], cutoffs)
     pl.store(pl.reshape(quota_packed, [16, 1]), [output_row, 0], quotas)
   return indices

 def specs():
  return [TensorSpec('scores', [M, N], torch.bfloat16, init_value=torch.randn), TensorSpec('lengths', [M], torch.int32, init_value=lambda: torch.randint(1, N + 1, (M,), dtype=torch.int32)), TensorSpec('block_table', [1, P], torch.int32, init_value=lambda: torch.randperm(P, dtype=torch.int64).to(torch.int32).reshape(1, P)), TensorSpec('wpairs', [M, WORKSPACE_WIDTH], torch.float32, init_value=lambda: torch.full((M, WORKSPACE_WIDTH), 17.0)), TensorSpec('cutoffs', [M, 1], torch.float32, init_value=lambda: torch.full((M, 1), 23.0)), TensorSpec('quotas', [M, 1], torch.int32, init_value=lambda: torch.full((M, 1), 29, dtype=torch.int32)), TensorSpec('indices', [M, K], torch.int32)]

 def golden(t):
  vis = t['lengths'].long().clamp(0, N)
  bt = t['block_table'][0].long()
  wp = torch.zeros(M, WORKSPACE_WIDTH)
  cut = torch.zeros(M, 1)
  quo = torch.zeros(M, 1, dtype=torch.int32)
  out_idx = torch.full((M, K), -1, dtype=torch.int32)
  allpos = torch.arange(NL * LEAF, dtype=torch.int64)
  for m in range(M):
   v = int(vis[m])
   w = max(v, 1)
   padded = torch.full((NL * LEAF,), SORT_FLOOR, dtype=torch.float32)
   padded[:w] = t['scores'][m, :w].float().clamp_min(SORT_FLOOR)
   lc = (w + LEAF - 1) // LEAF
   for leaf in range(lc):
    seg = padded[leaf * LEAF:(leaf + 1) * LEAF]
    order = torch.argsort(seg, descending=True, stable=True)
    top = order[:K]
    wp[m, leaf * PAIR_W:(leaf + 1) * PAIR_W] = torch.stack([seg[top], top.to(torch.int32).view(torch.float32)], dim=-1).reshape(-1)
   order = torch.argsort(padded, descending=True, stable=True)
   top = order[:K]
   vals = padded[top]
   wp[m, 0:PAIR_W] = torch.stack([vals, top.to(torch.int32).view(torch.float32)], dim=-1).reshape(-1)
   cutoff = torch.clamp(vals, min=FIN_FLOOR)[K - 1]
   above = int((padded > cutoff).sum())
   quota = K - above
   cut[m, 0] = cutoff
   quo[m, 0] = quota
   sel = []
   rem = quota
   for p in range(v):
    sp = padded[p]
    if sp > cutoff:
     sel.append(p)
    elif sp == cutoff and rem > 0:
     sel.append(p)
     rem -= 1
    if len(sel) == K:
     break
   if sel:
    selt = torch.tensor(sel, dtype=torch.int64)
    out_idx[m, :len(sel)] = (bt[selt // PAGE] * PAGE + selt % PAGE).to(torch.int32)
  t['wpairs'][:] = wp
  t['cutoffs'][:] = cut
  t['quotas'][:] = quo
  t['indices'][:] = out_idx
 return (dsa41_topk_guarded_root, specs(), golden, RTOL, ATOL)

def build(shape):
 if shape.get('dtype', 'fp32') == 'bf16':
  return _build_bf16(shape)
 return _build_fp32(shape)
