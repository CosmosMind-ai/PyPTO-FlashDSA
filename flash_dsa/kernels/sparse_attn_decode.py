"""Packed E4M3 sparse decode with matched layout and softmax treatments."""
import math
import pypto.language as pl
RTOL = 0.02
ATOL = 0.002
HEADS = 64
DIM = 512
SLOTS = 512
GROUP = 32
GROUPS = DIM // GROUP
PACKED_BYTES = 528
PAYLOAD_PITCH = 544
GATHER_TILE = 32
NATURAL_DECODE_ROWS = 8
VECTOR_HEAD_TILE = 16
SLOT_TILE = 64
SOFTMAX_HEAD_TILE = 16
TERMINAL_HEAD_TILE = 16
ATTENTION_HEAD_TILE = 16
QK_DIM_TILE = 512
QK_SLOT_TILE = 128
EXP_SLOT_TILE = 128
PV_COLUMN_TILE = 512
PV_SLOT_TILE = 128
NEG_BIG = 1e+30
SCALE = 1.0 / math.sqrt(DIM)
EPILOGUE_HEAD_TILE = 8

COHORT_SIZE = 2

def build(shape):
    import torch
    from golden import TensorSpec
    batch, sequence, cache_rows = (shape['B'], shape['S'], shape['N'])
    tokens = batch * sequence
    TRANSPORT_HEADS = 64
    TERMINAL_HEAD_TILE = 8 if sequence % 2 == 0 else 16
    ATTENTION_HEAD_TILE = 32 if tokens >= 1 else 32
    LANE_HEADS = 16

    if tokens > COHORT_SIZE:
        if sequence == 1:
            @pl.jit
            def dsa41_cohort_release(query: pl.Tensor[[batch, sequence, HEADS, DIM], pl.BF16], cache: pl.Tensor[[cache_rows, PACKED_BYTES], pl.UINT8], indices: pl.Tensor[[batch, sequence, SLOTS], pl.INT32], sink: pl.Tensor[[HEADS], pl.FP32], compact_scale_index: pl.Tensor[[1, NATURAL_DECODE_ROWS * GROUPS], pl.INT32], e4m3_codebook: pl.Tensor[[1, 256], pl.FP32], natural_kv: pl.InOut[pl.Tensor[[tokens * SLOTS, DIM], pl.BF16]], out: pl.Out[pl.Tensor[[batch, sequence, HEADS, DIM], pl.BF16]], lse: pl.Out[pl.Tensor[[batch, HEADS, sequence], pl.FP32]]):
                query_flat = pl.reshape(query, [tokens * HEADS, DIM])
                indices_flat = pl.reshape(indices, [tokens, SLOTS])
                output_flat = pl.reshape(out, [tokens * HEADS, DIM])
                gather_tasks = pl.array.create((tokens + COHORT_SIZE - 1) // COHORT_SIZE, pl.TASK_ID)
                attention_tasks = pl.array.create((tokens + COHORT_SIZE - 1) // COHORT_SIZE, pl.TASK_ID)
                statistics_flat = pl.reshape(lse, [tokens, HEADS])
                for cohort_index in pl.unroll((tokens + COHORT_SIZE - 1) // COHORT_SIZE):
                    cohort_begin = cohort_index * COHORT_SIZE
                    cohort_count = pl.min(COHORT_SIZE, tokens - cohort_begin)
                    cohort_indices = pl.slice(indices_flat, [cohort_count, SLOTS], [cohort_begin, 0])
                    cohort_natural = pl.slice(natural_kv, [cohort_count * SLOTS, DIM], [cohort_begin * SLOTS, 0])
                    with pl.spmd(cohort_count * (SLOTS // 64), name_hint='gather_cohort') as gathered:
                        gather_dequant_work = pl.tile.get_block_idx()
                        token = gather_dequant_work // (SLOTS // 64)
                        chunk = gather_dequant_work % (SLOTS // 64)
                        decode_slot_fetch = chunk * 64
                        gather_ids_fetch = pl.load(cohort_indices, [token, decode_slot_fetch], [1, 64])
                        safe_ids_fetch = pl.maximum(gather_ids_fetch, 0)
                        zero_raw_fetch = pl.tile.create([1, PAYLOAD_PITCH], dtype=pl.UINT8)
                        zero_raw_fetch = pl.tile.gather_row(zero_raw_fetch, cache, [0, 0], [0, 0], [1, PACKED_BYTES])
                        zero_words_fetch = pl.tile.reinterpret_view(zero_raw_fetch, pl.INT32)
                        replicated_words_fetch = pl.tile.create([64, PAYLOAD_PITCH // 4], dtype=pl.INT32)
                        for replication_lane_fetch in pl.unroll(64):
                            replicated_words_fetch = pl.tile.assemble(replicated_words_fetch, zero_words_fetch, [replication_lane_fetch, 0])
                        payload_fetch = pl.tile.reinterpret_view(replicated_words_fetch, pl.UINT8)
                        for issue_group_fetch in pl.range(8):
                            for issue_offset_fetch in pl.unroll(8):
                                lane_fetch = issue_group_fetch * 8 + issue_offset_fetch
                                original_fetch = pl.tile.read(gather_ids_fetch, [0, lane_fetch])
                                if original_fetch != -1:
                                    physical_fetch = pl.cast(original_fetch, pl.INDEX)
                                    payload_fetch = pl.tile.gather_row(payload_fetch, cache, [lane_fetch, 0], [physical_fetch, 0], [1, PACKED_BYTES])
                        codebook_values = pl.load(e4m3_codebook, [0, 0], [1, 256])
                        compact_scale_map = pl.load(compact_scale_index, [0, 0], [1, NATURAL_DECODE_ROWS * GROUPS])
                        for natural_chunk in pl.pipeline(64 // NATURAL_DECODE_ROWS, stage=2):
                            natural_slot = chunk * 64 + natural_chunk * NATURAL_DECODE_ROWS
                            natural_carrier_words = pl.tile.reinterpret_view(payload_fetch, pl.INT32)
                            natural_payload_word_view = pl.tile.slice(natural_carrier_words, [NATURAL_DECODE_ROWS, DIM // 4], [natural_chunk * NATURAL_DECODE_ROWS, 0])
                            natural_payload_word_mask = pl.tile.full([NATURAL_DECODE_ROWS, DIM // 4], dtype=pl.INT32, value=-1)
                            natural_payload_words = pl.tile.and_(natural_payload_word_view, natural_payload_word_mask)
                            natural_payload = pl.tile.reinterpret_view(natural_payload_words, pl.UINT8)
                            natural_half_codes = pl.cast(natural_payload, pl.FP16)
                            natural_integer_codes = pl.cast(natural_half_codes, pl.INT32, mode='rint')
                            natural_codebook_scratch = pl.tile.create([NATURAL_DECODE_ROWS, DIM], dtype=pl.INT32)
                            natural_decoded = pl.tile.gather(codebook_values, natural_integer_codes, natural_codebook_scratch)
                            natural_scale_bytes = pl.tile.slice(payload_fetch, [NATURAL_DECODE_ROWS, 32], [natural_chunk * NATURAL_DECODE_ROWS, DIM], valid_shape=[NATURAL_DECODE_ROWS, GROUPS], pad_value=0)
                            natural_scale_half = pl.cast(natural_scale_bytes, pl.FP16)
                            natural_scale_codes = pl.cast(natural_scale_half, pl.INT32, mode='rint')
                            compact_scale_scratch = pl.tile.create([1, NATURAL_DECODE_ROWS * GROUPS], dtype=pl.INT32)
                            compact_scale_codes = pl.tile.gather(natural_scale_codes, compact_scale_map, compact_scale_scratch)
                            compact_scale_bits = pl.shls(compact_scale_codes, 23)
                            compact_scale_values = pl.tile.reinterpret_view(compact_scale_bits, pl.FP32)
                            group_scale_column = pl.reshape(compact_scale_values, [NATURAL_DECODE_ROWS * GROUPS, 1])
                            grouped_decoded = pl.reshape(natural_decoded, [NATURAL_DECODE_ROWS * GROUPS, GROUP])
                            grouped_scaled = pl.row_expand_mul(grouped_decoded, group_scale_column)
                            natural_scaled = pl.reshape(grouped_scaled, [NATURAL_DECODE_ROWS, DIM])
                            natural_rounded = pl.cast(natural_scaled, pl.BF16, mode='rint')
                            pl.store(natural_rounded, [token * SLOTS + natural_slot, 0], cohort_natural)
                    gather_tasks[cohort_index] = gathered
                    cohort_begin = cohort_index * COHORT_SIZE
                    cohort_count = pl.min(COHORT_SIZE, tokens - cohort_begin)
                    cohort_indices = pl.slice(indices_flat, [cohort_count, SLOTS], [cohort_begin, 0])
                    cohort_natural = pl.slice(natural_kv, [cohort_count * SLOTS, DIM], [cohort_begin * SLOTS, 0])
                    cohort_query = pl.slice(query_flat, [cohort_count * HEADS, DIM], [cohort_begin * HEADS, 0])
                    cohort_output = pl.slice(output_flat, [cohort_count * HEADS, DIM], [cohort_begin * HEADS, 0])
                    cohort_statistics = pl.slice(statistics_flat, [cohort_count, HEADS], [cohort_begin, 0])
                    with pl.spmd(cohort_count * (HEADS // ATTENTION_HEAD_TILE), name_hint='attend_cohort', optimizations=[pl.cross_core_slot(slot_num=4, local_slot_num=1)], deps=[gather_tasks[cohort_index]]) as attended:
                        work = pl.tile.get_block_idx()
                        token = work // (HEADS // ATTENTION_HEAD_TILE)
                        head = work % (HEADS // ATTENTION_HEAD_TILE) * ATTENTION_HEAD_TILE
                        row = token * HEADS + head
                        for setup_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                            panel_maxima = pl.full([SLOTS // QK_SLOT_TILE, LANE_HEADS], dtype=pl.FP32, value=-NEG_BIG)
                            panel_sums = pl.full([SLOTS // QK_SLOT_TILE, LANE_HEADS], dtype=pl.FP32, value=0.0)
                            local_scores = pl.full([LANE_HEADS, SLOTS], dtype=pl.FP32, value=0.0)
                        resident_query = cohort_query[row:row + ATTENTION_HEAD_TILE, 0:DIM]
                        for column in pl.range(SLOTS // QK_SLOT_TILE):
                            slot = column * QK_SLOT_TILE
                            score_accumulator = pl.create_tensor([ATTENTION_HEAD_TILE, QK_SLOT_TILE], dtype=pl.FP32)
                            for reduction in pl.range(DIM // QK_DIM_TILE):
                                query_tile = resident_query[0:ATTENTION_HEAD_TILE, reduction * QK_DIM_TILE:reduction * QK_DIM_TILE + QK_DIM_TILE]
                                key_tile = cohort_natural[token * SLOTS + slot:token * SLOTS + slot + QK_SLOT_TILE, reduction * QK_DIM_TILE:reduction * QK_DIM_TILE + QK_DIM_TILE]
                                score_accumulator = pl.matmul_acc(score_accumulator, query_tile, key_tile, b_trans=True, init_cond=reduction == 0)
                            for score_lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                                score_vector = pl.aiv_shard(score_accumulator)
                                score_scaled = pl.mul(score_vector, SCALE)
                                score_ids = pl.cast(cohort_indices[token:token + 1, slot:slot + QK_SLOT_TILE], pl.FP32)
                                score_invalid = pl.cmp(score_ids, -1.0, cmp_type=0)
                                score_removed = pl.col_expand_mul(score_scaled, score_invalid)
                                score_masked = pl.col_expand_sub(pl.sub(score_scaled, score_removed), pl.mul(score_invalid, NEG_BIG))
                                slot_maximum = pl.reshape(pl.row_max(score_masked), [1, LANE_HEADS])
                                panel_maxima = pl.assemble(panel_maxima, slot_maximum, [column, 0])
                                panel_denominator = pl.full([1, LANE_HEADS], dtype=pl.FP32, value=0.0)
                                for statistics_head in pl.range(LANE_HEADS // SOFTMAX_HEAD_TILE):
                                    statistics_offset = statistics_head * SOFTMAX_HEAD_TILE
                                    statistics_maximum = slot_maximum[0:1, statistics_offset:statistics_offset + SOFTMAX_HEAD_TILE]
                                    statistics_denominator = pl.full([1, SOFTMAX_HEAD_TILE], dtype=pl.FP32, value=0.0)
                                    statistics_maximum_column = pl.reshape(statistics_maximum, [SOFTMAX_HEAD_TILE, 1])
                                    for exponential_column in pl.range(QK_SLOT_TILE // EXP_SLOT_TILE):
                                        exponential_slot = exponential_column * EXP_SLOT_TILE
                                        exponential_scores = score_masked[statistics_offset:statistics_offset + SOFTMAX_HEAD_TILE, exponential_slot:exponential_slot + EXP_SLOT_TILE]
                                        exponential_values = pl.exp(pl.row_expand_sub(exponential_scores, statistics_maximum_column))
                                        exponential_partial = pl.reshape(pl.row_sum(exponential_values), [1, SOFTMAX_HEAD_TILE])
                                        statistics_denominator = pl.add(statistics_denominator, exponential_partial)
                                        local_scores = pl.assemble(local_scores, exponential_values, [statistics_offset, slot + exponential_slot])
                                    panel_denominator = pl.assemble(panel_denominator, statistics_denominator, [0, statistics_offset])
                                panel_sums = pl.assemble(panel_sums, panel_denominator, [column, 0])
                        for merge_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                            softmax_maximum = pl.full([1, LANE_HEADS], dtype=pl.FP32, value=-NEG_BIG)
                            for merge_panel in pl.range(SLOTS // QK_SLOT_TILE):
                                merge_maximum = panel_maxima[merge_panel:merge_panel + 1, 0:LANE_HEADS]
                                softmax_maximum = pl.maximum(softmax_maximum, merge_maximum)
                            panel_corrections = pl.exp(pl.col_expand_sub(panel_maxima, softmax_maximum))
                            softmax_denominator = pl.full([1, LANE_HEADS], dtype=pl.FP32, value=0.0)
                            for merge_panel in pl.range(SLOTS // QK_SLOT_TILE):
                                merge_correction = panel_corrections[merge_panel:merge_panel + 1, 0:LANE_HEADS]
                                merge_sum = panel_sums[merge_panel:merge_panel + 1, 0:LANE_HEADS]
                                softmax_denominator = pl.add(softmax_denominator, pl.mul(merge_correction, merge_sum))
                        pv_accumulator_0 = pl.create_tensor([ATTENTION_HEAD_TILE, PV_COLUMN_TILE], dtype=pl.FP32)
                        for pv_reduction in pl.range(SLOTS // PV_SLOT_TILE):
                            pv_slot = pv_reduction * PV_SLOT_TILE
                            for probability_lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                                pv_probability = pl.full([LANE_HEADS, PV_SLOT_TILE], dtype=pl.BF16, value=0.0)
                                for probability_subslot in pl.range(PV_SLOT_TILE // SLOT_TILE):
                                    normalization_slot = pv_slot + probability_subslot * SLOT_TILE
                                    for normalization_head in pl.range(LANE_HEADS // SOFTMAX_HEAD_TILE):
                                        normalization_offset = normalization_head * SOFTMAX_HEAD_TILE
                                        normalization_exponentials = local_scores[normalization_offset:normalization_offset + SOFTMAX_HEAD_TILE, normalization_slot:normalization_slot + SLOT_TILE]
                                        normalization_denominator = pl.reshape(softmax_denominator[0:1, normalization_offset:normalization_offset + SOFTMAX_HEAD_TILE], [SOFTMAX_HEAD_TILE, 1])
                                        normalization_panel = normalization_slot // QK_SLOT_TILE
                                        normalization_correction = pl.reshape(panel_corrections[normalization_panel:normalization_panel + 1, normalization_offset:normalization_offset + SOFTMAX_HEAD_TILE], [SOFTMAX_HEAD_TILE, 1])
                                        normalization_rebased = pl.row_expand_mul(normalization_exponentials, normalization_correction)
                                        normalization_result = pl.row_expand_div(normalization_rebased, normalization_denominator)
                                        normalization_rounded = pl.cast(normalization_result, pl.BF16, mode='rint')
                                        pv_probability = pl.assemble(pv_probability, normalization_rounded, [normalization_offset, probability_subslot * SLOT_TILE])
                                        local_scores = pl.assemble(local_scores, normalization_result, [normalization_offset, normalization_slot])
                                pv_probability_cube = pl.aic_gather(pv_probability)
                            pv_value_0 = cohort_natural[token * SLOTS + pv_slot:token * SLOTS + pv_slot + PV_SLOT_TILE, 0:DIM]
                            pv_accumulator_0 = pl.matmul_acc(pv_accumulator_0, pv_probability_cube, pv_value_0, init_cond=pv_reduction == 0)
                        for validity_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                            fused_validity_ids = pl.cast(cohort_indices[token:token + 1, 0:SLOTS], pl.FP32)
                            fused_validity_invalid = pl.cmp(fused_validity_ids, -1.0, cmp_type=0)
                            fused_partial_count = pl.col_sum(pl.reshape(fused_validity_invalid, [4, 128]))
                            fused_counts = pl.col_sum(pl.reshape(fused_partial_count, [8, 16]))
                            fused_count_carrier = pl.full([8, 16], dtype=pl.FP32, value=0.0)
                            fused_count_carrier = pl.assemble(fused_count_carrier, fused_counts, [0, 0])
                            fused_count_total = pl.row_sum(fused_count_carrier)
                            fused_invalid_count = pl.cast(pl.cast(pl.read(fused_count_total, [0, 0]), pl.INT32), pl.INDEX)
                            fused_valid_count = SLOTS - fused_invalid_count
                        for epilogue_lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                            pv_complete_vector = pl.aiv_shard(pv_accumulator_0)
                            pv_committed = pl.cast(pv_complete_vector, pl.BF16, mode='rint')
                            fused_finite_lse = pl.add(softmax_maximum, pl.log(softmax_denominator, high_precision=True))
                            fused_exp = pl.exp(fused_finite_lse)
                            fused_sink = pl.reshape(sink[head + epilogue_lane * LANE_HEADS:head + (epilogue_lane + 1) * LANE_HEADS], [1, LANE_HEADS])
                            fused_factor = pl.div(fused_exp, pl.add(fused_exp, pl.exp(fused_sink)), high_precision=True)
                            fused_public_lse = fused_finite_lse
                            if fused_valid_count == 0:
                                fused_public_lse = pl.reinterpret_view(pl.full([1, LANE_HEADS], dtype=pl.INT32, value=2139095040), pl.FP32)
                            cohort_statistics[token:token + 1, head + epilogue_lane * LANE_HEADS:head + (epilogue_lane + 1) * LANE_HEADS] = fused_public_lse
                            for fused_piece in pl.range(LANE_HEADS // EPILOGUE_HEAD_TILE):
                                fused_offset = fused_piece * EPILOGUE_HEAD_TILE
                                fused_row = row + epilogue_lane * LANE_HEADS + fused_offset
                                fused_rounded_piece = pv_committed[fused_offset:fused_offset + EPILOGUE_HEAD_TILE, 0:DIM]
                                fused_values = pl.cast(fused_rounded_piece, pl.FP32)
                                fused_factor_piece = pl.reshape(fused_factor[0:1, fused_offset:fused_offset + EPILOGUE_HEAD_TILE], [EPILOGUE_HEAD_TILE, 1])
                                fused_scaled = pl.row_expand_mul(fused_values, fused_factor_piece)
                                if fused_valid_count == 0:
                                    fused_scaled = pl.full([EPILOGUE_HEAD_TILE, DIM], dtype=pl.FP32, value=0.0)
                                cohort_output[fused_row:fused_row + EPILOGUE_HEAD_TILE, 0:DIM] = pl.cast(fused_scaled, pl.BF16, mode='rint')
                    attention_tasks[cohort_index] = attended
                return (out, lse)
        else:
            @pl.jit
            def dsa41_cohort_release(query: pl.Tensor[[batch, sequence, HEADS, DIM], pl.BF16], cache: pl.Tensor[[cache_rows, PACKED_BYTES], pl.UINT8], indices: pl.Tensor[[batch, sequence, SLOTS], pl.INT32], sink: pl.Tensor[[HEADS], pl.FP32], compact_scale_index: pl.Tensor[[1, NATURAL_DECODE_ROWS * GROUPS], pl.INT32], e4m3_codebook: pl.Tensor[[1, 256], pl.FP32], natural_kv: pl.InOut[pl.Tensor[[tokens * SLOTS, DIM], pl.BF16]], lse_pack_index: pl.Tensor[[1, TRANSPORT_HEADS * sequence], pl.INT32], lse_stage: pl.InOut[pl.Tensor[[tokens, HEADS], pl.FP32]], out: pl.Out[pl.Tensor[[batch, sequence, HEADS, DIM], pl.BF16]], lse: pl.Out[pl.Tensor[[batch, HEADS, sequence], pl.FP32]]):
                query_flat = pl.reshape(query, [tokens * HEADS, DIM])
                indices_flat = pl.reshape(indices, [tokens, SLOTS])
                output_flat = pl.reshape(out, [tokens * HEADS, DIM])
                gather_tasks = pl.array.create((tokens + COHORT_SIZE - 1) // COHORT_SIZE, pl.TASK_ID)
                attention_tasks = pl.array.create((tokens + COHORT_SIZE - 1) // COHORT_SIZE, pl.TASK_ID)
                statistics_flat = lse_stage
                for cohort_index in pl.unroll((tokens + COHORT_SIZE - 1) // COHORT_SIZE):
                    cohort_begin = cohort_index * COHORT_SIZE
                    cohort_count = pl.min(COHORT_SIZE, tokens - cohort_begin)
                    cohort_indices = pl.slice(indices_flat, [cohort_count, SLOTS], [cohort_begin, 0])
                    cohort_natural = pl.slice(natural_kv, [cohort_count * SLOTS, DIM], [cohort_begin * SLOTS, 0])
                    with pl.spmd(cohort_count * (SLOTS // 64), name_hint='gather_cohort') as gathered:
                        gather_dequant_work = pl.tile.get_block_idx()
                        token = gather_dequant_work // (SLOTS // 64)
                        chunk = gather_dequant_work % (SLOTS // 64)
                        decode_slot_fetch = chunk * 64
                        gather_ids_fetch = pl.load(cohort_indices, [token, decode_slot_fetch], [1, 64])
                        safe_ids_fetch = pl.maximum(gather_ids_fetch, 0)
                        zero_raw_fetch = pl.tile.create([1, PAYLOAD_PITCH], dtype=pl.UINT8)
                        zero_raw_fetch = pl.tile.gather_row(zero_raw_fetch, cache, [0, 0], [0, 0], [1, PACKED_BYTES])
                        zero_words_fetch = pl.tile.reinterpret_view(zero_raw_fetch, pl.INT32)
                        replicated_words_fetch = pl.tile.create([64, PAYLOAD_PITCH // 4], dtype=pl.INT32)
                        for replication_lane_fetch in pl.unroll(64):
                            replicated_words_fetch = pl.tile.assemble(replicated_words_fetch, zero_words_fetch, [replication_lane_fetch, 0])
                        payload_fetch = pl.tile.reinterpret_view(replicated_words_fetch, pl.UINT8)
                        for issue_group_fetch in pl.range(8):
                            for issue_offset_fetch in pl.unroll(8):
                                lane_fetch = issue_group_fetch * 8 + issue_offset_fetch
                                original_fetch = pl.tile.read(gather_ids_fetch, [0, lane_fetch])
                                if original_fetch != -1:
                                    physical_fetch = pl.cast(original_fetch, pl.INDEX)
                                    payload_fetch = pl.tile.gather_row(payload_fetch, cache, [lane_fetch, 0], [physical_fetch, 0], [1, PACKED_BYTES])
                        codebook_values = pl.load(e4m3_codebook, [0, 0], [1, 256])
                        compact_scale_map = pl.load(compact_scale_index, [0, 0], [1, NATURAL_DECODE_ROWS * GROUPS])
                        for natural_chunk in pl.pipeline(64 // NATURAL_DECODE_ROWS, stage=2):
                            natural_slot = chunk * 64 + natural_chunk * NATURAL_DECODE_ROWS
                            natural_carrier_words = pl.tile.reinterpret_view(payload_fetch, pl.INT32)
                            natural_payload_word_view = pl.tile.slice(natural_carrier_words, [NATURAL_DECODE_ROWS, DIM // 4], [natural_chunk * NATURAL_DECODE_ROWS, 0])
                            natural_payload_word_mask = pl.tile.full([NATURAL_DECODE_ROWS, DIM // 4], dtype=pl.INT32, value=-1)
                            natural_payload_words = pl.tile.and_(natural_payload_word_view, natural_payload_word_mask)
                            natural_payload = pl.tile.reinterpret_view(natural_payload_words, pl.UINT8)
                            natural_half_codes = pl.cast(natural_payload, pl.FP16)
                            natural_integer_codes = pl.cast(natural_half_codes, pl.INT32, mode='rint')
                            natural_codebook_scratch = pl.tile.create([NATURAL_DECODE_ROWS, DIM], dtype=pl.INT32)
                            natural_decoded = pl.tile.gather(codebook_values, natural_integer_codes, natural_codebook_scratch)
                            natural_scale_bytes = pl.tile.slice(payload_fetch, [NATURAL_DECODE_ROWS, 32], [natural_chunk * NATURAL_DECODE_ROWS, DIM], valid_shape=[NATURAL_DECODE_ROWS, GROUPS], pad_value=0)
                            natural_scale_half = pl.cast(natural_scale_bytes, pl.FP16)
                            natural_scale_codes = pl.cast(natural_scale_half, pl.INT32, mode='rint')
                            compact_scale_scratch = pl.tile.create([1, NATURAL_DECODE_ROWS * GROUPS], dtype=pl.INT32)
                            compact_scale_codes = pl.tile.gather(natural_scale_codes, compact_scale_map, compact_scale_scratch)
                            compact_scale_bits = pl.shls(compact_scale_codes, 23)
                            compact_scale_values = pl.tile.reinterpret_view(compact_scale_bits, pl.FP32)
                            group_scale_column = pl.reshape(compact_scale_values, [NATURAL_DECODE_ROWS * GROUPS, 1])
                            grouped_decoded = pl.reshape(natural_decoded, [NATURAL_DECODE_ROWS * GROUPS, GROUP])
                            grouped_scaled = pl.row_expand_mul(grouped_decoded, group_scale_column)
                            natural_scaled = pl.reshape(grouped_scaled, [NATURAL_DECODE_ROWS, DIM])
                            natural_rounded = pl.cast(natural_scaled, pl.BF16, mode='rint')
                            pl.store(natural_rounded, [token * SLOTS + natural_slot, 0], cohort_natural)
                    gather_tasks[cohort_index] = gathered
                    cohort_begin = cohort_index * COHORT_SIZE
                    cohort_count = pl.min(COHORT_SIZE, tokens - cohort_begin)
                    cohort_indices = pl.slice(indices_flat, [cohort_count, SLOTS], [cohort_begin, 0])
                    cohort_natural = pl.slice(natural_kv, [cohort_count * SLOTS, DIM], [cohort_begin * SLOTS, 0])
                    cohort_query = pl.slice(query_flat, [cohort_count * HEADS, DIM], [cohort_begin * HEADS, 0])
                    cohort_output = pl.slice(output_flat, [cohort_count * HEADS, DIM], [cohort_begin * HEADS, 0])
                    cohort_statistics = pl.slice(statistics_flat, [cohort_count, HEADS], [cohort_begin, 0])
                    with pl.spmd(cohort_count * (HEADS // ATTENTION_HEAD_TILE), name_hint='attend_cohort', optimizations=[pl.cross_core_slot(slot_num=4, local_slot_num=1)], deps=[gather_tasks[cohort_index]]) as attended:
                        work = pl.tile.get_block_idx()
                        token = work // (HEADS // ATTENTION_HEAD_TILE)
                        head = work % (HEADS // ATTENTION_HEAD_TILE) * ATTENTION_HEAD_TILE
                        row = token * HEADS + head
                        for setup_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                            panel_maxima = pl.full([SLOTS // QK_SLOT_TILE, LANE_HEADS], dtype=pl.FP32, value=-NEG_BIG)
                            panel_sums = pl.full([SLOTS // QK_SLOT_TILE, LANE_HEADS], dtype=pl.FP32, value=0.0)
                            local_scores = pl.full([LANE_HEADS, SLOTS], dtype=pl.FP32, value=0.0)
                        resident_query = cohort_query[row:row + ATTENTION_HEAD_TILE, 0:DIM]
                        for column in pl.range(SLOTS // QK_SLOT_TILE):
                            slot = column * QK_SLOT_TILE
                            score_accumulator = pl.create_tensor([ATTENTION_HEAD_TILE, QK_SLOT_TILE], dtype=pl.FP32)
                            for reduction in pl.range(DIM // QK_DIM_TILE):
                                query_tile = resident_query[0:ATTENTION_HEAD_TILE, reduction * QK_DIM_TILE:reduction * QK_DIM_TILE + QK_DIM_TILE]
                                key_tile = cohort_natural[token * SLOTS + slot:token * SLOTS + slot + QK_SLOT_TILE, reduction * QK_DIM_TILE:reduction * QK_DIM_TILE + QK_DIM_TILE]
                                score_accumulator = pl.matmul_acc(score_accumulator, query_tile, key_tile, b_trans=True, init_cond=reduction == 0)
                            for score_lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                                score_vector = pl.aiv_shard(score_accumulator)
                                score_scaled = pl.mul(score_vector, SCALE)
                                score_ids = pl.cast(cohort_indices[token:token + 1, slot:slot + QK_SLOT_TILE], pl.FP32)
                                score_invalid = pl.cmp(score_ids, -1.0, cmp_type=0)
                                score_removed = pl.col_expand_mul(score_scaled, score_invalid)
                                score_masked = pl.col_expand_sub(pl.sub(score_scaled, score_removed), pl.mul(score_invalid, NEG_BIG))
                                slot_maximum = pl.reshape(pl.row_max(score_masked), [1, LANE_HEADS])
                                panel_maxima = pl.assemble(panel_maxima, slot_maximum, [column, 0])
                                panel_denominator = pl.full([1, LANE_HEADS], dtype=pl.FP32, value=0.0)
                                for statistics_head in pl.range(LANE_HEADS // SOFTMAX_HEAD_TILE):
                                    statistics_offset = statistics_head * SOFTMAX_HEAD_TILE
                                    statistics_maximum = slot_maximum[0:1, statistics_offset:statistics_offset + SOFTMAX_HEAD_TILE]
                                    statistics_denominator = pl.full([1, SOFTMAX_HEAD_TILE], dtype=pl.FP32, value=0.0)
                                    statistics_maximum_column = pl.reshape(statistics_maximum, [SOFTMAX_HEAD_TILE, 1])
                                    for exponential_column in pl.range(QK_SLOT_TILE // EXP_SLOT_TILE):
                                        exponential_slot = exponential_column * EXP_SLOT_TILE
                                        exponential_scores = score_masked[statistics_offset:statistics_offset + SOFTMAX_HEAD_TILE, exponential_slot:exponential_slot + EXP_SLOT_TILE]
                                        exponential_values = pl.exp(pl.row_expand_sub(exponential_scores, statistics_maximum_column))
                                        exponential_partial = pl.reshape(pl.row_sum(exponential_values), [1, SOFTMAX_HEAD_TILE])
                                        statistics_denominator = pl.add(statistics_denominator, exponential_partial)
                                        local_scores = pl.assemble(local_scores, exponential_values, [statistics_offset, slot + exponential_slot])
                                    panel_denominator = pl.assemble(panel_denominator, statistics_denominator, [0, statistics_offset])
                                panel_sums = pl.assemble(panel_sums, panel_denominator, [column, 0])
                        for merge_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                            softmax_maximum = pl.full([1, LANE_HEADS], dtype=pl.FP32, value=-NEG_BIG)
                            for merge_panel in pl.range(SLOTS // QK_SLOT_TILE):
                                merge_maximum = panel_maxima[merge_panel:merge_panel + 1, 0:LANE_HEADS]
                                softmax_maximum = pl.maximum(softmax_maximum, merge_maximum)
                            panel_corrections = pl.exp(pl.col_expand_sub(panel_maxima, softmax_maximum))
                            softmax_denominator = pl.full([1, LANE_HEADS], dtype=pl.FP32, value=0.0)
                            for merge_panel in pl.range(SLOTS // QK_SLOT_TILE):
                                merge_correction = panel_corrections[merge_panel:merge_panel + 1, 0:LANE_HEADS]
                                merge_sum = panel_sums[merge_panel:merge_panel + 1, 0:LANE_HEADS]
                                softmax_denominator = pl.add(softmax_denominator, pl.mul(merge_correction, merge_sum))
                        pv_accumulator_0 = pl.create_tensor([ATTENTION_HEAD_TILE, PV_COLUMN_TILE], dtype=pl.FP32)
                        for pv_reduction in pl.range(SLOTS // PV_SLOT_TILE):
                            pv_slot = pv_reduction * PV_SLOT_TILE
                            for probability_lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                                pv_probability = pl.full([LANE_HEADS, PV_SLOT_TILE], dtype=pl.BF16, value=0.0)
                                for probability_subslot in pl.range(PV_SLOT_TILE // SLOT_TILE):
                                    normalization_slot = pv_slot + probability_subslot * SLOT_TILE
                                    for normalization_head in pl.range(LANE_HEADS // SOFTMAX_HEAD_TILE):
                                        normalization_offset = normalization_head * SOFTMAX_HEAD_TILE
                                        normalization_exponentials = local_scores[normalization_offset:normalization_offset + SOFTMAX_HEAD_TILE, normalization_slot:normalization_slot + SLOT_TILE]
                                        normalization_denominator = pl.reshape(softmax_denominator[0:1, normalization_offset:normalization_offset + SOFTMAX_HEAD_TILE], [SOFTMAX_HEAD_TILE, 1])
                                        normalization_panel = normalization_slot // QK_SLOT_TILE
                                        normalization_correction = pl.reshape(panel_corrections[normalization_panel:normalization_panel + 1, normalization_offset:normalization_offset + SOFTMAX_HEAD_TILE], [SOFTMAX_HEAD_TILE, 1])
                                        normalization_rebased = pl.row_expand_mul(normalization_exponentials, normalization_correction)
                                        normalization_result = pl.row_expand_div(normalization_rebased, normalization_denominator)
                                        normalization_rounded = pl.cast(normalization_result, pl.BF16, mode='rint')
                                        pv_probability = pl.assemble(pv_probability, normalization_rounded, [normalization_offset, probability_subslot * SLOT_TILE])
                                        local_scores = pl.assemble(local_scores, normalization_result, [normalization_offset, normalization_slot])
                                pv_probability_cube = pl.aic_gather(pv_probability)
                            pv_value_0 = cohort_natural[token * SLOTS + pv_slot:token * SLOTS + pv_slot + PV_SLOT_TILE, 0:DIM]
                            pv_accumulator_0 = pl.matmul_acc(pv_accumulator_0, pv_probability_cube, pv_value_0, init_cond=pv_reduction == 0)
                        for validity_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                            fused_validity_ids = pl.cast(cohort_indices[token:token + 1, 0:SLOTS], pl.FP32)
                            fused_validity_invalid = pl.cmp(fused_validity_ids, -1.0, cmp_type=0)
                            fused_partial_count = pl.col_sum(pl.reshape(fused_validity_invalid, [4, 128]))
                            fused_counts = pl.col_sum(pl.reshape(fused_partial_count, [8, 16]))
                            fused_count_carrier = pl.full([8, 16], dtype=pl.FP32, value=0.0)
                            fused_count_carrier = pl.assemble(fused_count_carrier, fused_counts, [0, 0])
                            fused_count_total = pl.row_sum(fused_count_carrier)
                            fused_invalid_count = pl.cast(pl.cast(pl.read(fused_count_total, [0, 0]), pl.INT32), pl.INDEX)
                            fused_valid_count = SLOTS - fused_invalid_count
                        for epilogue_lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                            pv_complete_vector = pl.aiv_shard(pv_accumulator_0)
                            pv_committed = pl.cast(pv_complete_vector, pl.BF16, mode='rint')
                            fused_finite_lse = pl.add(softmax_maximum, pl.log(softmax_denominator, high_precision=True))
                            fused_exp = pl.exp(fused_finite_lse)
                            fused_sink = pl.reshape(sink[head + epilogue_lane * LANE_HEADS:head + (epilogue_lane + 1) * LANE_HEADS], [1, LANE_HEADS])
                            fused_factor = pl.div(fused_exp, pl.add(fused_exp, pl.exp(fused_sink)), high_precision=True)
                            fused_public_lse = fused_finite_lse
                            if fused_valid_count == 0:
                                fused_public_lse = pl.reinterpret_view(pl.full([1, LANE_HEADS], dtype=pl.INT32, value=2139095040), pl.FP32)
                            cohort_statistics[token:token + 1, head + epilogue_lane * LANE_HEADS:head + (epilogue_lane + 1) * LANE_HEADS] = fused_public_lse
                            for fused_piece in pl.range(LANE_HEADS // EPILOGUE_HEAD_TILE):
                                fused_offset = fused_piece * EPILOGUE_HEAD_TILE
                                fused_row = row + epilogue_lane * LANE_HEADS + fused_offset
                                fused_rounded_piece = pv_committed[fused_offset:fused_offset + EPILOGUE_HEAD_TILE, 0:DIM]
                                fused_values = pl.cast(fused_rounded_piece, pl.FP32)
                                fused_factor_piece = pl.reshape(fused_factor[0:1, fused_offset:fused_offset + EPILOGUE_HEAD_TILE], [EPILOGUE_HEAD_TILE, 1])
                                fused_scaled = pl.row_expand_mul(fused_values, fused_factor_piece)
                                if fused_valid_count == 0:
                                    fused_scaled = pl.full([EPILOGUE_HEAD_TILE, DIM], dtype=pl.FP32, value=0.0)
                                cohort_output[fused_row:fused_row + EPILOGUE_HEAD_TILE, 0:DIM] = pl.cast(fused_scaled, pl.BF16, mode='rint')
                    attention_tasks[cohort_index] = attended
                terminal_lse_flat = pl.reshape(lse, [1, tokens * HEADS])
                with pl.spmd(batch * (HEADS // TRANSPORT_HEADS), name_hint='bulk_lse_pack', deps=[attention_tasks]) as packed:
                    transport_owner = pl.tile.get_block_idx()
                    transport_batch = transport_owner // (HEADS // TRANSPORT_HEADS)
                    transport_head = transport_owner % (HEADS // TRANSPORT_HEADS) * TRANSPORT_HEADS
                    transport_rectangle = pl.load(lse_stage, [transport_batch * sequence, transport_head], [sequence, TRANSPORT_HEADS])
                    transport_permutation = pl.load(lse_pack_index, [0, 0], [1, TRANSPORT_HEADS * sequence])
                    transport_scratch = pl.tile.create([1, TRANSPORT_HEADS * sequence], dtype=pl.INT32)
                    transport_packed = pl.tile.gather(transport_rectangle, transport_permutation, transport_scratch)
                    pl.store(transport_packed, [0, (transport_batch * HEADS + transport_head) * sequence], terminal_lse_flat)
                return (out, lse)
        dsa41_layout_softmax_factorial = dsa41_cohort_release

    elif sequence == 1:
        @pl.jit
        def dsa41_layout_softmax_factorial(query: pl.Tensor[[batch, sequence, HEADS, DIM], pl.BF16], cache: pl.Tensor[[cache_rows, PACKED_BYTES], pl.UINT8], indices: pl.Tensor[[batch, sequence, SLOTS], pl.INT32], sink: pl.Tensor[[HEADS], pl.FP32], compact_scale_index: pl.Tensor[[1, NATURAL_DECODE_ROWS * GROUPS], pl.INT32], e4m3_codebook: pl.Tensor[[1, 256], pl.FP32], natural_kv: pl.InOut[pl.Tensor[[tokens * SLOTS, DIM], pl.BF16]], out: pl.Out[pl.Tensor[[batch, sequence, HEADS, DIM], pl.BF16]], lse: pl.Out[pl.Tensor[[batch, HEADS, sequence], pl.FP32]]):
            query_flat = pl.reshape(query, [tokens * HEADS, DIM])
            indices_flat = pl.reshape(indices, [tokens, SLOTS])
            output_flat = pl.reshape(out, [tokens * HEADS, DIM])
            lse_flat = pl.reshape(lse, [batch * HEADS, sequence])
            direct_lse_elements = pl.reshape(lse, [1, tokens * HEADS])
            for gather_dequant_work in pl.spmd(tokens * (SLOTS // 64), name_hint='gather_dequant'):
                token = gather_dequant_work // (SLOTS // 64)
                chunk = gather_dequant_work % (SLOTS // 64)
                decode_slot_fetch = chunk * 64
                gather_ids_fetch = pl.load(indices_flat, [token, decode_slot_fetch], [1, 64])
                safe_ids_fetch = pl.maximum(gather_ids_fetch, 0)
                zero_raw_fetch = pl.tile.create([1, PAYLOAD_PITCH], dtype=pl.UINT8)
                zero_raw_fetch = pl.tile.gather_row(zero_raw_fetch, cache, [0, 0], [0, 0], [1, PACKED_BYTES])
                zero_words_fetch = pl.tile.reinterpret_view(zero_raw_fetch, pl.INT32)
                replicated_words_fetch = pl.tile.create([64, PAYLOAD_PITCH // 4], dtype=pl.INT32)
                for replication_lane_fetch in pl.unroll(64):
                    replicated_words_fetch = pl.tile.assemble(replicated_words_fetch, zero_words_fetch, [replication_lane_fetch, 0])
                payload_fetch = pl.tile.reinterpret_view(replicated_words_fetch, pl.UINT8)
                for issue_group_fetch in pl.range(8):
                    for issue_offset_fetch in pl.unroll(8):
                        lane_fetch = issue_group_fetch * 8 + issue_offset_fetch
                        original_fetch = pl.tile.read(gather_ids_fetch, [0, lane_fetch])
                        if original_fetch != -1:
                            physical_fetch = pl.cast(original_fetch, pl.INDEX)
                            payload_fetch = pl.tile.gather_row(payload_fetch, cache, [lane_fetch, 0], [physical_fetch, 0], [1, PACKED_BYTES])
                codebook_values = pl.load(e4m3_codebook, [0, 0], [1, 256])
                compact_scale_map = pl.load(compact_scale_index, [0, 0], [1, NATURAL_DECODE_ROWS * GROUPS])
                for natural_chunk in pl.pipeline(64 // NATURAL_DECODE_ROWS, stage=2):
                    natural_slot = chunk * 64 + natural_chunk * NATURAL_DECODE_ROWS
                    natural_carrier_words = pl.tile.reinterpret_view(payload_fetch, pl.INT32)
                    natural_payload_word_view = pl.tile.slice(natural_carrier_words, [NATURAL_DECODE_ROWS, DIM // 4], [natural_chunk * NATURAL_DECODE_ROWS, 0])
                    natural_payload_word_mask = pl.tile.full([NATURAL_DECODE_ROWS, DIM // 4], dtype=pl.INT32, value=-1)
                    natural_payload_words = pl.tile.and_(natural_payload_word_view, natural_payload_word_mask)
                    natural_payload = pl.tile.reinterpret_view(natural_payload_words, pl.UINT8)
                    natural_half_codes = pl.cast(natural_payload, pl.FP16)
                    natural_integer_codes = pl.cast(natural_half_codes, pl.INT32, mode='rint')
                    natural_codebook_scratch = pl.tile.create([NATURAL_DECODE_ROWS, DIM], dtype=pl.INT32)
                    natural_decoded = pl.tile.gather(codebook_values, natural_integer_codes, natural_codebook_scratch)
                    natural_scale_bytes = pl.tile.slice(payload_fetch, [NATURAL_DECODE_ROWS, 32], [natural_chunk * NATURAL_DECODE_ROWS, DIM], valid_shape=[NATURAL_DECODE_ROWS, GROUPS], pad_value=0)
                    natural_scale_half = pl.cast(natural_scale_bytes, pl.FP16)
                    natural_scale_codes = pl.cast(natural_scale_half, pl.INT32, mode='rint')
                    compact_scale_scratch = pl.tile.create([1, NATURAL_DECODE_ROWS * GROUPS], dtype=pl.INT32)
                    compact_scale_codes = pl.tile.gather(natural_scale_codes, compact_scale_map, compact_scale_scratch)
                    compact_scale_bits = pl.shls(compact_scale_codes, 23)
                    compact_scale_values = pl.tile.reinterpret_view(compact_scale_bits, pl.FP32)
                    group_scale_column = pl.reshape(compact_scale_values, [NATURAL_DECODE_ROWS * GROUPS, 1])
                    grouped_decoded = pl.reshape(natural_decoded, [NATURAL_DECODE_ROWS * GROUPS, GROUP])
                    grouped_scaled = pl.row_expand_mul(grouped_decoded, group_scale_column)
                    natural_scaled = pl.reshape(grouped_scaled, [NATURAL_DECODE_ROWS, DIM])
                    natural_rounded = pl.cast(natural_scaled, pl.BF16, mode='rint')
                    pl.store(natural_rounded, [token * SLOTS + natural_slot, 0], natural_kv)
            for work in pl.spmd(tokens * (HEADS // ATTENTION_HEAD_TILE), name_hint='full_row_attention', optimizations=[pl.cross_core_slot(slot_num=4, local_slot_num=1)]):
                token = work // (HEADS // ATTENTION_HEAD_TILE)
                head = work % (HEADS // ATTENTION_HEAD_TILE) * ATTENTION_HEAD_TILE
                row = token * HEADS + head
                for setup_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    panel_maxima = pl.full([SLOTS // QK_SLOT_TILE, LANE_HEADS], dtype=pl.FP32, value=-NEG_BIG)
                    panel_sums = pl.full([SLOTS // QK_SLOT_TILE, LANE_HEADS], dtype=pl.FP32, value=0.0)
                    local_scores = pl.full([LANE_HEADS, SLOTS], dtype=pl.FP32, value=0.0)
                resident_query = query_flat[row:row + ATTENTION_HEAD_TILE, 0:DIM]
                for column in pl.range(SLOTS // QK_SLOT_TILE):
                    slot = column * QK_SLOT_TILE
                    score_accumulator = pl.create_tensor([ATTENTION_HEAD_TILE, QK_SLOT_TILE], dtype=pl.FP32)
                    for reduction in pl.range(DIM // QK_DIM_TILE):
                        query_tile = resident_query[0:ATTENTION_HEAD_TILE, reduction * QK_DIM_TILE:reduction * QK_DIM_TILE + QK_DIM_TILE]
                        key_tile = natural_kv[token * SLOTS + slot:token * SLOTS + slot + QK_SLOT_TILE, reduction * QK_DIM_TILE:reduction * QK_DIM_TILE + QK_DIM_TILE]
                        score_accumulator = pl.matmul_acc(score_accumulator, query_tile, key_tile, b_trans=True, init_cond=reduction == 0)
                    for score_lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        score_vector = pl.aiv_shard(score_accumulator)
                        score_scaled = pl.mul(score_vector, SCALE)
                        score_ids = pl.cast(indices_flat[token:token + 1, slot:slot + QK_SLOT_TILE], pl.FP32)
                        score_invalid = pl.cmp(score_ids, -1.0, cmp_type=0)
                        score_removed = pl.col_expand_mul(score_scaled, score_invalid)
                        score_masked = pl.col_expand_sub(pl.sub(score_scaled, score_removed), pl.mul(score_invalid, NEG_BIG))
                        slot_maximum = pl.reshape(pl.row_max(score_masked), [1, LANE_HEADS])
                        panel_maxima = pl.assemble(panel_maxima, slot_maximum, [column, 0])
                        panel_denominator = pl.full([1, LANE_HEADS], dtype=pl.FP32, value=0.0)
                        for statistics_head in pl.range(LANE_HEADS // SOFTMAX_HEAD_TILE):
                            statistics_offset = statistics_head * SOFTMAX_HEAD_TILE
                            statistics_maximum = slot_maximum[0:1, statistics_offset:statistics_offset + SOFTMAX_HEAD_TILE]
                            statistics_denominator = pl.full([1, SOFTMAX_HEAD_TILE], dtype=pl.FP32, value=0.0)
                            statistics_maximum_column = pl.reshape(statistics_maximum, [SOFTMAX_HEAD_TILE, 1])
                            for exponential_column in pl.range(QK_SLOT_TILE // EXP_SLOT_TILE):
                                exponential_slot = exponential_column * EXP_SLOT_TILE
                                exponential_scores = score_masked[statistics_offset:statistics_offset + SOFTMAX_HEAD_TILE, exponential_slot:exponential_slot + EXP_SLOT_TILE]
                                exponential_values = pl.exp(pl.row_expand_sub(exponential_scores, statistics_maximum_column))
                                exponential_partial = pl.reshape(pl.row_sum(exponential_values), [1, SOFTMAX_HEAD_TILE])
                                statistics_denominator = pl.add(statistics_denominator, exponential_partial)
                                local_scores = pl.assemble(local_scores, exponential_values, [statistics_offset, slot + exponential_slot])
                            panel_denominator = pl.assemble(panel_denominator, statistics_denominator, [0, statistics_offset])
                        panel_sums = pl.assemble(panel_sums, panel_denominator, [column, 0])
                for merge_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    softmax_maximum = pl.full([1, LANE_HEADS], dtype=pl.FP32, value=-NEG_BIG)
                    for merge_panel in pl.range(SLOTS // QK_SLOT_TILE):
                        merge_maximum = panel_maxima[merge_panel:merge_panel + 1, 0:LANE_HEADS]
                        softmax_maximum = pl.maximum(softmax_maximum, merge_maximum)
                    panel_corrections = pl.exp(pl.col_expand_sub(panel_maxima, softmax_maximum))
                    softmax_denominator = pl.full([1, LANE_HEADS], dtype=pl.FP32, value=0.0)
                    for merge_panel in pl.range(SLOTS // QK_SLOT_TILE):
                        merge_correction = panel_corrections[merge_panel:merge_panel + 1, 0:LANE_HEADS]
                        merge_sum = panel_sums[merge_panel:merge_panel + 1, 0:LANE_HEADS]
                        softmax_denominator = pl.add(softmax_denominator, pl.mul(merge_correction, merge_sum))
                pv_accumulator_0 = pl.create_tensor([ATTENTION_HEAD_TILE, PV_COLUMN_TILE], dtype=pl.FP32)
                for pv_reduction in pl.range(SLOTS // PV_SLOT_TILE):
                    pv_slot = pv_reduction * PV_SLOT_TILE
                    for probability_lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        pv_probability = pl.full([LANE_HEADS, PV_SLOT_TILE], dtype=pl.BF16, value=0.0)
                        for probability_subslot in pl.range(PV_SLOT_TILE // SLOT_TILE):
                            normalization_slot = pv_slot + probability_subslot * SLOT_TILE
                            for normalization_head in pl.range(LANE_HEADS // SOFTMAX_HEAD_TILE):
                                normalization_offset = normalization_head * SOFTMAX_HEAD_TILE
                                normalization_exponentials = local_scores[normalization_offset:normalization_offset + SOFTMAX_HEAD_TILE, normalization_slot:normalization_slot + SLOT_TILE]
                                normalization_denominator = pl.reshape(softmax_denominator[0:1, normalization_offset:normalization_offset + SOFTMAX_HEAD_TILE], [SOFTMAX_HEAD_TILE, 1])
                                normalization_panel = normalization_slot // QK_SLOT_TILE
                                normalization_correction = pl.reshape(panel_corrections[normalization_panel:normalization_panel + 1, normalization_offset:normalization_offset + SOFTMAX_HEAD_TILE], [SOFTMAX_HEAD_TILE, 1])
                                normalization_rebased = pl.row_expand_mul(normalization_exponentials, normalization_correction)
                                normalization_result = pl.row_expand_div(normalization_rebased, normalization_denominator)
                                normalization_rounded = pl.cast(normalization_result, pl.BF16, mode='rint')
                                pv_probability = pl.assemble(pv_probability, normalization_rounded, [normalization_offset, probability_subslot * SLOT_TILE])
                                local_scores = pl.assemble(local_scores, normalization_result, [normalization_offset, normalization_slot])
                        pv_probability_cube = pl.aic_gather(pv_probability)
                    pv_value_0 = natural_kv[token * SLOTS + pv_slot:token * SLOTS + pv_slot + PV_SLOT_TILE, 0:DIM]
                    pv_accumulator_0 = pl.matmul_acc(pv_accumulator_0, pv_probability_cube, pv_value_0, init_cond=pv_reduction == 0)
                for validity_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    fused_validity_ids = pl.cast(indices_flat[token:token + 1, 0:SLOTS], pl.FP32)
                    fused_validity_invalid = pl.cmp(fused_validity_ids, -1.0, cmp_type=0)
                    fused_partial_count = pl.col_sum(pl.reshape(fused_validity_invalid, [4, 128]))
                    fused_counts = pl.col_sum(pl.reshape(fused_partial_count, [8, 16]))
                    fused_count_carrier = pl.full([8, 16], dtype=pl.FP32, value=0.0)
                    fused_count_carrier = pl.assemble(fused_count_carrier, fused_counts, [0, 0])
                    fused_count_total = pl.row_sum(fused_count_carrier)
                    fused_invalid_count = pl.cast(pl.cast(pl.read(fused_count_total, [0, 0]), pl.INT32), pl.INDEX)
                    fused_valid_count = SLOTS - fused_invalid_count
                for epilogue_lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    pv_complete_vector = pl.aiv_shard(pv_accumulator_0)
                    pv_committed = pl.cast(pv_complete_vector, pl.BF16, mode='rint')
                    fused_finite_lse = pl.add(softmax_maximum, pl.log(softmax_denominator, high_precision=True))
                    fused_exp = pl.exp(fused_finite_lse)
                    fused_sink = pl.reshape(sink[head + epilogue_lane * LANE_HEADS:head + (epilogue_lane + 1) * LANE_HEADS], [1, LANE_HEADS])
                    fused_factor = pl.div(fused_exp, pl.add(fused_exp, pl.exp(fused_sink)), high_precision=True)
                    fused_public_lse = fused_finite_lse
                    if fused_valid_count == 0:
                        fused_public_lse = pl.reinterpret_view(pl.full([1, LANE_HEADS], dtype=pl.INT32, value=2139095040), pl.FP32)
                    if sequence == 1:
                        direct_lse_elements[0:1, token * HEADS + head + epilogue_lane * LANE_HEADS:token * HEADS + head + (epilogue_lane + 1) * LANE_HEADS] = fused_public_lse
                    else:
                        direct_lse_expand_target = pl.create_tensor([LANE_HEADS, 8], dtype=pl.FP32)
                        direct_lse_contiguous_column = pl.reshape(fused_public_lse, [LANE_HEADS, 1])
                        direct_lse_padded_carrier = pl.row_expand(direct_lse_expand_target, direct_lse_contiguous_column)
                        direct_lse_narrow_column = direct_lse_padded_carrier[0:LANE_HEADS, 0:1]
                        lse_flat[token // sequence * HEADS + head + epilogue_lane * LANE_HEADS:token // sequence * HEADS + head + (epilogue_lane + 1) * LANE_HEADS, token % sequence:token % sequence + 1] = direct_lse_narrow_column
                    for fused_piece in pl.range(LANE_HEADS // EPILOGUE_HEAD_TILE):
                        fused_offset = fused_piece * EPILOGUE_HEAD_TILE
                        fused_row = row + epilogue_lane * LANE_HEADS + fused_offset
                        fused_rounded_piece = pv_committed[fused_offset:fused_offset + EPILOGUE_HEAD_TILE, 0:DIM]
                        fused_values = pl.cast(fused_rounded_piece, pl.FP32)
                        fused_factor_piece = pl.reshape(fused_factor[0:1, fused_offset:fused_offset + EPILOGUE_HEAD_TILE], [EPILOGUE_HEAD_TILE, 1])
                        fused_scaled = pl.row_expand_mul(fused_values, fused_factor_piece)
                        if fused_valid_count == 0:
                            fused_scaled = pl.full([EPILOGUE_HEAD_TILE, DIM], dtype=pl.FP32, value=0.0)
                        output_flat[fused_row:fused_row + EPILOGUE_HEAD_TILE, 0:DIM] = pl.cast(fused_scaled, pl.BF16, mode='rint')
            return (out, lse)

    else:
        @pl.jit
        def dsa41_layout_softmax_factorial(query: pl.Tensor[[batch, sequence, HEADS, DIM], pl.BF16], cache: pl.Tensor[[cache_rows, PACKED_BYTES], pl.UINT8], indices: pl.Tensor[[batch, sequence, SLOTS], pl.INT32], sink: pl.Tensor[[HEADS], pl.FP32], compact_scale_index: pl.Tensor[[1, NATURAL_DECODE_ROWS * GROUPS], pl.INT32], e4m3_codebook: pl.Tensor[[1, 256], pl.FP32], natural_kv: pl.InOut[pl.Tensor[[tokens * SLOTS, DIM], pl.BF16]], lse_pack_index: pl.Tensor[[1, TRANSPORT_HEADS * sequence], pl.INT32], lse_stage: pl.InOut[pl.Tensor[[tokens, HEADS], pl.FP32]], out: pl.Out[pl.Tensor[[batch, sequence, HEADS, DIM], pl.BF16]], lse: pl.Out[pl.Tensor[[batch, HEADS, sequence], pl.FP32]]):
            query_flat = pl.reshape(query, [tokens * HEADS, DIM])
            indices_flat = pl.reshape(indices, [tokens, SLOTS])
            output_flat = pl.reshape(out, [tokens * HEADS, DIM])
            lse_flat = pl.reshape(lse, [batch * HEADS, sequence])
            direct_lse_elements = pl.reshape(lse, [1, tokens * HEADS])
            for gather_dequant_work in pl.spmd(tokens * (SLOTS // 64), name_hint='gather_dequant'):
                token = gather_dequant_work // (SLOTS // 64)
                chunk = gather_dequant_work % (SLOTS // 64)
                decode_slot_fetch = chunk * 64
                gather_ids_fetch = pl.load(indices_flat, [token, decode_slot_fetch], [1, 64])
                safe_ids_fetch = pl.maximum(gather_ids_fetch, 0)
                zero_raw_fetch = pl.tile.create([1, PAYLOAD_PITCH], dtype=pl.UINT8)
                zero_raw_fetch = pl.tile.gather_row(zero_raw_fetch, cache, [0, 0], [0, 0], [1, PACKED_BYTES])
                zero_words_fetch = pl.tile.reinterpret_view(zero_raw_fetch, pl.INT32)
                replicated_words_fetch = pl.tile.create([64, PAYLOAD_PITCH // 4], dtype=pl.INT32)
                for replication_lane_fetch in pl.unroll(64):
                    replicated_words_fetch = pl.tile.assemble(replicated_words_fetch, zero_words_fetch, [replication_lane_fetch, 0])
                payload_fetch = pl.tile.reinterpret_view(replicated_words_fetch, pl.UINT8)
                for issue_group_fetch in pl.range(8):
                    for issue_offset_fetch in pl.unroll(8):
                        lane_fetch = issue_group_fetch * 8 + issue_offset_fetch
                        original_fetch = pl.tile.read(gather_ids_fetch, [0, lane_fetch])
                        if original_fetch != -1:
                            physical_fetch = pl.cast(original_fetch, pl.INDEX)
                            payload_fetch = pl.tile.gather_row(payload_fetch, cache, [lane_fetch, 0], [physical_fetch, 0], [1, PACKED_BYTES])
                codebook_values = pl.load(e4m3_codebook, [0, 0], [1, 256])
                compact_scale_map = pl.load(compact_scale_index, [0, 0], [1, NATURAL_DECODE_ROWS * GROUPS])
                for natural_chunk in pl.pipeline(64 // NATURAL_DECODE_ROWS, stage=2):
                    natural_slot = chunk * 64 + natural_chunk * NATURAL_DECODE_ROWS
                    natural_carrier_words = pl.tile.reinterpret_view(payload_fetch, pl.INT32)
                    natural_payload_word_view = pl.tile.slice(natural_carrier_words, [NATURAL_DECODE_ROWS, DIM // 4], [natural_chunk * NATURAL_DECODE_ROWS, 0])
                    natural_payload_word_mask = pl.tile.full([NATURAL_DECODE_ROWS, DIM // 4], dtype=pl.INT32, value=-1)
                    natural_payload_words = pl.tile.and_(natural_payload_word_view, natural_payload_word_mask)
                    natural_payload = pl.tile.reinterpret_view(natural_payload_words, pl.UINT8)
                    natural_half_codes = pl.cast(natural_payload, pl.FP16)
                    natural_integer_codes = pl.cast(natural_half_codes, pl.INT32, mode='rint')
                    natural_codebook_scratch = pl.tile.create([NATURAL_DECODE_ROWS, DIM], dtype=pl.INT32)
                    natural_decoded = pl.tile.gather(codebook_values, natural_integer_codes, natural_codebook_scratch)
                    natural_scale_bytes = pl.tile.slice(payload_fetch, [NATURAL_DECODE_ROWS, 32], [natural_chunk * NATURAL_DECODE_ROWS, DIM], valid_shape=[NATURAL_DECODE_ROWS, GROUPS], pad_value=0)
                    natural_scale_half = pl.cast(natural_scale_bytes, pl.FP16)
                    natural_scale_codes = pl.cast(natural_scale_half, pl.INT32, mode='rint')
                    compact_scale_scratch = pl.tile.create([1, NATURAL_DECODE_ROWS * GROUPS], dtype=pl.INT32)
                    compact_scale_codes = pl.tile.gather(natural_scale_codes, compact_scale_map, compact_scale_scratch)
                    compact_scale_bits = pl.shls(compact_scale_codes, 23)
                    compact_scale_values = pl.tile.reinterpret_view(compact_scale_bits, pl.FP32)
                    group_scale_column = pl.reshape(compact_scale_values, [NATURAL_DECODE_ROWS * GROUPS, 1])
                    grouped_decoded = pl.reshape(natural_decoded, [NATURAL_DECODE_ROWS * GROUPS, GROUP])
                    grouped_scaled = pl.row_expand_mul(grouped_decoded, group_scale_column)
                    natural_scaled = pl.reshape(grouped_scaled, [NATURAL_DECODE_ROWS, DIM])
                    natural_rounded = pl.cast(natural_scaled, pl.BF16, mode='rint')
                    pl.store(natural_rounded, [token * SLOTS + natural_slot, 0], natural_kv)
            for work in pl.spmd(tokens * (HEADS // ATTENTION_HEAD_TILE), name_hint='full_row_attention', optimizations=[pl.cross_core_slot(slot_num=4, local_slot_num=1)]):
                token = work // (HEADS // ATTENTION_HEAD_TILE)
                head = work % (HEADS // ATTENTION_HEAD_TILE) * ATTENTION_HEAD_TILE
                row = token * HEADS + head
                for setup_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    panel_maxima = pl.full([SLOTS // QK_SLOT_TILE, LANE_HEADS], dtype=pl.FP32, value=-NEG_BIG)
                    panel_sums = pl.full([SLOTS // QK_SLOT_TILE, LANE_HEADS], dtype=pl.FP32, value=0.0)
                    local_scores = pl.full([LANE_HEADS, SLOTS], dtype=pl.FP32, value=0.0)
                resident_query = query_flat[row:row + ATTENTION_HEAD_TILE, 0:DIM]
                for column in pl.range(SLOTS // QK_SLOT_TILE):
                    slot = column * QK_SLOT_TILE
                    score_accumulator = pl.create_tensor([ATTENTION_HEAD_TILE, QK_SLOT_TILE], dtype=pl.FP32)
                    for reduction in pl.range(DIM // QK_DIM_TILE):
                        query_tile = resident_query[0:ATTENTION_HEAD_TILE, reduction * QK_DIM_TILE:reduction * QK_DIM_TILE + QK_DIM_TILE]
                        key_tile = natural_kv[token * SLOTS + slot:token * SLOTS + slot + QK_SLOT_TILE, reduction * QK_DIM_TILE:reduction * QK_DIM_TILE + QK_DIM_TILE]
                        score_accumulator = pl.matmul_acc(score_accumulator, query_tile, key_tile, b_trans=True, init_cond=reduction == 0)
                    for score_lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        score_vector = pl.aiv_shard(score_accumulator)
                        score_scaled = pl.mul(score_vector, SCALE)
                        score_ids = pl.cast(indices_flat[token:token + 1, slot:slot + QK_SLOT_TILE], pl.FP32)
                        score_invalid = pl.cmp(score_ids, -1.0, cmp_type=0)
                        score_removed = pl.col_expand_mul(score_scaled, score_invalid)
                        score_masked = pl.col_expand_sub(pl.sub(score_scaled, score_removed), pl.mul(score_invalid, NEG_BIG))
                        slot_maximum = pl.reshape(pl.row_max(score_masked), [1, LANE_HEADS])
                        panel_maxima = pl.assemble(panel_maxima, slot_maximum, [column, 0])
                        panel_denominator = pl.full([1, LANE_HEADS], dtype=pl.FP32, value=0.0)
                        for statistics_head in pl.range(LANE_HEADS // SOFTMAX_HEAD_TILE):
                            statistics_offset = statistics_head * SOFTMAX_HEAD_TILE
                            statistics_maximum = slot_maximum[0:1, statistics_offset:statistics_offset + SOFTMAX_HEAD_TILE]
                            statistics_denominator = pl.full([1, SOFTMAX_HEAD_TILE], dtype=pl.FP32, value=0.0)
                            statistics_maximum_column = pl.reshape(statistics_maximum, [SOFTMAX_HEAD_TILE, 1])
                            for exponential_column in pl.range(QK_SLOT_TILE // EXP_SLOT_TILE):
                                exponential_slot = exponential_column * EXP_SLOT_TILE
                                exponential_scores = score_masked[statistics_offset:statistics_offset + SOFTMAX_HEAD_TILE, exponential_slot:exponential_slot + EXP_SLOT_TILE]
                                exponential_values = pl.exp(pl.row_expand_sub(exponential_scores, statistics_maximum_column))
                                exponential_partial = pl.reshape(pl.row_sum(exponential_values), [1, SOFTMAX_HEAD_TILE])
                                statistics_denominator = pl.add(statistics_denominator, exponential_partial)
                                local_scores = pl.assemble(local_scores, exponential_values, [statistics_offset, slot + exponential_slot])
                            panel_denominator = pl.assemble(panel_denominator, statistics_denominator, [0, statistics_offset])
                        panel_sums = pl.assemble(panel_sums, panel_denominator, [column, 0])
                for merge_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    softmax_maximum = pl.full([1, LANE_HEADS], dtype=pl.FP32, value=-NEG_BIG)
                    for merge_panel in pl.range(SLOTS // QK_SLOT_TILE):
                        merge_maximum = panel_maxima[merge_panel:merge_panel + 1, 0:LANE_HEADS]
                        softmax_maximum = pl.maximum(softmax_maximum, merge_maximum)
                    panel_corrections = pl.exp(pl.col_expand_sub(panel_maxima, softmax_maximum))
                    softmax_denominator = pl.full([1, LANE_HEADS], dtype=pl.FP32, value=0.0)
                    for merge_panel in pl.range(SLOTS // QK_SLOT_TILE):
                        merge_correction = panel_corrections[merge_panel:merge_panel + 1, 0:LANE_HEADS]
                        merge_sum = panel_sums[merge_panel:merge_panel + 1, 0:LANE_HEADS]
                        softmax_denominator = pl.add(softmax_denominator, pl.mul(merge_correction, merge_sum))
                pv_accumulator_0 = pl.create_tensor([ATTENTION_HEAD_TILE, PV_COLUMN_TILE], dtype=pl.FP32)
                for pv_reduction in pl.range(SLOTS // PV_SLOT_TILE):
                    pv_slot = pv_reduction * PV_SLOT_TILE
                    for probability_lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        pv_probability = pl.full([LANE_HEADS, PV_SLOT_TILE], dtype=pl.BF16, value=0.0)
                        for probability_subslot in pl.range(PV_SLOT_TILE // SLOT_TILE):
                            normalization_slot = pv_slot + probability_subslot * SLOT_TILE
                            for normalization_head in pl.range(LANE_HEADS // SOFTMAX_HEAD_TILE):
                                normalization_offset = normalization_head * SOFTMAX_HEAD_TILE
                                normalization_exponentials = local_scores[normalization_offset:normalization_offset + SOFTMAX_HEAD_TILE, normalization_slot:normalization_slot + SLOT_TILE]
                                normalization_denominator = pl.reshape(softmax_denominator[0:1, normalization_offset:normalization_offset + SOFTMAX_HEAD_TILE], [SOFTMAX_HEAD_TILE, 1])
                                normalization_panel = normalization_slot // QK_SLOT_TILE
                                normalization_correction = pl.reshape(panel_corrections[normalization_panel:normalization_panel + 1, normalization_offset:normalization_offset + SOFTMAX_HEAD_TILE], [SOFTMAX_HEAD_TILE, 1])
                                normalization_rebased = pl.row_expand_mul(normalization_exponentials, normalization_correction)
                                normalization_result = pl.row_expand_div(normalization_rebased, normalization_denominator)
                                normalization_rounded = pl.cast(normalization_result, pl.BF16, mode='rint')
                                pv_probability = pl.assemble(pv_probability, normalization_rounded, [normalization_offset, probability_subslot * SLOT_TILE])
                                local_scores = pl.assemble(local_scores, normalization_result, [normalization_offset, normalization_slot])
                        pv_probability_cube = pl.aic_gather(pv_probability)
                    pv_value_0 = natural_kv[token * SLOTS + pv_slot:token * SLOTS + pv_slot + PV_SLOT_TILE, 0:DIM]
                    pv_accumulator_0 = pl.matmul_acc(pv_accumulator_0, pv_probability_cube, pv_value_0, init_cond=pv_reduction == 0)
                for validity_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    fused_validity_ids = pl.cast(indices_flat[token:token + 1, 0:SLOTS], pl.FP32)
                    fused_validity_invalid = pl.cmp(fused_validity_ids, -1.0, cmp_type=0)
                    fused_partial_count = pl.col_sum(pl.reshape(fused_validity_invalid, [4, 128]))
                    fused_counts = pl.col_sum(pl.reshape(fused_partial_count, [8, 16]))
                    fused_count_carrier = pl.full([8, 16], dtype=pl.FP32, value=0.0)
                    fused_count_carrier = pl.assemble(fused_count_carrier, fused_counts, [0, 0])
                    fused_count_total = pl.row_sum(fused_count_carrier)
                    fused_invalid_count = pl.cast(pl.cast(pl.read(fused_count_total, [0, 0]), pl.INT32), pl.INDEX)
                    fused_valid_count = SLOTS - fused_invalid_count
                for epilogue_lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    pv_complete_vector = pl.aiv_shard(pv_accumulator_0)
                    pv_committed = pl.cast(pv_complete_vector, pl.BF16, mode='rint')
                    fused_finite_lse = pl.add(softmax_maximum, pl.log(softmax_denominator, high_precision=True))
                    fused_exp = pl.exp(fused_finite_lse)
                    fused_sink = pl.reshape(sink[head + epilogue_lane * LANE_HEADS:head + (epilogue_lane + 1) * LANE_HEADS], [1, LANE_HEADS])
                    fused_factor = pl.div(fused_exp, pl.add(fused_exp, pl.exp(fused_sink)), high_precision=True)
                    fused_public_lse = fused_finite_lse
                    if fused_valid_count == 0:
                        fused_public_lse = pl.reinterpret_view(pl.full([1, LANE_HEADS], dtype=pl.INT32, value=2139095040), pl.FP32)
                    lse_stage[token:token + 1, head + epilogue_lane * LANE_HEADS:head + (epilogue_lane + 1) * LANE_HEADS] = fused_public_lse
                    for fused_piece in pl.range(LANE_HEADS // EPILOGUE_HEAD_TILE):
                        fused_offset = fused_piece * EPILOGUE_HEAD_TILE
                        fused_row = row + epilogue_lane * LANE_HEADS + fused_offset
                        fused_rounded_piece = pv_committed[fused_offset:fused_offset + EPILOGUE_HEAD_TILE, 0:DIM]
                        fused_values = pl.cast(fused_rounded_piece, pl.FP32)
                        fused_factor_piece = pl.reshape(fused_factor[0:1, fused_offset:fused_offset + EPILOGUE_HEAD_TILE], [EPILOGUE_HEAD_TILE, 1])
                        fused_scaled = pl.row_expand_mul(fused_values, fused_factor_piece)
                        if fused_valid_count == 0:
                            fused_scaled = pl.full([EPILOGUE_HEAD_TILE, DIM], dtype=pl.FP32, value=0.0)
                        output_flat[fused_row:fused_row + EPILOGUE_HEAD_TILE, 0:DIM] = pl.cast(fused_scaled, pl.BF16, mode='rint')
            terminal_lse_flat = pl.reshape(lse, [1, tokens * HEADS])
            for transport_owner in pl.spmd(batch * (HEADS // TRANSPORT_HEADS), name_hint='bulk_lse_pack'):
                transport_batch = transport_owner // (HEADS // TRANSPORT_HEADS)
                transport_head = transport_owner % (HEADS // TRANSPORT_HEADS) * TRANSPORT_HEADS
                transport_rectangle = pl.load(lse_stage, [transport_batch * sequence, transport_head], [sequence, TRANSPORT_HEADS])
                transport_permutation = pl.load(lse_pack_index, [0, 0], [1, TRANSPORT_HEADS * sequence])
                transport_scratch = pl.tile.create([1, TRANSPORT_HEADS * sequence], dtype=pl.INT32)
                transport_packed = pl.tile.gather(transport_rectangle, transport_permutation, transport_scratch)
                pl.store(transport_packed, [0, (transport_batch * HEADS + transport_head) * sequence], terminal_lse_flat)
            return (out, lse)

    def pack_cache():
        values = torch.randn(cache_rows, DIM)
        maximum = values.abs().view(cache_rows, GROUPS, GROUP).amax(dim=-1)
        scales = torch.pow(2.0, torch.clamp_min(maximum / 448.0, 0.0001).log2().ceil())
        quantized = (values.view(cache_rows, GROUPS, GROUP) / scales.unsqueeze(-1)).view(cache_rows, DIM).to(torch.float8_e4m3fn)
        scale_bytes = (scales.log2() + 127.0).to(torch.int32).to(torch.uint8)
        return torch.cat([quantized.view(torch.uint8), scale_bytes], dim=1).contiguous()

    def init_indices():
        result = torch.full((batch, sequence, SLOTS), -1, dtype=torch.int32)
        for batch_index in range(batch):
            for sequence_index in range(sequence):
                if tokens > 1 and batch_index == 0 and (sequence_index == 0):
                    continue
                valid_count = int(torch.randint(1, SLOTS + 1, (1,)))
                positions = torch.randperm(SLOTS)[:valid_count]
                rows = torch.randint(cache_rows, (valid_count,), dtype=torch.int32)
                if valid_count > 1:
                    rows[1] = rows[0]
                result[batch_index, sequence_index, positions] = rows
        return result

    def scale_indices():
        groups = torch.arange(GROUPS)
        pattern = groups % 4 * (GATHER_TILE * 8) + groups // 4
        return (torch.arange(GATHER_TILE).view(GATHER_TILE, 1) * 8 + pattern.view(1, GROUPS)).to(torch.int32).reshape(1, GATHER_TILE * GROUPS)

    def tensor_specs():
        dimensions = torch.arange(DIM)
        columns = dimensions % 128
        planes = dimensions // 128
        specs = [TensorSpec('query', [batch, sequence, HEADS, DIM], torch.bfloat16, init_value=torch.randn), TensorSpec('cache', [cache_rows, PACKED_BYTES], torch.uint8, init_value=pack_cache), TensorSpec('indices', [batch, sequence, SLOTS], torch.int32, init_value=init_indices), TensorSpec('sink', [HEADS], torch.float32, init_value=torch.randn), TensorSpec('compact_scale_index', [1, NATURAL_DECODE_ROWS * GROUPS], torch.int32, init_value=(torch.arange(NATURAL_DECODE_ROWS).view(-1, 1) * 32 + torch.arange(GROUPS).view(1, -1)).reshape(1, NATURAL_DECODE_ROWS * GROUPS).to(torch.int32)), TensorSpec('e4m3_codebook', [1, 256], torch.float32, init_value=torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().reshape(1, 256)), TensorSpec('natural_kv', [tokens * SLOTS, DIM], torch.bfloat16), TensorSpec('out', [batch, sequence, HEADS, DIM], torch.bfloat16), TensorSpec('lse', [batch, HEADS, sequence], torch.float32)]

        if sequence != 1:
            specs[-2:-2] = [
                TensorSpec('lse_pack_index', [1, TRANSPORT_HEADS * sequence], torch.int32, init_value=(torch.arange(TRANSPORT_HEADS).view(TRANSPORT_HEADS, 1) + torch.arange(sequence).view(1, sequence) * TRANSPORT_HEADS).reshape(1, TRANSPORT_HEADS * sequence).to(torch.int32)),
                TensorSpec('lse_stage', [tokens, HEADS], torch.float32),
            ]
        return specs

    def golden(tensors):
        raw = tensors['cache']
        data = raw[:, :DIM].view(torch.float8_e4m3fn).float()
        scales = (raw[:, DIM:].to(torch.int32) << 23).view(torch.float32)
        decoded = (data.view(cache_rows, GROUPS, GROUP) * scales.unsqueeze(-1)).view(cache_rows, DIM)
        decoded_bf16 = decoded.to(torch.bfloat16)
        ids = tensors['indices'].view(tokens, SLOTS).long()
        invalid = ids == -1
        safe = ids.masked_fill(invalid, 0)
        gathered = decoded_bf16[safe].float()
        queries = tensors['query'].view(tokens, HEADS, DIM).float()
        scores = torch.einsum('thd,tkd->thk', queries, gathered) * SCALE
        scores = scores.masked_fill(invalid.unsqueeze(1), -NEG_BIG)
        maximum = scores.max(dim=-1).values
        exponentials = torch.exp(scores - maximum.unsqueeze(-1))
        denominator = exponentials.sum(dim=-1)
        probabilities = (exponentials / denominator.unsqueeze(-1)).to(torch.bfloat16)
        weighted = torch.einsum('thk,tkd->thd', probabilities.float(), gathered).to(torch.bfloat16)
        lse_rows = maximum + torch.log(denominator)
        exponentials_lse = torch.exp(lse_rows)
        factor = exponentials_lse / (exponentials_lse + torch.exp(tensors['sink'].view(1, HEADS)))
        output = weighted.float() * factor.unsqueeze(-1)
        valid_count = (~invalid).sum(dim=-1)
        lse_rows = torch.where((valid_count == 0).unsqueeze(-1), torch.full_like(lse_rows, float('inf')), lse_rows)
        tensors['natural_kv'][:] = gathered.reshape(tokens * SLOTS, DIM).to(torch.bfloat16)
        tensors['out'][:] = output.reshape(batch, sequence, HEADS, DIM).to(torch.bfloat16)
        tensors['lse'][:] = lse_rows.reshape(batch, sequence, HEADS).transpose(1, 2)
        if sequence != 1:
            tensors['lse_stage'][:] = lse_rows
    return (dsa41_layout_softmax_factorial, tensor_specs(), golden, RTOL, ATOL)
