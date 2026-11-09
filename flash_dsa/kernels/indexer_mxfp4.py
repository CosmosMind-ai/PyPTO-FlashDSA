"""Paged MXFP4 indexer with shared decoding and a live FP32 score epilogue."""
import pypto.language as pl

RTOL = 2e-2
ATOL = 2e-2
H = 32
D = 128
PAGE = 128
DECODE_ROWS = 128
MICRO_ROWS = 64
PARTITION_MICRO_ROWS = 128
DECODE_CHUNKS = 4
DECODE_LUT_SLOTS = 4
G = 32
_FP4_MAGS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_FP4_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def _quantize_mxfp4_rows(rows):
    """Pack group-32 E2M1 values with upward-rounded E8M0 scales."""
    import torch
    grouped = rows.float().unflatten(-1, (-1, G))
    amax = grouped.abs().amax(dim=-1)
    raw = (amax / 6.0).clamp_min(2.0 ** -126)
    exponent = torch.ceil(torch.log2(raw)).clamp(-127, 128)
    scale = torch.exp2(exponent)
    norm = (grouped / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    table = torch.tensor(_FP4_MAGS)
    order = torch.tensor(_FP4_ORDER)
    sel = (norm.abs().unsqueeze(-1) - table[order]).abs().argmin(dim=-1)
    code = order[sel] | (torch.signbit(norm).to(torch.uint8) << 3)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).flatten(-2).to(torch.uint8)
    return packed, (exponent + 127).to(torch.uint8)


def _dequantize_mxfp4_rows(packed, scale_bytes):
    """Decode packed E2M1 pairs and apply E8M0 group scales in FP32."""
    import torch

    table = torch.tensor(_FP4_MAGS)
    code_lo = (packed & 0x0F).long()
    code_hi = (packed >> 4).long()
    mag_lo, mag_hi = table[code_lo & 7], table[code_hi & 7]
    val_lo = torch.where((code_lo & 8) != 0, -mag_lo, mag_lo)
    val_hi = torch.where((code_hi & 8) != 0, -mag_hi, mag_hi)
    vals = torch.stack((val_lo, val_hi), dim=-1).flatten(-2)
    scales = torch.exp2(scale_bytes.float() - 127.0)
    return vals * scales.repeat_interleave(G, dim=-1)

QUERY_PER_WORKER = 4
QUERY_TILE = 4
COL_TILE = 256
PACKET_PAGES = 2
FRAGMENT_COLS = 256
QUERY_GROUP = 4
PAGE_PARTS = 3
LANE_QUERIES = 2
LANE_COLS = 256
PAGE_SPAN = 256
CROSS_CORE_SLOTS = 2
OWNERSHIP_MODE = 2
OWNER_PACKET_STRIDE = 3
OWNER_RUN_PACKETS = 4

@pl.jit
def dsa41_indexer_mxfp4_fused(
    q: pl.Tensor,
    cache: pl.Tensor,
    scales: pl.Tensor,
    block_table: pl.Tensor,
    lengths: pl.Tensor,
    w: pl.Tensor,
    pidx: pl.Tensor,
    sidx: pl.Tensor,
    dots: pl.InOut[pl.Tensor],
    score: pl.InOut[pl.Tensor],
):
    M = q.shape[0]
    N = score.shape[1]
    NB = cache.shape[0]
    P = block_table.shape[1]
    q2 = pl.reshape(q, [M * H, D])
    packed2 = pl.reshape(cache, [NB * PAGE, D // 2])
    scales_w = pl.reshape(scales, [NB * 16, 32])
    kb2 = pl.create_tensor([NB * PAGE, D], dtype=pl.BF16)
    decode_chunks = NB * PAGE // DECODE_ROWS
    kb_pages = pl.reshape(kb2, [decode_chunks, DECODE_ROWS * D])
    for decode_task in pl.spmd((decode_chunks + DECODE_CHUNKS - 1) // DECODE_CHUNKS, name_hint="mxfp4_decode_persistent"):
        for decode_visit in pl.range(DECODE_CHUNKS):
            chunk_idx = decode_task * DECODE_CHUNKS + decode_visit
            if chunk_idx < decode_chunks:
                decoded_page = pl.tile.create([DECODE_ROWS, D], dtype=pl.BF16)
                for micro_visit in pl.range(DECODE_ROWS // MICRO_ROWS):
                    micro_base = chunk_idx * DECODE_ROWS + micro_visit * MICRO_ROWS
                    packed_micro = pl.load(packed2, [micro_base, 0], [MICRO_ROWS, D // 2])
                    byte_indices = pl.cast(pl.cast(packed_micro, pl.FP16), pl.INT32)
                    pair_low = pl.tile.and_(byte_indices, pl.tile.full([MICRO_ROWS, D // 2], dtype=pl.INT32, value=15))
                    pair_high = pl.shls(pl.tile.and_(byte_indices, pl.tile.full([MICRO_ROWS, D // 2], dtype=pl.INT32, value=240)), 12)
                    pair_spread = pl.or_(pair_low, pair_high)
                    pair_magnitudes = pl.tile.and_(pair_spread, pl.tile.full([MICRO_ROWS, D // 2], dtype=pl.INT32, value=458759))
                    pair_magnitude_bits = pl.shls(pair_magnitudes, 9)
                    pair_signs = pl.shls(pl.tile.and_(pair_spread, pl.tile.full([MICRO_ROWS, D // 2], dtype=pl.INT32, value=524296)), 12)
                    pair_words = pl.or_(pair_magnitude_bits, pair_signs)
                    unscaled_pairs = pl.tile.reinterpret_view(pair_words, pl.FP16)
                    biased_values = pl.cast(unscaled_pairs, pl.FP32)
                    unscaled_values = pl.mul(biased_values, 16384.0)
                    grouped_values = pl.reshape(unscaled_values, [MICRO_ROWS * 4, 32])
                    scale_bytes = pl.load(scales_w, [micro_base // 8, 0], [MICRO_ROWS // 8, 32])
                    scale_codes = pl.reshape(pl.cast(pl.cast(scale_bytes, pl.FP16), pl.INT32), [1, MICRO_ROWS * 4])
                    scale_normal = pl.tile.reinterpret_view(pl.shls(scale_codes, 23), pl.FP32)
                    flat_scales = pl.tile.select(pl.tile.cmps(scale_codes, 0, cmp_type=0), 5.877471754111438e-39, scale_normal)
                    group_scales = pl.reshape(flat_scales, [MICRO_ROWS * 4, 1])
                    scaled_values = pl.row_expand_mul(grouped_values, group_scales)
                    final_values = pl.reshape(pl.cast(scaled_values, pl.BF16, mode="rint"), [MICRO_ROWS, D])
                    decoded_page = pl.tile.assemble(decoded_page, final_values, [micro_visit * MICRO_ROWS, 0])
                decoded_flat = pl.reshape(decoded_page, [1, DECODE_ROWS * D])
                pl.store(decoded_flat, [chunk_idx, 0], kb_pages)
    for block_idx in pl.spmd(M // (QUERY_GROUP * 2) * PAGE_PARTS,
            name_hint="persistent_grouped_score",
            optimizations=[pl.cross_core_slot(slot_num=3, local_slot_num=1)]):
        super_query_base = (block_idx % (M // (QUERY_GROUP * 2))) * QUERY_GROUP * 2
        packet_count = (P + PACKET_PAGES - 1) // PACKET_PAGES
        packets_per_partition = (packet_count + PAGE_PARTS - 1) // PAGE_PARTS
        packet_base = (block_idx // (M // (QUERY_GROUP * 2))) * packets_per_partition
        packet_end = pl.min(packet_base + packets_per_partition, packet_count)
        resident_queries = pl.load(q2, [super_query_base * H, 0], [QUERY_GROUP * 2 * H, D], target_memory=pl.Mem.Mat)
        group_visible_0 = pl.min(pl.max(pl.read(lengths, [super_query_base + 0]), 0), N)
        group_visible_1 = pl.min(pl.max(pl.read(lengths, [super_query_base + 1]), 0), N)
        group_visible_2 = pl.min(pl.max(pl.read(lengths, [super_query_base + 2]), 0), N)
        group_visible_3 = pl.min(pl.max(pl.read(lengths, [super_query_base + 3]), 0), N)
        group_visible_4 = pl.min(pl.max(pl.read(lengths, [super_query_base + 4]), 0), N)
        group_visible_5 = pl.min(pl.max(pl.read(lengths, [super_query_base + 5]), 0), N)
        group_visible_6 = pl.min(pl.max(pl.read(lengths, [super_query_base + 6]), 0), N)
        group_visible_7 = pl.min(pl.max(pl.read(lengths, [super_query_base + 7]), 0), N)
        group_max_0 = pl.max(pl.max(group_visible_0, group_visible_1), pl.max(group_visible_2, group_visible_3))
        group_max_1 = pl.max(pl.max(group_visible_4, group_visible_5), pl.max(group_visible_6, group_visible_7))
        for metadata_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            lane_query_base = super_query_base + metadata_lane * LANE_QUERIES
            lane_visible0 = pl.min(pl.max(pl.read(lengths, [lane_query_base]), 0), N)
            lane_visible1 = pl.min(pl.max(pl.read(lengths, [lane_query_base + 1]), 0), N)
            lane_visible2 = pl.min(pl.max(pl.read(lengths, [lane_query_base + QUERY_GROUP]), 0), N)
            lane_visible3 = pl.min(pl.max(pl.read(lengths, [lane_query_base + QUERY_GROUP + 1]), 0), N)
            all_weights = pl.reshape(pl.load(w, [super_query_base, 0], [QUERY_GROUP * 2, H]), [QUERY_GROUP * 2 * H, 1])
            negative_inf = pl.reinterpret_view(pl.tile.full([1, FRAGMENT_COLS], dtype=pl.INT32, value=-8388608), pl.FP32)
            position_offset = pl.cast(pl.tile.arange(0, [1, FRAGMENT_COLS], dtype=pl.INT32), pl.FP32)
        local_packet_count = pl.max(packet_end - packet_base, 0)
        packet_phase = (block_idx % (M // (QUERY_GROUP * 2))) % 2
        packet_offset = (local_packet_count * packet_phase) // 2
        for visit_idx, (cursor_packet,) in pl.range(local_packet_count, init_values=(packet_base + packet_offset,)):
            packet_idx = cursor_packet
            page_idx = packet_idx * PACKET_PAGES
            valid_cols = pl.min(COL_TILE, N - page_idx * PAGE)
            physical_page = pl.read(block_table, [0, page_idx])
            cache_row = pl.cast(physical_page, pl.INDEX) * PAGE
            key_tile = pl.tile.create([COL_TILE, D], pl.BF16, target_memory=pl.Mem.Mat)
            key_tile = pl.tile.gather_row(key_tile, kb2, [0, 0], [cache_row, 0], [PAGE, D])
            second_page = pl.min(page_idx + 1, P - 1)
            second_physical = pl.read(block_table, [0, second_page])
            second_row = pl.cast(second_physical, pl.INDEX) * PAGE
            key_tile = pl.tile.gather_row(key_tile, kb2, [PAGE, 0], [second_row, 0], [PAGE, D])
            key_right = pl.tile.move(pl.tile.transpose_view(key_tile), target_memory=pl.Mem.Right)
            for subgroup_idx in pl.range(2):
                query_base = super_query_base + subgroup_idx * QUERY_GROUP
                subgroup_visible = group_max_0 + subgroup_idx * (group_max_1 - group_max_0)
                subgroup_packet_end = (subgroup_visible + COL_TILE - 1) // COL_TILE
                query_tile = pl.tile.slice(resident_queries, [QUERY_GROUP * H, D], [subgroup_idx * QUERY_GROUP * H, 0])
                grouped_dots = pl.matmul(query_tile, key_right, out_dtype=pl.FP32)
                if packet_idx < subgroup_packet_end:
                    for transfer_lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        sharded_dots = pl.aiv_shard(grouped_dots)
                    canonical_dots = pl.tile.slice(grouped_dots, [QUERY_GROUP * H, COL_TILE], [0, 0], valid_shape=[QUERY_GROUP * H, valid_cols])
                    pl.store(canonical_dots, [query_base * H, page_idx * PAGE], dots)
                    for vector_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                        visible_first = lane_visible0 + subgroup_idx * (lane_visible2 - lane_visible0)
                        visible_second = lane_visible1 + subgroup_idx * (lane_visible3 - lane_visible1)
                        grouped_weights = pl.tile.slice(all_weights, [LANE_QUERIES * H, 1], [(subgroup_idx * QUERY_GROUP + vector_lane * LANE_QUERIES) * H, 0])
                        output_query_base = query_base + vector_lane * LANE_QUERIES
                        query0_weights = pl.tile.slice(grouped_weights, [H, 1], [0, 0])
                        query0_frag0_raw = pl.tile.slice(sharded_dots, [H, FRAGMENT_COLS], [0, 0])
                        query0_frag0_activation = pl.maximum(query0_frag0_raw, 0.0)
                        query0_frag0_weighted = pl.row_expand_mul(query0_frag0_activation, query0_weights)
                        query0_frag0_scratch: pl.Tile[[H, 128], pl.FP32, pl.MemRef("head_tree_scratch"), pl.Mem.Vec] = pl.tile.create([H, 128], pl.FP32, target_memory=pl.Mem.Vec)
                        query0_frag0_half0 = pl.tile.slice(query0_frag0_weighted, [H, 128], [0, 0], valid_shape=[H, 128])
                        query0_frag0_sum0 = pl.tile.col_sum(query0_frag0_half0, query0_frag0_scratch)
                        query0_frag0_half1 = pl.tile.slice(query0_frag0_weighted, [H, 128], [0, 128], valid_shape=[H, 128])
                        query0_frag0_sum1 = pl.tile.col_sum(query0_frag0_half1, query0_frag0_scratch)
                        query0_frag0_reduced = pl.tile.create([1, FRAGMENT_COLS], pl.FP32, target_memory=pl.Mem.Vec)
                        query0_frag0_reduced = pl.tile.assemble(query0_frag0_reduced, query0_frag0_sum0, [0, 0])
                        query0_frag0_reduced = pl.tile.assemble(query0_frag0_reduced, query0_frag0_sum1, [0, 128])
                        if visible_first >= page_idx * PAGE + valid_cols:
                            query0_frag0_masked = query0_frag0_reduced
                        else:
                            query0_frag0_positions = pl.add(position_offset, pl.cast(pl.cast(page_idx * PAGE + 0, pl.INT32), pl.FP32))
                            query0_frag0_masked = pl.tile.select(pl.tile.cmps(query0_frag0_positions, pl.cast(pl.cast(visible_first, pl.INT32), pl.FP32), cmp_type=2), query0_frag0_reduced, negative_inf)
                        query0_score_stage = pl.tile.create([1, COL_TILE], pl.FP32, target_memory=pl.Mem.Vec)
                        query0_score_stage = pl.tile.assemble(query0_score_stage, query0_frag0_masked, [0, 0])
                        query0_output = pl.tile.slice(query0_score_stage, [1, COL_TILE], [0, 0], valid_shape=[1, valid_cols])
                        pl.store(query0_output, [output_query_base + 0, page_idx * PAGE], score)
                        query1_weights = pl.tile.slice(grouped_weights, [H, 1], [32, 0])
                        query1_frag0_raw = pl.tile.slice(sharded_dots, [H, FRAGMENT_COLS], [32, 0])
                        query1_frag0_activation = pl.maximum(query1_frag0_raw, 0.0)
                        query1_frag0_weighted = pl.row_expand_mul(query1_frag0_activation, query1_weights)
                        query1_frag0_scratch: pl.Tile[[H, 128], pl.FP32, pl.MemRef("head_tree_scratch"), pl.Mem.Vec] = pl.tile.create([H, 128], pl.FP32, target_memory=pl.Mem.Vec)
                        query1_frag0_half0 = pl.tile.slice(query1_frag0_weighted, [H, 128], [0, 0], valid_shape=[H, 128])
                        query1_frag0_sum0 = pl.tile.col_sum(query1_frag0_half0, query1_frag0_scratch)
                        query1_frag0_half1 = pl.tile.slice(query1_frag0_weighted, [H, 128], [0, 128], valid_shape=[H, 128])
                        query1_frag0_sum1 = pl.tile.col_sum(query1_frag0_half1, query1_frag0_scratch)
                        query1_frag0_reduced = pl.tile.create([1, FRAGMENT_COLS], pl.FP32, target_memory=pl.Mem.Vec)
                        query1_frag0_reduced = pl.tile.assemble(query1_frag0_reduced, query1_frag0_sum0, [0, 0])
                        query1_frag0_reduced = pl.tile.assemble(query1_frag0_reduced, query1_frag0_sum1, [0, 128])
                        if visible_second >= page_idx * PAGE + valid_cols:
                            query1_frag0_masked = query1_frag0_reduced
                        else:
                            query1_frag0_positions = pl.add(position_offset, pl.cast(pl.cast(page_idx * PAGE + 0, pl.INT32), pl.FP32))
                            query1_frag0_masked = pl.tile.select(pl.tile.cmps(query1_frag0_positions, pl.cast(pl.cast(visible_second, pl.INT32), pl.FP32), cmp_type=2), query1_frag0_reduced, negative_inf)
                        query1_score_stage = pl.tile.create([1, COL_TILE], pl.FP32, target_memory=pl.Mem.Vec)
                        query1_score_stage = pl.tile.assemble(query1_score_stage, query1_frag0_masked, [0, 0])
                        query1_output = pl.tile.slice(query1_score_stage, [1, COL_TILE], [0, 0], valid_shape=[1, valid_cols])
                        pl.store(query1_output, [output_query_base + 1, page_idx * PAGE], score)
                else:
                    canonical_dots = pl.tile.slice(grouped_dots, [QUERY_GROUP * H, COL_TILE], [0, 0], valid_shape=[QUERY_GROUP * H, valid_cols])
                    pl.store(canonical_dots, [query_base * H, page_idx * PAGE], dots)
                    for tail_vector_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                        tail_query_base = query_base + tail_vector_lane * LANE_QUERIES
                        tail_negative_inf = pl.reinterpret_view(pl.tile.full([1, COL_TILE], dtype=pl.INT32, value=-8388608), pl.FP32)
                        tail_score = pl.tile.slice(tail_negative_inf, [1, COL_TILE], [0, 0], valid_shape=[1, valid_cols])
                        for tail_local_query in pl.range(LANE_QUERIES):
                            pl.store(tail_score, [tail_query_base + tail_local_query, page_idx * PAGE], score)
            cursor_packet = packet_idx + 1
            if cursor_packet >= packet_end:
                cursor_packet = packet_base
            cursor_packet = pl.yield_(cursor_packet)
    return score


@pl.jit
def dsa41_indexer_mxfp4_partitioned(
    q: pl.Tensor,
    cache: pl.Tensor,
    scales: pl.Tensor,
    block_table: pl.Tensor,
    lengths: pl.Tensor,
    w: pl.Tensor,
    pidx: pl.Tensor,
    sidx: pl.Tensor,
    dots: pl.InOut[pl.Tensor],
    score: pl.InOut[pl.Tensor],
):
    M = q.shape[0]
    N = score.shape[1]
    NB = cache.shape[0]
    P = block_table.shape[1]
    q2 = pl.reshape(q, [M * H, D])
    packed2 = pl.reshape(cache, [NB * PAGE, D // 2])
    scales_w = pl.reshape(scales, [NB * 16, 32])
    packet_count = (P + PACKET_PAGES - 1) // PACKET_PAGES
    packets_per_partition = (packet_count + PAGE_PARTS - 1) // PAGE_PARTS
    owner_packet_base_0 = 0 * OWNER_RUN_PACKETS
    owner_packet_count_0 = (packet_count // (PAGE_PARTS * OWNER_RUN_PACKETS)) * OWNER_RUN_PACKETS + pl.min(OWNER_RUN_PACKETS, pl.max(0, packet_count % (PAGE_PARTS * OWNER_RUN_PACKETS) - owner_packet_base_0))
    owner_last_packet_0 = owner_packet_base_0 + ((owner_packet_count_0 - 1) // OWNER_RUN_PACKETS) * PAGE_PARTS * OWNER_RUN_PACKETS + (owner_packet_count_0 - 1) % OWNER_RUN_PACKETS
    owner_page_count_0 = owner_packet_count_0 * PACKET_PAGES - pl.max(0, (owner_last_packet_0 + 1) * PACKET_PAGES - P)
    decode_chunks_0 = owner_page_count_0 * (PAGE // DECODE_ROWS)
    kb2_0 = pl.create_tensor([owner_page_count_0 * PAGE, D], dtype=pl.BF16)
    owner_packet_base_1 = 1 * OWNER_RUN_PACKETS
    owner_packet_count_1 = (packet_count // (PAGE_PARTS * OWNER_RUN_PACKETS)) * OWNER_RUN_PACKETS + pl.min(OWNER_RUN_PACKETS, pl.max(0, packet_count % (PAGE_PARTS * OWNER_RUN_PACKETS) - owner_packet_base_1))
    owner_last_packet_1 = owner_packet_base_1 + ((owner_packet_count_1 - 1) // OWNER_RUN_PACKETS) * PAGE_PARTS * OWNER_RUN_PACKETS + (owner_packet_count_1 - 1) % OWNER_RUN_PACKETS
    owner_page_count_1 = owner_packet_count_1 * PACKET_PAGES - pl.max(0, (owner_last_packet_1 + 1) * PACKET_PAGES - P)
    decode_chunks_1 = owner_page_count_1 * (PAGE // DECODE_ROWS)
    kb2_1 = pl.create_tensor([owner_page_count_1 * PAGE, D], dtype=pl.BF16)
    owner_packet_base_2 = 2 * OWNER_RUN_PACKETS
    owner_packet_count_2 = (packet_count // (PAGE_PARTS * OWNER_RUN_PACKETS)) * OWNER_RUN_PACKETS + pl.min(OWNER_RUN_PACKETS, pl.max(0, packet_count % (PAGE_PARTS * OWNER_RUN_PACKETS) - owner_packet_base_2))
    owner_last_packet_2 = owner_packet_base_2 + ((owner_packet_count_2 - 1) // OWNER_RUN_PACKETS) * PAGE_PARTS * OWNER_RUN_PACKETS + (owner_packet_count_2 - 1) % OWNER_RUN_PACKETS
    owner_page_count_2 = owner_packet_count_2 * PACKET_PAGES - pl.max(0, (owner_last_packet_2 + 1) * PACKET_PAGES - P)
    decode_chunks_2 = owner_page_count_2 * (PAGE // DECODE_ROWS)
    kb2_2 = pl.create_tensor([owner_page_count_2 * PAGE, D], dtype=pl.BF16)
    with pl.manual_scope():
        with pl.spmd((decode_chunks_0 + DECODE_CHUNKS - 1) // DECODE_CHUNKS, name_hint="partition_decode_0", allow_early_resolve=False) as decode_ready_0:
            decode_task_decoder_0 = pl.tile.get_block_idx()
            for decode_visit_decoder_0 in pl.range(DECODE_CHUNKS):
                chunk_idx_decoder_0 = decode_task_decoder_0 * DECODE_CHUNKS + decode_visit_decoder_0
                if chunk_idx_decoder_0 < decode_chunks_0:
                    local_page_decoder_0 = chunk_idx_decoder_0 // (PAGE // DECODE_ROWS)
                    local_packet_decoder_0 = local_page_decoder_0 // PACKET_PAGES
                    logical_packet_decoder_0 = owner_packet_base_0 + (local_packet_decoder_0 // OWNER_RUN_PACKETS) * PAGE_PARTS * OWNER_RUN_PACKETS + local_packet_decoder_0 % OWNER_RUN_PACKETS
                    logical_page_decoder_0 = logical_packet_decoder_0 * PACKET_PAGES + local_page_decoder_0 % PACKET_PAGES
                    physical_page_decoder_0 = pl.read(block_table, [0, logical_page_decoder_0])
                    physical_chunk_decoder_0 = pl.cast(physical_page_decoder_0, pl.INDEX) * (PAGE // DECODE_ROWS) + chunk_idx_decoder_0 % (PAGE // DECODE_ROWS)
                    for micro_visit_decoder_0 in pl.range(DECODE_ROWS // PARTITION_MICRO_ROWS):
                        micro_base_decoder_0 = physical_chunk_decoder_0 * DECODE_ROWS + micro_visit_decoder_0 * PARTITION_MICRO_ROWS
                        packed_micro_decoder_0 = pl.load(packed2, [micro_base_decoder_0, 0], [PARTITION_MICRO_ROWS, D // 2])
                        byte_indices_decoder_0 = pl.cast(pl.cast(packed_micro_decoder_0, pl.FP16), pl.INT32)
                        pair_low_decoder_0 = pl.tile.and_(byte_indices_decoder_0, pl.tile.full([PARTITION_MICRO_ROWS, D // 2], dtype=pl.INT32, value=15))
                        pair_high_decoder_0 = pl.shls(pl.tile.and_(byte_indices_decoder_0, pl.tile.full([PARTITION_MICRO_ROWS, D // 2], dtype=pl.INT32, value=240)), 12)
                        pair_spread_decoder_0 = pl.or_(pair_low_decoder_0, pair_high_decoder_0)
                        pair_magnitudes_decoder_0 = pl.tile.and_(pair_spread_decoder_0, pl.tile.full([PARTITION_MICRO_ROWS, D // 2], dtype=pl.INT32, value=458759))
                        pair_magnitude_bits_decoder_0 = pl.shls(pair_magnitudes_decoder_0, 9)
                        pair_signs_decoder_0 = pl.shls(pl.tile.and_(pair_spread_decoder_0, pl.tile.full([PARTITION_MICRO_ROWS, D // 2], dtype=pl.INT32, value=524296)), 12)
                        pair_words_decoder_0 = pl.or_(pair_magnitude_bits_decoder_0, pair_signs_decoder_0)
                        unscaled_pairs_decoder_0 = pl.tile.reinterpret_view(pair_words_decoder_0, pl.FP16)
                        biased_values_decoder_0 = pl.cast(unscaled_pairs_decoder_0, pl.FP32)
                        unscaled_values_decoder_0 = pl.mul(biased_values_decoder_0, 16384.0)
                        grouped_values_decoder_0 = pl.reshape(unscaled_values_decoder_0, [PARTITION_MICRO_ROWS * 4, 32])
                        scale_bytes_decoder_0 = pl.load(scales_w, [micro_base_decoder_0 // 8, 0], [PARTITION_MICRO_ROWS // 8, 32])
                        scale_codes_decoder_0 = pl.reshape(pl.cast(pl.cast(scale_bytes_decoder_0, pl.FP16), pl.INT32), [1, PARTITION_MICRO_ROWS * 4])
                        scale_normal_decoder_0 = pl.tile.reinterpret_view(pl.shls(scale_codes_decoder_0, 23), pl.FP32)
                        flat_scales_decoder_0 = pl.tile.select(pl.tile.cmps(scale_codes_decoder_0, 0, cmp_type=0), 5.877471754111438e-39, scale_normal_decoder_0)
                        group_scales_decoder_0 = pl.reshape(flat_scales_decoder_0, [PARTITION_MICRO_ROWS * 4, 1])
                        scaled_values_decoder_0 = pl.row_expand_mul(grouped_values_decoder_0, group_scales_decoder_0)
                        final_values_decoder_0 = pl.reshape(pl.cast(scaled_values_decoder_0, pl.BF16, mode="rint"), [PARTITION_MICRO_ROWS, D])
                        pl.store(final_values_decoder_0, [chunk_idx_decoder_0 * DECODE_ROWS + micro_visit_decoder_0 * PARTITION_MICRO_ROWS, 0], kb2_0)
        with pl.spmd((decode_chunks_1 + DECODE_CHUNKS - 1) // DECODE_CHUNKS, name_hint="partition_decode_1", allow_early_resolve=False) as decode_ready_1:
            decode_task_decoder_1 = pl.tile.get_block_idx()
            for decode_visit_decoder_1 in pl.range(DECODE_CHUNKS):
                chunk_idx_decoder_1 = decode_task_decoder_1 * DECODE_CHUNKS + decode_visit_decoder_1
                if chunk_idx_decoder_1 < decode_chunks_1:
                    local_page_decoder_1 = chunk_idx_decoder_1 // (PAGE // DECODE_ROWS)
                    local_packet_decoder_1 = local_page_decoder_1 // PACKET_PAGES
                    logical_packet_decoder_1 = owner_packet_base_1 + (local_packet_decoder_1 // OWNER_RUN_PACKETS) * PAGE_PARTS * OWNER_RUN_PACKETS + local_packet_decoder_1 % OWNER_RUN_PACKETS
                    logical_page_decoder_1 = logical_packet_decoder_1 * PACKET_PAGES + local_page_decoder_1 % PACKET_PAGES
                    physical_page_decoder_1 = pl.read(block_table, [0, logical_page_decoder_1])
                    physical_chunk_decoder_1 = pl.cast(physical_page_decoder_1, pl.INDEX) * (PAGE // DECODE_ROWS) + chunk_idx_decoder_1 % (PAGE // DECODE_ROWS)
                    for micro_visit_decoder_1 in pl.range(DECODE_ROWS // PARTITION_MICRO_ROWS):
                        micro_base_decoder_1 = physical_chunk_decoder_1 * DECODE_ROWS + micro_visit_decoder_1 * PARTITION_MICRO_ROWS
                        packed_micro_decoder_1 = pl.load(packed2, [micro_base_decoder_1, 0], [PARTITION_MICRO_ROWS, D // 2])
                        byte_indices_decoder_1 = pl.cast(pl.cast(packed_micro_decoder_1, pl.FP16), pl.INT32)
                        pair_low_decoder_1 = pl.tile.and_(byte_indices_decoder_1, pl.tile.full([PARTITION_MICRO_ROWS, D // 2], dtype=pl.INT32, value=15))
                        pair_high_decoder_1 = pl.shls(pl.tile.and_(byte_indices_decoder_1, pl.tile.full([PARTITION_MICRO_ROWS, D // 2], dtype=pl.INT32, value=240)), 12)
                        pair_spread_decoder_1 = pl.or_(pair_low_decoder_1, pair_high_decoder_1)
                        pair_magnitudes_decoder_1 = pl.tile.and_(pair_spread_decoder_1, pl.tile.full([PARTITION_MICRO_ROWS, D // 2], dtype=pl.INT32, value=458759))
                        pair_magnitude_bits_decoder_1 = pl.shls(pair_magnitudes_decoder_1, 9)
                        pair_signs_decoder_1 = pl.shls(pl.tile.and_(pair_spread_decoder_1, pl.tile.full([PARTITION_MICRO_ROWS, D // 2], dtype=pl.INT32, value=524296)), 12)
                        pair_words_decoder_1 = pl.or_(pair_magnitude_bits_decoder_1, pair_signs_decoder_1)
                        unscaled_pairs_decoder_1 = pl.tile.reinterpret_view(pair_words_decoder_1, pl.FP16)
                        biased_values_decoder_1 = pl.cast(unscaled_pairs_decoder_1, pl.FP32)
                        unscaled_values_decoder_1 = pl.mul(biased_values_decoder_1, 16384.0)
                        grouped_values_decoder_1 = pl.reshape(unscaled_values_decoder_1, [PARTITION_MICRO_ROWS * 4, 32])
                        scale_bytes_decoder_1 = pl.load(scales_w, [micro_base_decoder_1 // 8, 0], [PARTITION_MICRO_ROWS // 8, 32])
                        scale_codes_decoder_1 = pl.reshape(pl.cast(pl.cast(scale_bytes_decoder_1, pl.FP16), pl.INT32), [1, PARTITION_MICRO_ROWS * 4])
                        scale_normal_decoder_1 = pl.tile.reinterpret_view(pl.shls(scale_codes_decoder_1, 23), pl.FP32)
                        flat_scales_decoder_1 = pl.tile.select(pl.tile.cmps(scale_codes_decoder_1, 0, cmp_type=0), 5.877471754111438e-39, scale_normal_decoder_1)
                        group_scales_decoder_1 = pl.reshape(flat_scales_decoder_1, [PARTITION_MICRO_ROWS * 4, 1])
                        scaled_values_decoder_1 = pl.row_expand_mul(grouped_values_decoder_1, group_scales_decoder_1)
                        final_values_decoder_1 = pl.reshape(pl.cast(scaled_values_decoder_1, pl.BF16, mode="rint"), [PARTITION_MICRO_ROWS, D])
                        pl.store(final_values_decoder_1, [chunk_idx_decoder_1 * DECODE_ROWS + micro_visit_decoder_1 * PARTITION_MICRO_ROWS, 0], kb2_1)
        with pl.spmd((decode_chunks_2 + DECODE_CHUNKS - 1) // DECODE_CHUNKS, name_hint="partition_decode_2", allow_early_resolve=False) as decode_ready_2:
            decode_task_decoder_2 = pl.tile.get_block_idx()
            for decode_visit_decoder_2 in pl.range(DECODE_CHUNKS):
                chunk_idx_decoder_2 = decode_task_decoder_2 * DECODE_CHUNKS + decode_visit_decoder_2
                if chunk_idx_decoder_2 < decode_chunks_2:
                    local_page_decoder_2 = chunk_idx_decoder_2 // (PAGE // DECODE_ROWS)
                    local_packet_decoder_2 = local_page_decoder_2 // PACKET_PAGES
                    logical_packet_decoder_2 = owner_packet_base_2 + (local_packet_decoder_2 // OWNER_RUN_PACKETS) * PAGE_PARTS * OWNER_RUN_PACKETS + local_packet_decoder_2 % OWNER_RUN_PACKETS
                    logical_page_decoder_2 = logical_packet_decoder_2 * PACKET_PAGES + local_page_decoder_2 % PACKET_PAGES
                    physical_page_decoder_2 = pl.read(block_table, [0, logical_page_decoder_2])
                    physical_chunk_decoder_2 = pl.cast(physical_page_decoder_2, pl.INDEX) * (PAGE // DECODE_ROWS) + chunk_idx_decoder_2 % (PAGE // DECODE_ROWS)
                    for micro_visit_decoder_2 in pl.range(DECODE_ROWS // PARTITION_MICRO_ROWS):
                        micro_base_decoder_2 = physical_chunk_decoder_2 * DECODE_ROWS + micro_visit_decoder_2 * PARTITION_MICRO_ROWS
                        packed_micro_decoder_2 = pl.load(packed2, [micro_base_decoder_2, 0], [PARTITION_MICRO_ROWS, D // 2])
                        byte_indices_decoder_2 = pl.cast(pl.cast(packed_micro_decoder_2, pl.FP16), pl.INT32)
                        pair_low_decoder_2 = pl.tile.and_(byte_indices_decoder_2, pl.tile.full([PARTITION_MICRO_ROWS, D // 2], dtype=pl.INT32, value=15))
                        pair_high_decoder_2 = pl.shls(pl.tile.and_(byte_indices_decoder_2, pl.tile.full([PARTITION_MICRO_ROWS, D // 2], dtype=pl.INT32, value=240)), 12)
                        pair_spread_decoder_2 = pl.or_(pair_low_decoder_2, pair_high_decoder_2)
                        pair_magnitudes_decoder_2 = pl.tile.and_(pair_spread_decoder_2, pl.tile.full([PARTITION_MICRO_ROWS, D // 2], dtype=pl.INT32, value=458759))
                        pair_magnitude_bits_decoder_2 = pl.shls(pair_magnitudes_decoder_2, 9)
                        pair_signs_decoder_2 = pl.shls(pl.tile.and_(pair_spread_decoder_2, pl.tile.full([PARTITION_MICRO_ROWS, D // 2], dtype=pl.INT32, value=524296)), 12)
                        pair_words_decoder_2 = pl.or_(pair_magnitude_bits_decoder_2, pair_signs_decoder_2)
                        unscaled_pairs_decoder_2 = pl.tile.reinterpret_view(pair_words_decoder_2, pl.FP16)
                        biased_values_decoder_2 = pl.cast(unscaled_pairs_decoder_2, pl.FP32)
                        unscaled_values_decoder_2 = pl.mul(biased_values_decoder_2, 16384.0)
                        grouped_values_decoder_2 = pl.reshape(unscaled_values_decoder_2, [PARTITION_MICRO_ROWS * 4, 32])
                        scale_bytes_decoder_2 = pl.load(scales_w, [micro_base_decoder_2 // 8, 0], [PARTITION_MICRO_ROWS // 8, 32])
                        scale_codes_decoder_2 = pl.reshape(pl.cast(pl.cast(scale_bytes_decoder_2, pl.FP16), pl.INT32), [1, PARTITION_MICRO_ROWS * 4])
                        scale_normal_decoder_2 = pl.tile.reinterpret_view(pl.shls(scale_codes_decoder_2, 23), pl.FP32)
                        flat_scales_decoder_2 = pl.tile.select(pl.tile.cmps(scale_codes_decoder_2, 0, cmp_type=0), 5.877471754111438e-39, scale_normal_decoder_2)
                        group_scales_decoder_2 = pl.reshape(flat_scales_decoder_2, [PARTITION_MICRO_ROWS * 4, 1])
                        scaled_values_decoder_2 = pl.row_expand_mul(grouped_values_decoder_2, group_scales_decoder_2)
                        final_values_decoder_2 = pl.reshape(pl.cast(scaled_values_decoder_2, pl.BF16, mode="rint"), [PARTITION_MICRO_ROWS, D])
                        pl.store(final_values_decoder_2, [chunk_idx_decoder_2 * DECODE_ROWS + micro_visit_decoder_2 * PARTITION_MICRO_ROWS, 0], kb2_2)
        with pl.spmd(M // (QUERY_GROUP * 2), name_hint="partition_score_0", deps=[decode_ready_0], optimizations=[pl.cross_core_slot(slot_num=3, local_slot_num=1)]) as score_ready_0:
            block_idx_score_0 = pl.tile.get_block_idx()
            super_query_base_score_0 = (block_idx_score_0 % (M // (QUERY_GROUP * 2))) * QUERY_GROUP * 2
            packet_count_score_0 = (P + PACKET_PAGES - 1) // PACKET_PAGES
            packets_per_partition_score_0 = (packet_count_score_0 + PAGE_PARTS - 1) // PAGE_PARTS
            packet_base_score_0 = 0
            packet_end_score_0 = owner_packet_count_0
            resident_queries_score_0 = pl.load(q2, [super_query_base_score_0 * H, 0], [QUERY_GROUP * 2 * H, D], target_memory=pl.Mem.Mat)
            group_visible_0_score_0 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_0 + 0]), 0), N)
            group_visible_1_score_0 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_0 + 1]), 0), N)
            group_visible_2_score_0 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_0 + 2]), 0), N)
            group_visible_3_score_0 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_0 + 3]), 0), N)
            group_visible_4_score_0 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_0 + 4]), 0), N)
            group_visible_5_score_0 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_0 + 5]), 0), N)
            group_visible_6_score_0 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_0 + 6]), 0), N)
            group_visible_7_score_0 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_0 + 7]), 0), N)
            group_max_0_score_0 = pl.max(pl.max(group_visible_0_score_0, group_visible_1_score_0), pl.max(group_visible_2_score_0, group_visible_3_score_0))
            group_max_1_score_0 = pl.max(pl.max(group_visible_4_score_0, group_visible_5_score_0), pl.max(group_visible_6_score_0, group_visible_7_score_0))
            for metadata_lane_score_0 in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                lane_query_base_score_0 = super_query_base_score_0 + metadata_lane_score_0 * LANE_QUERIES
                lane_visible0_score_0 = pl.min(pl.max(pl.read(lengths, [lane_query_base_score_0]), 0), N)
                lane_visible1_score_0 = pl.min(pl.max(pl.read(lengths, [lane_query_base_score_0 + 1]), 0), N)
                lane_visible2_score_0 = pl.min(pl.max(pl.read(lengths, [lane_query_base_score_0 + QUERY_GROUP]), 0), N)
                lane_visible3_score_0 = pl.min(pl.max(pl.read(lengths, [lane_query_base_score_0 + QUERY_GROUP + 1]), 0), N)
                all_weights_score_0 = pl.reshape(pl.load(w, [super_query_base_score_0, 0], [QUERY_GROUP * 2, H]), [QUERY_GROUP * 2 * H, 1])
                fragment_negative_inf_score_0 = pl.reinterpret_view(pl.tile.full([1, 128], dtype=pl.INT32, value=-8388608), pl.FP32)
                fragment_positions_score_0 = pl.cast(pl.tile.arange(0, [1, 128], dtype=pl.INT32), pl.FP32)
            local_packet_count_score_0 = pl.max(packet_end_score_0 - packet_base_score_0, 0)
            packet_phase_score_0 = (block_idx_score_0 % (M // (QUERY_GROUP * 2))) % 2
            packet_offset_score_0 = (local_packet_count_score_0 * packet_phase_score_0) // 2
            for visit_idx_score_0, (cursor_packet_score_0,) in pl.range(local_packet_count_score_0, init_values=(packet_base_score_0 + packet_offset_score_0,)):
                local_packet_score_0 = cursor_packet_score_0
                local_page_score_0 = local_packet_score_0 * PACKET_PAGES
                packet_idx_score_0 = owner_packet_base_0 + (local_packet_score_0 // OWNER_RUN_PACKETS) * PAGE_PARTS * OWNER_RUN_PACKETS + local_packet_score_0 % OWNER_RUN_PACKETS
                page_idx_score_0 = packet_idx_score_0 * PACKET_PAGES
                valid_cols_score_0 = pl.min(COL_TILE, N - page_idx_score_0 * PAGE)
                physical_page_score_0 = pl.read(block_table, [0, page_idx_score_0])
                cache_row_score_0 = local_page_score_0 * PAGE
                key_tile_score_0 = pl.tile.create([COL_TILE, D], pl.BF16, target_memory=pl.Mem.Mat)
                key_tile_score_0 = pl.tile.gather_row(key_tile_score_0, kb2_0, [0, 0], [cache_row_score_0, 0], [PAGE, D])
                second_page_score_0 = pl.min(page_idx_score_0 + 1, P - 1)
                second_physical_score_0 = pl.read(block_table, [0, second_page_score_0])
                second_local_page_score_0 = pl.min(local_page_score_0 + 1, owner_page_count_0 - 1)
                second_row_score_0 = second_local_page_score_0 * PAGE
                key_tile_score_0 = pl.tile.gather_row(key_tile_score_0, kb2_0, [PAGE, 0], [second_row_score_0, 0], [PAGE, D])
                key_right_score_0 = pl.tile.move(pl.tile.transpose_view(key_tile_score_0), target_memory=pl.Mem.Right)
                for subgroup_idx_score_0 in pl.range(2):
                    query_base_score_0 = super_query_base_score_0 + subgroup_idx_score_0 * QUERY_GROUP
                    subgroup_visible_score_0 = group_max_0_score_0 + subgroup_idx_score_0 * (group_max_1_score_0 - group_max_0_score_0)
                    subgroup_packet_end_score_0 = (subgroup_visible_score_0 + COL_TILE - 1) // COL_TILE
                    query_tile_score_0 = pl.tile.slice(resident_queries_score_0, [QUERY_GROUP * H, D], [subgroup_idx_score_0 * QUERY_GROUP * H, 0])
                    grouped_dots_score_0 = pl.matmul(query_tile_score_0, key_right_score_0, out_dtype=pl.FP32)
                    if packet_idx_score_0 < subgroup_packet_end_score_0:
                        for transfer_lane_score_0 in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                            sharded_dots_score_0 = pl.aiv_shard(grouped_dots_score_0)
                        canonical_dots_score_0 = pl.tile.slice(grouped_dots_score_0, [QUERY_GROUP * H, COL_TILE], [0, 0], valid_shape=[QUERY_GROUP * H, valid_cols_score_0])
                        pl.store(canonical_dots_score_0, [query_base_score_0 * H, page_idx_score_0 * PAGE], dots)
                        for vector_lane_score_0 in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                            visible_first_score_0 = lane_visible0_score_0 + subgroup_idx_score_0 * (lane_visible2_score_0 - lane_visible0_score_0)
                            visible_second_score_0 = lane_visible1_score_0 + subgroup_idx_score_0 * (lane_visible3_score_0 - lane_visible1_score_0)
                            grouped_weights_score_0 = pl.tile.slice(all_weights_score_0, [LANE_QUERIES * H, 1], [(subgroup_idx_score_0 * QUERY_GROUP + vector_lane_score_0 * LANE_QUERIES) * H, 0])
                            output_query_base_score_0 = query_base_score_0 + vector_lane_score_0 * LANE_QUERIES
                            query0_score_empty_score_0 = pl.tile.create([1, COL_TILE], pl.FP32, target_memory=pl.Mem.Vec)
                            query1_score_empty_score_0 = pl.tile.create([1, COL_TILE], pl.FP32, target_memory=pl.Mem.Vec)
                            fragment_base_0_score_0 = page_idx_score_0 * PAGE + 0
                            fragment_valid_cols_0_score_0 = pl.max(0, pl.min(128, N - fragment_base_0_score_0))
                            if fragment_valid_cols_0_score_0 > 0:
                                query0_frag0_weights_score_0 = pl.tile.slice(grouped_weights_score_0, [H, 1], [0, 0])
                                query0_frag0_raw_score_0 = pl.tile.slice(sharded_dots_score_0, [H, 128], [0, 0])
                                query0_frag0_activation_score_0 = pl.maximum(query0_frag0_raw_score_0, 0.0)
                                query0_frag0_weighted_score_0 = pl.row_expand_mul(query0_frag0_activation_score_0, query0_frag0_weights_score_0)
                                query1_frag0_weights_score_0 = pl.tile.slice(grouped_weights_score_0, [H, 1], [32, 0])
                                query1_frag0_raw_score_0 = pl.tile.slice(sharded_dots_score_0, [H, 128], [32, 0])
                                query1_frag0_activation_score_0 = pl.maximum(query1_frag0_raw_score_0, 0.0)
                                query1_frag0_weighted_score_0 = pl.row_expand_mul(query1_frag0_activation_score_0, query1_frag0_weights_score_0)
                                query0_frag0_scratch_score_0: pl.Tile[[H // 2, 128], pl.FP32, pl.MemRef("head_tree_scratch"), pl.Mem.Vec] = pl.tile.create([H // 2, 128], pl.FP32, target_memory=pl.Mem.Vec)
                                query0_frag0_reduced_score_0 = pl.tile.col_sum(query0_frag0_weighted_score_0, query0_frag0_scratch_score_0)
                                if visible_first_score_0 >= fragment_base_0_score_0 + fragment_valid_cols_0_score_0:
                                    query0_frag0_masked_score_0 = query0_frag0_reduced_score_0
                                else:
                                    query0_frag0_positions_score_0 = pl.add(fragment_positions_score_0, pl.cast(pl.cast(fragment_base_0_score_0, pl.INT32), pl.FP32))
                                    query0_frag0_masked_score_0 = pl.tile.select(pl.tile.cmps(query0_frag0_positions_score_0, pl.cast(pl.cast(visible_first_score_0, pl.INT32), pl.FP32), cmp_type=2), query0_frag0_reduced_score_0, fragment_negative_inf_score_0)
                                query0_stage_frag0_score_0 = pl.tile.assemble(query0_score_empty_score_0, query0_frag0_masked_score_0, [0, 0])
                                query1_frag0_scratch_score_0: pl.Tile[[H // 2, 128], pl.FP32, pl.MemRef("head_tree_scratch"), pl.Mem.Vec] = pl.tile.create([H // 2, 128], pl.FP32, target_memory=pl.Mem.Vec)
                                query1_frag0_reduced_score_0 = pl.tile.col_sum(query1_frag0_weighted_score_0, query1_frag0_scratch_score_0)
                                if visible_second_score_0 >= fragment_base_0_score_0 + fragment_valid_cols_0_score_0:
                                    query1_frag0_masked_score_0 = query1_frag0_reduced_score_0
                                else:
                                    query1_frag0_positions_score_0 = pl.add(fragment_positions_score_0, pl.cast(pl.cast(fragment_base_0_score_0, pl.INT32), pl.FP32))
                                    query1_frag0_masked_score_0 = pl.tile.select(pl.tile.cmps(query1_frag0_positions_score_0, pl.cast(pl.cast(visible_second_score_0, pl.INT32), pl.FP32), cmp_type=2), query1_frag0_reduced_score_0, fragment_negative_inf_score_0)
                                query1_stage_frag0_score_0 = pl.tile.assemble(query1_score_empty_score_0, query1_frag0_masked_score_0, [0, 0])
                            else:
                                query0_stage_frag0_score_0 = query0_score_empty_score_0
                                query1_stage_frag0_score_0 = query1_score_empty_score_0
                            fragment_base_1_score_0 = page_idx_score_0 * PAGE + 128
                            fragment_valid_cols_1_score_0 = pl.max(0, pl.min(128, N - fragment_base_1_score_0))
                            if fragment_valid_cols_1_score_0 > 0:
                                query0_frag1_weights_score_0 = pl.tile.slice(grouped_weights_score_0, [H, 1], [0, 0])
                                query0_frag1_raw_score_0 = pl.tile.slice(sharded_dots_score_0, [H, 128], [0, 128])
                                query0_frag1_activation_score_0 = pl.maximum(query0_frag1_raw_score_0, 0.0)
                                query0_frag1_weighted_score_0 = pl.row_expand_mul(query0_frag1_activation_score_0, query0_frag1_weights_score_0)
                                query1_frag1_weights_score_0 = pl.tile.slice(grouped_weights_score_0, [H, 1], [32, 0])
                                query1_frag1_raw_score_0 = pl.tile.slice(sharded_dots_score_0, [H, 128], [32, 128])
                                query1_frag1_activation_score_0 = pl.maximum(query1_frag1_raw_score_0, 0.0)
                                query1_frag1_weighted_score_0 = pl.row_expand_mul(query1_frag1_activation_score_0, query1_frag1_weights_score_0)
                                query0_frag1_scratch_score_0: pl.Tile[[H // 2, 128], pl.FP32, pl.MemRef("head_tree_scratch"), pl.Mem.Vec] = pl.tile.create([H // 2, 128], pl.FP32, target_memory=pl.Mem.Vec)
                                query0_frag1_reduced_score_0 = pl.tile.col_sum(query0_frag1_weighted_score_0, query0_frag1_scratch_score_0)
                                if visible_first_score_0 >= fragment_base_1_score_0 + fragment_valid_cols_1_score_0:
                                    query0_frag1_masked_score_0 = query0_frag1_reduced_score_0
                                else:
                                    query0_frag1_positions_score_0 = pl.add(fragment_positions_score_0, pl.cast(pl.cast(fragment_base_1_score_0, pl.INT32), pl.FP32))
                                    query0_frag1_masked_score_0 = pl.tile.select(pl.tile.cmps(query0_frag1_positions_score_0, pl.cast(pl.cast(visible_first_score_0, pl.INT32), pl.FP32), cmp_type=2), query0_frag1_reduced_score_0, fragment_negative_inf_score_0)
                                query0_stage_frag1_score_0 = pl.tile.assemble(query0_stage_frag0_score_0, query0_frag1_masked_score_0, [0, 128])
                                query1_frag1_scratch_score_0: pl.Tile[[H // 2, 128], pl.FP32, pl.MemRef("head_tree_scratch"), pl.Mem.Vec] = pl.tile.create([H // 2, 128], pl.FP32, target_memory=pl.Mem.Vec)
                                query1_frag1_reduced_score_0 = pl.tile.col_sum(query1_frag1_weighted_score_0, query1_frag1_scratch_score_0)
                                if visible_second_score_0 >= fragment_base_1_score_0 + fragment_valid_cols_1_score_0:
                                    query1_frag1_masked_score_0 = query1_frag1_reduced_score_0
                                else:
                                    query1_frag1_positions_score_0 = pl.add(fragment_positions_score_0, pl.cast(pl.cast(fragment_base_1_score_0, pl.INT32), pl.FP32))
                                    query1_frag1_masked_score_0 = pl.tile.select(pl.tile.cmps(query1_frag1_positions_score_0, pl.cast(pl.cast(visible_second_score_0, pl.INT32), pl.FP32), cmp_type=2), query1_frag1_reduced_score_0, fragment_negative_inf_score_0)
                                query1_stage_frag1_score_0 = pl.tile.assemble(query1_stage_frag0_score_0, query1_frag1_masked_score_0, [0, 128])
                            else:
                                query0_stage_frag1_score_0 = query0_stage_frag0_score_0
                                query1_stage_frag1_score_0 = query1_stage_frag0_score_0
                            query0_score_stage_score_0 = query0_stage_frag1_score_0
                            query1_score_stage_score_0 = query1_stage_frag1_score_0
                            query0_output_score_0 = pl.tile.slice(query0_score_stage_score_0, [1, COL_TILE], [0, 0], valid_shape=[1, valid_cols_score_0])
                            pl.store(query0_output_score_0, [output_query_base_score_0 + 0, page_idx_score_0 * PAGE], score)
                            query1_output_score_0 = pl.tile.slice(query1_score_stage_score_0, [1, COL_TILE], [0, 0], valid_shape=[1, valid_cols_score_0])
                            pl.store(query1_output_score_0, [output_query_base_score_0 + 1, page_idx_score_0 * PAGE], score)
                    else:
                        canonical_dots_score_0 = pl.tile.slice(grouped_dots_score_0, [QUERY_GROUP * H, COL_TILE], [0, 0], valid_shape=[QUERY_GROUP * H, valid_cols_score_0])
                        pl.store(canonical_dots_score_0, [query_base_score_0 * H, page_idx_score_0 * PAGE], dots)
                        for tail_vector_lane_score_0 in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                            tail_query_base_score_0 = query_base_score_0 + tail_vector_lane_score_0 * LANE_QUERIES
                            tail_negative_inf_score_0 = pl.reinterpret_view(pl.tile.full([1, COL_TILE], dtype=pl.INT32, value=-8388608), pl.FP32)
                            tail_score_score_0 = pl.tile.slice(tail_negative_inf_score_0, [1, COL_TILE], [0, 0], valid_shape=[1, valid_cols_score_0])
                            for tail_local_query_score_0 in pl.range(LANE_QUERIES):
                                pl.store(tail_score_score_0, [tail_query_base_score_0 + tail_local_query_score_0, page_idx_score_0 * PAGE], score)
                cursor_packet_score_0 = local_packet_score_0 + 1
                if cursor_packet_score_0 >= packet_end_score_0:
                    cursor_packet_score_0 = packet_base_score_0
                cursor_packet_score_0 = pl.yield_(cursor_packet_score_0)
        with pl.spmd(M // (QUERY_GROUP * 2), name_hint="partition_score_1", deps=[decode_ready_1], optimizations=[pl.cross_core_slot(slot_num=3, local_slot_num=1)]) as score_ready_1:
            block_idx_score_1 = pl.tile.get_block_idx()
            super_query_base_score_1 = (block_idx_score_1 % (M // (QUERY_GROUP * 2))) * QUERY_GROUP * 2
            packet_count_score_1 = (P + PACKET_PAGES - 1) // PACKET_PAGES
            packets_per_partition_score_1 = (packet_count_score_1 + PAGE_PARTS - 1) // PAGE_PARTS
            packet_base_score_1 = 0
            packet_end_score_1 = owner_packet_count_1
            resident_queries_score_1 = pl.load(q2, [super_query_base_score_1 * H, 0], [QUERY_GROUP * 2 * H, D], target_memory=pl.Mem.Mat)
            group_visible_0_score_1 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_1 + 0]), 0), N)
            group_visible_1_score_1 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_1 + 1]), 0), N)
            group_visible_2_score_1 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_1 + 2]), 0), N)
            group_visible_3_score_1 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_1 + 3]), 0), N)
            group_visible_4_score_1 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_1 + 4]), 0), N)
            group_visible_5_score_1 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_1 + 5]), 0), N)
            group_visible_6_score_1 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_1 + 6]), 0), N)
            group_visible_7_score_1 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_1 + 7]), 0), N)
            group_max_0_score_1 = pl.max(pl.max(group_visible_0_score_1, group_visible_1_score_1), pl.max(group_visible_2_score_1, group_visible_3_score_1))
            group_max_1_score_1 = pl.max(pl.max(group_visible_4_score_1, group_visible_5_score_1), pl.max(group_visible_6_score_1, group_visible_7_score_1))
            for metadata_lane_score_1 in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                lane_query_base_score_1 = super_query_base_score_1 + metadata_lane_score_1 * LANE_QUERIES
                lane_visible0_score_1 = pl.min(pl.max(pl.read(lengths, [lane_query_base_score_1]), 0), N)
                lane_visible1_score_1 = pl.min(pl.max(pl.read(lengths, [lane_query_base_score_1 + 1]), 0), N)
                lane_visible2_score_1 = pl.min(pl.max(pl.read(lengths, [lane_query_base_score_1 + QUERY_GROUP]), 0), N)
                lane_visible3_score_1 = pl.min(pl.max(pl.read(lengths, [lane_query_base_score_1 + QUERY_GROUP + 1]), 0), N)
                all_weights_score_1 = pl.reshape(pl.load(w, [super_query_base_score_1, 0], [QUERY_GROUP * 2, H]), [QUERY_GROUP * 2 * H, 1])
                fragment_negative_inf_score_1 = pl.reinterpret_view(pl.tile.full([1, 128], dtype=pl.INT32, value=-8388608), pl.FP32)
                fragment_positions_score_1 = pl.cast(pl.tile.arange(0, [1, 128], dtype=pl.INT32), pl.FP32)
            local_packet_count_score_1 = pl.max(packet_end_score_1 - packet_base_score_1, 0)
            packet_phase_score_1 = (block_idx_score_1 % (M // (QUERY_GROUP * 2))) % 2
            packet_offset_score_1 = (local_packet_count_score_1 * packet_phase_score_1) // 2
            for visit_idx_score_1, (cursor_packet_score_1,) in pl.range(local_packet_count_score_1, init_values=(packet_base_score_1 + packet_offset_score_1,)):
                local_packet_score_1 = cursor_packet_score_1
                local_page_score_1 = local_packet_score_1 * PACKET_PAGES
                packet_idx_score_1 = owner_packet_base_1 + (local_packet_score_1 // OWNER_RUN_PACKETS) * PAGE_PARTS * OWNER_RUN_PACKETS + local_packet_score_1 % OWNER_RUN_PACKETS
                page_idx_score_1 = packet_idx_score_1 * PACKET_PAGES
                valid_cols_score_1 = pl.min(COL_TILE, N - page_idx_score_1 * PAGE)
                physical_page_score_1 = pl.read(block_table, [0, page_idx_score_1])
                cache_row_score_1 = local_page_score_1 * PAGE
                key_tile_score_1 = pl.tile.create([COL_TILE, D], pl.BF16, target_memory=pl.Mem.Mat)
                key_tile_score_1 = pl.tile.gather_row(key_tile_score_1, kb2_1, [0, 0], [cache_row_score_1, 0], [PAGE, D])
                second_page_score_1 = pl.min(page_idx_score_1 + 1, P - 1)
                second_physical_score_1 = pl.read(block_table, [0, second_page_score_1])
                second_local_page_score_1 = pl.min(local_page_score_1 + 1, owner_page_count_1 - 1)
                second_row_score_1 = second_local_page_score_1 * PAGE
                key_tile_score_1 = pl.tile.gather_row(key_tile_score_1, kb2_1, [PAGE, 0], [second_row_score_1, 0], [PAGE, D])
                key_right_score_1 = pl.tile.move(pl.tile.transpose_view(key_tile_score_1), target_memory=pl.Mem.Right)
                for subgroup_idx_score_1 in pl.range(2):
                    query_base_score_1 = super_query_base_score_1 + subgroup_idx_score_1 * QUERY_GROUP
                    subgroup_visible_score_1 = group_max_0_score_1 + subgroup_idx_score_1 * (group_max_1_score_1 - group_max_0_score_1)
                    subgroup_packet_end_score_1 = (subgroup_visible_score_1 + COL_TILE - 1) // COL_TILE
                    query_tile_score_1 = pl.tile.slice(resident_queries_score_1, [QUERY_GROUP * H, D], [subgroup_idx_score_1 * QUERY_GROUP * H, 0])
                    grouped_dots_score_1 = pl.matmul(query_tile_score_1, key_right_score_1, out_dtype=pl.FP32)
                    if packet_idx_score_1 < subgroup_packet_end_score_1:
                        for transfer_lane_score_1 in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                            sharded_dots_score_1 = pl.aiv_shard(grouped_dots_score_1)
                        canonical_dots_score_1 = pl.tile.slice(grouped_dots_score_1, [QUERY_GROUP * H, COL_TILE], [0, 0], valid_shape=[QUERY_GROUP * H, valid_cols_score_1])
                        pl.store(canonical_dots_score_1, [query_base_score_1 * H, page_idx_score_1 * PAGE], dots)
                        for vector_lane_score_1 in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                            visible_first_score_1 = lane_visible0_score_1 + subgroup_idx_score_1 * (lane_visible2_score_1 - lane_visible0_score_1)
                            visible_second_score_1 = lane_visible1_score_1 + subgroup_idx_score_1 * (lane_visible3_score_1 - lane_visible1_score_1)
                            grouped_weights_score_1 = pl.tile.slice(all_weights_score_1, [LANE_QUERIES * H, 1], [(subgroup_idx_score_1 * QUERY_GROUP + vector_lane_score_1 * LANE_QUERIES) * H, 0])
                            output_query_base_score_1 = query_base_score_1 + vector_lane_score_1 * LANE_QUERIES
                            query0_score_empty_score_1 = pl.tile.create([1, COL_TILE], pl.FP32, target_memory=pl.Mem.Vec)
                            query1_score_empty_score_1 = pl.tile.create([1, COL_TILE], pl.FP32, target_memory=pl.Mem.Vec)
                            fragment_base_0_score_1 = page_idx_score_1 * PAGE + 0
                            fragment_valid_cols_0_score_1 = pl.max(0, pl.min(128, N - fragment_base_0_score_1))
                            if fragment_valid_cols_0_score_1 > 0:
                                query0_frag0_weights_score_1 = pl.tile.slice(grouped_weights_score_1, [H, 1], [0, 0])
                                query0_frag0_raw_score_1 = pl.tile.slice(sharded_dots_score_1, [H, 128], [0, 0])
                                query0_frag0_activation_score_1 = pl.maximum(query0_frag0_raw_score_1, 0.0)
                                query0_frag0_weighted_score_1 = pl.row_expand_mul(query0_frag0_activation_score_1, query0_frag0_weights_score_1)
                                query1_frag0_weights_score_1 = pl.tile.slice(grouped_weights_score_1, [H, 1], [32, 0])
                                query1_frag0_raw_score_1 = pl.tile.slice(sharded_dots_score_1, [H, 128], [32, 0])
                                query1_frag0_activation_score_1 = pl.maximum(query1_frag0_raw_score_1, 0.0)
                                query1_frag0_weighted_score_1 = pl.row_expand_mul(query1_frag0_activation_score_1, query1_frag0_weights_score_1)
                                query0_frag0_scratch_score_1: pl.Tile[[H // 2, 128], pl.FP32, pl.MemRef("head_tree_scratch"), pl.Mem.Vec] = pl.tile.create([H // 2, 128], pl.FP32, target_memory=pl.Mem.Vec)
                                query0_frag0_reduced_score_1 = pl.tile.col_sum(query0_frag0_weighted_score_1, query0_frag0_scratch_score_1)
                                if visible_first_score_1 >= fragment_base_0_score_1 + fragment_valid_cols_0_score_1:
                                    query0_frag0_masked_score_1 = query0_frag0_reduced_score_1
                                else:
                                    query0_frag0_positions_score_1 = pl.add(fragment_positions_score_1, pl.cast(pl.cast(fragment_base_0_score_1, pl.INT32), pl.FP32))
                                    query0_frag0_masked_score_1 = pl.tile.select(pl.tile.cmps(query0_frag0_positions_score_1, pl.cast(pl.cast(visible_first_score_1, pl.INT32), pl.FP32), cmp_type=2), query0_frag0_reduced_score_1, fragment_negative_inf_score_1)
                                query0_stage_frag0_score_1 = pl.tile.assemble(query0_score_empty_score_1, query0_frag0_masked_score_1, [0, 0])
                                query1_frag0_scratch_score_1: pl.Tile[[H // 2, 128], pl.FP32, pl.MemRef("head_tree_scratch"), pl.Mem.Vec] = pl.tile.create([H // 2, 128], pl.FP32, target_memory=pl.Mem.Vec)
                                query1_frag0_reduced_score_1 = pl.tile.col_sum(query1_frag0_weighted_score_1, query1_frag0_scratch_score_1)
                                if visible_second_score_1 >= fragment_base_0_score_1 + fragment_valid_cols_0_score_1:
                                    query1_frag0_masked_score_1 = query1_frag0_reduced_score_1
                                else:
                                    query1_frag0_positions_score_1 = pl.add(fragment_positions_score_1, pl.cast(pl.cast(fragment_base_0_score_1, pl.INT32), pl.FP32))
                                    query1_frag0_masked_score_1 = pl.tile.select(pl.tile.cmps(query1_frag0_positions_score_1, pl.cast(pl.cast(visible_second_score_1, pl.INT32), pl.FP32), cmp_type=2), query1_frag0_reduced_score_1, fragment_negative_inf_score_1)
                                query1_stage_frag0_score_1 = pl.tile.assemble(query1_score_empty_score_1, query1_frag0_masked_score_1, [0, 0])
                            else:
                                query0_stage_frag0_score_1 = query0_score_empty_score_1
                                query1_stage_frag0_score_1 = query1_score_empty_score_1
                            fragment_base_1_score_1 = page_idx_score_1 * PAGE + 128
                            fragment_valid_cols_1_score_1 = pl.max(0, pl.min(128, N - fragment_base_1_score_1))
                            if fragment_valid_cols_1_score_1 > 0:
                                query0_frag1_weights_score_1 = pl.tile.slice(grouped_weights_score_1, [H, 1], [0, 0])
                                query0_frag1_raw_score_1 = pl.tile.slice(sharded_dots_score_1, [H, 128], [0, 128])
                                query0_frag1_activation_score_1 = pl.maximum(query0_frag1_raw_score_1, 0.0)
                                query0_frag1_weighted_score_1 = pl.row_expand_mul(query0_frag1_activation_score_1, query0_frag1_weights_score_1)
                                query1_frag1_weights_score_1 = pl.tile.slice(grouped_weights_score_1, [H, 1], [32, 0])
                                query1_frag1_raw_score_1 = pl.tile.slice(sharded_dots_score_1, [H, 128], [32, 128])
                                query1_frag1_activation_score_1 = pl.maximum(query1_frag1_raw_score_1, 0.0)
                                query1_frag1_weighted_score_1 = pl.row_expand_mul(query1_frag1_activation_score_1, query1_frag1_weights_score_1)
                                query0_frag1_scratch_score_1: pl.Tile[[H // 2, 128], pl.FP32, pl.MemRef("head_tree_scratch"), pl.Mem.Vec] = pl.tile.create([H // 2, 128], pl.FP32, target_memory=pl.Mem.Vec)
                                query0_frag1_reduced_score_1 = pl.tile.col_sum(query0_frag1_weighted_score_1, query0_frag1_scratch_score_1)
                                if visible_first_score_1 >= fragment_base_1_score_1 + fragment_valid_cols_1_score_1:
                                    query0_frag1_masked_score_1 = query0_frag1_reduced_score_1
                                else:
                                    query0_frag1_positions_score_1 = pl.add(fragment_positions_score_1, pl.cast(pl.cast(fragment_base_1_score_1, pl.INT32), pl.FP32))
                                    query0_frag1_masked_score_1 = pl.tile.select(pl.tile.cmps(query0_frag1_positions_score_1, pl.cast(pl.cast(visible_first_score_1, pl.INT32), pl.FP32), cmp_type=2), query0_frag1_reduced_score_1, fragment_negative_inf_score_1)
                                query0_stage_frag1_score_1 = pl.tile.assemble(query0_stage_frag0_score_1, query0_frag1_masked_score_1, [0, 128])
                                query1_frag1_scratch_score_1: pl.Tile[[H // 2, 128], pl.FP32, pl.MemRef("head_tree_scratch"), pl.Mem.Vec] = pl.tile.create([H // 2, 128], pl.FP32, target_memory=pl.Mem.Vec)
                                query1_frag1_reduced_score_1 = pl.tile.col_sum(query1_frag1_weighted_score_1, query1_frag1_scratch_score_1)
                                if visible_second_score_1 >= fragment_base_1_score_1 + fragment_valid_cols_1_score_1:
                                    query1_frag1_masked_score_1 = query1_frag1_reduced_score_1
                                else:
                                    query1_frag1_positions_score_1 = pl.add(fragment_positions_score_1, pl.cast(pl.cast(fragment_base_1_score_1, pl.INT32), pl.FP32))
                                    query1_frag1_masked_score_1 = pl.tile.select(pl.tile.cmps(query1_frag1_positions_score_1, pl.cast(pl.cast(visible_second_score_1, pl.INT32), pl.FP32), cmp_type=2), query1_frag1_reduced_score_1, fragment_negative_inf_score_1)
                                query1_stage_frag1_score_1 = pl.tile.assemble(query1_stage_frag0_score_1, query1_frag1_masked_score_1, [0, 128])
                            else:
                                query0_stage_frag1_score_1 = query0_stage_frag0_score_1
                                query1_stage_frag1_score_1 = query1_stage_frag0_score_1
                            query0_score_stage_score_1 = query0_stage_frag1_score_1
                            query1_score_stage_score_1 = query1_stage_frag1_score_1
                            query0_output_score_1 = pl.tile.slice(query0_score_stage_score_1, [1, COL_TILE], [0, 0], valid_shape=[1, valid_cols_score_1])
                            pl.store(query0_output_score_1, [output_query_base_score_1 + 0, page_idx_score_1 * PAGE], score)
                            query1_output_score_1 = pl.tile.slice(query1_score_stage_score_1, [1, COL_TILE], [0, 0], valid_shape=[1, valid_cols_score_1])
                            pl.store(query1_output_score_1, [output_query_base_score_1 + 1, page_idx_score_1 * PAGE], score)
                    else:
                        canonical_dots_score_1 = pl.tile.slice(grouped_dots_score_1, [QUERY_GROUP * H, COL_TILE], [0, 0], valid_shape=[QUERY_GROUP * H, valid_cols_score_1])
                        pl.store(canonical_dots_score_1, [query_base_score_1 * H, page_idx_score_1 * PAGE], dots)
                        for tail_vector_lane_score_1 in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                            tail_query_base_score_1 = query_base_score_1 + tail_vector_lane_score_1 * LANE_QUERIES
                            tail_negative_inf_score_1 = pl.reinterpret_view(pl.tile.full([1, COL_TILE], dtype=pl.INT32, value=-8388608), pl.FP32)
                            tail_score_score_1 = pl.tile.slice(tail_negative_inf_score_1, [1, COL_TILE], [0, 0], valid_shape=[1, valid_cols_score_1])
                            for tail_local_query_score_1 in pl.range(LANE_QUERIES):
                                pl.store(tail_score_score_1, [tail_query_base_score_1 + tail_local_query_score_1, page_idx_score_1 * PAGE], score)
                cursor_packet_score_1 = local_packet_score_1 + 1
                if cursor_packet_score_1 >= packet_end_score_1:
                    cursor_packet_score_1 = packet_base_score_1
                cursor_packet_score_1 = pl.yield_(cursor_packet_score_1)
        with pl.spmd(M // (QUERY_GROUP * 2), name_hint="partition_score_2", deps=[decode_ready_2], optimizations=[pl.cross_core_slot(slot_num=3, local_slot_num=1)]) as score_ready_2:
            block_idx_score_2 = pl.tile.get_block_idx()
            super_query_base_score_2 = (block_idx_score_2 % (M // (QUERY_GROUP * 2))) * QUERY_GROUP * 2
            packet_count_score_2 = (P + PACKET_PAGES - 1) // PACKET_PAGES
            packets_per_partition_score_2 = (packet_count_score_2 + PAGE_PARTS - 1) // PAGE_PARTS
            packet_base_score_2 = 0
            packet_end_score_2 = owner_packet_count_2
            resident_queries_score_2 = pl.load(q2, [super_query_base_score_2 * H, 0], [QUERY_GROUP * 2 * H, D], target_memory=pl.Mem.Mat)
            group_visible_0_score_2 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_2 + 0]), 0), N)
            group_visible_1_score_2 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_2 + 1]), 0), N)
            group_visible_2_score_2 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_2 + 2]), 0), N)
            group_visible_3_score_2 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_2 + 3]), 0), N)
            group_visible_4_score_2 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_2 + 4]), 0), N)
            group_visible_5_score_2 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_2 + 5]), 0), N)
            group_visible_6_score_2 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_2 + 6]), 0), N)
            group_visible_7_score_2 = pl.min(pl.max(pl.read(lengths, [super_query_base_score_2 + 7]), 0), N)
            group_max_0_score_2 = pl.max(pl.max(group_visible_0_score_2, group_visible_1_score_2), pl.max(group_visible_2_score_2, group_visible_3_score_2))
            group_max_1_score_2 = pl.max(pl.max(group_visible_4_score_2, group_visible_5_score_2), pl.max(group_visible_6_score_2, group_visible_7_score_2))
            for metadata_lane_score_2 in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                lane_query_base_score_2 = super_query_base_score_2 + metadata_lane_score_2 * LANE_QUERIES
                lane_visible0_score_2 = pl.min(pl.max(pl.read(lengths, [lane_query_base_score_2]), 0), N)
                lane_visible1_score_2 = pl.min(pl.max(pl.read(lengths, [lane_query_base_score_2 + 1]), 0), N)
                lane_visible2_score_2 = pl.min(pl.max(pl.read(lengths, [lane_query_base_score_2 + QUERY_GROUP]), 0), N)
                lane_visible3_score_2 = pl.min(pl.max(pl.read(lengths, [lane_query_base_score_2 + QUERY_GROUP + 1]), 0), N)
                all_weights_score_2 = pl.reshape(pl.load(w, [super_query_base_score_2, 0], [QUERY_GROUP * 2, H]), [QUERY_GROUP * 2 * H, 1])
                fragment_negative_inf_score_2 = pl.reinterpret_view(pl.tile.full([1, 128], dtype=pl.INT32, value=-8388608), pl.FP32)
                fragment_positions_score_2 = pl.cast(pl.tile.arange(0, [1, 128], dtype=pl.INT32), pl.FP32)
            local_packet_count_score_2 = pl.max(packet_end_score_2 - packet_base_score_2, 0)
            packet_phase_score_2 = (block_idx_score_2 % (M // (QUERY_GROUP * 2))) % 2
            packet_offset_score_2 = (local_packet_count_score_2 * packet_phase_score_2) // 2
            for visit_idx_score_2, (cursor_packet_score_2,) in pl.range(local_packet_count_score_2, init_values=(packet_base_score_2 + packet_offset_score_2,)):
                local_packet_score_2 = cursor_packet_score_2
                local_page_score_2 = local_packet_score_2 * PACKET_PAGES
                packet_idx_score_2 = owner_packet_base_2 + (local_packet_score_2 // OWNER_RUN_PACKETS) * PAGE_PARTS * OWNER_RUN_PACKETS + local_packet_score_2 % OWNER_RUN_PACKETS
                page_idx_score_2 = packet_idx_score_2 * PACKET_PAGES
                valid_cols_score_2 = pl.min(COL_TILE, N - page_idx_score_2 * PAGE)
                physical_page_score_2 = pl.read(block_table, [0, page_idx_score_2])
                cache_row_score_2 = local_page_score_2 * PAGE
                key_tile_score_2 = pl.tile.create([COL_TILE, D], pl.BF16, target_memory=pl.Mem.Mat)
                key_tile_score_2 = pl.tile.gather_row(key_tile_score_2, kb2_2, [0, 0], [cache_row_score_2, 0], [PAGE, D])
                second_page_score_2 = pl.min(page_idx_score_2 + 1, P - 1)
                second_physical_score_2 = pl.read(block_table, [0, second_page_score_2])
                second_local_page_score_2 = pl.min(local_page_score_2 + 1, owner_page_count_2 - 1)
                second_row_score_2 = second_local_page_score_2 * PAGE
                key_tile_score_2 = pl.tile.gather_row(key_tile_score_2, kb2_2, [PAGE, 0], [second_row_score_2, 0], [PAGE, D])
                key_right_score_2 = pl.tile.move(pl.tile.transpose_view(key_tile_score_2), target_memory=pl.Mem.Right)
                for subgroup_idx_score_2 in pl.range(2):
                    query_base_score_2 = super_query_base_score_2 + subgroup_idx_score_2 * QUERY_GROUP
                    subgroup_visible_score_2 = group_max_0_score_2 + subgroup_idx_score_2 * (group_max_1_score_2 - group_max_0_score_2)
                    subgroup_packet_end_score_2 = (subgroup_visible_score_2 + COL_TILE - 1) // COL_TILE
                    query_tile_score_2 = pl.tile.slice(resident_queries_score_2, [QUERY_GROUP * H, D], [subgroup_idx_score_2 * QUERY_GROUP * H, 0])
                    grouped_dots_score_2 = pl.matmul(query_tile_score_2, key_right_score_2, out_dtype=pl.FP32)
                    if packet_idx_score_2 < subgroup_packet_end_score_2:
                        for transfer_lane_score_2 in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                            sharded_dots_score_2 = pl.aiv_shard(grouped_dots_score_2)
                        canonical_dots_score_2 = pl.tile.slice(grouped_dots_score_2, [QUERY_GROUP * H, COL_TILE], [0, 0], valid_shape=[QUERY_GROUP * H, valid_cols_score_2])
                        pl.store(canonical_dots_score_2, [query_base_score_2 * H, page_idx_score_2 * PAGE], dots)
                        for vector_lane_score_2 in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                            visible_first_score_2 = lane_visible0_score_2 + subgroup_idx_score_2 * (lane_visible2_score_2 - lane_visible0_score_2)
                            visible_second_score_2 = lane_visible1_score_2 + subgroup_idx_score_2 * (lane_visible3_score_2 - lane_visible1_score_2)
                            grouped_weights_score_2 = pl.tile.slice(all_weights_score_2, [LANE_QUERIES * H, 1], [(subgroup_idx_score_2 * QUERY_GROUP + vector_lane_score_2 * LANE_QUERIES) * H, 0])
                            output_query_base_score_2 = query_base_score_2 + vector_lane_score_2 * LANE_QUERIES
                            query0_score_empty_score_2 = pl.tile.create([1, COL_TILE], pl.FP32, target_memory=pl.Mem.Vec)
                            query1_score_empty_score_2 = pl.tile.create([1, COL_TILE], pl.FP32, target_memory=pl.Mem.Vec)
                            fragment_base_0_score_2 = page_idx_score_2 * PAGE + 0
                            fragment_valid_cols_0_score_2 = pl.max(0, pl.min(128, N - fragment_base_0_score_2))
                            if fragment_valid_cols_0_score_2 > 0:
                                query0_frag0_weights_score_2 = pl.tile.slice(grouped_weights_score_2, [H, 1], [0, 0])
                                query0_frag0_raw_score_2 = pl.tile.slice(sharded_dots_score_2, [H, 128], [0, 0])
                                query0_frag0_activation_score_2 = pl.maximum(query0_frag0_raw_score_2, 0.0)
                                query0_frag0_weighted_score_2 = pl.row_expand_mul(query0_frag0_activation_score_2, query0_frag0_weights_score_2)
                                query1_frag0_weights_score_2 = pl.tile.slice(grouped_weights_score_2, [H, 1], [32, 0])
                                query1_frag0_raw_score_2 = pl.tile.slice(sharded_dots_score_2, [H, 128], [32, 0])
                                query1_frag0_activation_score_2 = pl.maximum(query1_frag0_raw_score_2, 0.0)
                                query1_frag0_weighted_score_2 = pl.row_expand_mul(query1_frag0_activation_score_2, query1_frag0_weights_score_2)
                                query0_frag0_scratch_score_2: pl.Tile[[H // 2, 128], pl.FP32, pl.MemRef("head_tree_scratch"), pl.Mem.Vec] = pl.tile.create([H // 2, 128], pl.FP32, target_memory=pl.Mem.Vec)
                                query0_frag0_reduced_score_2 = pl.tile.col_sum(query0_frag0_weighted_score_2, query0_frag0_scratch_score_2)
                                if visible_first_score_2 >= fragment_base_0_score_2 + fragment_valid_cols_0_score_2:
                                    query0_frag0_masked_score_2 = query0_frag0_reduced_score_2
                                else:
                                    query0_frag0_positions_score_2 = pl.add(fragment_positions_score_2, pl.cast(pl.cast(fragment_base_0_score_2, pl.INT32), pl.FP32))
                                    query0_frag0_masked_score_2 = pl.tile.select(pl.tile.cmps(query0_frag0_positions_score_2, pl.cast(pl.cast(visible_first_score_2, pl.INT32), pl.FP32), cmp_type=2), query0_frag0_reduced_score_2, fragment_negative_inf_score_2)
                                query0_stage_frag0_score_2 = pl.tile.assemble(query0_score_empty_score_2, query0_frag0_masked_score_2, [0, 0])
                                query1_frag0_scratch_score_2: pl.Tile[[H // 2, 128], pl.FP32, pl.MemRef("head_tree_scratch"), pl.Mem.Vec] = pl.tile.create([H // 2, 128], pl.FP32, target_memory=pl.Mem.Vec)
                                query1_frag0_reduced_score_2 = pl.tile.col_sum(query1_frag0_weighted_score_2, query1_frag0_scratch_score_2)
                                if visible_second_score_2 >= fragment_base_0_score_2 + fragment_valid_cols_0_score_2:
                                    query1_frag0_masked_score_2 = query1_frag0_reduced_score_2
                                else:
                                    query1_frag0_positions_score_2 = pl.add(fragment_positions_score_2, pl.cast(pl.cast(fragment_base_0_score_2, pl.INT32), pl.FP32))
                                    query1_frag0_masked_score_2 = pl.tile.select(pl.tile.cmps(query1_frag0_positions_score_2, pl.cast(pl.cast(visible_second_score_2, pl.INT32), pl.FP32), cmp_type=2), query1_frag0_reduced_score_2, fragment_negative_inf_score_2)
                                query1_stage_frag0_score_2 = pl.tile.assemble(query1_score_empty_score_2, query1_frag0_masked_score_2, [0, 0])
                            else:
                                query0_stage_frag0_score_2 = query0_score_empty_score_2
                                query1_stage_frag0_score_2 = query1_score_empty_score_2
                            fragment_base_1_score_2 = page_idx_score_2 * PAGE + 128
                            fragment_valid_cols_1_score_2 = pl.max(0, pl.min(128, N - fragment_base_1_score_2))
                            if fragment_valid_cols_1_score_2 > 0:
                                query0_frag1_weights_score_2 = pl.tile.slice(grouped_weights_score_2, [H, 1], [0, 0])
                                query0_frag1_raw_score_2 = pl.tile.slice(sharded_dots_score_2, [H, 128], [0, 128])
                                query0_frag1_activation_score_2 = pl.maximum(query0_frag1_raw_score_2, 0.0)
                                query0_frag1_weighted_score_2 = pl.row_expand_mul(query0_frag1_activation_score_2, query0_frag1_weights_score_2)
                                query1_frag1_weights_score_2 = pl.tile.slice(grouped_weights_score_2, [H, 1], [32, 0])
                                query1_frag1_raw_score_2 = pl.tile.slice(sharded_dots_score_2, [H, 128], [32, 128])
                                query1_frag1_activation_score_2 = pl.maximum(query1_frag1_raw_score_2, 0.0)
                                query1_frag1_weighted_score_2 = pl.row_expand_mul(query1_frag1_activation_score_2, query1_frag1_weights_score_2)
                                query0_frag1_scratch_score_2: pl.Tile[[H // 2, 128], pl.FP32, pl.MemRef("head_tree_scratch"), pl.Mem.Vec] = pl.tile.create([H // 2, 128], pl.FP32, target_memory=pl.Mem.Vec)
                                query0_frag1_reduced_score_2 = pl.tile.col_sum(query0_frag1_weighted_score_2, query0_frag1_scratch_score_2)
                                if visible_first_score_2 >= fragment_base_1_score_2 + fragment_valid_cols_1_score_2:
                                    query0_frag1_masked_score_2 = query0_frag1_reduced_score_2
                                else:
                                    query0_frag1_positions_score_2 = pl.add(fragment_positions_score_2, pl.cast(pl.cast(fragment_base_1_score_2, pl.INT32), pl.FP32))
                                    query0_frag1_masked_score_2 = pl.tile.select(pl.tile.cmps(query0_frag1_positions_score_2, pl.cast(pl.cast(visible_first_score_2, pl.INT32), pl.FP32), cmp_type=2), query0_frag1_reduced_score_2, fragment_negative_inf_score_2)
                                query0_stage_frag1_score_2 = pl.tile.assemble(query0_stage_frag0_score_2, query0_frag1_masked_score_2, [0, 128])
                                query1_frag1_scratch_score_2: pl.Tile[[H // 2, 128], pl.FP32, pl.MemRef("head_tree_scratch"), pl.Mem.Vec] = pl.tile.create([H // 2, 128], pl.FP32, target_memory=pl.Mem.Vec)
                                query1_frag1_reduced_score_2 = pl.tile.col_sum(query1_frag1_weighted_score_2, query1_frag1_scratch_score_2)
                                if visible_second_score_2 >= fragment_base_1_score_2 + fragment_valid_cols_1_score_2:
                                    query1_frag1_masked_score_2 = query1_frag1_reduced_score_2
                                else:
                                    query1_frag1_positions_score_2 = pl.add(fragment_positions_score_2, pl.cast(pl.cast(fragment_base_1_score_2, pl.INT32), pl.FP32))
                                    query1_frag1_masked_score_2 = pl.tile.select(pl.tile.cmps(query1_frag1_positions_score_2, pl.cast(pl.cast(visible_second_score_2, pl.INT32), pl.FP32), cmp_type=2), query1_frag1_reduced_score_2, fragment_negative_inf_score_2)
                                query1_stage_frag1_score_2 = pl.tile.assemble(query1_stage_frag0_score_2, query1_frag1_masked_score_2, [0, 128])
                            else:
                                query0_stage_frag1_score_2 = query0_stage_frag0_score_2
                                query1_stage_frag1_score_2 = query1_stage_frag0_score_2
                            query0_score_stage_score_2 = query0_stage_frag1_score_2
                            query1_score_stage_score_2 = query1_stage_frag1_score_2
                            query0_output_score_2 = pl.tile.slice(query0_score_stage_score_2, [1, COL_TILE], [0, 0], valid_shape=[1, valid_cols_score_2])
                            pl.store(query0_output_score_2, [output_query_base_score_2 + 0, page_idx_score_2 * PAGE], score)
                            query1_output_score_2 = pl.tile.slice(query1_score_stage_score_2, [1, COL_TILE], [0, 0], valid_shape=[1, valid_cols_score_2])
                            pl.store(query1_output_score_2, [output_query_base_score_2 + 1, page_idx_score_2 * PAGE], score)
                    else:
                        canonical_dots_score_2 = pl.tile.slice(grouped_dots_score_2, [QUERY_GROUP * H, COL_TILE], [0, 0], valid_shape=[QUERY_GROUP * H, valid_cols_score_2])
                        pl.store(canonical_dots_score_2, [query_base_score_2 * H, page_idx_score_2 * PAGE], dots)
                        for tail_vector_lane_score_2 in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                            tail_query_base_score_2 = query_base_score_2 + tail_vector_lane_score_2 * LANE_QUERIES
                            tail_negative_inf_score_2 = pl.reinterpret_view(pl.tile.full([1, COL_TILE], dtype=pl.INT32, value=-8388608), pl.FP32)
                            tail_score_score_2 = pl.tile.slice(tail_negative_inf_score_2, [1, COL_TILE], [0, 0], valid_shape=[1, valid_cols_score_2])
                            for tail_local_query_score_2 in pl.range(LANE_QUERIES):
                                pl.store(tail_score_score_2, [tail_query_base_score_2 + tail_local_query_score_2, page_idx_score_2 * PAGE], score)
                cursor_packet_score_2 = local_packet_score_2 + 1
                if cursor_packet_score_2 >= packet_end_score_2:
                    cursor_packet_score_2 = packet_base_score_2
                cursor_packet_score_2 = pl.yield_(cursor_packet_score_2)
        score_join = pl.system.task_dummy(deps=[score_ready_0, score_ready_1, score_ready_2])
    return score


def build(shape):
    import torch
    from golden import TensorSpec

    M, N = shape["M"], shape["N"]
    P = N // PAGE
    NB = P + 8
    _keys = torch.randn(NB * PAGE, D).bfloat16()
    _packed, _scale_bytes = _quantize_mxfp4_rows(_keys)
    _pidx = torch.tensor(
        [[head * D + 8 * (column % 16) + 2 * (column // 32) + (column // 16) % 2 for column in range(D)]
         for head in range(H)], dtype=torch.int32)
    _sidx = torch.tensor(
        [[128 * (word // 4) + row for word in range(D // 8)] for row in range(PAGE)],
        dtype=torch.int32)

    def specs():
        return [
            TensorSpec("q", [M, H * D], torch.bfloat16, init_value=torch.randn),
            TensorSpec("cache", [NB, PAGE, D // 2], torch.uint8,
                       init_value=_packed.view(NB, PAGE, D // 2)),
            TensorSpec("scales", [NB, PAGE, D // G], torch.uint8,
                       init_value=_scale_bytes.view(NB, PAGE, D // G)),
            TensorSpec(
                "block_table", [1, P], torch.int32,
                init_value=lambda: torch.randperm(NB, dtype=torch.int64)[:P]
                .to(torch.int32).reshape(1, P),
            ),
            TensorSpec(
                "lengths", [M], torch.int32,
                init_value=lambda: torch.randint(1, N + 1, (M,), dtype=torch.int32),
            ),
            TensorSpec(
                "w", [M, H], torch.float32,
                init_value=lambda: torch.randn(M, H) * (D ** -0.5 * H ** -0.5),
            ),
            TensorSpec("pidx", [H, D], torch.int32, init_value=_pidx),
            TensorSpec("sidx", [PAGE, D // 8], torch.int32, init_value=_sidx),
            TensorSpec("dots", [M * H, N], torch.float32),
            TensorSpec("score", [M, N], torch.float32),
        ]

    def golden(tensors):
        decoded = _dequantize_mxfp4_rows(
            tensors["cache"].view(NB * PAGE, D // 2),
            tensors["scales"].view(NB * PAGE, D // G))
        physical_rows = (tensors["block_table"][0].long().unsqueeze(1) * PAGE
                         + torch.arange(PAGE).unsqueeze(0)).reshape(-1)
        logical_keys = decoded[physical_rows]
        queries = tensors["q"].view(M, H, D).float()
        products = torch.einsum("mhd,nd->mhn", queries, logical_keys)
        tensors["dots"][:] = products.reshape(M * H, N)
        scores = (torch.relu(products) * tensors["w"].unsqueeze(-1)).sum(dim=1)
        visible_lengths = tensors["lengths"].long().clamp(0, N)
        mask = torch.arange(N).unsqueeze(0) >= visible_lengths.unsqueeze(1)
        tensors["score"][:] = scores.masked_fill(mask, float("-inf"))

    fn = dsa41_indexer_mxfp4_fused if P < 9 or (P + PACKET_PAGES - 1) // PACKET_PAGES <= 2 * OWNER_RUN_PACKETS else dsa41_indexer_mxfp4_partitioned
    return fn, specs(), golden, RTOL, ATOL
