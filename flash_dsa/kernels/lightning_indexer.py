import pypto.language as pl

H = 32
D = 128
PAGE = 128
COL_TILE = 256
PACKET_PAGES = 2
FRAGMENT_COLS = 256
M = pl.dynamic("M")
N = pl.dynamic("N")
P = pl.dynamic("P")
NB = pl.dynamic("NB")
MH = pl.dynamic("MH")
QUERY_GROUP = 4
PAGE_PARTS = 3
LANE_QUERIES = 2
LANE_COLS = 256


@pl.jit
def grouped_score(
    q: pl.Tensor[[M, H * D], pl.BF16],
    cache: pl.Tensor[[NB, PAGE, D], pl.BF16],
    block_table: pl.Tensor[[1, P], pl.INT32],
    lengths: pl.Tensor[[M], pl.INT32],
    w: pl.Tensor[[M, H], pl.FP32],
    dots: pl.InOut[pl.Tensor[[MH, N], pl.FP32]],
    score: pl.InOut[pl.Tensor[[M, N], pl.FP32]],
):
    cache_flat = pl.reshape(cache, [NB * PAGE, D])
    query_flat = pl.reshape(q, [M * H, D])
    for block_idx in pl.spmd(M // (QUERY_GROUP * 2) * PAGE_PARTS,
            name_hint="persistent_grouped_score",
            optimizations=[pl.cross_core_slot(slot_num=3, local_slot_num=1)]):
        super_query_base = (block_idx % (M // (QUERY_GROUP * 2))) * QUERY_GROUP * 2
        packet_count = (P + PACKET_PAGES - 1) // PACKET_PAGES
        packets_per_partition = (packet_count + PAGE_PARTS - 1) // PAGE_PARTS
        packet_base = (block_idx // (M // (QUERY_GROUP * 2))) * packets_per_partition
        packet_end = pl.min(packet_base + packets_per_partition, packet_count)
        resident_queries = pl.load(query_flat, [super_query_base * H, 0], [QUERY_GROUP * 2 * H, D], target_memory=pl.Mem.Mat)
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
        for packet_idx in pl.range(packet_base, packet_end):
            page_idx = packet_idx * PACKET_PAGES
            valid_cols = pl.min(COL_TILE, N - page_idx * PAGE)
            physical_page = pl.read(block_table, [0, page_idx])
            cache_row = pl.cast(physical_page, pl.INDEX) * PAGE
            key_tile = pl.tile.create([COL_TILE, D], pl.BF16, target_memory=pl.Mem.Mat)
            key_tile = pl.tile.gather_row(key_tile, cache_flat, [0, 0], [cache_row, 0], [PAGE, D])
            second_page = pl.min(page_idx + 1, P - 1)
            second_physical = pl.read(block_table, [0, second_page])
            second_row = pl.cast(second_physical, pl.INDEX) * PAGE
            key_tile = pl.tile.gather_row(key_tile, cache_flat, [PAGE, 0], [second_row, 0], [PAGE, D])
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
                        query0_frag0_reduced = pl.col_sum(query0_frag0_weighted)
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
                        query1_frag0_reduced = pl.col_sum(query1_frag0_weighted)
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
    return score


def build(shape):
    import torch
    from golden import TensorSpec

    M, N = shape["M"], shape["N"]
    P = N // PAGE
    NB = P + 8

    specs = [
        TensorSpec("q", [M, H * D], torch.bfloat16, init_value=torch.randn),
        TensorSpec("cache", [NB, PAGE, D], torch.bfloat16, init_value=torch.randn),
        TensorSpec("block_table", [1, P], torch.int32,
            init_value=lambda: torch.randperm(NB, dtype=torch.int64)[:P]
                .to(torch.int32).reshape(1, P)),
        TensorSpec("lengths", [M], torch.int32,
            init_value=lambda: torch.randint(1, N + 1, (M,), dtype=torch.int32)),
        TensorSpec("w", [M, H], torch.float32,
            init_value=lambda: torch.randn(M, H) * (D ** -0.5 * H ** -0.5)),
        TensorSpec("dots", [M * H, N], torch.float32),
        TensorSpec("score", [M, N], torch.float32),
    ]

    def golden(tensors):
        physical = (tensors["block_table"][0].long().unsqueeze(1) * PAGE
                    + torch.arange(PAGE).unsqueeze(0)).reshape(-1)
        logical_keys = tensors["cache"].reshape(NB * PAGE, D).float()[physical]
        all_dots = torch.einsum("mhd,nd->mhn",
            tensors["q"].reshape(M, H, D).float(), logical_keys)
        tensors["dots"][:] = all_dots.reshape(M * H, N)
        raw_score = (torch.relu(all_dots) * tensors["w"].unsqueeze(-1)).sum(dim=1)
        visible = tensors["lengths"].long().clamp(0, N)
        invalid = torch.arange(N).unsqueeze(0) >= visible.unsqueeze(1)
        tensors["score"][:] = raw_score.masked_fill(invalid, float("-inf"))

    return grouped_score, specs, golden, 2e-2, 2e-2
