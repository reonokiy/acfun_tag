# AcFun Word2Vec + BM25

这个仓库现在有两条可直接运行的检索链路：

- `scripts/query_word2vec_terms.py`
  - 从 `title + description` 训练出的 `gensim word2vec` 模型里扩展相关词
- `scripts/duckdb_bm25_search.py`
  - 从 `data/acfun.videoinfo.20260307.full.flattened.parquet` 建 DuckDB BM25 索引
  - 主要索引字段：`tags`、`title`、`description`、`payload_channel_parentName`、`payload_channel_name`

## 1. 环境

项目使用 `pixi`：

```bash
pixi run python --version
```

## 2. 训练好的 Word2Vec

当前模型文件：

- `data/acfun.videoinfo.20260307.full.word2vec.model`
- `data/acfun.videoinfo.20260307.full.word2vec.kv`

查询相关词：

```bash
pixi run python scripts/query_word2vec_terms.py 原神
```

只输出可直接拿去检索的关键词：

```bash
pixi run python scripts/query_word2vec_terms.py 原神 --topn 8 --min-score 0.7 --keywords-only
```

导出给 Neo4j / Bloom 用的关键词关系图：

```bash
pixi run python scripts/export_word2vec_neo4j.py 原神 --topn 8 --min-score 0.7 --depth 1
```

默认会写出：

- `data/neo4j_word2vec/nodes.csv`
- `data/neo4j_word2vec/edges.csv`

如果你想导出一个更大的通用词图，而不是围绕种子词扩展：

```bash
pixi run python scripts/export_word2vec_neo4j.py --max-vocab 5000 --topn 8 --min-score 0.75
```

## 3. 建 BM25 索引

首次需要从 flattened parquet 建一个 DuckDB 库：

```bash
pixi run python scripts/duckdb_bm25_search.py build \
  --input data/acfun.videoinfo.20260307.full.flattened.parquet \
  --db data/acfun.videoinfo.20260307.full.bm25.duckdb \
  --overwrite
```

说明：

- 脚本会先对 `title/description/tags/两个分区` 做中日文分词
- 然后把分词结果写入 DuckDB 表
- 最后用 DuckDB `fts` 扩展创建 BM25 索引

## 4. 做 BM25 检索

直接查：

```bash
pixi run python scripts/duckdb_bm25_search.py query 原神 枫丹 草神 --tokenize-query --top-k 20
```

如果想让所有词都必须命中：

```bash
pixi run python scripts/duckdb_bm25_search.py query 原神 枫丹 草神 \
  --tokenize-query \
  --conjunctive \
  --top-k 20
```

默认字段权重：

- `tags = 4.0`
- `title = 3.0`
- `channel = 2.0`
- `parent_channel = 1.5`
- `description = 1.0`

可以手动调整，例如更强调标题：

```bash
pixi run python scripts/duckdb_bm25_search.py query 原神 枫丹 草神 \
  --tokenize-query \
  --title-weight 5 \
  --tags-weight 4 \
  --description-weight 0.5
```

把 BM25 结果导出成 parquet，给后续 pipeline 用：

```bash
pixi run python scripts/duckdb_bm25_search.py query 原神 \
  --tokenize-query \
  --top-k 2000 \
  --output-parquet data/tmp/yuanshen.bm25.parquet
```

默认导出的 parquet 至少包含：

- `id`
- `rank`
- `query_text`
- `query_terms`
- `total_score`
- `title_score`
- `description_score`
- `tags_score`
- `parent_score`
- `channel_score`

如果你还想把标题、描述、tags、频道这些便于人工查看的字段也一起写进去：

```bash
pixi run python scripts/duckdb_bm25_search.py query 原神 \
  --tokenize-query \
  --top-k 2000 \
  --output-parquet data/tmp/yuanshen.bm25.parquet \
  --include-info-columns
```

说明：

- `parent_score` / `channel_score` 为 `null` 是正常的
- 这表示该视频在 `parent_terms` / `channel_terms` 这两个字段上没有命中查询词
- 总分不会漏算，因为脚本内部在计算 `total_score` 时已经对它们做了 `coalesce(..., 0)`

## 5. 从 id parquet 做综合打分

现在正式 pipeline 是：

1. BM25 查询输出一个 `id parquet`
2. `score` 脚本读取这个 `id parquet`
3. 脚本再去 `data/acfun.videoinfo.20260307.full.flattened.parquet` 回表拿互动指标和时间字段
4. 输出候选打分 parquet 和最终入选 parquet

打分脚本：

- `scripts/score_videos_from_ids.py`

最小输出模式：只保留 `id + 打分字段 + 透传的 BM25 分数字段`

```bash
pixi run python scripts/score_videos_from_ids.py data/tmp/yuanshen.bm25.parquet \
  --top-k 1000 \
  --selection-mode time-quota \
  --time-bucket month \
  --min-per-bucket 1 \
  --output-parquet data/tmp/yuanshen.scored.parquet \
  --selected-output-parquet data/tmp/yuanshen.selected.parquet
```

这时输出 parquet 默认包含：

- `id`
- `score_before_decay`
- `time_decay`
- `score`
- `age_days`
- `time_bucket`
- 如果输入的 `id parquet` 里有这些字段，也会一起透传：
  - `rank`
  - `query_text`
  - `query_terms`
  - `total_score`
  - `title_score`
  - `description_score`
  - `tags_score`
  - `parent_score`
  - `channel_score`

如果你想把标题、作者、发布时间、播放、点赞、评论、banana、弹幕、收藏、分享这些信息列也带上，便于人工查看：

```bash
pixi run python scripts/score_videos_from_ids.py data/tmp/yuanshen.bm25.parquet \
  --top-k 1000 \
  --selection-mode time-quota \
  --time-bucket month \
  --min-per-bucket 1 \
  --include-info-columns \
  --output-parquet data/tmp/yuanshen.scored.with_info.parquet \
  --selected-output-parquet data/tmp/yuanshen.selected.with_info.parquet
```

当前打分默认使用的互动字段：

- `payload_viewCount`
- `payload_likeCount`
- `payload_commentCountRealValue`
- `payload_bananaCount`
- `payload_danmakuCount`
- `payload_stowCount`
- `payload_shareCount`

时间相关逻辑：

- 默认有时间衰减：`score = score_before_decay * time_decay`
- `time_decay = 0.5 ** (age_days / half_life_days)`
- 默认 `half_life_days = 540`

时间分布控制：

- `--selection-mode time-quota`
  - 不是单纯按分数取前 `k`
  - 而是先算分，再按时间桶占比分配名额
- `--time-bucket month`
  - 以月为单位控制时间分布
- `--min-per-bucket 1`
  - 只要某个月在候选集里出现过，就至少给这个月 1 个名额

## 6. 推荐工作流

推荐先用 word2vec 扩词，再把多个词一起丢给 BM25：

```bash
pixi run python scripts/query_word2vec_terms.py 原神 --topn 8 --min-score 0.7 --keywords-only
```

可能得到：

```text
原神 真境 剧诗 草神 枫丹 那维
```

然后用这些词做 BM25：

```bash
pixi run python scripts/duckdb_bm25_search.py query 原神 真境 剧诗 草神 枫丹 那维 \
  --tokenize-query \
  --top-k 50
```

这个流程的核心就是：

1. 用 `word2vec` 从种子词找语义相关词
2. 把相关词并成一组关键词
3. 用 DuckDB BM25 在 `tags/title/description/分区` 上联合检索

## 7. 脚本列表

- `scripts/train_word2vec.py`
  - 训练 `word2vec`
- `scripts/query_word2vec_terms.py`
  - 查询相似词
- `scripts/export_word2vec_neo4j.py`
  - 导出 `Neo4j` / `Bloom` 可导入的关键词节点和相似关系
- `scripts/duckdb_bm25_search.py`
  - `build` 建索引
  - `query` 查 BM25
  - 支持导出 BM25 结果 parquet
- `scripts/score_videos_from_ids.py`
  - 读取 `id parquet`
  - 回表 `flattened parquet`
  - 输出综合打分 parquet 和最终入选 parquet
