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

## 5. 推荐工作流

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

## 6. 脚本列表

- `scripts/train_word2vec.py`
  - 训练 `word2vec`
- `scripts/query_word2vec_terms.py`
  - 查询相似词
- `scripts/duckdb_bm25_search.py`
  - `build` 建索引
  - `query` 查 BM25
