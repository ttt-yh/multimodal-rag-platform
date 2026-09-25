# 离线评测数据集

`retrieval_gold_v1.jsonl` 是绑定当前激活索引的开发评测集，共 9 条：8 条可回答问题，1 条无答案问题。每条可回答问题通过 `evidence_chunk_ids` 标注能够支撑答案的 Chunk；无答案问题使用空证据集合验证拒答边界。

当前所有记录都是 `pending_human_review`，`gold_eligible=false`。它们是根据当前索引内容整理的候选标注，不代表已经完成正式人工审核。只有审核者确认问题、参考答案和证据 Chunk 后，才能将记录更新为 `review_status=approved`、`gold_eligible=true`，再生成正式指标。

评测集与索引版本绑定。Chunk 发生重新切分或文档版本变化后，需要重新核对证据，不应直接复用旧标注。工程合成数据 `data/annotations/engineering/qa.jsonl` 仍仅用于回归测试，不作为本数据集的替代品。

`retrieval_stage3_text_gold_v1.jsonl` 是阶段3正式文本检索集，共11条：8条可回答文本题和3条无答案/澄清题，均已完成证据或行为核对并标记 `gold_eligible=true`。它绑定累计索引中的现有Chunk编号，当前正式结果为 Recall@5 0.9375、Precision@5 0.20、MRR 1.00。两条视觉题继续保留在候选文件中，人工金标准发布前不进入正式指标。

`run_stage3_retrieval_eval.py` 只允许对 `review_status=approved` 且 `gold_eligible=true` 的数据生成正式指标。`run_stage4_qa_eval.py` 复用同一正式集验证完整问答链路，只把模型答案中实际出现的 `[n]` 作为已使用引用，计算引用证据召回、引用有效性、完整证据覆盖和无答案拒答。`--only-failed` 只重试失败样本，`--report-only` 不调用模型、仅重新汇总当前索引下的逐题报告。

`qa_stage4_required_facts_v1.jsonl`为8条可回答题提供确定性必要术语组，用于发现LLM评分可能忽略的要点缺失；它不替代语义判断。`qa_stage4_question_revisions_v1.jsonl`保存两条歧义问题的修订记录，两条均已对照原文完成问题修订审核并发布到文本Gold；审核对象是问题与证据一致性，不是对8条模型答案逐条人工评分。`run_stage4_semantic_eval.py`读取既有问答报告，使用参考答案、模型实际引用的证据和必要术语进行开发级评分；参考答案不会输入被测RAG生成链路。

`qa_stage5_visual_facts_v1.jsonl`为两条视觉候选保存必要答案要点和已知数据问题，两条记录均不是正式Gold。`run_stage5_visual_qa_eval.py`只在生成完成后读取答案要点进行确定性核验，不把参考答案或目标图片编号输入检索和VLM。当前开发结果绑定`idx_e3ee3ea7...`，正式发布前仍需明确QPS题的目标截图，并为TiCDC题补充可替代图片证据。
