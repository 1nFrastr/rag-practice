"""Query decomposer module for handling multi-entity and complex queries.

This module decomposes complex queries (like comparisons between multiple entities)
into simpler sub-queries that can be retrieved independently and merged.
"""

from dataclasses import dataclass
from typing import List, Optional
import json
import re

from .query_engine import get_llm
from .tracing import traced_operation, SpanKind


@dataclass
class DecomposedQuery:
    """Result of query decomposition."""
    
    original_query: str
    # Whether the query was decomposed
    is_decomposed: bool
    # List of sub-queries (if decomposed)
    sub_queries: List[str]
    # Type of decomposition (comparison, multi_entity, aggregation, etc.)
    decomposition_type: Optional[str] = None
    # Entities identified in the query
    entities: Optional[List[str]] = None


# Prompt template for query decomposition
DECOMPOSE_PROMPT = """你是一个查询分析助手。分析用户的问题，判断是否需要分解成多个子查询。

用户问题: {question}

判断规则：
1. **需要分解的情况**：
   - 比较类问题：如"A和B的共同点/区别是什么？"
   - 多实体问题：如"A、B、C分别怎么样？"
   - 聚合类问题：如"所有产品的价格是多少？"

2. **不需要分解的情况**：
   - 单一实体的简单问题：如"苹果公司什么时候成立的？"
   - 已经足够具体的问题

请以JSON格式返回分析结果：
{{
    "need_decompose": true/false,
    "type": "comparison" | "multi_entity" | "aggregation" | "simple",
    "entities": ["实体1", "实体2", ...],
    "sub_queries": ["子查询1", "子查询2", ...]
}}

注意事项：
- 如果是比较类问题，为每个实体生成一个独立的子查询，用于检索该实体的相关信息
- 子查询应该是完整的、独立的问题
- 只返回JSON，不要添加其他解释

JSON结果："""


def decompose_query(question: str) -> DecomposedQuery:
    """
    Decompose a complex query into simpler sub-queries.
    
    Args:
        question: The original user question
        
    Returns:
        DecomposedQuery with sub-queries if decomposition is needed
    """
    with traced_operation(
        "query_decompose",
        span_kind=SpanKind.LLM,
        input_value=question,
        attributes={"operation": "decompose"}
    ):
        # First, use a fast heuristic check
        if not _needs_decomposition_heuristic(question):
            return DecomposedQuery(
                original_query=question,
                is_decomposed=False,
                sub_queries=[question],
                decomposition_type="simple"
            )
        
        # Use LLM for complex decomposition
        llm = get_llm()
        prompt = DECOMPOSE_PROMPT.format(question=question)
        
        try:
            response = llm.complete(prompt)
            result_text = str(response).strip()
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                result_text = json_match.group()
            
            result = json.loads(result_text)
            
            need_decompose = result.get("need_decompose", False)
            decomposition_type = result.get("type", "simple")
            entities = result.get("entities", [])
            sub_queries = result.get("sub_queries", [question])
            
            # Validate sub_queries
            if not sub_queries or not isinstance(sub_queries, list):
                sub_queries = [question]
            
            return DecomposedQuery(
                original_query=question,
                is_decomposed=need_decompose and len(sub_queries) > 1,
                sub_queries=sub_queries if need_decompose else [question],
                decomposition_type=decomposition_type,
                entities=entities if entities else None
            )
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Fallback to original query on parsing error
            return DecomposedQuery(
                original_query=question,
                is_decomposed=False,
                sub_queries=[question],
                decomposition_type="simple"
            )


def _needs_decomposition_heuristic(question: str) -> bool:
    """
    Fast heuristic check to determine if decomposition might be needed.
    
    This avoids unnecessary LLM calls for simple queries.
    """
    # Keywords that suggest comparison or multi-entity queries
    comparison_keywords = [
        "和", "与", "跟", "同", "比较", "对比", "区别", "差异", "不同",
        "共同", "相同", "类似", "相似", "一样",
        "分别", "各自", "都", "所有", "每个",
        "之间", "两者", "三者", "它们",
    ]
    
    # Check if any comparison keywords exist
    for keyword in comparison_keywords:
        if keyword in question:
            return True
    
    # Check for enumeration patterns (A、B、C or A，B，C)
    if re.search(r'[\u4e00-\u9fff]+[、，,][\u4e00-\u9fff]+[、，,]?[\u4e00-\u9fff]*', question):
        return True
    
    return False


def merge_retrieval_results(
    all_results: List[List],
    strategy: str = "interleave",
    max_per_query: int = 5
) -> List:
    """
    Merge retrieval results from multiple sub-queries.
    
    Args:
        all_results: List of retrieval results for each sub-query
        strategy: Merge strategy - "interleave" or "rrf"
        max_per_query: Maximum results to take from each sub-query
        
    Returns:
        Merged list of results
    """
    if not all_results:
        return []
    
    if len(all_results) == 1:
        return all_results[0][:max_per_query * 2]
    
    if strategy == "interleave":
        return _interleave_results(all_results, max_per_query)
    elif strategy == "rrf":
        return _rrf_merge_results(all_results, max_per_query)
    else:
        return _interleave_results(all_results, max_per_query)


def _interleave_results(all_results: List[List], max_per_query: int) -> List:
    """
    Interleave results from multiple queries.
    
    Takes results round-robin from each query to ensure diversity.
    """
    merged = []
    seen_ids = set()
    max_iterations = max_per_query
    
    for i in range(max_iterations):
        for results in all_results:
            if i < len(results):
                node = results[i]
                # Get unique identifier (using node_id or hash)
                node_id = node.node.node_id if hasattr(node, 'node') else str(node)
                if node_id not in seen_ids:
                    seen_ids.add(node_id)
                    merged.append(node)
    
    return merged


def _rrf_merge_results(all_results: List[List], max_per_query: int, k: int = 60) -> List:
    """
    Merge results using Reciprocal Rank Fusion.
    
    RRF score = sum(1 / (k + rank)) for each result across all queries.
    """
    from llama_index.core.schema import NodeWithScore
    
    # Build RRF scores
    rrf_scores = {}  # node_id -> (node, score)
    
    for results in all_results:
        for rank, node in enumerate(results[:max_per_query * 2]):
            node_id = node.node.node_id if hasattr(node, 'node') else str(node)
            rrf_score = 1.0 / (k + rank + 1)
            
            if node_id in rrf_scores:
                # Accumulate RRF score
                existing_node, existing_score = rrf_scores[node_id]
                rrf_scores[node_id] = (existing_node, existing_score + rrf_score)
            else:
                rrf_scores[node_id] = (node, rrf_score)
    
    # Sort by RRF score and return
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x[1], reverse=True)
    
    # Create new NodeWithScore with RRF scores
    merged = []
    for node, score in sorted_results:
        if hasattr(node, 'node'):
            new_node = NodeWithScore(node=node.node, score=score)
            merged.append(new_node)
        else:
            merged.append(node)
    
    return merged[:max_per_query * len(all_results)]
